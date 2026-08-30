"""Frozen CTS-style and VB-LoRA-style shared code representations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import cbor2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from scipy.optimize import nnls  # type: ignore[import-untyped]

from ratemem.baselines.backbones import BackboneRunner
from ratemem.baselines.ledger import (
    decode_state,
    empty_components,
    export_state,
    ledger_from_export,
)
from ratemem.baselines.protocol import (
    CausalEventView,
    EventReceipt,
    ExactByteLedger,
    FrozenComparisonContract,
    MethodSnapshot,
    ProbeResult,
    validate_operational_event_order,
)
from ratemem.baselines.shared_inputs import SharedInputReader
from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256
from ratemem.evaluation.traces import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.evaluation.types import Sha256

Float32 = NDArray[np.float32]
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class LeakageError(ValueError):
    """Raised before fitting shared state from a non-training corpus."""


def truncated_basis(train_codes: Float32, rank: int) -> Float32:
    if (
        train_codes.ndim != 2
        or train_codes.dtype != np.float32
        or not np.isfinite(train_codes).all()
        or not 0 < rank <= min(train_codes.shape)
    ):
        raise ValueError("invalid static basis shape, values, dtype, or rank")
    _u, _singular_values, vh = np.linalg.svd(
        train_codes.astype(np.float64),
        full_matrices=False,
    )
    basis = vh[:rank].astype(np.float32)
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return basis


def vb_encode_subvector(
    vector: Float32,
    bank: Float32,
    top_k: int,
) -> tuple[tuple[int, ...], Float32]:
    if (
        vector.ndim != 1
        or bank.ndim != 2
        or bank.shape[1] != vector.shape[0]
        or vector.dtype != np.float32
        or bank.dtype != np.float32
        or not np.isfinite(vector).all()
        or not np.isfinite(bank).all()
        or not 0 < top_k <= len(bank)
    ):
        raise ValueError("invalid VB subvector, bank, or top-k")
    target = vector.astype(np.float64)
    residual = target.copy()
    selected: list[int] = []
    for _ in range(top_k):
        scores = bank.astype(np.float64) @ residual
        candidates = [index for index in range(len(scores)) if index not in selected]
        index = min(candidates, key=lambda item: (-abs(float(scores[item])), item))
        selected.append(index)
        design = bank[selected].astype(np.float64).T
        weights, _residual_norm = nnls(design, target)
        residual = target - design @ weights
    return tuple(selected), weights.astype(np.float32)


@dataclass(frozen=True, slots=True)
class CtsEncoded:
    coordinates: tuple[Float32, ...]


@dataclass(frozen=True, slots=True)
class VbEncoded:
    indices: tuple[tuple[int, ...], ...]
    weights: tuple[Float32, ...]


class CtsCodebook:
    def __init__(self, group_bases: tuple[Float32, ...], quantization_bits: int) -> None:
        if quantization_bits not in {8, 16, 32}:
            raise ValueError("CTS coordinate precision must be 8, 16, or 32 bits")
        if not group_bases:
            raise ValueError("CTS requires at least one group basis")
        widths: list[int] = []
        owned: list[Float32] = []
        for basis in group_bases:
            if basis.ndim != 2 or basis.dtype != np.float32 or not np.isfinite(basis).all():
                raise ValueError("CTS bases must be finite float32 matrices")
            if basis.shape[0] < 1 or basis.shape[1] < 1:
                raise ValueError("CTS basis dimensions must be positive")
            widths.append(basis.shape[1])
            owned.append(basis.copy())
        self.group_bases = tuple(owned)
        self.group_widths = tuple(widths)
        self.quantization_bits = quantization_bits

    @classmethod
    def from_fixture(
        cls,
        *,
        group_bases: tuple[Float32, ...],
        quantization_bits: int,
    ) -> CtsCodebook:
        return cls(group_bases, quantization_bits)

    def _quantize(self, value: Float32) -> Float32:
        if self.quantization_bits == 16:
            return value.astype(np.float16).astype(np.float32)
        if self.quantization_bits == 8:
            scale = max(
                float(np.max(np.abs(value))) / 127.0,
                float(np.finfo(np.float32).eps),
            )
            return cast(
                Float32,
                (np.round(value / scale).clip(-127, 127) * scale).astype(np.float32),
            )
        return value.astype(np.float32)

    def encode(self, code: Float32) -> CtsEncoded:
        if code.dtype != np.float32 or code.shape != (sum(self.group_widths),):
            raise ValueError("CTS code has an invalid layout")
        coordinates: list[Float32] = []
        start = 0
        for basis, width in zip(self.group_bases, self.group_widths, strict=True):
            segment = code[start : start + width]
            coordinates.append(self._quantize((basis @ segment).astype(np.float32)))
            start += width
        return CtsEncoded(tuple(coordinates))

    def decode(self, encoded: CtsEncoded) -> Float32:
        if len(encoded.coordinates) != len(self.group_bases):
            raise ValueError("CTS encoded group count mismatch")
        segments = [
            (basis.T @ coordinate).astype(np.float32)
            for basis, coordinate in zip(
                self.group_bases,
                encoded.coordinates,
                strict=True,
            )
        ]
        return np.concatenate(segments).astype(np.float32)


class VbCodebook:
    def __init__(
        self,
        *,
        bank: Float32,
        subvector_size: int,
        top_k: int,
        weight_bits: int,
    ) -> None:
        if (
            bank.ndim != 2
            or bank.dtype != np.float32
            or not np.isfinite(bank).all()
            or bank.shape[1] != subvector_size
        ):
            raise ValueError("VB bank has an invalid layout")
        if not 0 < top_k <= len(bank):
            raise ValueError("VB top-k is outside the bank")
        if weight_bits not in {8, 16, 32}:
            raise ValueError("VB weight precision must be 8, 16, or 32 bits")
        self.bank = bank.copy()
        self.subvector_size = subvector_size
        self.top_k = top_k
        self.weight_bits = weight_bits

    @classmethod
    def from_fixture(
        cls,
        *,
        bank: Float32,
        subvector_size: int,
        top_k: int,
        weight_bits: int,
    ) -> VbCodebook:
        return cls(
            bank=bank,
            subvector_size=subvector_size,
            top_k=top_k,
            weight_bits=weight_bits,
        )

    def _quantize_weights(self, weights: Float32) -> Float32:
        if self.weight_bits == 16:
            return weights.astype(np.float16).astype(np.float32)
        if self.weight_bits == 8:
            scale = max(
                float(np.max(np.abs(weights))) / 255.0,
                float(np.finfo(np.float32).eps),
            )
            return cast(
                Float32,
                (np.round(weights / scale).clip(0, 255) * scale).astype(np.float32),
            )
        return weights.astype(np.float32)

    def encode(self, code: Float32) -> VbEncoded:
        if (
            code.ndim != 1
            or code.dtype != np.float32
            or len(code) % self.subvector_size
            or not np.isfinite(code).all()
        ):
            raise ValueError("VB code must be a divisible finite float32 vector")
        indices: list[tuple[int, ...]] = []
        weights: list[Float32] = []
        for start in range(0, len(code), self.subvector_size):
            selected, values = vb_encode_subvector(
                code[start : start + self.subvector_size],
                self.bank,
                self.top_k,
            )
            indices.append(selected)
            weights.append(self._quantize_weights(values))
        return VbEncoded(tuple(indices), tuple(weights))

    def decode(self, encoded: VbEncoded) -> Float32:
        if len(encoded.indices) != len(encoded.weights):
            raise ValueError("VB index and weight group counts differ")
        segments: list[Float32] = []
        for indices, weights in zip(encoded.indices, encoded.weights, strict=True):
            if len(indices) != len(weights):
                raise ValueError("VB index and weight counts differ")
            segment = weights.astype(np.float32) @ self.bank[list(indices)]
            segments.append(segment.astype(np.float32))
        return np.concatenate(segments).astype(np.float32)


@dataclass(frozen=True, slots=True)
class CodeCorpus:
    split: Literal["train", "validation", "final_test"]
    codes: Float32
    manifest_sha256: Sha256


class StaticCodebookArtifact(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    family: Literal["cts_style_static", "vb_lora_style_static"]
    training_manifest_sha256: Sha256
    code_corpus_sha256: Sha256
    seed: int = Field(ge=0)
    group_boundaries: tuple[PositiveInt, ...]
    rank: PositiveInt | None
    bank_size: PositiveInt | None
    top_k: PositiveInt | None
    quantization_bits: Literal[8, 16, 32]
    tensor_file: Path
    tensor_sha256: Sha256
    codebook_sha256: Sha256


def fit_static_codebook(
    corpus: CodeCorpus,
    *,
    family: Literal["cts_style_static", "vb_lora_style_static"],
    rank: int = 8,
    bank_size: int = 256,
    subvector_size: int = 16,
    top_k: int = 2,
    quantization_bits: int = 16,
) -> CtsCodebook | VbCodebook:
    """Fit deterministic shared state from train codes only."""

    if corpus.split != "train":
        raise LeakageError("static codebook accepts train split only")
    codes = corpus.codes
    if codes.ndim != 2 or codes.dtype != np.float32 or not np.isfinite(codes).all():
        raise ValueError("static code corpus must be a finite float32 matrix")
    if family == "cts_style_static":
        return CtsCodebook((truncated_basis(codes, rank),), quantization_bits)
    if len(codes[0]) % subvector_size:
        raise ValueError("VB corpus width must be divisible by subvector size")
    subvectors = codes.reshape(-1, subvector_size)
    if bank_size > len(subvectors):
        raise ValueError("VB bank size exceeds the training subvector count")
    keys = [
        hashlib.sha256(row.tobytes(order="C")).digest()
        for row in np.ascontiguousarray(subvectors)
    ]
    order = sorted(range(len(keys)), key=lambda index: (keys[index], index))
    bank = subvectors[order[:bank_size]].copy()
    return VbCodebook(
        bank=bank,
        subvector_size=subvector_size,
        top_k=top_k,
        weight_bits=quantization_bits,
    )


def codebook_semantic_sha256(codebook: CtsCodebook | VbCodebook) -> str:
    if isinstance(codebook, CtsCodebook):
        payload = {
            "family": "cts_style_static",
            "quantization_bits": codebook.quantization_bits,
            "bases": [
                {
                    "shape": list(basis.shape),
                    "sha256": hashlib.sha256(basis.tobytes(order="C")).hexdigest(),
                }
                for basis in codebook.group_bases
            ],
        }
    else:
        payload = {
            "family": "vb_lora_style_static",
            "subvector_size": codebook.subvector_size,
            "top_k": codebook.top_k,
            "weight_bits": codebook.weight_bits,
            "bank_sha256": hashlib.sha256(codebook.bank.tobytes(order="C")).hexdigest(),
        }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


Outcome = Literal[
    "created",
    "updated",
    "read",
    "deleted",
    "rejected",
    "evicted",
    "stale_handle",
]


@dataclass(frozen=True, slots=True)
class StaticCodeRecord:
    handle: str
    encoded_payload: bytes
    created_event: int
    last_read_event: int
    reads: int
    update_count: int


@dataclass(frozen=True, slots=True)
class StaticSharedStateView:
    records: dict[str, StaticCodeRecord]
    codebook_sha256: str
    ledger: ExactByteLedger
    optimizer_state_present: Literal[False] = False


def _array_record(array: Float32) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(order="C"),
    }


def _array_from_record(record: object) -> Float32:
    if not isinstance(record, dict):
        raise ValueError("static encoded tensor record must be an object")
    raw = record.get("data")
    shape = record.get("shape")
    dtype = record.get("dtype")
    if type(raw) is not bytes or not isinstance(shape, list) or type(dtype) is not str:
        raise ValueError("static encoded tensor record is invalid")
    array = (
        np.frombuffer(raw, dtype=np.dtype(dtype))
        .reshape(tuple(int(value) for value in shape))
        .copy()
    )
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("static encoded tensors must be finite float32")
    return cast(Float32, array)


def _serialize_encoded(encoded: CtsEncoded | VbEncoded) -> bytes:
    if isinstance(encoded, CtsEncoded):
        payload: dict[str, object] = {
            "family": "cts_style_static",
            "coordinates": [_array_record(value) for value in encoded.coordinates],
        }
    else:
        payload = {
            "family": "vb_lora_style_static",
            "indices": [list(values) for values in encoded.indices],
            "weights": [_array_record(value) for value in encoded.weights],
        }
    return cbor2.dumps(payload, canonical=True)


def _deserialize_encoded(
    payload: bytes,
    family: Literal["cts_style_static", "vb_lora_style_static"],
) -> CtsEncoded | VbEncoded:
    try:
        decoded = cbor2.loads(payload)
    except Exception as error:
        raise ValueError("static encoded state is not valid CBOR") from error
    if type(decoded) is not dict or decoded.get("family") != family:
        raise ValueError("static encoded family mismatch")
    if cbor2.dumps(decoded, canonical=True) != payload:
        raise ValueError("static encoded state is not canonical CBOR")
    if family == "cts_style_static":
        coordinates = decoded.get("coordinates")
        if not isinstance(coordinates, list):
            raise ValueError("CTS coordinate payload is invalid")
        return CtsEncoded(tuple(_array_from_record(value) for value in coordinates))
    indices = decoded.get("indices")
    weights = decoded.get("weights")
    if not isinstance(indices, list) or not isinstance(weights, list):
        raise ValueError("VB encoded payload is invalid")
    return VbEncoded(
        tuple(tuple(int(value) for value in row) for row in indices),
        tuple(_array_from_record(value) for value in weights),
    )


class StaticSharedAdapter:
    """Frozen shared codebook plus exact per-concept online coordinates."""

    role: Literal["causal"] = "causal"
    external_support_bytes = 0

    def __init__(
        self,
        method_id: Literal["cts_style_static", "vb_lora_style_static"],
        *,
        codebook: CtsCodebook | VbCodebook,
        codebook_file: Path,
        shared_inputs: SharedInputReader | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        if (method_id == "cts_style_static") != isinstance(codebook, CtsCodebook):
            raise ValueError("CTS method requires a CTS codebook")
        if (method_id == "vb_lora_style_static") != isinstance(codebook, VbCodebook):
            raise ValueError("VB method requires a VB codebook")
        if codebook_file.is_symlink() or not codebook_file.is_file():
            raise ValueError("static codebook file must be a real immutable artifact")
        self.method_id = method_id
        self.codebook = codebook
        self.codebook_file_sha256 = file_sha256(codebook_file)
        self.codebook_sha256 = codebook_semantic_sha256(codebook)
        self.shared_trained_bytes = codebook_file.stat().st_size
        self._reader = shared_inputs
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._records: dict[str, StaticCodeRecord] = {}
        self._last_event_index: int | None = None
        self._snapshots: dict[str, bytes] = {}
        self._closed = False

    def bind_shared_inputs(self, reader: SharedInputReader) -> None:
        if self._contract is not None:
            raise RuntimeError("shared inputs must be bound before initialization")
        self._reader = reader

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("adapter cannot be initialized twice or after close")
        if self._reader is None:
            raise RuntimeError("static adapter has no shared-input reader")
        if self._reader.manifest.trace_id != contract.trace_id:
            raise ValueError("shared-input trace differs from the contract")
        if self._reader.manifest.candidate_stream_sha256 != contract.candidate_stream_sha256:
            raise ValueError("shared-input candidate stream differs from the contract")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty state")

    def _require_active(self) -> tuple[FrozenComparisonContract, SharedInputReader]:
        if self._contract is None or self._reader is None or self._closed:
            raise RuntimeError("static adapter is not active")
        return self._contract, self._reader

    def _components(
        self,
        records: dict[str, StaticCodeRecord] | None = None,
    ) -> dict[str, list[object]]:
        selected = self._records if records is None else records
        components = empty_components()
        for handle in sorted(selected):
            record = selected[handle]
            components["base_codes"].append(
                {"handle": handle, "family": self.method_id, "data": record.encoded_payload}
            )
            components["handles"].append(handle)
            components["usage_age"].append(
                {
                    "handle": handle,
                    "created_event": record.created_event,
                    "last_read_event": record.last_read_event,
                    "reads": record.reads,
                    "update_count": record.update_count,
                }
            )
            components["checksums"].append(
                {
                    "handle": handle,
                    "sha256": hashlib.sha256(record.encoded_payload).hexdigest(),
                }
            )
        components["controller_state"].append(
            {
                "policy": "lru",
                "family": self.method_id,
                "codebook_sha256": self.codebook_sha256,
                "codebook_file_sha256": self.codebook_file_sha256,
                "last_event_index": self._last_event_index,
            }
        )
        return components

    def _export(self, records: dict[str, StaticCodeRecord]) -> bytes:
        return export_state(self._components(records))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export(self._records)

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(
            self.export_online_state(),
            self.shared_trained_bytes,
            0,
        )

    def _encode_event(self, event_index: int, handle: str, current_index: int) -> bytes:
        _contract, reader = self._require_active()
        loaded = reader.load_event(event_index, current_index)
        if loaded.record.handle != handle:
            raise ValueError("shared-input handle differs from lifecycle event")
        return _serialize_encoded(self.codebook.encode(loaded.target_code))

    def _fit_with_eviction(
        self,
        records: dict[str, StaticCodeRecord],
        *,
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, StaticCodeRecord] | None, tuple[str, ...]]:
        selected = dict(records)
        evicted: list[str] = []
        while len(self._export(selected)) > budget:
            eligible = [row for handle, row in selected.items() if handle != protected_handle]
            if not eligible:
                return None, ()
            victim = min(
                eligible,
                key=lambda row: (row.last_read_event, row.created_event, row.handle),
            )
            del selected[victim.handle]
            evicted.append(victim.handle)
        return selected, tuple(evicted)

    def _decode(self, payload: bytes) -> Float32:
        encoded = _deserialize_encoded(payload, self.method_id)
        if isinstance(self.codebook, CtsCodebook) and isinstance(encoded, CtsEncoded):
            return self.codebook.decode(encoded)
        if isinstance(self.codebook, VbCodebook) and isinstance(encoded, VbEncoded):
            return self.codebook.decode(encoded)
        raise ValueError("static encoded payload and codebook families differ")

    def _sample_sha(self, payload: bytes, prompt: str, seed: int) -> tuple[str, str]:
        code = self._decode(payload)
        code_sha = hashlib.sha256(code.tobytes(order="C")).hexdigest()
        if self._backbone is None:
            sample_sha = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_backbone": True,
                        "code_sha256": code_sha,
                        "prompt": prompt,
                        "seed": seed,
                    }
                )
            ).hexdigest()
            return code_sha, sample_sha
        contract, _reader = self._require_active()
        self._backbone.install_code(torch.from_numpy(code))
        try:
            sample = self._backbone.generate(
                prompt,
                seed,
                sampler_id=contract.sampler_id,
                cfg_scale=contract.cfg_scale,
                steps=contract.denoising_steps,
            )
        finally:
            self._backbone.clear_code()
        array = sample.detach().cpu().contiguous().numpy()
        return code_sha, hashlib.sha256(array.tobytes(order="C")).hexdigest()

    def _admit(
        self,
        records: dict[str, StaticCodeRecord],
        handle: str,
        budget: int,
    ) -> tuple[dict[str, StaticCodeRecord] | None, tuple[str, ...]]:
        return self._fit_with_eviction(
            records,
            protected_handle=handle,
            budget=budget,
        )

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract, _reader = self._require_active()
        if isinstance(event, ProbeEvent):
            raise TypeError("probe events must use score_probe")
        if event.event_index != view.current_index or view.at(event.event_index) != event:
            raise ValueError("event and causal view are not aligned")
        validate_operational_event_order(self._last_event_index, event, view)
        before = self.state_ledger()
        self._last_event_index = event.event_index
        records = dict(self._records)
        affected: tuple[str, ...] = ()
        evicted: tuple[str, ...] = ()
        decoded_sha: str | None = None
        generated_sha: str | None = None
        outcome: Outcome
        if isinstance(event, CreateEvent):
            if event.handle in records:
                outcome = "rejected"
            else:
                records[event.handle] = StaticCodeRecord(
                    event.handle,
                    self._encode_event(event.event_index, event.handle, view.current_index),
                    event.event_index,
                    event.event_index,
                    0,
                    0,
                )
                fitted, evicted = self._admit(records, event.handle, contract.byte_budget)
                if fitted is None:
                    records.pop(event.handle)
                    outcome = "rejected"
                    evicted = ()
                else:
                    affected = tuple(sorted(set(records) | set(fitted)))
                    records = fitted
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                records[event.handle] = StaticCodeRecord(
                    event.handle,
                    self._encode_event(event.event_index, event.handle, view.current_index),
                    previous.created_event,
                    previous.last_read_event,
                    previous.reads,
                    previous.update_count + 1,
                )
                fitted, evicted = self._admit(records, event.handle, contract.byte_budget)
                if fitted is None:
                    records[event.handle] = previous
                    outcome = "rejected"
                    evicted = ()
                else:
                    affected = tuple(sorted(set(records) | set(fitted)))
                    records = fitted
                    outcome = "updated"
        elif isinstance(event, ReadEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                decoded_sha, generated_sha = self._sample_sha(
                    previous.encoded_payload,
                    event.prompt_id,
                    event.generation_seed,
                )
                records[event.handle] = replace(
                    previous,
                    last_read_event=event.event_index,
                    reads=previous.reads + 1,
                )
                affected = (event.handle,)
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in records:
                outcome = "stale_handle"
            else:
                del records[event.handle]
                affected = (event.handle,)
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported static adapter event: {type(event).__name__}")
        self._records = records
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("static adapter exceeded the exact byte budget")
        event_sha = hashlib.sha256(canonical_json_bytes(event.model_dump(mode="json"))).hexdigest()
        return EventReceipt(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=event.event_index,
            event_kind=event.kind,
            input_commitment_sha256=event_sha,
            method_state_sha256_before=before.online_state_sha256,
            method_state_sha256_after=after.online_state_sha256,
            candidate_stream_sha256=contract.candidate_stream_sha256,
            outcome=outcome,
            affected_handles=affected,
            evicted_handles=evicted,
            decoded_code_sha256=decoded_sha,
            generated_sample_sha256=generated_sha,
            ledger=after,
        )

    def copy_snapshot(self) -> MethodSnapshot:
        contract, _reader = self._require_active()
        if self._last_event_index is None:
            raise RuntimeError("cannot snapshot before the first event")
        payload = self.export_online_state()
        state_sha = hashlib.sha256(payload).hexdigest()
        token = hashlib.sha256(b"static-shared-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    def _records_from_payload(
        self,
        payload: bytes,
    ) -> tuple[dict[str, StaticCodeRecord], int | None]:
        components = decode_state(payload)
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("static controller state is invalid")
        row = controller[0]
        if (
            row.get("family") != self.method_id
            or row.get("codebook_sha256") != self.codebook_sha256
            or row.get("codebook_file_sha256") != self.codebook_file_sha256
        ):
            raise ValueError("static codebook identity differs from exported state")
        last_event = row.get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("static last event index is invalid")
        code_rows = {
            str(item["handle"]): item
            for item in components["base_codes"]
            if isinstance(item, dict)
        }
        usage_rows = {
            str(item["handle"]): item
            for item in components["usage_age"]
            if isinstance(item, dict)
        }
        if set(code_rows) != set(usage_rows):
            raise ValueError("static code and metadata handles differ")
        records: dict[str, StaticCodeRecord] = {}
        for handle in sorted(code_rows):
            raw = code_rows[handle].get("data")
            if type(raw) is not bytes:
                raise ValueError("static encoded record is invalid")
            _deserialize_encoded(raw, self.method_id)
            meta = usage_rows[handle]
            records[handle] = StaticCodeRecord(
                handle,
                raw,
                int(meta["created_event"]),
                int(meta["last_read_event"]),
                int(meta["reads"]),
                int(meta["update_count"]),
            )
        return records, last_event

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        records, last_event = self._records_from_payload(payload)
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export(records) != payload:
            self._last_event_index = prior_event
            raise ValueError("static state does not roundtrip canonically")
        self._records = records

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract, _reader = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        records, _last_event = self._records_from_payload(payload)
        record = records.get(probe.handle)
        if record is None:
            raise ValueError("probe handle is absent from static snapshot")
        _code_sha, sample_sha = self._sample_sha(
            record.encoded_payload,
            probe.prompt_id,
            probe.generation_seed,
        )
        input_sha = hashlib.sha256(canonical_json_bytes(probe.model_dump(mode="json"))).hexdigest()
        return ProbeResult(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            probe_event_index=probe.event_index,
            snapshot_state_sha256=snapshot.state_sha256,
            input_commitment_sha256=input_sha,
            generated_sample_sha256=sample_sha,
            update_usage=False,
        )

    def inspect_state(self) -> StaticSharedStateView:
        self._require_active()
        return StaticSharedStateView(
            records=dict(self._records),
            codebook_sha256=self.codebook_sha256,
            ledger=self.state_ledger(),
        )

    def close(self) -> None:
        self._records.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "CodeCorpus",
    "CtsCodebook",
    "CtsEncoded",
    "LeakageError",
    "StaticCodebookArtifact",
    "StaticCodeRecord",
    "StaticSharedAdapter",
    "StaticSharedStateView",
    "VbCodebook",
    "VbEncoded",
    "codebook_semantic_sha256",
    "fit_static_codebook",
    "truncated_basis",
    "vb_encode_subvector",
]
