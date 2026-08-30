"""Exact append-only and full-future upper references for matched evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
from torch import Tensor

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
    FutureAccessError,
    MethodSnapshot,
    ProbeResult,
    validate_operational_event_order,
)
from ratemem.baselines.shared_inputs import SharedInputReader
from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.traces import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)

Float32 = NDArray[np.float32]
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_HEX = frozenset("0123456789abcdef")
Outcome = Literal[
    "created",
    "updated",
    "read",
    "deleted",
    "rejected",
    "evicted",
    "stale_handle",
]


class OracleNotOptimal(RuntimeError):
    """Raised when an upper reference cannot prove global optimality."""


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class QuantizedTeacherCode:
    quantizer_id: str
    payload: bytes
    decoded_code: Float32
    squared_error: float

    def __post_init__(self) -> None:
        if not self.quantizer_id:
            raise ValueError("teacher quantizer id is required")
        if type(self.payload) is not bytes or not self.payload:
            raise ValueError("teacher quantizer payload must be non-empty bytes")
        if (
            not isinstance(self.decoded_code, np.ndarray)
            or self.decoded_code.dtype != np.float32
            or self.decoded_code.shape != (480,)
            or not np.isfinite(self.decoded_code).all()
        ):
            raise ValueError("decoded teacher code must be one finite float32 480-vector")
        if not math.isfinite(self.squared_error) or self.squared_error < 0.0:
            raise ValueError("teacher quantizer distortion must be finite and nonnegative")
        object.__setattr__(self, "decoded_code", self.decoded_code.copy())


class TeacherQuantizer(Protocol):
    quantizer_id: str

    def encode(self, code: Float32) -> QuantizedTeacherCode: ...

    def decode(self, payload: bytes) -> Float32: ...


class SymmetricTeacherQuantizer:
    """Deterministic signed 8/16-bit symmetric teacher-code quantizer."""

    def __init__(self, bits: Literal[8, 16]) -> None:
        if bits not in {8, 16}:
            raise ValueError("teacher quantizer supports only signed int8 or int16")
        self.bits = bits
        self.quantizer_id = f"symmetric_int{bits}_v1"

    def encode(self, code: Float32) -> QuantizedTeacherCode:
        checked = np.asarray(code, dtype=np.float32)
        if checked.shape != (480,) or not np.isfinite(checked).all():
            raise ValueError("teacher code must be one finite float32 480-vector")
        limit = (1 << (self.bits - 1)) - 1
        maximum = float(np.max(np.abs(checked)))
        scale = np.float32(maximum / limit if maximum > 0.0 else 1.0)
        integer_dtype = np.dtype("<i1" if self.bits == 8 else "<i2")
        quantized = np.clip(np.rint(checked / scale), -limit, limit).astype(integer_dtype)
        payload = scale.astype("<f4").tobytes() + quantized.tobytes(order="C")
        decoded = (quantized.astype(np.float32) * scale).astype(np.float32)
        error = float(np.mean(np.square(checked.astype(np.float64) - decoded)))
        return QuantizedTeacherCode(self.quantizer_id, payload, decoded, error)

    def decode(self, payload: bytes) -> Float32:
        width = 1 if self.bits == 8 else 2
        if len(payload) != 4 + 480 * width:
            raise ValueError("teacher quantizer payload has the wrong length")
        scale = np.frombuffer(payload[:4], dtype="<f4")[0]
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("teacher quantizer scale is invalid")
        dtype = np.dtype("<i1" if self.bits == 8 else "<i2")
        values = np.frombuffer(payload[4:], dtype=dtype).astype(np.float32)
        return np.ascontiguousarray(values * scale, dtype=np.float32)


def choose_append_option(
    options: Sequence[QuantizedTeacherCode],
    remaining_bytes: int,
) -> QuantizedTeacherCode | None:
    """Choose by distortion, bytes, then stable quantizer identity."""

    if type(remaining_bytes) is not int or remaining_bytes < 0:
        raise ValueError("remaining bytes must be a nonnegative integer")
    feasible = [option for option in options if len(option.payload) <= remaining_bytes]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda option: (
            option.squared_error,
            len(option.payload),
            option.quantizer_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class AppendRecord:
    handle: str
    quantizer_id: str
    payload: bytes
    decoded_code_sha256: str
    squared_error: float
    description_id: str
    created_event: int
    last_read_event: int
    update_count: int


@dataclass(frozen=True, slots=True)
class AppendOnlyStateView:
    records: dict[str, AppendRecord]
    record_hashes: dict[str, str]
    quantizer_id: dict[str, str]


class ExactAppendOnlyAdapter:
    """Teacher-code upper reference that never evicts an admitted record."""

    method_id = "exact_append_only_quantized"
    role: Literal["upper_reference"] = "upper_reference"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        quantizers: Sequence[TeacherQuantizer],
        *,
        shared_inputs: SharedInputReader | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        selected = tuple(quantizers)
        identifiers = tuple(row.quantizer_id for row in selected)
        if not selected or len(identifiers) != len(set(identifiers)):
            raise ValueError("teacher quantizers must be non-empty and uniquely identified")
        self.quantizers = {row.quantizer_id: row for row in selected}
        self._reader = shared_inputs
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._records: dict[str, AppendRecord] = {}
        self._last_event_index: int | None = None
        self._snapshots: dict[str, bytes] = {}
        self._closed = False

    def bind_shared_inputs(self, reader: SharedInputReader) -> None:
        if self._contract is not None:
            raise RuntimeError("shared inputs must be bound before initialization")
        self._reader = reader

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("append-only adapter cannot be initialized twice or after close")
        if self._reader is None:
            raise RuntimeError("append-only adapter has no teacher-code reader")
        if self._reader.manifest.trace_id != contract.trace_id:
            raise ValueError("teacher-code trace differs from the contract")
        if self._reader.manifest.candidate_stream_sha256 != contract.candidate_stream_sha256:
            raise ValueError("teacher-code stream differs from the contract")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty oracle state")

    def _require_active(self) -> tuple[FrozenComparisonContract, SharedInputReader]:
        if self._contract is None or self._reader is None or self._closed:
            raise RuntimeError("append-only adapter is not active")
        return self._contract, self._reader

    def _components(
        self,
        records: Mapping[str, AppendRecord] | None = None,
    ) -> dict[str, list[object]]:
        selected = self._records if records is None else records
        components = empty_components()
        for handle in sorted(selected):
            row = selected[handle]
            components["base_codes"].append(
                {
                    "handle": handle,
                    "quantizer_id": row.quantizer_id,
                    "data": row.payload,
                }
            )
            components["handles"].append(handle)
            components["optional_tokens"].append(
                {"handle": handle, "description_id": row.description_id}
            )
            components["usage_age"].append(
                {
                    "handle": handle,
                    "created_event": row.created_event,
                    "last_read_event": row.last_read_event,
                    "update_count": row.update_count,
                }
            )
            components["checksums"].append(
                {
                    "handle": handle,
                    "decoded_code_sha256": row.decoded_code_sha256,
                    "payload_sha256": hashlib.sha256(row.payload).hexdigest(),
                }
            )
            components["allocator_state"].append(
                {
                    "handle": handle,
                    "squared_error": row.squared_error,
                    "immutable_until_delete": True,
                }
            )
        components["controller_state"].append(
            {
                "policy": "exact_append_only",
                "quantizer_ids": sorted(self.quantizers),
                "last_event_index": self._last_event_index,
            }
        )
        return components

    def _export_records(self, records: Mapping[str, AppendRecord]) -> bytes:
        return export_state(self._components(records))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export_records(self._records)

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(
            self.export_online_state(),
            self.shared_trained_bytes,
            self.external_support_bytes,
        )

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller_rows = components["controller_state"]
        if len(controller_rows) != 1 or not isinstance(controller_rows[0], dict):
            raise ValueError("append-only controller state is invalid")
        controller = controller_rows[0]
        if (
            controller.get("policy") != "exact_append_only"
            or controller.get("quantizer_ids") != sorted(self.quantizers)
        ):
            raise ValueError("append-only controller state differs from this adapter")
        last_event = controller.get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("append-only last event index is invalid")
        handles_raw = tuple(components["handles"])
        if any(type(handle) is not str for handle in handles_raw):
            raise ValueError("append-only handles are invalid")
        handles = tuple(str(handle) for handle in handles_raw)
        if handles != tuple(sorted(set(handles))):
            raise ValueError("append-only handles must be sorted and unique")
        codes = {
            str(row["handle"]): row
            for row in components["base_codes"]
            if isinstance(row, dict)
        }
        metadata = {
            str(row["handle"]): row
            for row in components["usage_age"]
            if isinstance(row, dict)
        }
        descriptions = {
            str(row["handle"]): row
            for row in components["optional_tokens"]
            if isinstance(row, dict)
        }
        distortions = {
            str(row["handle"]): row
            for row in components["allocator_state"]
            if isinstance(row, dict)
        }
        checksums = {
            str(row["handle"]): row
            for row in components["checksums"]
            if isinstance(row, dict)
        }
        expected = set(handles)
        exported_rows = (codes, metadata, descriptions, distortions, checksums)
        if any(set(rows) != expected for rows in exported_rows):
            raise ValueError("append-only exported handle sets differ")
        restored: dict[str, AppendRecord] = {}
        for handle in handles:
            code = codes[handle]
            quantizer_id = code.get("quantizer_id")
            raw = code.get("data")
            if type(quantizer_id) is not str or quantizer_id not in self.quantizers:
                raise ValueError("append-only quantizer identity is invalid")
            if type(raw) is not bytes:
                raise ValueError("append-only quantized payload is invalid")
            decoded = self.quantizers[quantizer_id].decode(raw)
            decoded_sha = hashlib.sha256(decoded.tobytes(order="C")).hexdigest()
            if decoded_sha != checksums[handle].get("decoded_code_sha256"):
                raise ValueError("append-only decoded-code checksum changed")
            meta = metadata[handle]
            restored[handle] = AppendRecord(
                handle=handle,
                quantizer_id=quantizer_id,
                payload=raw,
                decoded_code_sha256=decoded_sha,
                squared_error=float(distortions[handle]["squared_error"]),
                description_id=str(descriptions[handle]["description_id"]),
                created_event=int(meta["created_event"]),
                last_read_event=int(meta["last_read_event"]),
                update_count=int(meta["update_count"]),
            )
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export_records(restored) != payload:
            self._last_event_index = prior_event
            raise ValueError("append-only state does not roundtrip canonically")
        self._records = restored

    def _teacher_options(
        self,
        event_index: int,
        current_index: int,
    ) -> tuple[QuantizedTeacherCode, ...]:
        _contract, reader = self._require_active()
        target = reader.load_event(event_index, current_index).target_code
        return tuple(self.quantizers[key].encode(target) for key in sorted(self.quantizers))

    def _candidate_record(
        self,
        event: CreateEvent | UpdateEvent,
        description_id: str,
        option: QuantizedTeacherCode,
        previous: AppendRecord | None,
    ) -> AppendRecord:
        return AppendRecord(
            handle=event.handle,
            quantizer_id=option.quantizer_id,
            payload=option.payload,
            decoded_code_sha256=hashlib.sha256(
                option.decoded_code.tobytes(order="C")
            ).hexdigest(),
            squared_error=option.squared_error,
            description_id=description_id,
            created_event=event.event_index if previous is None else previous.created_event,
            last_read_event=event.event_index if previous is None else previous.last_read_event,
            update_count=0 if previous is None else previous.update_count + 1,
        )

    def _choose_exact_fit(
        self,
        records: Mapping[str, AppendRecord],
        event: CreateEvent | UpdateEvent,
        description_id: str,
        previous: AppendRecord | None,
        current_index: int,
        budget: int,
    ) -> tuple[AppendRecord, dict[str, AppendRecord]] | None:
        feasible: list[tuple[QuantizedTeacherCode, AppendRecord, dict[str, AppendRecord]]] = []
        for option in self._teacher_options(event.event_index, current_index):
            record = self._candidate_record(event, description_id, option, previous)
            candidate = dict(records)
            candidate[event.handle] = record
            if len(self._export_records(candidate)) <= budget:
                feasible.append((option, record, candidate))
        if not feasible:
            return None
        _option, record, candidate = min(
            feasible,
            key=lambda row: (
                row[0].squared_error,
                len(row[0].payload),
                row[0].quantizer_id,
            ),
        )
        return record, candidate

    def _decode(self, record: AppendRecord) -> Float32:
        return self.quantizers[record.quantizer_id].decode(record.payload)

    def _sample_sha(self, record: AppendRecord, prompt_id: str, seed: int) -> str:
        code = self._decode(record)
        if self._backbone is None:
            return hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_backbone": True,
                        "code_sha256": record.decoded_code_sha256,
                        "prompt_id": prompt_id,
                        "seed": seed,
                    }
                )
            ).hexdigest()
        contract, _reader = self._require_active()
        self._backbone.install_code(torch.from_numpy(code.copy()))
        try:
            generated = self._backbone.generate(
                prompt_id,
                seed,
                sampler_id=contract.sampler_id,
                cfg_scale=contract.cfg_scale,
                steps=contract.denoising_steps,
            )
        finally:
            self._backbone.clear_code()
        return hashlib.sha256(
            generated.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest()

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
        decoded_sha: str | None = None
        generated_sha: str | None = None
        outcome: Outcome
        if isinstance(event, CreateEvent):
            if event.handle in records:
                outcome = "rejected"
            else:
                selected = self._choose_exact_fit(
                    records,
                    event,
                    event.description_id,
                    None,
                    view.current_index,
                    contract.byte_budget,
                )
                if selected is None:
                    outcome = "rejected"
                else:
                    record, records = selected
                    affected = (event.handle,)
                    decoded_sha = record.decoded_code_sha256
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                selected = self._choose_exact_fit(
                    records,
                    event,
                    previous.description_id,
                    previous,
                    view.current_index,
                    contract.byte_budget,
                )
                if selected is None:
                    outcome = "rejected"
                else:
                    record, records = selected
                    affected = (event.handle,)
                    decoded_sha = record.decoded_code_sha256
                    outcome = "updated"
        elif isinstance(event, ReadEvent):
            read_record = records.get(event.handle)
            if read_record is None:
                outcome = "stale_handle"
            else:
                generated_sha = self._sample_sha(
                    read_record,
                    event.prompt_id,
                    event.generation_seed,
                )
                records[event.handle] = replace(
                    read_record,
                    last_read_event=event.event_index,
                )
                affected = (event.handle,)
                decoded_sha = read_record.decoded_code_sha256
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in records:
                outcome = "stale_handle"
            else:
                del records[event.handle]
                affected = (event.handle,)
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported append-only event: {type(event).__name__}")
        self._records = records
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("append-only adapter exceeded the exact byte budget")
        input_sha = hashlib.sha256(
            canonical_json_bytes(event.model_dump(mode="json"))
        ).hexdigest()
        return EventReceipt(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=event.event_index,
            event_kind=event.kind,
            input_commitment_sha256=input_sha,
            method_state_sha256_before=before.online_state_sha256,
            method_state_sha256_after=after.online_state_sha256,
            candidate_stream_sha256=contract.candidate_stream_sha256,
            outcome=outcome,
            affected_handles=affected,
            evicted_handles=(),
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
        token = hashlib.sha256(b"append-oracle-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    def _snapshot_record(self, payload: bytes, handle: str) -> AppendRecord:
        original = self.export_online_state()
        try:
            self.import_online_state(payload)
            record = self._records.get(handle)
            if record is None:
                raise ValueError("probe references a handle absent from its snapshot")
            return record
        finally:
            self.import_online_state(original)

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract, _reader = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        generated_sha = self._sample_sha(
            self._snapshot_record(payload, probe.handle),
            probe.prompt_id,
            probe.generation_seed,
        )
        input_sha = hashlib.sha256(
            canonical_json_bytes(probe.model_dump(mode="json"))
        ).hexdigest()
        return ProbeResult(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            probe_event_index=probe.event_index,
            snapshot_state_sha256=snapshot.state_sha256,
            input_commitment_sha256=input_sha,
            generated_sample_sha256=generated_sha,
            update_usage=False,
        )

    def inspect_state(self) -> AppendOnlyStateView:
        self._require_active()
        return AppendOnlyStateView(
            records=dict(self._records),
            record_hashes={
                handle: hashlib.sha256(self._export_records({handle: row})).hexdigest()
                for handle, row in self._records.items()
            },
            quantizer_id={handle: row.quantizer_id for handle, row in self._records.items()},
        )

    def close(self) -> None:
        self._records.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


class FutureHandle(BaseModel):
    model_config = _MODEL_CONFIG

    handle: str = Field(min_length=1)
    base_bytes: PositiveInt
    create_event: NonNegativeInt
    delete_event: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_lifetime(self) -> FutureHandle:
        if self.delete_event is not None and self.delete_event <= self.create_event:
            raise ValueError("future handle delete must follow create")
        return self


class FuturePacket(BaseModel):
    model_config = _MODEL_CONFIG

    packet_id: str = Field(min_length=1)
    cost_bytes: PositiveInt
    proposal_event: NonNegativeInt
    dependent_handles: tuple[str, ...]
    gain_by_handle: dict[str, NonNegativeInt]

    @model_validator(mode="after")
    def validate_incidence(self) -> FuturePacket:
        if self.dependent_handles != tuple(sorted(set(self.dependent_handles))):
            raise ValueError("future packet dependencies must be sorted and unique")
        if not self.dependent_handles or set(self.gain_by_handle) != set(self.dependent_handles):
            raise ValueError("future packet gains must match dependencies")
        return self


class FutureUtility(BaseModel):
    model_config = _MODEL_CONFIG

    event_index: NonNegativeInt
    request_weight_by_handle: dict[str, NonNegativeInt]
    coverage_cap_by_handle: dict[str, NonNegativeInt]
    base_gain_by_handle: dict[str, NonNegativeInt]


class FutureTraceProblem(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    byte_budget: PositiveInt
    handles: tuple[FutureHandle, ...]
    packets: tuple[FuturePacket, ...]
    utilities: tuple[FutureUtility, ...]
    switching_penalty: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_problem(self) -> FutureTraceProblem:
        handles = tuple(row.handle for row in self.handles)
        packets = tuple(row.packet_id for row in self.packets)
        if (
            not handles
            or handles != tuple(sorted(set(handles)))
            or packets != tuple(sorted(set(packets)))
        ):
            raise ValueError("future handles and packets must be sorted and unique")
        indices = tuple(row.event_index for row in self.utilities)
        if not indices or indices != tuple(range(len(indices))):
            raise ValueError("future utilities must be contiguous from event zero")
        handle_set = set(handles)
        for packet in self.packets:
            if not set(packet.dependent_handles) <= handle_set:
                raise ValueError("future packet depends on an unknown handle")
            if packet.proposal_event >= len(self.utilities):
                raise ValueError("future packet proposal is outside the trace")
        for handle in self.handles:
            if handle.create_event >= len(self.utilities):
                raise ValueError("future handle create is outside the trace")
            if handle.delete_event is not None and handle.delete_event >= len(self.utilities):
                raise ValueError("future handle delete is outside the trace")
        for utility in self.utilities:
            dictionaries = (
                utility.request_weight_by_handle,
                utility.coverage_cap_by_handle,
                utility.base_gain_by_handle,
            )
            if any(set(values) != handle_set for values in dictionaries):
                raise ValueError("future utility must cover every handle exactly")
            if any(
                utility.base_gain_by_handle[handle] > utility.coverage_cap_by_handle[handle]
                for handle in handles
            ):
                raise ValueError("future base gain exceeds its coverage cap")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class FutureAllocation(BaseModel):
    model_config = _MODEL_CONFIG

    event_index: NonNegativeInt
    admitted_handles: tuple[str, ...]
    packet_ids: tuple[str, ...]
    serialized_bytes: NonNegativeInt
    utility_integer: int


class OracleCertificate(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    method_id: Literal["exact_future_trace_packets"] = "exact_future_trace_packets"
    problem_sha256: str
    solver: Literal["scipy-highs-milp"]
    status: Literal["optimal"]
    mip_gap: float
    objective_integer: int
    allocations_sha256: str
    certificate_sha256: str

    @model_validator(mode="after")
    def validate_hashes(self) -> OracleCertificate:
        _require_sha256(self.problem_sha256, "problem_sha256")
        _require_sha256(self.allocations_sha256, "allocations_sha256")
        _require_sha256(self.certificate_sha256, "certificate_sha256")
        if self.mip_gap != 0.0:
            raise ValueError("an exact oracle certificate requires zero MIP gap")
        return self

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("certificate_sha256")
        return canonical_json_bytes(payload)


class FutureTraceResult(BaseModel):
    model_config = _MODEL_CONFIG

    status: Literal["optimal"]
    objective_integer: int
    allocations: tuple[FutureAllocation, ...]
    certificate: OracleCertificate


@dataclass(frozen=True, slots=True)
class _MilpLayout:
    y: dict[tuple[int, str], int]
    x: dict[tuple[int, str], int]
    z: dict[tuple[int, str], int]
    d: dict[tuple[int, str], int]
    size: int


def _layout(problem: FutureTraceProblem) -> _MilpLayout:
    cursor = 0
    y: dict[tuple[int, str], int] = {}
    x: dict[tuple[int, str], int] = {}
    z: dict[tuple[int, str], int] = {}
    d: dict[tuple[int, str], int] = {}
    for event in range(len(problem.utilities)):
        for handle in (row.handle for row in problem.handles):
            y[event, handle] = cursor
            cursor += 1
        for packet in (row.packet_id for row in problem.packets):
            x[event, packet] = cursor
            cursor += 1
        for handle in (row.handle for row in problem.handles):
            z[event, handle] = cursor
            cursor += 1
    if problem.switching_penalty:
        for event in range(1, len(problem.utilities)):
            for packet in (row.packet_id for row in problem.packets):
                d[event, packet] = cursor
                cursor += 1
    return _MilpLayout(y=y, x=x, z=z, d=d, size=cursor)


def _active(handle: FutureHandle, event: int) -> bool:
    return event >= handle.create_event and (
        handle.delete_event is None or event < handle.delete_event
    )


def _build_milp(
    problem: FutureTraceProblem,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], object, _MilpLayout]:
    layout = _layout(problem)
    lower = np.zeros(layout.size, dtype=np.float64)
    upper = np.ones(layout.size, dtype=np.float64)
    objective = np.zeros(layout.size, dtype=np.float64)
    handle_by_id = {row.handle: row for row in problem.handles}
    packet_by_id = {row.packet_id: row for row in problem.packets}
    for utility in problem.utilities:
        event = utility.event_index
        for handle_id, handle in handle_by_id.items():
            if not _active(handle, event):
                upper[layout.y[event, handle_id]] = 0.0
            z_index = layout.z[event, handle_id]
            upper[z_index] = float(utility.coverage_cap_by_handle[handle_id])
            objective[z_index] = -float(utility.request_weight_by_handle[handle_id])
        for packet_id, packet in packet_by_id.items():
            if event < packet.proposal_event:
                upper[layout.x[event, packet_id]] = 0.0
    for index in layout.d.values():
        objective[index] = float(problem.switching_penalty)

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def constraint(coefficients: dict[int, float], low: float, high: float) -> None:
        rows.append(coefficients)
        row_lower.append(low)
        row_upper.append(high)

    for utility in problem.utilities:
        event = utility.event_index
        capacity: dict[int, float] = {}
        for handle in problem.handles:
            capacity[layout.y[event, handle.handle]] = float(handle.base_bytes)
        for packet in problem.packets:
            capacity[layout.x[event, packet.packet_id]] = float(packet.cost_bytes)
        constraint(capacity, -np.inf, float(problem.byte_budget))

        for packet in problem.packets:
            x_index = layout.x[event, packet.packet_id]
            for dependent_handle in packet.dependent_handles:
                constraint(
                    {x_index: 1.0, layout.y[event, dependent_handle]: -1.0},
                    -np.inf,
                    0.0,
                )

        for handle in problem.handles:
            handle_id = handle.handle
            z_index = layout.z[event, handle_id]
            y_index = layout.y[event, handle_id]
            cap = float(utility.coverage_cap_by_handle[handle_id])
            constraint({z_index: 1.0, y_index: -cap}, -np.inf, 0.0)
            coverage = {
                z_index: 1.0,
                y_index: -float(utility.base_gain_by_handle[handle_id]),
            }
            for packet in problem.packets:
                gain = packet.gain_by_handle.get(handle_id, 0)
                if gain:
                    coverage[layout.x[event, packet.packet_id]] = -float(gain)
            constraint(coverage, -np.inf, 0.0)

    for handle in problem.handles:
        end = handle.delete_event or len(problem.utilities)
        for event in range(handle.create_event + 1, end):
            constraint(
                {
                    layout.y[event, handle.handle]: 1.0,
                    layout.y[event - 1, handle.handle]: -1.0,
                },
                -np.inf,
                0.0,
            )

    if problem.switching_penalty:
        for event in range(1, len(problem.utilities)):
            for packet in problem.packets:
                current = layout.x[event, packet.packet_id]
                previous = layout.x[event - 1, packet.packet_id]
                changed = layout.d[event, packet.packet_id]
                constraint({current: 1.0, previous: -1.0, changed: -1.0}, -np.inf, 0.0)
                constraint({previous: 1.0, current: -1.0, changed: -1.0}, -np.inf, 0.0)

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), layout.size),
        dtype=np.float64,
    ).tocsr()
    constraints = LinearConstraint(
        matrix,
        np.asarray(row_lower, dtype=np.float64),
        np.asarray(row_upper, dtype=np.float64),
    )
    return objective, lower, upper, constraints, layout


def _decode_future_solution(
    problem: FutureTraceProblem,
    layout: _MilpLayout,
    solution: NDArray[np.float64],
) -> tuple[FutureAllocation, ...]:
    def binary(index: int) -> bool:
        value = float(solution[index])
        rounded = round(value)
        if abs(value - rounded) > 1e-6 or rounded not in {0, 1}:
            raise OracleNotOptimal("future oracle returned a nonintegral binary variable")
        return bool(rounded)

    allocations: list[FutureAllocation] = []
    previous_packets: frozenset[str] = frozenset()
    for utility in problem.utilities:
        event = utility.event_index
        handles = tuple(
            row.handle
            for row in problem.handles
            if binary(layout.y[event, row.handle])
        )
        packets = tuple(
            row.packet_id
            for row in problem.packets
            if binary(layout.x[event, row.packet_id])
        )
        handle_set = set(handles)
        packet_set = set(packets)
        serialized = sum(
            row.base_bytes for row in problem.handles if row.handle in handle_set
        ) + sum(row.cost_bytes for row in problem.packets if row.packet_id in packet_set)
        utility_integer = 0
        for handle in handles:
            coverage = utility.base_gain_by_handle[handle] + sum(
                packet.gain_by_handle.get(handle, 0)
                for packet in problem.packets
                if packet.packet_id in packet_set
            )
            coverage = min(coverage, utility.coverage_cap_by_handle[handle])
            utility_integer += utility.request_weight_by_handle[handle] * coverage
        switches = len(packet_set ^ previous_packets) if event > 0 else 0
        utility_integer -= problem.switching_penalty * switches
        allocations.append(
            FutureAllocation(
                event_index=event,
                admitted_handles=handles,
                packet_ids=packets,
                serialized_bytes=serialized,
                utility_integer=utility_integer,
            )
        )
        previous_packets = frozenset(packets)
    return tuple(allocations)


def verify_future_trace_result(
    problem: FutureTraceProblem,
    allocations: Sequence[FutureAllocation],
) -> int:
    """Recompute every integer constraint without trusting floating solver output."""

    if tuple(row.event_index for row in allocations) != tuple(range(len(problem.utilities))):
        raise OracleNotOptimal("future allocation indices differ from the trace")
    handles = {row.handle: row for row in problem.handles}
    packets = {row.packet_id: row for row in problem.packets}
    previous_handles: set[str] = set()
    previous_packets: set[str] = set()
    total = 0
    for allocation in allocations:
        event = allocation.event_index
        admitted = set(allocation.admitted_handles)
        selected = set(allocation.packet_ids)
        if len(admitted) != len(allocation.admitted_handles) or len(selected) != len(
            allocation.packet_ids
        ):
            raise OracleNotOptimal("future allocation repeats an item")
        if not admitted <= set(handles) or not selected <= set(packets):
            raise OracleNotOptimal("future allocation names an unknown item")
        if any(not _active(handles[handle], event) for handle in admitted):
            raise OracleNotOptimal("future allocation admits an unavailable handle")
        for handle, spec in handles.items():
            if (
                event > spec.create_event
                and _active(spec, event)
                and handle not in previous_handles
                and handle in admitted
            ):
                raise OracleNotOptimal("future allocation readmits an evicted handle")
        for packet_id in selected:
            packet = packets[packet_id]
            if event < packet.proposal_event or not set(packet.dependent_handles) <= admitted:
                raise OracleNotOptimal("future allocation selects an unavailable packet")
        exact_bytes = sum(handles[row].base_bytes for row in admitted) + sum(
            packets[row].cost_bytes for row in selected
        )
        if exact_bytes != allocation.serialized_bytes or exact_bytes > problem.byte_budget:
            raise OracleNotOptimal("future allocation violates exact capacity")
        utility = problem.utilities[event]
        exact_utility = 0
        for handle in admitted:
            coverage = utility.base_gain_by_handle[handle] + sum(
                packets[packet].gain_by_handle.get(handle, 0) for packet in selected
            )
            exact_utility += utility.request_weight_by_handle[handle] * min(
                coverage,
                utility.coverage_cap_by_handle[handle],
            )
        if event:
            exact_utility -= problem.switching_penalty * len(selected ^ previous_packets)
        if exact_utility != allocation.utility_integer:
            raise OracleNotOptimal("future allocation utility does not recompute exactly")
        total += exact_utility
        previous_handles = admitted
        previous_packets = selected
    return total


def solve_future_trace(problem: FutureTraceProblem) -> FutureTraceResult:
    """Solve the bounded future trace and emit a self-verifying optimal certificate."""

    if type(problem) is not FutureTraceProblem:
        raise TypeError("problem must be an exact FutureTraceProblem")
    objective, lower, upper, constraints, layout = _build_milp(problem)
    result = milp(
        c=objective,
        integrality=np.ones(layout.size, dtype=np.uint8),
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"mip_rel_gap": 0.0, "presolve": True, "disp": False},
    )
    gap = getattr(result, "mip_gap", None)
    if result.status != 0 or result.x is None or gap is None or float(gap) > 1e-9:
        raise OracleNotOptimal(
            f"future trace oracle did not prove optimality: status={result.status} gap={gap}"
        )
    solution = cast(NDArray[np.float64], result.x)
    allocations = _decode_future_solution(problem, layout, solution)
    objective_integer = verify_future_trace_result(problem, allocations)
    allocations_sha = hashlib.sha256(
        canonical_json_bytes([row.model_dump(mode="json") for row in allocations])
    ).hexdigest()
    provisional = OracleCertificate(
        problem_sha256=problem.sha256,
        solver="scipy-highs-milp",
        status="optimal",
        mip_gap=0.0,
        objective_integer=objective_integer,
        allocations_sha256=allocations_sha,
        certificate_sha256="0" * 64,
    )
    certificate = provisional.model_copy(
        update={
            "certificate_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
        }
    )
    return FutureTraceResult(
        status="optimal",
        objective_integer=objective_integer,
        allocations=allocations,
        certificate=certificate,
    )


def exhaustive_future_trace(problem: FutureTraceProblem) -> FutureTraceResult:
    """Verification-only exact enumeration for very small oracle instances."""

    if len(problem.handles) > 8 or len(problem.packets) > 12 or len(problem.utilities) > 8:
        raise ValueError("exhaustive future oracle supports only tiny verification instances")
    handle_ids = tuple(row.handle for row in problem.handles)
    packet_ids = tuple(row.packet_id for row in problem.packets)
    handle_specs = {row.handle: row for row in problem.handles}
    packet_specs = {row.packet_id: row for row in problem.packets}
    StateKey = tuple[frozenset[str], frozenset[str]]
    StateValue = tuple[int, tuple[FutureAllocation, ...]]

    def state_order(value: StateValue) -> tuple[int, int, bytes]:
        score, rows = value
        total_bytes = sum(row.serialized_bytes for row in rows)
        serialized = canonical_json_bytes(
            [row.model_dump(mode="json") for row in rows]
        )
        return score, -total_bytes, bytes(255 - byte for byte in serialized)

    states: dict[StateKey, StateValue] = {
        (frozenset(), frozenset()): (0, ())
    }
    for utility in problem.utilities:
        event = utility.event_index
        next_states: dict[
            tuple[frozenset[str], frozenset[str]],
            tuple[int, tuple[FutureAllocation, ...]],
        ] = {}
        for previous_key, (prior_score, prior_rows) in states.items():
            previous_handles, previous_packets = previous_key
            for handle_mask in product((False, True), repeat=len(handle_ids)):
                admitted = frozenset(
                    handle for handle, keep in zip(handle_ids, handle_mask, strict=True) if keep
                )
                if any(not _active(handle_specs[handle], event) for handle in admitted):
                    continue
                if any(
                    event > handle_specs[handle].create_event
                    and _active(handle_specs[handle], event)
                    and handle not in previous_handles
                    and handle in admitted
                    for handle in handle_ids
                ):
                    continue
                for packet_mask in product((False, True), repeat=len(packet_ids)):
                    selected = frozenset(
                        packet
                        for packet, keep in zip(packet_ids, packet_mask, strict=True)
                        if keep
                    )
                    if any(
                        event < packet_specs[packet].proposal_event
                        or not set(packet_specs[packet].dependent_handles) <= admitted
                        for packet in selected
                    ):
                        continue
                    used = sum(handle_specs[row].base_bytes for row in admitted) + sum(
                        packet_specs[row].cost_bytes for row in selected
                    )
                    if used > problem.byte_budget:
                        continue
                    value = 0
                    for handle in admitted:
                        coverage = utility.base_gain_by_handle[handle] + sum(
                            packet_specs[packet].gain_by_handle.get(handle, 0)
                            for packet in selected
                        )
                        value += utility.request_weight_by_handle[handle] * min(
                            coverage,
                            utility.coverage_cap_by_handle[handle],
                        )
                    if event:
                        value -= problem.switching_penalty * len(selected ^ previous_packets)
                    row = FutureAllocation(
                        event_index=event,
                        admitted_handles=tuple(sorted(admitted)),
                        packet_ids=tuple(sorted(selected)),
                        serialized_bytes=used,
                        utility_integer=value,
                    )
                    candidate_score = prior_score + value
                    key = (admitted, selected)
                    existing = next_states.get(key)
                    candidate_rows = prior_rows + (row,)
                    candidate = (candidate_score, candidate_rows)
                    if existing is None or state_order(candidate) > state_order(existing):
                        next_states[key] = candidate
        states = next_states
    if not states:
        raise OracleNotOptimal("tiny future problem has no feasible trajectory")
    objective_integer, allocations = max(states.values(), key=state_order)
    if verify_future_trace_result(problem, allocations) != objective_integer:
        raise OracleNotOptimal("exhaustive future result failed exact verification")
    allocations_sha = hashlib.sha256(
        canonical_json_bytes([row.model_dump(mode="json") for row in allocations])
    ).hexdigest()
    provisional = OracleCertificate(
        problem_sha256=problem.sha256,
        solver="scipy-highs-milp",
        status="optimal",
        mip_gap=0.0,
        objective_integer=objective_integer,
        allocations_sha256=allocations_sha,
        certificate_sha256="0" * 64,
    )
    certificate = provisional.model_copy(
        update={
            "certificate_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
        }
    )
    return FutureTraceResult(
        status="optimal",
        objective_integer=objective_integer,
        allocations=allocations,
        certificate=certificate,
    )


def require_upper_reference_role(role: str) -> None:
    if role != "upper_reference":
        raise FutureAccessError("full trace requires upper_reference role")


class FutureCodeDecoder(Protocol):
    """Reconstruct one SANA adapter code from a certified future allocation."""

    def decode(
        self,
        handle: str,
        base_payload: bytes,
        packet_payloads: Mapping[str, bytes],
    ) -> Tensor: ...


class FutureTracePacketAdapter:
    """Replay a proved full-trace allocation as a non-causal upper reference."""

    method_id = "exact_future_trace_packets"
    role: Literal["upper_reference"] = "upper_reference"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        problem: FutureTraceProblem,
        result: FutureTraceResult,
        *,
        base_payload_by_handle: Mapping[str, bytes],
        packet_payload_by_id: Mapping[str, bytes],
        requesting_role: str,
        decoder: FutureCodeDecoder | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        require_upper_reference_role(requesting_role)
        if result.status != "optimal" or result.certificate.problem_sha256 != problem.sha256:
            raise OracleNotOptimal("future adapter requires a matching optimal certificate")
        if verify_future_trace_result(problem, result.allocations) != result.objective_integer:
            raise OracleNotOptimal("future adapter result failed integer verification")
        expected_handles = {row.handle for row in problem.handles}
        expected_packets = {row.packet_id for row in problem.packets}
        if set(base_payload_by_handle) != expected_handles:
            raise ValueError("future base payloads differ from the problem handles")
        if set(packet_payload_by_id) != expected_packets:
            raise ValueError("future packet payloads differ from the problem packets")
        if any(type(value) is not bytes or not value for value in base_payload_by_handle.values()):
            raise ValueError("future base payloads must be non-empty bytes")
        if any(type(value) is not bytes or not value for value in packet_payload_by_id.values()):
            raise ValueError("future packet payloads must be non-empty bytes")
        for handle_spec in problem.handles:
            if len(base_payload_by_handle[handle_spec.handle]) > handle_spec.base_bytes:
                raise ValueError("future base payload exceeds its certified byte cost")
        for packet_spec in problem.packets:
            if len(packet_payload_by_id[packet_spec.packet_id]) > packet_spec.cost_bytes:
                raise ValueError("future packet payload exceeds its certified byte cost")
        if (decoder is None) != (backbone is None):
            raise ValueError("future decoder and backbone must be supplied together")
        self.problem = problem
        self.result = result
        self._base_payloads = dict(base_payload_by_handle)
        self._packet_payloads = dict(packet_payload_by_id)
        self._decoder = decoder
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._last_event_index: int | None = None
        self._snapshots: dict[str, bytes] = {}
        self._closed = False

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("future adapter cannot be initialized twice or after close")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty future state")

    def _require_active(self) -> FrozenComparisonContract:
        if self._contract is None or self._closed:
            raise RuntimeError("future adapter is not active")
        return self._contract

    def _allocation(self, event_index: int) -> FutureAllocation:
        try:
            allocation = self.result.allocations[event_index]
        except IndexError as error:
            raise IndexError("future allocation event is outside the certificate") from error
        if allocation.event_index != event_index:
            raise OracleNotOptimal("future allocation order changed")
        return allocation

    def _components(self, event_index: int | None) -> dict[str, list[object]]:
        components = empty_components()
        allocation = None if event_index is None else self._allocation(event_index)
        handles = () if allocation is None else allocation.admitted_handles
        packet_ids = () if allocation is None else allocation.packet_ids
        packets = {row.packet_id: row for row in self.problem.packets}
        for handle in handles:
            payload = self._base_payloads[handle]
            components["base_codes"].append({"handle": handle, "data": payload})
            components["handles"].append(handle)
            components["checksums"].append(
                {
                    "kind": "base",
                    "handle": handle,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        for packet_id in packet_ids:
            payload = self._packet_payloads[packet_id]
            packet = packets[packet_id]
            components["packet_payloads"].append(
                {"packet_id": packet_id, "data": payload}
            )
            components["packet_hashes"].append(
                {"packet_id": packet_id, "sha256": hashlib.sha256(payload).hexdigest()}
            )
            components["incidences_gains"].append(
                {
                    "packet_id": packet_id,
                    "dependent_handles": list(packet.dependent_handles),
                    "gain_by_handle": dict(packet.gain_by_handle),
                }
            )
            components["reference_counts"].append(
                {"packet_id": packet_id, "count": len(packet.dependent_handles)}
            )
        components["controller_state"].append(
            {
                "policy": "exact_future_trace",
                "problem_sha256": self.problem.sha256,
                "certificate_sha256": self.result.certificate.certificate_sha256,
                "last_event_index": event_index,
            }
        )
        if allocation is not None:
            components["allocator_state"].append(allocation.model_dump(mode="json"))
        return components

    def export_online_state(self) -> bytes:
        self._require_active()
        return export_state(self._components(self._last_event_index))

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller_rows = components["controller_state"]
        if len(controller_rows) != 1 or not isinstance(controller_rows[0], dict):
            raise ValueError("future controller state is invalid")
        controller = controller_rows[0]
        if (
            controller.get("policy") != "exact_future_trace"
            or controller.get("problem_sha256") != self.problem.sha256
            or controller.get("certificate_sha256")
            != self.result.certificate.certificate_sha256
        ):
            raise ValueError("future controller state differs from the certificate")
        event_index = controller.get("last_event_index")
        if event_index is not None and (
            type(event_index) is not int
            or event_index < 0
            or event_index >= len(self.result.allocations)
        ):
            raise ValueError("future state event index is invalid")
        expected = export_state(self._components(event_index))
        if expected != payload:
            raise ValueError("future state does not match its certified allocation")
        self._last_event_index = event_index

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(
            self.export_online_state(),
            self.shared_trained_bytes,
            self.external_support_bytes,
        )

    def _code_sha(self, allocation: FutureAllocation, handle: str) -> str:
        if handle not in allocation.admitted_handles:
            raise ValueError("future code requested for an absent handle")
        if self._decoder is None:
            return hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_decoder": True,
                        "handle": handle,
                        "base_sha256": hashlib.sha256(
                            self._base_payloads[handle]
                        ).hexdigest(),
                        "packet_ids": list(allocation.packet_ids),
                    }
                )
            ).hexdigest()
        relevant = {
            packet_id: self._packet_payloads[packet_id]
            for packet_id in allocation.packet_ids
            if handle
            in next(
                row.dependent_handles
                for row in self.problem.packets
                if row.packet_id == packet_id
            )
        }
        code = self._decoder.decode(handle, self._base_payloads[handle], relevant)
        if not isinstance(code, Tensor) or code.shape != (480,) or not torch.isfinite(code).all():
            raise ValueError("future decoder must return one finite 480-vector")
        return hashlib.sha256(
            code.detach().cpu().to(torch.float32).contiguous().numpy().tobytes(order="C")
        ).hexdigest()

    def _sample_sha(
        self,
        allocation: FutureAllocation,
        handle: str,
        prompt_id: str,
        seed: int,
    ) -> str:
        code_sha = self._code_sha(allocation, handle)
        if self._decoder is None or self._backbone is None:
            return hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_backbone": True,
                        "code_sha256": code_sha,
                        "prompt_id": prompt_id,
                        "seed": seed,
                    }
                )
            ).hexdigest()
        relevant = {
            packet_id: self._packet_payloads[packet_id]
            for packet_id in allocation.packet_ids
            if handle
            in next(
                row.dependent_handles
                for row in self.problem.packets
                if row.packet_id == packet_id
            )
        }
        code = self._decoder.decode(handle, self._base_payloads[handle], relevant)
        contract = self._require_active()
        self._backbone.install_code(code)
        try:
            generated = self._backbone.generate(
                prompt_id,
                seed,
                sampler_id=contract.sampler_id,
                cfg_scale=contract.cfg_scale,
                steps=contract.denoising_steps,
            )
        finally:
            self._backbone.clear_code()
        return hashlib.sha256(
            generated.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest()

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract = self._require_active()
        if isinstance(event, ProbeEvent):
            raise TypeError("probe events must use score_probe")
        if event.event_index != view.current_index or view.at(event.event_index) != event:
            raise ValueError("event and causal view are not aligned")
        validate_operational_event_order(self._last_event_index, event, view)
        before = self.state_ledger()
        previous_handles = (
            set()
            if self._last_event_index is None
            else set(self._allocation(self._last_event_index).admitted_handles)
        )
        allocation = self._allocation(event.event_index)
        current_handles = set(allocation.admitted_handles)
        self._last_event_index = event.event_index
        affected: tuple[str, ...] = ()
        decoded_sha: str | None = None
        generated_sha: str | None = None
        if isinstance(event, CreateEvent):
            outcome: Outcome = "created" if event.handle in current_handles else "rejected"
        elif isinstance(event, UpdateEvent):
            outcome = "updated" if event.handle in current_handles else "stale_handle"
        elif isinstance(event, ReadEvent):
            outcome = "read" if event.handle in current_handles else "stale_handle"
        elif isinstance(event, DeleteEvent):
            outcome = "deleted" if event.handle in previous_handles else "stale_handle"
        else:
            raise TypeError(f"unsupported future-oracle event: {type(event).__name__}")
        if outcome in {"created", "updated", "read", "deleted"}:
            affected = (event.handle,)
        code_event = isinstance(event, CreateEvent | UpdateEvent | ReadEvent)
        if event.handle in current_handles and code_event:
            decoded_sha = self._code_sha(allocation, event.handle)
        if outcome == "read" and isinstance(event, ReadEvent):
            generated_sha = self._sample_sha(
                allocation,
                event.handle,
                event.prompt_id,
                event.generation_seed,
            )
        explicit_delete = {event.handle} if isinstance(event, DeleteEvent) else set()
        evicted = tuple(sorted(previous_handles - current_handles - explicit_delete))
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise OracleNotOptimal("certified future allocation exceeds canonical byte budget")
        input_sha = hashlib.sha256(
            canonical_json_bytes(event.model_dump(mode="json"))
        ).hexdigest()
        return EventReceipt(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=event.event_index,
            event_kind=event.kind,
            input_commitment_sha256=input_sha,
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
        contract = self._require_active()
        if self._last_event_index is None:
            raise RuntimeError("cannot snapshot before the first event")
        payload = self.export_online_state()
        state_sha = hashlib.sha256(payload).hexdigest()
        token = hashlib.sha256(b"future-oracle-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        components = decode_state(payload)
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("future probe snapshot controller is invalid")
        event_index = controller[0].get("last_event_index")
        if type(event_index) is not int:
            raise ValueError("future probe snapshot has no event allocation")
        allocation = self._allocation(event_index)
        generated_sha = self._sample_sha(
            allocation,
            probe.handle,
            probe.prompt_id,
            probe.generation_seed,
        )
        input_sha = hashlib.sha256(
            canonical_json_bytes(probe.model_dump(mode="json"))
        ).hexdigest()
        return ProbeResult(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            probe_event_index=probe.event_index,
            snapshot_state_sha256=snapshot.state_sha256,
            input_commitment_sha256=input_sha,
            generated_sample_sha256=generated_sha,
            update_usage=False,
        )

    def close(self) -> None:
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "AppendOnlyStateView",
    "AppendRecord",
    "ExactAppendOnlyAdapter",
    "FutureAllocation",
    "FutureHandle",
    "FuturePacket",
    "FutureTraceProblem",
    "FutureTracePacketAdapter",
    "FutureTraceResult",
    "FutureUtility",
    "FutureCodeDecoder",
    "OracleCertificate",
    "OracleNotOptimal",
    "QuantizedTeacherCode",
    "SymmetricTeacherQuantizer",
    "TeacherQuantizer",
    "choose_append_option",
    "exhaustive_future_trace",
    "require_upper_reference_role",
    "solve_future_trace",
    "verify_future_trace_result",
]
