"""Content-addressed provider-neutral inputs shared by matched SANA controls."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator
from safetensors.numpy import load_file, save_file

from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256, write_json_atomic
from ratemem.evaluation.traces import LifecycleEvent
from ratemem.evaluation.types import GitCommit, Sha256

Float32 = NDArray[np.float32]
SignedInt16 = Annotated[int, Field(ge=-32768, le=32767)]
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SharedProviderMetadata:
    provider_id: str
    provider_revision_sha256: Sha256
    backbone_id: Literal["sana_1_5_1_6b"]
    backbone_revision: GitCommit
    adapter_layout_sha256: Sha256
    projection_count: Literal[120]
    code_dim: Literal[480]
    amortizer_sha256: Sha256
    adapter_basis_sha256: Sha256
    codec_dictionary_sha256: Sha256
    support_pool_sha256: Sha256
    incidence_gain_step: float

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider id is required")
        for field in (
            "provider_revision_sha256",
            "adapter_layout_sha256",
            "amortizer_sha256",
            "adapter_basis_sha256",
            "codec_dictionary_sha256",
            "support_pool_sha256",
        ):
            _require_sha256(str(getattr(self, field)), field)
        if self.projection_count != 120 or self.code_dim != 480:
            raise ValueError("shared provider must expose the frozen SANA 120/480 layout")
        if not np.isfinite(self.incidence_gain_step) or self.incidence_gain_step <= 0.0:
            raise ValueError("incidence gain step must be finite and positive")


@dataclass(frozen=True, slots=True, order=True)
class ProviderPacketKey:
    dictionary_revision_sha256: Sha256
    group: int
    stage: int
    entry: int

    def __post_init__(self) -> None:
        _require_sha256(self.dictionary_revision_sha256, "dictionary_revision_sha256")
        if min(self.group, self.stage, self.entry) < 0:
            raise ValueError("packet address indices must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProviderPacketCandidate:
    key: ProviderPacketKey
    packet_id: Sha256
    packet_payload: bytes
    gain_q: int

    def __post_init__(self) -> None:
        _require_sha256(self.packet_id, "packet_id")
        if hashlib.sha256(self.packet_payload).hexdigest() != self.packet_id:
            raise ValueError("provider packet content address does not match payload")
        if not -32768 <= self.gain_q <= 32767:
            raise ValueError("provider gain must fit signed int16")


@dataclass(frozen=True, slots=True)
class ProviderEventOutput:
    event_index: int
    handle: str
    target_code: Float32
    base_code: Float32
    quantizer_scales: Float32
    candidates: tuple[ProviderPacketCandidate, ...]

    def __post_init__(self) -> None:
        if self.event_index < 0 or not self.handle:
            raise ValueError("provider event identity is invalid")
        for name, value, shape in (
            ("target_code", self.target_code, (480,)),
            ("base_code", self.base_code, (480,)),
            ("quantizer_scales", self.quantizer_scales, (30,)),
        ):
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite float32 with shape {shape}")
        if len({candidate.key for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("provider event repeats a packet address")


@runtime_checkable
class SharedInputProvider(Protocol):
    def manifest_metadata(self) -> SharedProviderMetadata: ...

    def record_for_event(self, event: LifecycleEvent) -> ProviderEventOutput: ...


class CandidateAccessError(RuntimeError):
    """Raised when a causal method requests future shared inputs."""


class CandidatePacket(BaseModel):
    model_config = _MODEL_CONFIG

    packet_id: Sha256
    dictionary_revision_sha256: Sha256
    group: NonNegativeInt
    stage: NonNegativeInt
    entry: NonNegativeInt
    payload_sha256: Sha256
    payload_bytes: NonNegativeInt
    incidence_bytes: NonNegativeInt
    dependent_handles: tuple[str, ...]
    gain_q_by_handle: dict[str, SignedInt16]

    @model_validator(mode="after")
    def validate_incidence(self) -> CandidatePacket:
        if self.packet_id != self.payload_sha256:
            raise ValueError("packet id must equal its payload content address")
        if self.dependent_handles != tuple(sorted(set(self.dependent_handles))):
            raise ValueError("dependent handles must be sorted and unique")
        if set(self.gain_q_by_handle) != set(self.dependent_handles):
            raise ValueError("packet gains must cover exactly the dependent handles")
        return self


class SharedEventRecord(BaseModel):
    model_config = _MODEL_CONFIG

    event_index: NonNegativeInt
    handle: str = Field(min_length=1)
    tensor_path: str = Field(pattern=r"^events/[0-9]{6}\.safetensors$")
    tensor_sha256: Sha256
    target_code_sha256: Sha256
    base_code_sha256: Sha256
    candidate_packets: tuple[CandidatePacket, ...]
    record_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("record_sha256")
        return canonical_json_bytes(payload)


class SharedInputManifest(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    split: Literal["train", "validation", "final_test"]
    trace_id: Sha256
    provider_id: str = Field(min_length=1)
    provider_revision_sha256: Sha256
    backbone_id: Literal["sana_1_5_1_6b"]
    backbone_revision: GitCommit
    adapter_layout_sha256: Sha256
    projection_count: Literal[120]
    code_dim: Literal[480]
    amortizer_sha256: Sha256
    adapter_basis_sha256: Sha256
    codec_dictionary_sha256: Sha256
    support_pool_sha256: Sha256
    incidence_gain_step: float = Field(gt=0.0)
    event_records: tuple[SharedEventRecord, ...]
    candidate_stream_sha256: Sha256

    @model_validator(mode="after")
    def validate_records(self) -> SharedInputManifest:
        indices = tuple(record.event_index for record in self.event_records)
        if indices != tuple(range(len(indices))):
            raise ValueError("shared-input event records must be contiguous")
        expected = hashlib.sha256(
            canonical_json_bytes([record.record_sha256 for record in self.event_records])
        ).hexdigest()
        if expected != self.candidate_stream_sha256:
            raise ValueError("candidate stream hash mismatch")
        return self


class SharedInputBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    root: Path
    manifest: SharedInputManifest


@dataclass(frozen=True, slots=True)
class LoadedSharedEvent:
    record: SharedEventRecord
    target_code: Float32
    base_code: Float32
    quantizer_scales: Float32
    packet_payloads: dict[str, bytes]


def _array_sha256(value: NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _record_with_hash(payload: dict[str, object]) -> SharedEventRecord:
    provisional = SharedEventRecord.model_validate({**payload, "record_sha256": "0" * 64})
    return provisional.model_copy(
        update={"record_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )


def _write_safetensors_exclusive(path: Path, tensors: dict[str, NDArray[np.generic]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"shared-input tensor already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"shared-input temporary tensor already exists: {temporary}")
    try:
        save_file(tensors, temporary)
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_shared_input_bundle(
    root: Path,
    *,
    metadata: SharedProviderMetadata,
    outputs: Iterable[ProviderEventOutput],
    split: Literal["train", "validation", "final_test"],
    trace_id: str,
) -> SharedInputBundle:
    """Materialize one immutable candidate stream from provider outputs."""

    _require_sha256(trace_id, "trace_id")
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError("shared-input bundle root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, tuple[ProviderPacketCandidate, set[str], dict[str, int]]] = {}
    records: list[SharedEventRecord] = []
    for expected_index, output in enumerate(outputs):
        if output.event_index != expected_index:
            raise ValueError("provider outputs must be contiguous from event zero")
        tensors: dict[str, NDArray[np.generic]] = {
            "target_code": np.ascontiguousarray(output.target_code),
            "base_code": np.ascontiguousarray(output.base_code),
            "quantizer_scales": np.ascontiguousarray(output.quantizer_scales),
        }
        for candidate in output.candidates:
            if candidate.key.dictionary_revision_sha256 != metadata.codec_dictionary_sha256:
                raise ValueError("packet dictionary revision differs from provider metadata")
            existing = aggregate.get(candidate.packet_id)
            if existing is None:
                aggregate[candidate.packet_id] = (
                    candidate,
                    {output.handle},
                    {output.handle: candidate.gain_q},
                )
            else:
                original, handles, gains = existing
                if (
                    original.key != candidate.key
                    or original.packet_payload != candidate.packet_payload
                ):
                    raise ValueError("one packet id maps to inconsistent provider data")
                handles.add(output.handle)
                gains[output.handle] = candidate.gain_q
            tensors[f"packet_{candidate.packet_id}"] = np.frombuffer(
                candidate.packet_payload,
                dtype=np.uint8,
            ).copy()
        tensor_relative = Path("events") / f"{output.event_index:06d}.safetensors"
        tensor_path = root / tensor_relative
        _write_safetensors_exclusive(tensor_path, tensors)
        packets: list[CandidatePacket] = []
        for packet_id in sorted({candidate.packet_id for candidate in output.candidates}):
            candidate, handles, gains = aggregate[packet_id]
            dependent_handles = tuple(sorted(handles))
            incidence_payload = canonical_json_bytes(
                {"dependent_handles": dependent_handles, "gain_q_by_handle": gains}
            )
            packets.append(
                CandidatePacket(
                    packet_id=packet_id,
                    dictionary_revision_sha256=candidate.key.dictionary_revision_sha256,
                    group=candidate.key.group,
                    stage=candidate.key.stage,
                    entry=candidate.key.entry,
                    payload_sha256=packet_id,
                    payload_bytes=len(candidate.packet_payload),
                    incidence_bytes=len(incidence_payload),
                    dependent_handles=dependent_handles,
                    gain_q_by_handle={key: gains[key] for key in sorted(gains)},
                )
            )
        records.append(
            _record_with_hash(
                {
                    "event_index": output.event_index,
                    "handle": output.handle,
                    "tensor_path": tensor_relative.as_posix(),
                    "tensor_sha256": file_sha256(tensor_path),
                    "target_code_sha256": _array_sha256(output.target_code),
                    "base_code_sha256": _array_sha256(output.base_code),
                    "candidate_packets": [packet.model_dump(mode="json") for packet in packets],
                }
            )
        )
    stream_sha256 = hashlib.sha256(
        canonical_json_bytes([record.record_sha256 for record in records])
    ).hexdigest()
    manifest = SharedInputManifest(
        schema_version="1.0",
        split=split,
        trace_id=trace_id,
        provider_id=metadata.provider_id,
        provider_revision_sha256=metadata.provider_revision_sha256,
        backbone_id=metadata.backbone_id,
        backbone_revision=metadata.backbone_revision,
        adapter_layout_sha256=metadata.adapter_layout_sha256,
        projection_count=metadata.projection_count,
        code_dim=metadata.code_dim,
        amortizer_sha256=metadata.amortizer_sha256,
        adapter_basis_sha256=metadata.adapter_basis_sha256,
        codec_dictionary_sha256=metadata.codec_dictionary_sha256,
        support_pool_sha256=metadata.support_pool_sha256,
        incidence_gain_step=metadata.incidence_gain_step,
        event_records=tuple(records),
        candidate_stream_sha256=stream_sha256,
    )
    write_json_atomic(root / "manifest.json", manifest.model_dump(mode="json"))
    return SharedInputBundle(root=root, manifest=manifest)


class SharedInputReader:
    """Hash-verifying, prefix-only reader for one immutable shared bundle."""

    def __init__(self, root: Path | SharedInputBundle, method_id: str) -> None:
        if not method_id:
            raise ValueError("method id is required")
        self.root = root.root if isinstance(root, SharedInputBundle) else root
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("shared-input root must be a real directory")
        manifest_path = self.root / "manifest.json"
        self.manifest = SharedInputManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        self.method_id = method_id
        for record in self.manifest.event_records:
            if hashlib.sha256(record.semantic_bytes).hexdigest() != record.record_sha256:
                raise ValueError("shared-input record hash mismatch")
            tensor_path = self._safe_tensor_path(record)
            if file_sha256(tensor_path) != record.tensor_sha256:
                raise ValueError("shared-input file hash mismatch")
            self._validate_tensor_payload(record, tensor_path)

    def _safe_tensor_path(self, record: SharedEventRecord) -> Path:
        path = self.root / record.tensor_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("shared-input tensor path is missing or symbolic")
        if path.resolve().parent != (self.root / "events").resolve():
            raise ValueError("shared-input tensor path escapes the bundle")
        return path

    def _validate_tensor_payload(self, record: SharedEventRecord, path: Path) -> None:
        tensors = load_file(path)
        required = {"target_code", "base_code", "quantizer_scales"}
        required.update(f"packet_{packet.packet_id}" for packet in record.candidate_packets)
        if set(tensors) != required:
            raise ValueError("shared-input tensor keys differ from the manifest")
        target = tensors["target_code"]
        base = tensors["base_code"]
        scales = tensors["quantizer_scales"]
        if target.dtype != np.float32 or target.shape != (480,):
            raise ValueError("shared-input target code has an invalid layout")
        if base.dtype != np.float32 or base.shape != (480,):
            raise ValueError("shared-input base code has an invalid layout")
        if scales.dtype != np.float32 or scales.shape != (30,):
            raise ValueError("shared-input quantizer scales have an invalid layout")
        if _array_sha256(target) != record.target_code_sha256:
            raise ValueError("shared-input target code hash mismatch")
        if _array_sha256(base) != record.base_code_sha256:
            raise ValueError("shared-input base code hash mismatch")
        for packet in record.candidate_packets:
            payload = tensors[f"packet_{packet.packet_id}"]
            if payload.dtype != np.uint8 or payload.ndim != 1:
                raise ValueError("shared-input packet payload has an invalid layout")
            raw = payload.tobytes(order="C")
            if (
                len(raw) != packet.payload_bytes
                or hashlib.sha256(raw).hexdigest() != packet.payload_sha256
            ):
                raise ValueError("shared-input packet payload hash mismatch")

    def for_event(self, event_index: int, current_index: int) -> SharedEventRecord:
        if event_index > current_index:
            raise CandidateAccessError(f"future shared input event {event_index}")
        if event_index < 0:
            raise IndexError("shared input event is before the trace")
        try:
            return self.manifest.event_records[event_index]
        except IndexError as error:
            raise IndexError(f"shared input event does not exist: {event_index}") from error

    def load_event(self, event_index: int, current_index: int) -> LoadedSharedEvent:
        record = self.for_event(event_index, current_index)
        tensors = load_file(self._safe_tensor_path(record))
        return LoadedSharedEvent(
            record=record,
            target_code=tensors["target_code"],
            base_code=tensors["base_code"],
            quantizer_scales=tensors["quantizer_scales"],
            packet_payloads={
                packet.packet_id: tensors[f"packet_{packet.packet_id}"].tobytes(order="C")
                for packet in record.candidate_packets
            },
        )


def materialize_fixture_bundle(root: Path) -> SharedInputBundle:
    """Create a deterministic synthetic bundle; never publication-eligible evidence."""

    dictionary_sha = "d" * 64
    metadata = SharedProviderMetadata(
        provider_id="synthetic-fixture-v1",
        provider_revision_sha256="1" * 64,
        backbone_id="sana_1_5_1_6b",
        backbone_revision="b77948f2b4eed5c728e9b828ccff07f7427b43cc",
        adapter_layout_sha256="2" * 64,
        projection_count=120,
        code_dim=480,
        amortizer_sha256="3" * 64,
        adapter_basis_sha256="4" * 64,
        codec_dictionary_sha256=dictionary_sha,
        support_pool_sha256="5" * 64,
        incidence_gain_step=1.0 / 1024.0,
    )
    outputs: list[ProviderEventOutput] = []
    for index in range(2):
        raw = f"fixture-packet-{index}".encode()
        packet_id = hashlib.sha256(raw).hexdigest()
        outputs.append(
            ProviderEventOutput(
                event_index=index,
                handle=f"fixture-handle-{index}",
                target_code=np.linspace(0.0, 1.0, 480, dtype=np.float32) + index,
                base_code=np.linspace(1.0, 0.0, 480, dtype=np.float32) - index,
                quantizer_scales=np.full(30, 0.125 + index, dtype=np.float32),
                candidates=(
                    ProviderPacketCandidate(
                        key=ProviderPacketKey(dictionary_sha, 0, index, 0),
                        packet_id=packet_id,
                        packet_payload=raw,
                        gain_q=128 - index,
                    ),
                ),
            )
        )
    return write_shared_input_bundle(
        root,
        metadata=metadata,
        outputs=outputs,
        split="train",
        trace_id="6" * 64,
    )


__all__ = [
    "CandidateAccessError",
    "CandidatePacket",
    "LoadedSharedEvent",
    "ProviderEventOutput",
    "ProviderPacketCandidate",
    "ProviderPacketKey",
    "SharedEventRecord",
    "SharedInputBundle",
    "SharedInputManifest",
    "SharedInputProvider",
    "SharedInputReader",
    "SharedProviderMetadata",
    "materialize_fixture_bundle",
    "write_shared_input_bundle",
]
