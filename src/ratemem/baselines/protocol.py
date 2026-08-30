"""One causal lifecycle contract shared by RateMem and every comparator."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, overload, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

from ratemem.evaluation.traces import LifecycleEvent, ProbeEvent
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class FutureAccessError(RuntimeError):
    """Raised when a causal comparator attempts to inspect a future event."""


class FrozenComparisonContract(BaseModel):
    """All immutable inputs that must be identical in a paired comparison."""

    model_config = _MODEL_CONFIG

    contract_version: Literal["1.0"] = "1.0"
    trace_id: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_requirements_sha256: Sha256
    backbone_id: Literal["sana_1_5_1_6b"]
    backbone_revision: GitCommit
    adapter_layout_sha256: Sha256
    amortizer_sha256: Sha256
    adapter_basis_sha256: Sha256
    codec_dictionary_sha256: Sha256
    candidate_stream_sha256: Sha256
    prompt_pool_sha256: Sha256
    support_pool_sha256: Sha256
    noise_seed_manifest_sha256: Sha256
    sampler_id: str = Field(min_length=1, max_length=255)
    scheduler_revision: str = Field(min_length=1, max_length=255)
    cfg_scale: float = Field(gt=0.0)
    resolution: tuple[PositiveInt, PositiveInt]
    denoising_steps: PositiveInt
    byte_budget: PositiveInt
    request_regime: Literal["uniform", "zipf"]
    search_budget_sha256: Sha256


class ExactByteLedger(BaseModel):
    """Host-computed accounting for canonical exported online state."""

    model_config = _MODEL_CONFIG

    serializer_id: Literal["ratemem-baseline-cbor-v1"]
    online_state_bytes: NonNegativeInt
    online_state_sha256: Sha256
    component_bytes: dict[str, NonNegativeInt]
    shared_trained_bytes: NonNegativeInt
    external_support_bytes: NonNegativeInt

    @model_validator(mode="after")
    def totals_match(self) -> ExactByteLedger:
        if sum(self.component_bytes.values()) != self.online_state_bytes:
            raise ValueError("component bytes do not equal canonical online state bytes")
        return self


class EventReceipt(BaseModel):
    """Immutable event transition receipt produced by a comparator."""

    model_config = _MODEL_CONFIG

    method_id: str = Field(min_length=1, max_length=128)
    trace_id: Sha256
    event_index: NonNegativeInt
    event_kind: Literal["create", "update", "read", "delete"]
    input_commitment_sha256: Sha256
    method_state_sha256_before: Sha256
    method_state_sha256_after: Sha256
    candidate_stream_sha256: Sha256
    outcome: Literal[
        "created",
        "updated",
        "read",
        "deleted",
        "rejected",
        "evicted",
        "stale_handle",
    ]
    affected_handles: tuple[str, ...]
    evicted_handles: tuple[str, ...]
    decoded_code_sha256: Sha256 | None
    generated_sample_sha256: Sha256 | None
    ledger: ExactByteLedger


class MethodSnapshot(BaseModel):
    """Read-only snapshot token bound to one exact online state."""

    model_config = _MODEL_CONFIG

    method_id: str = Field(min_length=1, max_length=128)
    trace_id: Sha256
    event_index: NonNegativeInt
    state_sha256: Sha256
    online_state_bytes: NonNegativeInt
    opaque_snapshot_token: str = Field(min_length=1)


class ProbeResult(BaseModel):
    """Generation receipt from a copied snapshot; probes never update usage."""

    model_config = _MODEL_CONFIG

    method_id: str = Field(min_length=1, max_length=128)
    trace_id: Sha256
    probe_event_index: NonNegativeInt
    snapshot_state_sha256: Sha256
    input_commitment_sha256: Sha256
    generated_sample_sha256: Sha256
    update_usage: Literal[False] = False


class CausalEventView(Sequence[LifecycleEvent]):
    """A prefix-only facade that structurally prevents access to future events."""

    def __init__(self, events: Sequence[LifecycleEvent], current_index: int) -> None:
        if type(current_index) is not int or current_index < 0:
            raise ValueError("current event index must be a nonnegative integer")
        if current_index >= len(events):
            raise ValueError("current event index is outside the trace")
        self._events = tuple(events)
        self._current = current_index

    @property
    def current_index(self) -> int:
        return self._current

    def __len__(self) -> int:
        return self._current + 1

    @overload
    def __getitem__(self, index: int) -> LifecycleEvent: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[LifecycleEvent, ...]: ...

    def __getitem__(self, index: int | slice) -> LifecycleEvent | tuple[LifecycleEvent, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return tuple(self._events[position] for position in range(start, stop, step))
        normalized = index + len(self) if index < 0 else index
        return self.at(normalized)

    def __iter__(self) -> Iterator[LifecycleEvent]:
        return iter(self._events[: self._current + 1])

    def at(self, index: int) -> LifecycleEvent:
        if type(index) is not int:
            raise TypeError("event index must be an integer")
        if index < 0:
            raise IndexError("event index is before the trace")
        if index > self._current:
            raise FutureAccessError(f"causal adapter requested event {index}")
        return self._events[index]

    def history(self) -> tuple[LifecycleEvent, ...]:
        return self._events[: self._current + 1]


@runtime_checkable
class BaselineAdapter(Protocol):
    """Canonical causal adapter interface used by matched replay."""

    method_id: str
    role: Literal["causal", "upper_reference", "latency_control"]

    def initialize(self, contract: FrozenComparisonContract) -> None: ...

    def apply_event(
        self,
        event: LifecycleEvent,
        view: CausalEventView,
    ) -> EventReceipt: ...

    def copy_snapshot(self) -> MethodSnapshot: ...

    def score_probe(
        self,
        snapshot: MethodSnapshot,
        probe: ProbeEvent,
    ) -> ProbeResult: ...

    def export_online_state(self) -> bytes: ...

    def import_online_state(self, payload: bytes) -> None: ...

    def state_ledger(self) -> ExactByteLedger: ...

    def close(self) -> None: ...


__all__ = [
    "BaselineAdapter",
    "CausalEventView",
    "EventReceipt",
    "ExactByteLedger",
    "FrozenComparisonContract",
    "FutureAccessError",
    "MethodSnapshot",
    "ProbeResult",
]
