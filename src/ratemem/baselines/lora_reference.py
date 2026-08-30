"""Per-concept LoRA optimization reference on the frozen SANA backbone."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import torch
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, PositiveFloat, model_validator
from torch import Tensor

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
from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.traces import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_ALLOWED_TENSOR_SUFFIXES = tuple(
    f".{projection}.{side}"
    for projection in ("to_q", "to_k", "to_v")
    for side in ("lora_A", "lora_B")
)
Outcome = Literal[
    "created",
    "updated",
    "read",
    "deleted",
    "rejected",
    "evicted",
    "stale_handle",
]


class PrimaryBackboneError(ValueError):
    """Raised when a contextual backbone is offered as a matched reference."""


class LoRASearchSpace(BaseModel):
    model_config = _MODEL_CONFIG

    rank: tuple[Literal[2, 4, 8, 16], ...]
    learning_rate: tuple[PositiveFloat, ...]
    steps: tuple[Literal[50, 100], ...]
    prior_preservation_weight: tuple[float, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> LoRASearchSpace:
        dimensions = (
            self.rank,
            self.learning_rate,
            self.steps,
            self.prior_preservation_weight,
        )
        if any(not values or len(values) != len(set(values)) for values in dimensions):
            raise ValueError("LoRA search dimensions must be non-empty and unique")
        cells = math.prod(len(values) for values in dimensions)
        if cells != 24:
            raise ValueError("LoRA reference search grid must contain exactly 24 cells")
        if self.prior_preservation_weight != (0.0,):
            raise ValueError("matched LoRA disables prior-preservation loss")
        return self


class LoRASearchSelector(BaseModel):
    model_config = _MODEL_CONFIG

    split: Literal["validation"]
    endpoint: Literal["request_weighted_identity"]
    prompt_constraint_source: Literal["evaluation_lock"]


class LoRAReferenceConfig(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    method_id: Literal["per_concept_lora"]
    target_suffixes: tuple[Literal["to_q", "to_k", "to_v"], ...]
    precision: Literal["bfloat16"]
    gradient_checkpointing: Literal[True]
    optimizer: Literal["adamw"]
    discard_optimizer_after_event: Literal[True]
    create_initial_state: Literal["zero_lora"]
    update_initial_state: Literal["resident_lora"]
    update_support_rule: Literal["new_evidence_only"]
    search_space: LoRASearchSpace
    search_selector: LoRASearchSelector

    @model_validator(mode="after")
    def validate_targets(self) -> LoRAReferenceConfig:
        if self.target_suffixes != ("to_q", "to_k", "to_v"):
            raise ValueError("matched LoRA must target SANA q/k/v projections only")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class SelectedLoRAHyperparameters(BaseModel):
    model_config = _MODEL_CONFIG

    rank: Literal[2, 4, 8, 16]
    learning_rate: PositiveFloat
    steps: Literal[50, 100]
    prior_preservation_weight: float = 0.0

    @model_validator(mode="after")
    def validate_prior_weight(self) -> SelectedLoRAHyperparameters:
        if self.prior_preservation_weight != 0.0:
            raise ValueError("matched LoRA disables prior-preservation loss")
        return self


def load_lora_reference_config(path: str | Path) -> LoRAReferenceConfig:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return LoRAReferenceConfig.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"invalid LoRA reference config: {error}") from error


def require_primary_lora_contract(contract: object) -> None:
    """Fail before loading weights if a run is not the fixed primary SANA route."""

    if getattr(contract, "backbone_id", None) != "sana_1_5_1_6b":
        raise PrimaryBackboneError("primary comparisons require sana_1_5_1_6b")
    revision = getattr(contract, "backbone_revision", None)
    if (
        type(revision) is not str
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise PrimaryBackboneError("primary SANA revision must be immutable")


@dataclass(frozen=True, slots=True)
class LoRATensor:
    """One persistent little-endian BF16 LoRA tensor."""

    name: str
    shape: tuple[int, ...]
    bf16_payload: bytes

    def __post_init__(self) -> None:
        if not self.name.endswith(_ALLOWED_TENSOR_SUFFIXES):
            raise ValueError(f"LoRA tensor is outside frozen q/k/v targets: {self.name}")
        if not self.shape or any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("LoRA tensor shape must be non-empty and positive")
        elements = math.prod(self.shape)
        if type(self.bf16_payload) is not bytes or len(self.bf16_payload) != 2 * elements:
            raise ValueError("LoRA tensor payload is not exact BF16 storage")
        values = self.as_float32()
        if not torch.isfinite(values).all():
            raise ValueError("LoRA tensor payload must be finite")

    @classmethod
    def from_tensor(cls, name: str, value: Tensor) -> LoRATensor:
        if not isinstance(value, Tensor) or value.ndim < 1 or not torch.isfinite(value).all():
            raise ValueError("LoRA state tensors must be finite non-scalar tensors")
        bf16 = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        payload = bf16.view(torch.uint8).numpy().tobytes(order="C")
        return cls(name=name, shape=tuple(bf16.shape), bf16_payload=payload)

    def as_float32(self) -> Tensor:
        values = torch.frombuffer(bytearray(self.bf16_payload), dtype=torch.bfloat16).clone()
        return values.reshape(self.shape).to(torch.float32)


@dataclass(frozen=True, slots=True)
class LoRAState:
    tensors: tuple[LoRATensor, ...]

    def __post_init__(self) -> None:
        names = tuple(row.name for row in self.tensors)
        if not names or names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("LoRA tensors must be non-empty, uniquely named, and sorted")

    @classmethod
    def from_tensors(cls, tensors: dict[str, Tensor]) -> LoRAState:
        return cls(tuple(LoRATensor.from_tensor(name, tensors[name]) for name in sorted(tensors)))

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256(b"ratemem-lora-state-v1\0")
        for row in self.tensors:
            for value in (
                row.name.encode("utf-8"),
                canonical_json_bytes(row.shape),
                row.bf16_payload,
            ):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoRAFitResult:
    state: LoRAState
    frozen_parameter_sha256_before: str
    frozen_parameter_sha256_after: str
    changed_parameter_names: tuple[str, ...]
    optimizer_state_discarded: bool

    def __post_init__(self) -> None:
        for value in (
            self.frozen_parameter_sha256_before,
            self.frozen_parameter_sha256_after,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("trainer base-parameter hash is invalid")
        if self.frozen_parameter_sha256_before != self.frozen_parameter_sha256_after:
            raise ValueError("LoRA fitting changed frozen backbone parameters")
        state_names = tuple(row.name for row in self.state.tensors)
        if tuple(sorted(self.changed_parameter_names)) != state_names:
            raise ValueError("trainer changed-parameter list differs from persistent LoRA state")
        if not self.optimizer_state_discarded:
            raise ValueError("optimizer state must be discarded after every lifecycle event")


class PerConceptLoRATrainer(Protocol):
    backbone_id: str
    backbone_revision: str

    def fit(
        self,
        support_image_ids: Sequence[str],
        description_id: str,
        initial_lora: LoRAState | None,
        config: LoRAReferenceConfig,
        hyperparameters: SelectedLoRAHyperparameters,
        seed: int,
    ) -> LoRAFitResult: ...

    def generate(
        self,
        state: LoRAState,
        prompt_id: str,
        seed: int,
        contract: FrozenComparisonContract,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class LoRARecord:
    handle: str
    state: LoRAState
    description_id: str
    created_event: int
    last_read_event: int
    update_count: int


@dataclass(frozen=True, slots=True)
class LoRAStateView:
    records: dict[str, LoRARecord]
    optimizer_state_present: Literal[False] = False


class LoRAOptimizationAdapter:
    """Independent LRU store of event-trained per-concept SANA LoRAs."""

    method_id = "per_concept_lora"
    role: Literal["causal"] = "causal"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        config: LoRAReferenceConfig,
        trainer: PerConceptLoRATrainer,
        hyperparameters: SelectedLoRAHyperparameters,
    ) -> None:
        if config.method_id != self.method_id:
            raise ValueError("LoRA config has the wrong method id")
        if hyperparameters.rank not in config.search_space.rank:
            raise ValueError("selected LoRA rank is outside the frozen search grid")
        if hyperparameters.learning_rate not in config.search_space.learning_rate:
            raise ValueError("selected LoRA learning rate is outside the frozen search grid")
        if hyperparameters.steps not in config.search_space.steps:
            raise ValueError("selected LoRA steps are outside the frozen search grid")
        self.config = config
        self.trainer = trainer
        self.hyperparameters = hyperparameters
        self._contract: FrozenComparisonContract | None = None
        self._records: dict[str, LoRARecord] = {}
        self._last_event_index: int | None = None
        self._snapshots: dict[str, bytes] = {}
        self._closed = False

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("LoRA adapter cannot be initialized twice or after close")
        require_primary_lora_contract(contract)
        if self.trainer.backbone_id != contract.backbone_id:
            raise ValueError("LoRA trainer backbone differs from the contract")
        if self.trainer.backbone_revision != contract.backbone_revision:
            raise ValueError("LoRA trainer revision differs from the contract")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty LoRA state")

    def _require_active(self) -> FrozenComparisonContract:
        if self._contract is None or self._closed:
            raise RuntimeError("LoRA adapter is not active")
        return self._contract

    def _components(
        self,
        records: dict[str, LoRARecord] | None = None,
    ) -> dict[str, list[object]]:
        selected = self._records if records is None else records
        components = empty_components()
        for handle in sorted(selected):
            record = selected[handle]
            for tensor in record.state.tensors:
                components["base_codes"].append(
                    {
                        "handle": handle,
                        "name": tensor.name,
                        "dtype": "bfloat16",
                        "shape": list(tensor.shape),
                        "data": tensor.bf16_payload,
                    }
                )
            components["handles"].append(handle)
            components["optional_tokens"].append(
                {"handle": handle, "description_id": record.description_id}
            )
            components["usage_age"].append(
                {
                    "handle": handle,
                    "created_event": record.created_event,
                    "last_read_event": record.last_read_event,
                    "update_count": record.update_count,
                }
            )
            components["checksums"].append(
                {"handle": handle, "lora_state_sha256": record.state.sha256}
            )
        components["controller_state"].append(
            {
                "policy": "lru",
                "config_sha256": self.config.sha256,
                "hyperparameters": self.hyperparameters.model_dump(mode="json"),
                "last_event_index": self._last_event_index,
                "optimizer_state_present": False,
            }
        )
        return components

    def _export_records(self, records: dict[str, LoRARecord]) -> bytes:
        return export_state(self._components(records))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export_records(self._records)

    def state_ledger(self) -> ExactByteLedger:
        payload = self.export_online_state()
        return ledger_from_export(payload, self.shared_trained_bytes, self.external_support_bytes)

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller = components["controller_state"]
        expected_hyperparameters = self.hyperparameters.model_dump(mode="json")
        if (
            len(controller) != 1
            or not isinstance(controller[0], dict)
            or controller[0].get("policy") != "lru"
            or controller[0].get("config_sha256") != self.config.sha256
            or controller[0].get("hyperparameters") != expected_hyperparameters
            or controller[0].get("optimizer_state_present") is not False
        ):
            raise ValueError("LoRA controller state differs from this adapter")
        last_event = controller[0].get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("LoRA last event index is invalid")
        raw_handles = tuple(components["handles"])
        if any(type(handle) is not str for handle in raw_handles):
            raise ValueError("LoRA exported handles are invalid")
        handles = tuple(str(handle) for handle in raw_handles)
        if handles != tuple(
            sorted(set(handles))
        ):
            raise ValueError("LoRA exported handles are invalid")
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
        tensor_rows: dict[str, list[LoRATensor]] = {str(handle): [] for handle in handles}
        for raw in components["base_codes"]:
            if not isinstance(raw, dict):
                raise ValueError("LoRA tensor row is invalid")
            handle = raw.get("handle")
            shape = raw.get("shape")
            if (
                type(handle) is not str
                or handle not in tensor_rows
                or raw.get("dtype") != "bfloat16"
                or type(raw.get("name")) is not str
                or type(raw.get("data")) is not bytes
                or not isinstance(shape, list)
            ):
                raise ValueError("LoRA tensor row is invalid")
            tensor_rows[handle].append(
                LoRATensor(
                    name=raw["name"],
                    shape=tuple(int(size) for size in shape),
                    bf16_payload=raw["data"],
                )
            )
        if set(handles) != set(metadata) or set(handles) != set(descriptions):
            raise ValueError("LoRA exported handle sets differ")
        restored: dict[str, LoRARecord] = {}
        for handle in handles:
            meta = metadata[handle]
            restored[handle] = LoRARecord(
                handle=handle,
                state=LoRAState(tuple(tensor_rows[handle])),
                description_id=str(descriptions[handle]["description_id"]),
                created_event=int(meta["created_event"]),
                last_read_event=int(meta["last_read_event"]),
                update_count=int(meta["update_count"]),
            )
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export_records(restored) != payload:
            self._last_event_index = prior_event
            raise ValueError("LoRA state does not roundtrip canonically")
        self._records = restored

    def _fit_with_eviction(
        self,
        records: dict[str, LoRARecord],
        *,
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, LoRARecord] | None, tuple[str, ...]]:
        candidate = dict(records)
        evicted: list[str] = []
        while len(self._export_records(candidate)) > budget:
            eligible = [row for handle, row in candidate.items() if handle != protected_handle]
            if not eligible:
                return None, ()
            victim = min(
                eligible,
                key=lambda row: (row.last_read_event, row.created_event, row.handle),
            )
            del candidate[victim.handle]
            evicted.append(victim.handle)
        return candidate, tuple(evicted)

    def _training_seed(self, event: CreateEvent | UpdateEvent) -> int:
        contract = self._require_active()
        material = canonical_json_bytes(
            {
                "trace_id": contract.trace_id,
                "event_index": event.event_index,
                "handle": event.handle,
                "config_sha256": self.config.sha256,
                "hyperparameters": self.hyperparameters.model_dump(mode="json"),
            }
        )
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _fit(
        self,
        event: CreateEvent | UpdateEvent,
        *,
        description_id: str,
        initial: LoRAState | None,
    ) -> LoRAState:
        result = self.trainer.fit(
            event.support_image_ids,
            description_id,
            initial,
            self.config,
            self.hyperparameters,
            self._training_seed(event),
        )
        if type(result) is not LoRAFitResult:
            raise TypeError("LoRA trainer must return an exact LoRAFitResult")
        return result.state

    def _generate_sha(self, state: LoRAState, prompt_id: str, seed: int) -> str:
        generated = self.trainer.generate(
            state,
            prompt_id,
            seed,
            self._require_active(),
        )
        if not isinstance(generated, Tensor) or not torch.isfinite(generated).all():
            raise ValueError("LoRA trainer returned an invalid generated tensor")
        payload = generated.detach().cpu().contiguous().numpy().tobytes(order="C")
        return hashlib.sha256(payload).hexdigest()

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract = self._require_active()
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
                state = self._fit(event, description_id=event.description_id, initial=None)
                records[event.handle] = LoRARecord(
                    event.handle,
                    state,
                    event.description_id,
                    event.event_index,
                    event.event_index,
                    0,
                )
                fitted, evicted = self._fit_with_eviction(
                    records,
                    protected_handle=event.handle,
                    budget=contract.byte_budget,
                )
                if fitted is None:
                    records.pop(event.handle)
                    outcome = "rejected"
                    evicted = ()
                else:
                    records = fitted
                    affected = (event.handle,)
                    decoded_sha = state.sha256
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                state = self._fit(
                    event,
                    description_id=previous.description_id,
                    initial=previous.state,
                )
                records[event.handle] = replace(
                    previous,
                    state=state,
                    update_count=previous.update_count + 1,
                )
                fitted, evicted = self._fit_with_eviction(
                    records,
                    protected_handle=event.handle,
                    budget=contract.byte_budget,
                )
                if fitted is None:
                    records[event.handle] = previous
                    outcome = "rejected"
                    evicted = ()
                else:
                    records = fitted
                    affected = (event.handle,)
                    decoded_sha = state.sha256
                    outcome = "updated"
        elif isinstance(event, ReadEvent):
            record = records.get(event.handle)
            if record is None:
                outcome = "stale_handle"
            else:
                generated_sha = self._generate_sha(
                    record.state,
                    event.prompt_id,
                    event.generation_seed,
                )
                records[event.handle] = replace(record, last_read_event=event.event_index)
                affected = (event.handle,)
                decoded_sha = record.state.sha256
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in records:
                outcome = "stale_handle"
            else:
                del records[event.handle]
                affected = (event.handle,)
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported LoRA event: {type(event).__name__}")
        self._records = records
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("LoRA adapter exceeded the exact byte budget")
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
        token = hashlib.sha256(b"lora-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    def _state_from_snapshot(self, payload: bytes, handle: str) -> LoRAState:
        components = decode_state(payload)
        tensors: list[LoRATensor] = []
        for raw in components["base_codes"]:
            if isinstance(raw, dict) and raw.get("handle") == handle:
                shape = raw.get("shape")
                if (
                    type(raw.get("name")) is not str
                    or type(raw.get("data")) is not bytes
                    or not isinstance(shape, list)
                ):
                    raise ValueError("LoRA snapshot tensor is invalid")
                tensors.append(
                    LoRATensor(
                        raw["name"],
                        tuple(int(size) for size in shape),
                        raw["data"],
                    )
                )
        if not tensors:
            raise ValueError("probe references a handle absent from its snapshot")
        return LoRAState(tuple(sorted(tensors, key=lambda row: row.name)))

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        generated_sha = self._generate_sha(
            self._state_from_snapshot(payload, probe.handle),
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

    def inspect_state(self) -> LoRAStateView:
        self._require_active()
        return LoRAStateView(records=dict(self._records))

    def close(self) -> None:
        self._records.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "LoRAFitResult",
    "LoRAOptimizationAdapter",
    "LoRARecord",
    "LoRAReferenceConfig",
    "LoRASearchSelector",
    "LoRASearchSpace",
    "LoRAState",
    "LoRAStateView",
    "LoRATensor",
    "PerConceptLoRATrainer",
    "PrimaryBackboneError",
    "SelectedLoRAHyperparameters",
    "load_lora_reference_config",
    "require_primary_lora_contract",
]
