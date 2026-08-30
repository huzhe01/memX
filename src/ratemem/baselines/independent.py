"""Independent uncompressed-code FIFO, LRU, and LRUA controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

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

Policy = Literal["fifo", "lru", "lrua"]
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
class CodeRecord:
    handle: str
    bf16_payload: bytes
    created_event: int
    last_read_event: int
    decayed_usage: float
    update_count: int


@dataclass(frozen=True, slots=True)
class InspectedCode:
    dtype: Literal["bfloat16"]
    raw_sha256: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class IndependentStateView:
    codes: dict[str, InspectedCode]
    records: dict[str, CodeRecord]


def age_usage(records: dict[str, CodeRecord], decay: float) -> dict[str, CodeRecord]:
    return {
        handle: replace(record, decayed_usage=record.decayed_usage * decay)
        for handle, record in records.items()
    }


def victim_key(record: CodeRecord, policy: Policy) -> tuple[float, int, int, str]:
    if policy == "fifo":
        return (
            float(record.created_event),
            record.created_event,
            record.last_read_event,
            record.handle,
        )
    if policy == "lru":
        return (
            float(record.last_read_event),
            record.created_event,
            record.last_read_event,
            record.handle,
        )
    return (
        record.decayed_usage,
        record.last_read_event,
        record.created_event,
        record.handle,
    )


def _bf16_payload(code: NDArray[np.generic]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(code)).to(dtype=torch.bfloat16)
    return tensor.view(torch.uint8).contiguous().numpy().tobytes(order="C")


def _decode_bf16(payload: bytes) -> torch.Tensor:
    if len(payload) % 2:
        raise ValueError("BF16 code payload must contain whole elements")
    return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone().to(torch.float32)


class IndependentCodeCacheAdapter:
    """Byte-exact independent 480-code cache with a deterministic victim policy."""

    role: Literal["causal"] = "causal"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        method_id: str,
        policy: Policy,
        *,
        lrua_decay: float = 0.99,
        shared_inputs: SharedInputReader | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        expected_method = f"independent_{policy}"
        if method_id != expected_method:
            raise ValueError(f"method id must be {expected_method}")
        if policy not in {"fifo", "lru", "lrua"}:
            raise ValueError("unknown independent cache policy")
        if not np.isfinite(lrua_decay) or not 0.0 < lrua_decay <= 1.0:
            raise ValueError("LRUA decay must be finite in (0, 1]")
        self.method_id = method_id
        self.policy = policy
        self.lrua_decay = float(lrua_decay)
        self._reader = shared_inputs
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._records: dict[str, CodeRecord] = {}
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
            raise RuntimeError("independent adapter has no shared-input reader")
        if self._reader.manifest.candidate_stream_sha256 != contract.candidate_stream_sha256:
            raise ValueError("shared-input candidate stream differs from the contract")
        if self._reader.manifest.trace_id != contract.trace_id:
            raise ValueError("shared-input trace differs from the contract")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty state")

    def _require_active(self) -> tuple[FrozenComparisonContract, SharedInputReader]:
        if self._contract is None or self._reader is None or self._closed:
            raise RuntimeError("adapter is not active")
        return self._contract, self._reader

    def _components(self, records: dict[str, CodeRecord] | None = None) -> dict[str, list[object]]:
        selected = self._records if records is None else records
        components = empty_components()
        for handle in sorted(selected):
            record = selected[handle]
            components["base_codes"].append(
                {"handle": handle, "dtype": "bfloat16", "data": record.bf16_payload}
            )
            components["handles"].append(handle)
            components["usage_age"].append(
                {
                    "handle": handle,
                    "created_event": record.created_event,
                    "last_read_event": record.last_read_event,
                    "decayed_usage": record.decayed_usage,
                    "update_count": record.update_count,
                }
            )
            components["checksums"].append(
                {"handle": handle, "sha256": hashlib.sha256(record.bf16_payload).hexdigest()}
            )
        components["controller_state"].append(
            {
                "policy": self.policy,
                "lrua_decay": self.lrua_decay,
                "last_event_index": self._last_event_index,
            }
        )
        return components

    def _export_records(self, records: dict[str, CodeRecord]) -> bytes:
        return export_state(self._components(records))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export_records(self._records)

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("independent controller state is invalid")
        controller_row = controller[0]
        if (
            controller_row.get("policy") != self.policy
            or controller_row.get("lrua_decay") != self.lrua_decay
        ):
            raise ValueError("independent controller state differs from the adapter")
        last_event_index = controller_row.get("last_event_index")
        if last_event_index is not None and (
            type(last_event_index) is not int or last_event_index < 0
        ):
            raise ValueError("independent last event index is invalid")
        code_rows = {
            str(row["handle"]): row
            for row in components["base_codes"]
            if isinstance(row, dict)
        }
        metadata_rows = {
            str(row["handle"]): row
            for row in components["usage_age"]
            if isinstance(row, dict)
        }
        if set(code_rows) != set(metadata_rows):
            raise ValueError("independent code and metadata handles differ")
        restored: dict[str, CodeRecord] = {}
        for handle in sorted(code_rows):
            code = code_rows[handle]
            metadata = metadata_rows[handle]
            if code.get("dtype") != "bfloat16" or type(code.get("data")) is not bytes:
                raise ValueError("independent code record is invalid")
            restored[handle] = CodeRecord(
                handle=handle,
                bf16_payload=code["data"],
                created_event=int(metadata["created_event"]),
                last_read_event=int(metadata["last_read_event"]),
                decayed_usage=float(metadata["decayed_usage"]),
                update_count=int(metadata["update_count"]),
            )
        prior_last_event = self._last_event_index
        self._last_event_index = last_event_index
        if self._export_records(restored) != payload:
            self._last_event_index = prior_last_event
            raise ValueError("independent state does not roundtrip canonically")
        self._records = restored

    def state_ledger(self) -> ExactByteLedger:
        payload = self.export_online_state()
        return ledger_from_export(
            payload,
            shared_trained_bytes=self.shared_trained_bytes,
            external_support_bytes=self.external_support_bytes,
        )

    def _fit_with_eviction(
        self,
        records: dict[str, CodeRecord],
        *,
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, CodeRecord] | None, tuple[str, ...]]:
        candidate = dict(records)
        evicted: list[str] = []
        while len(self._export_records(candidate)) > budget:
            eligible = [
                record
                for handle, record in candidate.items()
                if handle != protected_handle
            ]
            if not eligible:
                return None, ()
            victim = min(eligible, key=lambda record: victim_key(record, self.policy))
            del candidate[victim.handle]
            evicted.append(victim.handle)
        return candidate, tuple(evicted)

    def _input_code(self, event_index: int, handle: str, current_index: int) -> bytes:
        _contract, reader = self._require_active()
        loaded = reader.load_event(event_index, current_index)
        if loaded.record.handle != handle:
            raise ValueError("shared-input handle differs from the lifecycle event")
        return _bf16_payload(loaded.target_code)

    def _sample_sha256(self, payload: bytes, prompt_id: str, seed: int) -> str:
        if self._backbone is None:
            return hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_backbone": True,
                        "code_sha256": hashlib.sha256(payload).hexdigest(),
                        "prompt_id": prompt_id,
                        "seed": seed,
                    }
                )
            ).hexdigest()
        contract, _reader = self._require_active()
        self._backbone.install_code(_decode_bf16(payload))
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
        array = generated.detach().cpu().contiguous().numpy()
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract, _reader = self._require_active()
        if isinstance(event, ProbeEvent):
            raise TypeError("probe events must use score_probe on a copied snapshot")
        if event.event_index != view.current_index or view.at(event.event_index) != event:
            raise ValueError("event and causal view are not aligned")
        if self._last_event_index is not None and event.event_index != self._last_event_index + 1:
            raise ValueError("events must be applied exactly once in trace order")
        before = self.state_ledger()
        self._last_event_index = event.event_index
        records = age_usage(self._records, self.lrua_decay)
        evicted: tuple[str, ...] = ()
        decoded_code_sha256: str | None = None
        generated_sample_sha256: str | None = None
        affected: tuple[str, ...] = ()
        outcome: Outcome
        if isinstance(event, CreateEvent):
            if event.handle in records:
                outcome = "rejected"
            else:
                payload = self._input_code(event.event_index, event.handle, view.current_index)
                decoded_code_sha256 = hashlib.sha256(payload).hexdigest()
                records[event.handle] = CodeRecord(
                    handle=event.handle,
                    bf16_payload=payload,
                    created_event=event.event_index,
                    last_read_event=event.event_index,
                    decayed_usage=0.0,
                    update_count=0,
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
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                payload = self._input_code(event.event_index, event.handle, view.current_index)
                decoded_code_sha256 = hashlib.sha256(payload).hexdigest()
                updated = replace(
                    previous,
                    bf16_payload=payload,
                    update_count=previous.update_count + 1,
                )
                records[event.handle] = updated
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
                    outcome = "updated"
        elif isinstance(event, ReadEvent):
            record = records.get(event.handle)
            if record is None:
                outcome = "stale_handle"
            else:
                records[event.handle] = replace(
                    record,
                    last_read_event=event.event_index,
                    decayed_usage=record.decayed_usage + 1.0,
                )
                affected = (event.handle,)
                decoded_code_sha256 = hashlib.sha256(record.bf16_payload).hexdigest()
                generated_sample_sha256 = self._sample_sha256(
                    record.bf16_payload,
                    event.prompt_id,
                    event.generation_seed,
                )
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in records:
                outcome = "stale_handle"
            else:
                del records[event.handle]
                affected = (event.handle,)
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported operational event: {type(event).__name__}")
        self._records = records
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget and affected:
            raise RuntimeError("independent adapter exceeded the exact byte budget")
        event_sha = hashlib.sha256(
            canonical_json_bytes(event.model_dump(mode="json"))
        ).hexdigest()
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
            decoded_code_sha256=decoded_code_sha256,
            generated_sample_sha256=generated_sample_sha256,
            ledger=after,
        )

    def copy_snapshot(self) -> MethodSnapshot:
        contract, _reader = self._require_active()
        if self._last_event_index is None:
            raise RuntimeError("cannot snapshot before the first event")
        payload = self.export_online_state()
        state_sha = hashlib.sha256(payload).hexdigest()
        token = hashlib.sha256(b"independent-snapshot-v1\0" + payload).hexdigest()
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
        contract, _reader = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        components = decode_state(payload)
        code_rows = {
            str(row["handle"]): row
            for row in components["base_codes"]
            if isinstance(row, dict)
        }
        row = code_rows.get(probe.handle)
        if row is None or type(row.get("data")) is not bytes:
            raise ValueError("probe references a handle absent from its snapshot")
        generated_sha = self._sample_sha256(
            row["data"],
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

    def inspect_state(self) -> IndependentStateView:
        self._require_active()
        return IndependentStateView(
            codes={
                handle: InspectedCode(
                    dtype="bfloat16",
                    raw_sha256=hashlib.sha256(record.bf16_payload).hexdigest(),
                    payload_bytes=len(record.bf16_payload),
                )
                for handle, record in self._records.items()
            },
            records=dict(self._records),
        )

    def close(self) -> None:
        self._records.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "CodeRecord",
    "IndependentCodeCacheAdapter",
    "IndependentStateView",
    "InspectedCode",
    "Policy",
    "age_usage",
    "victim_key",
]
