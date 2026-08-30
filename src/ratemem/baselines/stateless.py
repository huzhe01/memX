"""Explicitly nondeployable stateless-amortizer latency control."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import torch
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

Outcome = Literal[
    "created",
    "updated",
    "read",
    "deleted",
    "rejected",
    "evicted",
    "stale_handle",
]


class RetainedSupportProvider(Protocol):
    serialized_bytes: int

    def support_for_handle(self, handle: str) -> Sequence[Tensor]: ...

    def description_for_handle(self, handle: str) -> Tensor: ...


class Amortizer(Protocol):
    checkpoint_sha256: str
    shared_trained_bytes: int

    def __call__(self, support: Sequence[Tensor], description: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class StatelessHandle:
    handle: str
    description_id: str
    created_event: int
    last_read_event: int
    update_count: int


class StatelessAmortizerAdapter:
    """Recomputes a code on every read and discloses retained support storage."""

    method_id = "stateless_amortizer"
    role: Literal["latency_control"] = "latency_control"

    def __init__(
        self,
        support_provider: RetainedSupportProvider,
        amortizer: Amortizer,
        backbone: BackboneRunner,
    ) -> None:
        if (
            type(support_provider.serialized_bytes) is not int
            or support_provider.serialized_bytes < 0
        ):
            raise ValueError("retained support byte count must be a nonnegative integer")
        if type(amortizer.shared_trained_bytes) is not int or amortizer.shared_trained_bytes < 0:
            raise ValueError("amortizer shared trained bytes must be nonnegative")
        if len(amortizer.checkpoint_sha256) != 64:
            raise ValueError("amortizer checkpoint hash is invalid")
        self.support_provider = support_provider
        self.amortizer = amortizer
        self.backbone = backbone
        self.external_support_bytes = support_provider.serialized_bytes
        self.shared_trained_bytes = amortizer.shared_trained_bytes
        self._contract: FrozenComparisonContract | None = None
        self._handles: dict[str, StatelessHandle] = {}
        self._last_event_index: int | None = None
        self._snapshots: dict[str, bytes] = {}
        self._closed = False

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("stateless adapter cannot be initialized twice or after close")
        if self.amortizer.checkpoint_sha256 != contract.amortizer_sha256:
            raise ValueError("stateless amortizer differs from the comparison contract")
        if self.backbone.backbone_id != contract.backbone_id:
            raise ValueError("stateless backbone differs from the comparison contract")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty state")

    def _require_active(self) -> FrozenComparisonContract:
        if self._contract is None or self._closed:
            raise RuntimeError("stateless adapter is not active")
        return self._contract

    def _components(
        self,
        handles: dict[str, StatelessHandle] | None = None,
    ) -> dict[str, list[object]]:
        selected = self._handles if handles is None else handles
        components = empty_components()
        for handle in sorted(selected):
            record = selected[handle]
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
        components["controller_state"].append(
            {
                "policy": "lru",
                "last_event_index": self._last_event_index,
                "external_support_disclosed": True,
            }
        )
        return components

    def _export_handles(self, handles: dict[str, StatelessHandle]) -> bytes:
        return export_state(self._components(handles))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export_handles(self._handles)

    def state_ledger(self) -> ExactByteLedger:
        payload = self.export_online_state()
        return ledger_from_export(
            payload,
            shared_trained_bytes=self.shared_trained_bytes,
            external_support_bytes=self.external_support_bytes,
        )

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller = components["controller_state"]
        if (
            len(controller) != 1
            or not isinstance(controller[0], dict)
            or controller[0].get("policy") != "lru"
            or controller[0].get("external_support_disclosed") is not True
        ):
            raise ValueError("stateless controller state is invalid")
        last_event = controller[0].get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("stateless last event index is invalid")
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
        if set(metadata) != set(descriptions):
            raise ValueError("stateless exported handle sets differ")
        restored = {
            handle: StatelessHandle(
                handle=handle,
                description_id=str(descriptions[handle]["description_id"]),
                created_event=int(metadata[handle]["created_event"]),
                last_read_event=int(metadata[handle]["last_read_event"]),
                update_count=int(metadata[handle]["update_count"]),
            )
            for handle in sorted(metadata)
        }
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export_handles(restored) != payload:
            self._last_event_index = prior_event
            raise ValueError("stateless state does not roundtrip canonically")
        self._handles = restored

    def _fit_with_eviction(
        self,
        handles: dict[str, StatelessHandle],
        *,
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, StatelessHandle] | None, tuple[str, ...]]:
        candidate = dict(handles)
        evicted: list[str] = []
        while len(self._export_handles(candidate)) > budget:
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

    def _amortize(self, handle: str) -> Tensor:
        support = self.support_provider.support_for_handle(handle)
        description = self.support_provider.description_for_handle(handle)
        code = self.amortizer(support, description)
        if not isinstance(code, Tensor) or code.shape != (480,) or not torch.isfinite(code).all():
            raise ValueError("stateless amortizer must return one finite 480-vector")
        return code.detach()

    def _generate(self, handle: str, prompt_id: str, seed: int) -> tuple[str, str]:
        contract = self._require_active()
        code = self._amortize(handle)
        code_array = code.cpu().contiguous().numpy()
        code_sha = hashlib.sha256(code_array.tobytes(order="C")).hexdigest()
        self.backbone.install_code(code)
        try:
            sample = self.backbone.generate(
                prompt_id,
                seed,
                sampler_id=contract.sampler_id,
                cfg_scale=contract.cfg_scale,
                steps=contract.denoising_steps,
            )
        finally:
            self.backbone.clear_code()
        sample_array = sample.detach().cpu().contiguous().numpy()
        return code_sha, hashlib.sha256(sample_array.tobytes(order="C")).hexdigest()

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract = self._require_active()
        if isinstance(event, ProbeEvent):
            raise TypeError("probe events must use score_probe")
        if event.event_index != view.current_index or view.at(event.event_index) != event:
            raise ValueError("event and causal view are not aligned")
        validate_operational_event_order(self._last_event_index, event, view)
        before = self.state_ledger()
        self._last_event_index = event.event_index
        handles = dict(self._handles)
        affected: tuple[str, ...] = ()
        evicted: tuple[str, ...] = ()
        decoded_sha: str | None = None
        generated_sha: str | None = None
        outcome: Outcome
        if isinstance(event, CreateEvent):
            if event.handle in handles:
                outcome = "rejected"
            else:
                handles[event.handle] = StatelessHandle(
                    event.handle,
                    event.description_id,
                    event.event_index,
                    event.event_index,
                    0,
                )
                fitted, evicted = self._fit_with_eviction(
                    handles,
                    protected_handle=event.handle,
                    budget=contract.byte_budget,
                )
                if fitted is None:
                    handles.pop(event.handle)
                    outcome = "rejected"
                    evicted = ()
                else:
                    handles = fitted
                    affected = (event.handle,)
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = handles.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                handles[event.handle] = replace(
                    previous,
                    update_count=previous.update_count + 1,
                )
                affected = (event.handle,)
                outcome = "updated"
        elif isinstance(event, ReadEvent):
            previous = handles.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                decoded_sha, generated_sha = self._generate(
                    event.handle,
                    event.prompt_id,
                    event.generation_seed,
                )
                handles[event.handle] = replace(
                    previous,
                    last_read_event=event.event_index,
                )
                affected = (event.handle,)
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in handles:
                outcome = "stale_handle"
            else:
                del handles[event.handle]
                affected = (event.handle,)
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported stateless event: {type(event).__name__}")
        self._handles = handles
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("stateless adapter exceeded the exact byte budget")
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
        token = hashlib.sha256(b"stateless-snapshot-v1\0" + payload).hexdigest()
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
        handles = {
            row
            for row in decode_state(payload)["handles"]
            if isinstance(row, str)
        }
        if probe.handle not in handles:
            raise ValueError("probe handle is absent from the stateless snapshot")
        _code_sha, sample_sha = self._generate(
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
            generated_sample_sha256=sample_sha,
            update_usage=False,
        )

    def close(self) -> None:
        self._handles.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "Amortizer",
    "RetainedSupportProvider",
    "StatelessAmortizerAdapter",
    "StatelessHandle",
]
