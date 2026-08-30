"""RateMem implementation of the canonical causal baseline adapter protocol."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ratemem.baselines.ledger import (
    ONLINE_COMPONENT_NAMES,
    decode_state,
    export_state,
    ledger_from_export,
)
from ratemem.baselines.protocol import (
    BaselineAdapter,
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
from ratemem.method.codec import HardIncidence, RateMemHardCodec
from ratemem.method.controller import ControllerDecision, RateMemController
from ratemem.method.proposal import CausalCandidateProposer
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet


@runtime_checkable
class TargetCodePredictor(Protocol):
    def predict(
        self,
        support_image_ids: Sequence[str],
        description_id: str | None,
    ) -> NDArray[np.float32]: ...


@runtime_checkable
class GenerationBackend(Protocol):
    def generate(
        self,
        adapter_code: NDArray[np.float32],
        prompt_id: str,
        seed: int,
    ) -> bytes: ...


def _component_records(
    state: MemoryState,
    *,
    budget_bytes: int,
    last_event_index: int,
) -> Mapping[str, Sequence[object]]:
    handles = tuple(sorted(state.bases))
    packet_ids = tuple(sorted(state.packets))
    reference_counts = {
        packet_id: sum(
            edge.packet_id == packet_id for edge in state.incidences.values()
        )
        for packet_id in packet_ids
    }
    records: dict[str, Sequence[object]] = {
        name: () for name in ONLINE_COMPONENT_NAMES
    }
    records.update(
        {
            "base_codes": tuple(
                [state.bases[handle].payload] for handle in handles
            ),
            "packet_payloads": tuple(
                [state.packets[packet_id].payload] for packet_id in packet_ids
            ),
            "packet_hashes": tuple([packet_id] for packet_id in packet_ids),
            "incidences_gains": tuple(
                [edge.handle, edge.packet_id, edge.gain_q]
                for edge in sorted(
                    state.incidences.values(),
                    key=lambda row: (row.handle, row.packet_id),
                )
            ),
            "handles": tuple([handle] for handle in handles),
            "usage_age": tuple(
                [state.bases[handle].reads, state.bases[handle].created_at]
                for handle in handles
            ),
            "reference_counts": tuple(
                [packet_id, reference_counts[packet_id]]
                for packet_id in packet_ids
            ),
            "controller_state": (["budget_bytes", budget_bytes],),
            "allocator_state": (
                ["last_event_index", last_event_index],
                ["selected_packet_ids", list(packet_ids)],
            ),
        }
    )
    return records


def _restore_state(
    records: Mapping[str, Sequence[object]],
) -> tuple[MemoryState, int, int]:
    handles = [str(row[0]) for row in records["handles"]]  # type: ignore[index]
    base_payloads = [bytes(row[0]) for row in records["base_codes"]]  # type: ignore[index]
    usage = [
        (int(row[0]), int(row[1]))  # type: ignore[index]
        for row in records["usage_age"]
    ]
    if not len(handles) == len(base_payloads) == len(usage):
        raise ValueError("base, handle, and usage records are misaligned")
    bases = {
        handle: BaseRecord(handle, payload, reads, created_at)
        for handle, payload, (reads, created_at) in zip(
            handles,
            base_payloads,
            usage,
            strict=True,
        )
    }
    packet_ids = [str(row[0]) for row in records["packet_hashes"]]  # type: ignore[index]
    packet_payloads = [
        bytes(row[0]) for row in records["packet_payloads"]  # type: ignore[index]
    ]
    if len(packet_ids) != len(packet_payloads):
        raise ValueError("packet hashes and payloads are misaligned")
    packets = {
        packet_id: Packet(packet_id, packet_payload)
        for packet_id, packet_payload in zip(
            packet_ids,
            packet_payloads,
            strict=True,
        )
    }
    if any(
        hashlib.sha256(packet.payload).hexdigest() != packet.packet_id
        for packet in packets.values()
    ):
        raise ValueError("packet payload hash does not match its canonical identity")
    incidences = {
        (str(row[0]), str(row[1])): Incidence(  # type: ignore[index]
            str(row[0]),  # type: ignore[index]
            str(row[1]),  # type: ignore[index]
            int(row[2]),  # type: ignore[index]
        )
        for row in records["incidences_gains"]
    }
    expected_references = {
        str(row[0]): int(row[1])  # type: ignore[index]
        for row in records["reference_counts"]
    }
    actual_references = {
        packet_id: sum(
            edge.packet_id == packet_id for edge in incidences.values()
        )
        for packet_id in packets
    }
    if expected_references != actual_references:
        raise ValueError("packet reference counts do not match incidences")
    def pairs_to_mapping(rows: Sequence[object], label: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for row in rows:
            if type(row) is not list or len(row) != 2 or type(row[0]) is not str:
                raise ValueError(f"{label} must contain canonical key-value pairs")
            if row[0] in result:
                raise ValueError(f"{label} repeats a key")
            result[row[0]] = row[1]
        return result

    controller = pairs_to_mapping(records["controller_state"], "controller state")
    allocator = pairs_to_mapping(records["allocator_state"], "allocator state")
    if set(controller) != {"budget_bytes"} or set(allocator) != {
        "last_event_index",
        "selected_packet_ids",
    }:
        raise ValueError("controller or allocator state fields changed")
    if allocator["selected_packet_ids"] != packet_ids:
        raise ValueError("allocator packet selection differs from restored state")
    budget_value = controller["budget_bytes"]
    last_event_value = allocator["last_event_index"]
    if type(budget_value) is not int or type(last_event_value) is not int:
        raise ValueError("controller budget and last event index must be exact integers")
    return (
        MemoryState(bases=bases, packets=packets, incidences=incidences),
        budget_value,
        last_event_value,
    )


class RateMemAdapter(BaselineAdapter):
    method_id: str = "ratemem_v1"
    role: Literal["causal", "upper_reference", "latency_control"] = "causal"

    def __init__(
        self,
        predictor: TargetCodePredictor,
        generation_backend: GenerationBackend,
        codec: RateMemHardCodec,
        controller_factory: Callable[[int], RateMemController],
        *,
        shared_trained_bytes: int,
    ) -> None:
        if not isinstance(predictor, TargetCodePredictor):
            raise TypeError("predictor does not implement TargetCodePredictor")
        if not isinstance(generation_backend, GenerationBackend):
            raise TypeError("generation backend does not implement GenerationBackend")
        if type(codec) is not RateMemHardCodec:
            raise TypeError("codec must be an exact RateMemHardCodec")
        if not callable(controller_factory):
            raise TypeError("controller_factory must be callable")
        if type(shared_trained_bytes) is not int or shared_trained_bytes < 0:
            raise ValueError("shared_trained_bytes must be a nonnegative exact integer")
        self.predictor = predictor
        self.generation_backend = generation_backend
        self.codec = codec
        self.proposer = CausalCandidateProposer(codec)
        self.controller_factory = controller_factory
        self.shared_trained_bytes = shared_trained_bytes
        self.contract: FrozenComparisonContract | None = None
        self.controller: RateMemController | None = None
        self.state = MemoryState()
        self.last_event_index = -1
        self._snapshot_states: dict[str, MemoryState] = {}

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self.contract is not None:
            raise RuntimeError("adapter is already initialized")
        if type(contract) is not FrozenComparisonContract:
            raise TypeError("contract must be an exact FrozenComparisonContract")
        self.contract = contract
        self.controller = self.controller_factory(contract.byte_budget)
        if type(self.controller) is not RateMemController:
            raise TypeError("controller_factory must return an exact RateMemController")
        self.state = MemoryState()
        self.last_event_index = -1
        self._snapshot_states.clear()
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self.close()
            raise ValueError("byte budget cannot hold the canonical empty state")

    def _require_runtime(
        self,
    ) -> tuple[FrozenComparisonContract, RateMemController]:
        if self.contract is None or self.controller is None:
            raise RuntimeError("adapter is not initialized")
        return self.contract, self.controller

    def _export(self, state: MemoryState, last_event_index: int) -> bytes:
        contract, _ = self._require_runtime()
        return export_state(
            _component_records(
                state,
                budget_bytes=contract.byte_budget,
                last_event_index=last_event_index,
            )
        )

    def export_online_state(self) -> bytes:
        return self._export(self.state, self.last_event_index)

    def import_online_state(self, payload: bytes) -> None:
        contract, _ = self._require_runtime()
        records = decode_state(payload)
        state, budget_bytes, last_event_index = _restore_state(records)
        if export_state(records) != payload:
            raise ValueError("online state is not in canonical form")
        if budget_bytes != contract.byte_budget:
            raise ValueError("imported state belongs to another byte budget")
        if last_event_index < -1:
            raise ValueError("imported last event index is invalid")
        ledger = ledger_from_export(
            payload,
            shared_trained_bytes=self.shared_trained_bytes,
            external_support_bytes=0,
        )
        if ledger.online_state_bytes > contract.byte_budget:
            raise ValueError("imported canonical state exceeds the byte budget")
        self.state = state
        self.last_event_index = last_event_index
        self._snapshot_states.clear()

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(
            self.export_online_state(),
            shared_trained_bytes=self.shared_trained_bytes,
            external_support_bytes=0,
        )

    def copy_snapshot(self) -> MethodSnapshot:
        contract, _ = self._require_runtime()
        if self.last_event_index < 0:
            raise RuntimeError("cannot snapshot before the first operational event")
        ledger = self.state_ledger()
        token = "ratemem-snapshot-" + ledger.online_state_sha256
        self._snapshot_states[token] = self.state
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self.last_event_index,
            state_sha256=ledger.online_state_sha256,
            online_state_bytes=ledger.online_state_bytes,
            opaque_snapshot_token=token,
        )

    def _decoded_code(
        self,
        handle: str,
        state: MemoryState,
    ) -> NDArray[np.float32]:
        base = state.bases[handle]
        rows: list[HardIncidence] = []
        for edge in sorted(
            state.incidences.values(),
            key=lambda row: (row.handle, row.packet_id),
        ):
            if edge.handle == handle:
                packet = state.packets[edge.packet_id]
                group, stage, entry = self.codec.dictionary.validate_packet(packet)
                rows.append(
                    HardIncidence(
                        edge,
                        packet,
                        group,
                        stage,
                        entry,
                        0.0,
                    )
                )
        return self.codec.decode(base.payload, tuple(rows))

    def _bounded_decision(
        self,
        decision: ControllerDecision,
        event_index: int,
    ) -> ControllerDecision:
        contract, _ = self._require_runtime()
        payload = self._export(decision.state, event_index)
        if len(payload) <= contract.byte_budget:
            return decision
        return ControllerDecision(state=self.state, outcome="rejected")

    def apply_event(
        self,
        event: LifecycleEvent,
        view: CausalEventView,
    ) -> EventReceipt:
        contract, controller = self._require_runtime()
        validate_operational_event_order(
            None if self.last_event_index < 0 else self.last_event_index,
            event,
            view,
        )
        if len(view) == 0 or view.history()[-1] != event:
            raise ValueError("causal event view does not end at the supplied event")
        before = self.state_ledger().online_state_sha256
        generated: bytes | None = None
        decoded: NDArray[np.float32] | None = None
        if isinstance(event, CreateEvent):
            code = self.predictor.predict(
                event.support_image_ids,
                event.description_id,
            )
            decision = controller.apply_create(
                self.state,
                self.proposer.propose(
                    self.state,
                    event.handle,
                    code,
                    event.event_index,
                ),
            )
        elif isinstance(event, UpdateEvent):
            code = self.predictor.predict(event.support_image_ids, None)
            decision = controller.apply_update(
                self.state,
                self.proposer.propose(
                    self.state,
                    event.handle,
                    code,
                    event.event_index,
                ),
            )
        elif isinstance(event, ReadEvent):
            decision = controller.read(self.state, event.handle, update_usage=True)
            if decision.outcome != "stale_handle":
                decoded = self._decoded_code(event.handle, decision.state)
                generated = self.generation_backend.generate(
                    decoded,
                    event.prompt_id,
                    event.generation_seed,
                )
        elif isinstance(event, DeleteEvent):
            decision = controller.delete(self.state, event.handle)
        elif isinstance(event, ProbeEvent):
            raise ValueError("probe events must use score_probe")
        else:
            raise TypeError(f"unsupported lifecycle event: {type(event).__name__}")
        decision = self._bounded_decision(decision, event.event_index)
        self.state = decision.state
        self.last_event_index = event.event_index
        ledger = self.state_ledger()
        input_commitment = hashlib.sha256(
            canonical_json_bytes(
                {
                    "contract": contract.model_dump(mode="json"),
                    "event": event.model_dump(mode="json"),
                }
            )
        ).hexdigest()
        return EventReceipt(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=event.event_index,
            event_kind=event.kind,
            input_commitment_sha256=input_commitment,
            method_state_sha256_before=before,
            method_state_sha256_after=ledger.online_state_sha256,
            candidate_stream_sha256=contract.candidate_stream_sha256,
            outcome=decision.outcome,
            affected_handles=tuple(
                sorted({event.handle, *decision.evicted_handles})
            ),
            evicted_handles=decision.evicted_handles,
            decoded_code_sha256=(
                hashlib.sha256(
                    np.asarray(decoded, dtype="<f4").tobytes()
                ).hexdigest()
                if decoded is not None
                else None
            ),
            generated_sample_sha256=(
                hashlib.sha256(generated).hexdigest()
                if generated is not None
                else None
            ),
            ledger=ledger,
        )

    def score_probe(
        self,
        snapshot: MethodSnapshot,
        probe: ProbeEvent,
    ) -> ProbeResult:
        contract, _ = self._require_runtime()
        if (
            snapshot.method_id != self.method_id
            or snapshot.trace_id != contract.trace_id
        ):
            raise ValueError("snapshot identity does not match RateMem runtime")
        try:
            state = self._snapshot_states[snapshot.opaque_snapshot_token]
        except KeyError as error:
            raise ValueError("unknown or expired RateMem snapshot token") from error
        if snapshot.state_sha256 != hashlib.sha256(
            self._export(state, snapshot.event_index)
        ).hexdigest():
            raise ValueError("snapshot state commitment changed")
        if probe.handle not in state.bases:
            raise ValueError("probe references a stale snapshot handle")
        before = self.state_ledger()
        decoded = self._decoded_code(probe.handle, state)
        generated = self.generation_backend.generate(
            decoded,
            probe.prompt_id,
            probe.generation_seed,
        )
        after = self.state_ledger()
        if before != after:
            raise RuntimeError("probe mutated online RateMem state")
        return ProbeResult(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            probe_event_index=probe.event_index,
            snapshot_state_sha256=snapshot.state_sha256,
            input_commitment_sha256=hashlib.sha256(
                canonical_json_bytes(probe.model_dump(mode="json"))
            ).hexdigest(),
            generated_sample_sha256=hashlib.sha256(generated).hexdigest(),
            update_usage=False,
        )

    def close(self) -> None:
        self.state = MemoryState()
        self.last_event_index = -1
        self._snapshot_states.clear()
        self.controller = None
        self.contract = None
