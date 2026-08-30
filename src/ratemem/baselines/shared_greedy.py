"""Plain marginal-density greedy control over RateMem's shared packet stream."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ratemem.allocation.objective import CoverageOracle, PacketBundle
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
from ratemem.baselines.shared_inputs import CandidatePacket, SharedInputReader
from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.traces import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)


@dataclass(frozen=True, slots=True)
class GreedyResult:
    selected_packet_ids: tuple[str, ...]
    total_cost: int
    objective_value: float


def plain_density_greedy(oracle: CoverageOracle, budget_bytes: int) -> GreedyResult:
    """Run one deterministic pass with no seed enumeration or switching term."""

    if type(oracle) is not CoverageOracle:
        raise TypeError("oracle must be an exact CoverageOracle")
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget bytes must be a nonnegative integer")
    selected: tuple[str, ...] = ()
    remaining = set(oracle.bundles)
    used = 0
    while remaining:
        feasible = [
            packet_id
            for packet_id in remaining
            if used + oracle.bundles[packet_id].cost_bytes <= budget_bytes
        ]
        if not feasible:
            break
        selected_set = frozenset(selected)
        packet_id = min(
            feasible,
            key=lambda item: (
                -(
                    oracle.exact_marginal(selected_set, item)
                    / oracle.bundles[item].cost_bytes
                ),
                item,
            ),
        )
        if oracle.exact_marginal(selected_set, packet_id) <= 0:
            break
        selected += (packet_id,)
        remaining.remove(packet_id)
        used += oracle.bundles[packet_id].cost_bytes
    return GreedyResult(selected, used, oracle.value(frozenset(selected)))


Outcome = Literal[
    "created",
    "updated",
    "read",
    "deleted",
    "rejected",
    "evicted",
    "stale_handle",
]
_RESIDUAL_HEADER = struct.Struct("<II")


@dataclass(frozen=True, slots=True)
class SharedBaseRecord:
    handle: str
    base_payload: bytes
    created_event: int
    last_read_event: int
    reads: int
    update_count: int


@dataclass(frozen=True, slots=True)
class SharedPacketState:
    packet_id: str
    payload: bytes
    dictionary_revision_sha256: str
    group: int
    stage: int
    entry: int
    gain_q_by_handle: dict[str, int]


@dataclass(frozen=True, slots=True)
class SharedGreedyStateView:
    bases: dict[str, SharedBaseRecord]
    packets: dict[str, SharedPacketState]
    selected_packet_ids: tuple[str, ...]
    ledger: ExactByteLedger

    def payload_occurrences(self, payload_sha256: str) -> int:
        return sum(
            hashlib.sha256(packet.payload).hexdigest() == payload_sha256
            for packet in self.packets.values()
        )


def _bf16_payload(code: NDArray[np.generic]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(code)).to(dtype=torch.bfloat16)
    return tensor.view(torch.uint8).contiguous().numpy().tobytes(order="C")


def _decode_base(payload: bytes) -> NDArray[np.float32]:
    if len(payload) != 480 * 2:
        raise ValueError("shared greedy base must be one BF16 480-vector")
    return (
        torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
        .clone()
        .to(torch.float32)
        .numpy()
    )


class SharedPacketGreedyAdapter:
    """Causal shared-packet control using only plain marginal-density greedy."""

    method_id = "shared_packet_plain_greedy"
    role: Literal["causal"] = "causal"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        *,
        shared_inputs: SharedInputReader | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        self._reader = shared_inputs
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._bases: dict[str, SharedBaseRecord] = {}
        self._packets: dict[str, SharedPacketState] = {}
        self._selection_order: tuple[str, ...] = ()
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
            raise RuntimeError("shared greedy adapter has no shared-input reader")
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
            raise RuntimeError("shared greedy adapter is not active")
        return self._contract, self._reader

    def _components(
        self,
        bases: dict[str, SharedBaseRecord] | None = None,
        packets: dict[str, SharedPacketState] | None = None,
        selection_order: tuple[str, ...] | None = None,
    ) -> dict[str, list[object]]:
        selected_bases = self._bases if bases is None else bases
        selected_packets = self._packets if packets is None else packets
        order = self._selection_order if selection_order is None else selection_order
        components = empty_components()
        for handle in sorted(selected_bases):
            record = selected_bases[handle]
            components["base_codes"].append(
                {"handle": handle, "dtype": "bfloat16", "data": record.base_payload}
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
                {"handle": handle, "sha256": hashlib.sha256(record.base_payload).hexdigest()}
            )
        for packet_id in sorted(selected_packets):
            packet = selected_packets[packet_id]
            components["packet_payloads"].append(
                {"packet_id": packet_id, "data": packet.payload}
            )
            components["packet_hashes"].append(
                {
                    "packet_id": packet_id,
                    "dictionary_revision_sha256": packet.dictionary_revision_sha256,
                    "group": packet.group,
                    "stage": packet.stage,
                    "entry": packet.entry,
                }
            )
            for handle in sorted(packet.gain_q_by_handle):
                components["incidences_gains"].append(
                    {
                        "packet_id": packet_id,
                        "handle": handle,
                        "gain_q": packet.gain_q_by_handle[handle],
                    }
                )
        components["controller_state"].append(
            {
                "allocator": "plain_marginal_density",
                "seed_enumeration": False,
                "switching_penalty": False,
                "hysteresis": False,
                "selection_order": list(order),
                "last_event_index": self._last_event_index,
            }
        )
        return components

    def _export(
        self,
        bases: dict[str, SharedBaseRecord],
        packets: dict[str, SharedPacketState],
        order: tuple[str, ...],
    ) -> bytes:
        return export_state(self._components(bases, packets, order))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export(self._bases, self._packets, self._selection_order)

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(self.export_online_state(), 0, 0)

    def _base_for_event(self, event_index: int, handle: str, current_index: int) -> bytes:
        _contract, reader = self._require_active()
        loaded = reader.load_event(event_index, current_index)
        if loaded.record.handle != handle:
            raise ValueError("shared-input handle differs from lifecycle event")
        return _bf16_payload(loaded.base_code)

    def _candidate_catalog(
        self,
        current_index: int,
        active_handles: set[str],
    ) -> dict[str, tuple[CandidatePacket, bytes]]:
        _contract, reader = self._require_active()
        catalog: dict[str, tuple[CandidatePacket, bytes]] = {}
        for event_index in range(current_index + 1):
            loaded = reader.load_event(event_index, current_index)
            for packet in loaded.record.candidate_packets:
                gains = {
                    handle: gain
                    for handle, gain in packet.gain_q_by_handle.items()
                    if handle in active_handles
                }
                if not gains:
                    continue
                filtered = packet.model_copy(
                    update={
                        "dependent_handles": tuple(sorted(gains)),
                        "gain_q_by_handle": {key: gains[key] for key in sorted(gains)},
                    }
                )
                payload = loaded.packet_payloads[packet.packet_id]
                prior = catalog.get(packet.packet_id)
                if prior is not None:
                    prior_packet, prior_payload = prior
                    if (
                        prior_payload != payload
                        or prior_packet.dictionary_revision_sha256
                        != filtered.dictionary_revision_sha256
                        or (prior_packet.group, prior_packet.stage, prior_packet.entry)
                        != (filtered.group, filtered.stage, filtered.entry)
                    ):
                        raise ValueError(
                            "shared candidate content address maps to conflicting data"
                        )
                    merged_gains = dict(prior_packet.gain_q_by_handle)
                    merged_gains.update(filtered.gain_q_by_handle)
                    filtered = filtered.model_copy(
                        update={
                            "dependent_handles": tuple(sorted(merged_gains)),
                            "gain_q_by_handle": {
                                key: merged_gains[key] for key in sorted(merged_gains)
                            },
                        }
                    )
                catalog[packet.packet_id] = (filtered, payload)
        return catalog

    def _packet_state(self, packet: CandidatePacket, payload: bytes) -> SharedPacketState:
        if hashlib.sha256(payload).hexdigest() != packet.packet_id:
            raise ValueError("shared packet payload hash mismatch")
        return SharedPacketState(
            packet_id=packet.packet_id,
            payload=payload,
            dictionary_revision_sha256=packet.dictionary_revision_sha256,
            group=packet.group,
            stage=packet.stage,
            entry=packet.entry,
            gain_q_by_handle=dict(packet.gain_q_by_handle),
        )

    def _allocate(
        self,
        bases: dict[str, SharedBaseRecord],
        *,
        current_index: int,
        protected_handle: str | None,
        budget: int,
    ) -> tuple[
        dict[str, SharedBaseRecord] | None,
        dict[str, SharedPacketState],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        selected_bases = dict(bases)
        evicted: list[str] = []
        while len(self._export(selected_bases, {}, ())) > budget:
            eligible = [
                row
                for handle, row in selected_bases.items()
                if handle != protected_handle
            ]
            if not eligible:
                return None, {}, (), ()
            victim = min(
                eligible,
                key=lambda row: (row.last_read_event, row.created_event, row.handle),
            )
            del selected_bases[victim.handle]
            evicted.append(victim.handle)
        catalog = self._candidate_catalog(current_index, set(selected_bases))
        if not catalog:
            return selected_bases, {}, (), tuple(evicted)
        maximum_group = max(packet.group for packet, _payload in catalog.values())
        group_width = maximum_group + 1
        gain_step = self._reader.manifest.incidence_gain_step if self._reader else 1.0
        bundles: dict[str, PacketBundle] = {}
        for packet_id, (packet, _payload) in catalog.items():
            gains: dict[str, tuple[float, ...]] = {}
            for handle, gain_q in packet.gain_q_by_handle.items():
                vector = [0.0] * group_width
                vector[packet.group] = max(0.0, min(1.0, gain_q * gain_step))
                gains[handle] = tuple(vector)
            bundles[packet_id] = PacketBundle(
                packet_id,
                max(1, packet.payload_bytes + packet.incidence_bytes),
                gains,
            )
        oracle = CoverageOracle(
            bundles=bundles,
            request_weights={
                handle: float(row.reads + 1)
                for handle, row in selected_bases.items()
            },
            group_weights={handle: (1.0,) * group_width for handle in selected_bases},
        )
        base_bytes = len(self._export(selected_bases, {}, ()))
        result = plain_density_greedy(oracle, max(0, budget - base_bytes))
        order = result.selected_packet_ids
        packets = {
            packet_id: self._packet_state(*catalog[packet_id])
            for packet_id in order
        }
        while len(self._export(selected_bases, packets, order)) > budget and order:
            removed = order[-1]
            order = order[:-1]
            del packets[removed]
        if len(self._export(selected_bases, packets, order)) > budget:
            return None, {}, (), ()
        return selected_bases, packets, order, tuple(evicted)

    @staticmethod
    def _decoded_code(
        base: SharedBaseRecord,
        packets: dict[str, SharedPacketState],
    ) -> NDArray[np.float32]:
        code = _decode_base(base.base_payload)
        applicable = sorted(
            (packet for packet in packets.values() if base.handle in packet.gain_q_by_handle),
            key=lambda row: (row.group, row.stage, row.entry, row.packet_id),
        )
        for packet in applicable:
            if len(packet.payload) < _RESIDUAL_HEADER.size:
                raise ValueError("shared greedy residual packet is truncated")
            group, start = _RESIDUAL_HEADER.unpack(packet.payload[: _RESIDUAL_HEADER.size])
            if group != packet.group:
                raise ValueError("shared greedy residual group mismatch")
            body = packet.payload[_RESIDUAL_HEADER.size :]
            if len(body) % 2:
                raise ValueError("shared greedy residual body is truncated")
            values = np.frombuffer(body, dtype="<f2").astype(np.float32)
            if start + len(values) > len(code):
                raise ValueError("shared greedy residual exceeds code width")
            code[start : start + len(values)] += values
        return code

    def _sample_sha(self, base: SharedBaseRecord, prompt: str, seed: int) -> tuple[str, str]:
        try:
            code = self._decoded_code(base, self._packets)
        except ValueError:
            if self._backbone is not None:
                raise
            code_sha = hashlib.sha256(
                base.base_payload
                + b"".join(
                    packet.payload
                    for packet in self._packets.values()
                    if base.handle in packet.gain_q_by_handle
                )
            ).hexdigest()
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

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract, _reader = self._require_active()
        if isinstance(event, ProbeEvent):
            raise TypeError("probe events must use score_probe")
        if event.event_index != view.current_index or view.at(event.event_index) != event:
            raise ValueError("event and causal view are not aligned")
        validate_operational_event_order(self._last_event_index, event, view)
        before = self.state_ledger()
        self._last_event_index = event.event_index
        bases = dict(self._bases)
        affected: tuple[str, ...] = ()
        evicted: tuple[str, ...] = ()
        decoded_sha: str | None = None
        generated_sha: str | None = None
        outcome: Outcome
        protected: str | None = None
        if isinstance(event, CreateEvent):
            if event.handle in bases:
                outcome = "rejected"
            else:
                bases[event.handle] = SharedBaseRecord(
                    event.handle,
                    self._base_for_event(event.event_index, event.handle, view.current_index),
                    event.event_index,
                    event.event_index,
                    0,
                    0,
                )
                protected = event.handle
                outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = bases.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                bases[event.handle] = SharedBaseRecord(
                    event.handle,
                    self._base_for_event(event.event_index, event.handle, view.current_index),
                    previous.created_event,
                    previous.last_read_event,
                    previous.reads,
                    previous.update_count + 1,
                )
                protected = event.handle
                outcome = "updated"
        elif isinstance(event, ReadEvent):
            previous = bases.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                decoded_sha, generated_sha = self._sample_sha(
                    previous,
                    event.prompt_id,
                    event.generation_seed,
                )
                bases[event.handle] = SharedBaseRecord(
                    previous.handle,
                    previous.base_payload,
                    previous.created_event,
                    event.event_index,
                    previous.reads + 1,
                    previous.update_count,
                )
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in bases:
                outcome = "stale_handle"
            else:
                del bases[event.handle]
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported shared greedy event: {type(event).__name__}")
        if outcome not in {"rejected", "stale_handle"} or isinstance(event, ReadEvent):
            allocated, packets, order, evicted = self._allocate(
                bases,
                current_index=event.event_index,
                protected_handle=protected,
                budget=contract.byte_budget,
            )
            if allocated is None:
                if protected is not None:
                    bases = dict(self._bases)
                    packets = dict(self._packets)
                    order = self._selection_order
                    outcome = "rejected"
                    evicted = ()
                else:
                    raise RuntimeError("shared greedy existing state became infeasible")
            else:
                affected = tuple(sorted(set(self._bases) | set(allocated)))
                bases = allocated
            self._packets = packets
            self._selection_order = order
        self._bases = bases
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("shared greedy adapter exceeded the exact byte budget")
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
        token = hashlib.sha256(b"shared-greedy-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    def _state_from_payload(
        self,
        payload: bytes,
    ) -> tuple[
        dict[str, SharedBaseRecord],
        dict[str, SharedPacketState],
        tuple[str, ...],
        int | None,
    ]:
        components = decode_state(payload)
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("shared greedy controller state is invalid")
        if controller[0].get("allocator") != "plain_marginal_density":
            raise ValueError("shared greedy allocator identity mismatch")
        order_raw = controller[0].get("selection_order")
        if not isinstance(order_raw, list) or any(type(item) is not str for item in order_raw):
            raise ValueError("shared greedy selection order is invalid")
        last_event = controller[0].get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("shared greedy last event index is invalid")
        base_rows = {
            str(row["handle"]): row
            for row in components["base_codes"]
            if isinstance(row, dict)
        }
        usage_rows = {
            str(row["handle"]): row
            for row in components["usage_age"]
            if isinstance(row, dict)
        }
        if set(base_rows) != set(usage_rows):
            raise ValueError("shared greedy base and metadata handles differ")
        bases: dict[str, SharedBaseRecord] = {}
        for handle in sorted(base_rows):
            raw = base_rows[handle].get("data")
            if type(raw) is not bytes:
                raise ValueError("shared greedy base payload is invalid")
            meta = usage_rows[handle]
            bases[handle] = SharedBaseRecord(
                handle,
                raw,
                int(meta["created_event"]),
                int(meta["last_read_event"]),
                int(meta["reads"]),
                int(meta["update_count"]),
            )
        payload_rows = {
            str(row["packet_id"]): row
            for row in components["packet_payloads"]
            if isinstance(row, dict)
        }
        hash_rows = {
            str(row["packet_id"]): row
            for row in components["packet_hashes"]
            if isinstance(row, dict)
        }
        gains: dict[str, dict[str, int]] = {}
        for row in components["incidences_gains"]:
            if isinstance(row, dict):
                gains.setdefault(str(row["packet_id"]), {})[str(row["handle"])] = int(
                    row["gain_q"]
                )
        if set(payload_rows) != set(hash_rows) or set(payload_rows) != set(gains):
            raise ValueError("shared greedy packet metadata is incomplete")
        packets: dict[str, SharedPacketState] = {}
        for packet_id in sorted(payload_rows):
            raw = payload_rows[packet_id].get("data")
            if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != packet_id:
                raise ValueError("shared greedy packet content address is invalid")
            meta = hash_rows[packet_id]
            packets[packet_id] = SharedPacketState(
                packet_id,
                raw,
                str(meta["dictionary_revision_sha256"]),
                int(meta["group"]),
                int(meta["stage"]),
                int(meta["entry"]),
                gains[packet_id],
            )
        order = tuple(order_raw)
        if set(order) != set(packets) or len(order) != len(set(order)):
            raise ValueError("shared greedy selection order differs from packets")
        return bases, packets, order, last_event

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        bases, packets, order, last_event = self._state_from_payload(payload)
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export(bases, packets, order) != payload:
            self._last_event_index = prior_event
            raise ValueError("shared greedy state does not roundtrip canonically")
        self._bases = bases
        self._packets = packets
        self._selection_order = order

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract, _reader = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        bases, packets, _order, _last_event = self._state_from_payload(payload)
        base = bases.get(probe.handle)
        if base is None:
            raise ValueError("probe handle is absent from shared greedy snapshot")
        current_packets = self._packets
        self._packets = packets
        try:
            _code_sha, sample_sha = self._sample_sha(
                base,
                probe.prompt_id,
                probe.generation_seed,
            )
        finally:
            self._packets = current_packets
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

    def inspect_state(self) -> SharedGreedyStateView:
        self._require_active()
        return SharedGreedyStateView(
            bases=dict(self._bases),
            packets=dict(self._packets),
            selected_packet_ids=self._selection_order,
            ledger=self.state_ledger(),
        )

    def close(self) -> None:
        self._bases.clear()
        self._packets.clear()
        self._selection_order = ()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "GreedyResult",
    "SharedBaseRecord",
    "SharedGreedyStateView",
    "SharedPacketGreedyAdapter",
    "SharedPacketState",
    "plain_density_greedy",
]
