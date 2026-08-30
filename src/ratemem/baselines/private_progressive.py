"""Private progressive-code rate allocation controls."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
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
class RateChoice:
    handle: str
    prefix_length: int
    serialized_bytes: int
    value: Decimal

    def __post_init__(self) -> None:
        if not self.handle:
            raise ValueError("rate choice handle must be non-empty")
        if type(self.prefix_length) is not int or self.prefix_length < 0:
            raise ValueError("rate choice prefix must be nonnegative")
        if type(self.serialized_bytes) is not int or self.serialized_bytes < 0:
            raise ValueError("rate choice bytes must be nonnegative")
        if type(self.value) is not Decimal or not self.value.is_finite() or self.value < 0:
            raise ValueError("rate choice value must be a finite nonnegative Decimal")


@dataclass(frozen=True, slots=True)
class RateAllocation:
    prefix_by_handle: dict[str, int]
    total_bytes: int
    total_value: Decimal


def _validate_options(options: dict[str, Sequence[RateChoice]]) -> None:
    if not options:
        raise ValueError("rate options must contain at least one handle")
    for handle, choices in options.items():
        if not handle or not choices:
            raise ValueError("every handle requires at least one rate choice")
        if any(choice.handle != handle for choice in choices):
            raise ValueError("rate option map key differs from its choice handle")
        prefixes = tuple(choice.prefix_length for choice in choices)
        if prefixes != tuple(range(len(prefixes))):
            raise ValueError("rate choices must enumerate every legal prefix from zero")
        if any(
            right.serialized_bytes < left.serialized_bytes
            for left, right in zip(choices, choices[1:], strict=False)
        ):
            raise ValueError("longer progressive prefixes cannot use fewer bytes")


def exact_separable_allocation(
    options: dict[str, Sequence[RateChoice]],
    budget: int,
) -> RateAllocation:
    """Solve the multiple-choice prefix knapsack exactly with sparse dominance pruning."""

    _validate_options(options)
    if type(budget) is not int or budget < 0:
        raise ValueError("rate budget must be a nonnegative integer")
    frontier: dict[int, tuple[Decimal, tuple[tuple[str, int], ...]]] = {
        0: (Decimal(0), ())
    }
    for handle in sorted(options):
        expanded: dict[int, tuple[Decimal, tuple[tuple[str, int], ...]]] = {}
        for used, (value, choices) in frontier.items():
            for choice in options[handle]:
                candidate_bytes = used + choice.serialized_bytes
                if candidate_bytes > budget:
                    continue
                candidate = (
                    value + choice.value,
                    choices + ((handle, choice.prefix_length),),
                )
                incumbent = expanded.get(candidate_bytes)
                if incumbent is None or candidate[0] > incumbent[0] or (
                    candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
                ):
                    expanded[candidate_bytes] = candidate
        if not expanded:
            raise ValueError("no feasible separable rate allocation")
        best_value = Decimal("-Infinity")
        frontier = {}
        for used in sorted(expanded):
            value, choices = expanded[used]
            if value > best_value:
                frontier[used] = (value, choices)
                best_value = value
    used, (value, choices) = min(
        frontier.items(),
        key=lambda row: (-row[1][0], row[0], row[1][1]),
    )
    return RateAllocation(dict(choices), used, value)


Policy = Literal["size_aware", "separable_rate"]
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
class PrivatePacketState:
    packet_id: str
    payload: bytes
    dictionary_revision_sha256: str
    group: int
    stage: int
    entry: int
    gain_q: int
    serialized_key: str


@dataclass(frozen=True, slots=True)
class PrivateRecord:
    handle: str
    base_payload: bytes
    packets: tuple[PrivatePacketState, ...]
    created_event: int
    last_read_event: int
    reads: int
    update_count: int


@dataclass(frozen=True, slots=True)
class PrivateProgressiveStateView:
    private_packets: dict[tuple[str, int], PrivatePacketState]
    prefixes: dict[str, tuple[int, ...]]
    records: dict[str, PrivateRecord]
    ledger: ExactByteLedger


def _bf16_payload(code: NDArray[np.generic]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(code)).to(dtype=torch.bfloat16)
    return tensor.view(torch.uint8).contiguous().numpy().tobytes(order="C")


def _decode_base(payload: bytes) -> NDArray[np.float32]:
    if len(payload) != 480 * 2:
        raise ValueError("private progressive base must be one BF16 480-vector")
    return (
        torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
        .clone()
        .to(torch.float32)
        .numpy()
    )


def _packet_state(
    handle: str,
    index: int,
    packet: CandidatePacket,
    payload: bytes,
) -> PrivatePacketState:
    if hashlib.sha256(payload).hexdigest() != packet.packet_id:
        raise ValueError("private packet payload hash mismatch")
    gain = packet.gain_q_by_handle.get(handle)
    if gain is None:
        raise ValueError("private packet lacks an incidence for its handle")
    serialized_key = hashlib.sha256(
        canonical_json_bytes(
            {
                "scope": "private",
                "handle": handle,
                "prefix_index": index,
                "packet_id": packet.packet_id,
            }
        )
    ).hexdigest()
    return PrivatePacketState(
        packet_id=packet.packet_id,
        payload=payload,
        dictionary_revision_sha256=packet.dictionary_revision_sha256,
        group=packet.group,
        stage=packet.stage,
        entry=packet.entry,
        gain_q=gain,
        serialized_key=serialized_key,
    )


class PrivateProgressiveAdapter:
    """Private packet streams with causal or exact separable rate allocation."""

    role: Literal["causal"] = "causal"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        method_id: str,
        *,
        policy: Policy,
        shared_inputs: SharedInputReader | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        expected = f"private_progressive_{policy}"
        if method_id != expected:
            raise ValueError(f"private progressive method id must be {expected}")
        self.method_id = method_id
        self.policy = policy
        self._reader = shared_inputs
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._records: dict[str, PrivateRecord] = {}
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
            raise RuntimeError("private progressive adapter has no shared inputs")
        manifest = self._reader.manifest
        if manifest.trace_id != contract.trace_id:
            raise ValueError("shared-input trace differs from the contract")
        if manifest.candidate_stream_sha256 != contract.candidate_stream_sha256:
            raise ValueError("shared-input candidate stream differs from the contract")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty state")

    def _require_active(self) -> tuple[FrozenComparisonContract, SharedInputReader]:
        if self._contract is None or self._reader is None or self._closed:
            raise RuntimeError("private progressive adapter is not active")
        return self._contract, self._reader

    def _components(
        self,
        records: dict[str, PrivateRecord] | None = None,
    ) -> dict[str, list[object]]:
        selected = self._records if records is None else records
        components = empty_components()
        for handle in sorted(selected):
            record = selected[handle]
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
            for index, packet in enumerate(record.packets):
                components["packet_payloads"].append(
                    {
                        "key": packet.serialized_key,
                        "handle": handle,
                        "prefix_index": index,
                        "packet_id": packet.packet_id,
                        "data": packet.payload,
                    }
                )
                components["packet_hashes"].append(
                    {
                        "key": packet.serialized_key,
                        "dictionary_revision_sha256": packet.dictionary_revision_sha256,
                        "group": packet.group,
                        "stage": packet.stage,
                        "entry": packet.entry,
                    }
                )
                components["incidences_gains"].append(
                    {"handle": handle, "key": packet.serialized_key, "gain_q": packet.gain_q}
                )
        components["controller_state"].append(
            {
                "policy": self.policy,
                "last_event_index": self._last_event_index,
                "incidence_gain_step": self._reader.manifest.incidence_gain_step
                if self._reader is not None
                else None,
            }
        )
        return components

    def _export_records(self, records: dict[str, PrivateRecord]) -> bytes:
        return export_state(self._components(records))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export_records(self._records)

    def state_ledger(self) -> ExactByteLedger:
        payload = self.export_online_state()
        return ledger_from_export(payload, 0, 0)

    def _record_for_event(self, event_index: int, handle: str, current_index: int) -> PrivateRecord:
        _contract, reader = self._require_active()
        loaded = reader.load_event(event_index, current_index)
        if loaded.record.handle != handle:
            raise ValueError("shared-input handle differs from the lifecycle event")
        ordered = sorted(
            loaded.record.candidate_packets,
            key=lambda row: (row.group, row.stage, row.entry, row.packet_id),
        )
        packets = tuple(
            _packet_state(handle, index, packet, loaded.packet_payloads[packet.packet_id])
            for index, packet in enumerate(ordered)
        )
        return PrivateRecord(
            handle=handle,
            base_payload=_bf16_payload(loaded.base_code),
            packets=packets,
            created_event=event_index,
            last_read_event=event_index,
            reads=0,
            update_count=0,
        )

    @staticmethod
    def _degrade_record(record: PrivateRecord) -> PrivateRecord | None:
        if record.packets:
            return replace(record, packets=record.packets[:-1])
        return None

    def _request_weight(self, record: PrivateRecord) -> Decimal:
        return Decimal(record.reads + 1)

    def _loss(self, record: PrivateRecord) -> Decimal:
        if record.packets:
            raw = max(record.packets[-1].gain_q, 0)
            return self._request_weight(record) * Decimal(raw)
        return self._request_weight(record) * Decimal(32768)

    def _fit_size_aware(
        self,
        records: dict[str, PrivateRecord],
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, PrivateRecord] | None, tuple[str, ...]]:
        candidate = dict(records)
        evicted: list[str] = []
        while len(self._export_records(candidate)) > budget:
            actions: list[tuple[Decimal, str, int, PrivateRecord | None]] = []
            before_bytes = len(self._export_records(candidate))
            for handle in sorted(candidate):
                replacement = self._degrade_record(candidate[handle])
                if replacement is None and handle == protected_handle:
                    continue
                trial = dict(candidate)
                if replacement is None:
                    del trial[handle]
                    resulting_prefix = -1
                else:
                    trial[handle] = replacement
                    resulting_prefix = len(replacement.packets)
                reclaimed = before_bytes - len(self._export_records(trial))
                if reclaimed <= 0:
                    continue
                density = self._loss(candidate[handle]) / Decimal(reclaimed)
                actions.append((density, handle, resulting_prefix, replacement))
            if not actions:
                return None, ()
            _density, handle, _prefix, replacement = min(actions)
            if replacement is None:
                del candidate[handle]
                evicted.append(handle)
            else:
                candidate[handle] = replacement
        if protected_handle not in candidate:
            return None, ()
        return candidate, tuple(evicted)

    def _fit_separable(
        self,
        records: dict[str, PrivateRecord],
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, PrivateRecord] | None, tuple[str, ...]]:
        handles = sorted(records)
        frontier: list[tuple[Decimal, dict[str, PrivateRecord]]] = [(Decimal(0), {})]
        for handle in handles:
            record = records[handle]
            choices: list[PrivateRecord | None] = []
            if handle != protected_handle:
                choices.append(None)
            choices.extend(
                replace(record, packets=record.packets[:count])
                for count in range(len(record.packets) + 1)
            )
            expanded: dict[bytes, tuple[Decimal, dict[str, PrivateRecord]]] = {}
            for value, partial in frontier:
                for choice in choices:
                    selected = dict(partial)
                    if choice is not None:
                        selected[handle] = choice
                    if len(self._export_records(selected)) > budget:
                        continue
                    prefix_value = Decimal(0)
                    if choice is not None:
                        prefix_value = self._request_weight(choice) * (
                            Decimal(32768)
                            + sum(
                                (
                                    Decimal(max(packet.gain_q, 0))
                                    for packet in choice.packets
                                ),
                                Decimal(0),
                            )
                        )
                    key = self._export_records(selected)
                    candidate = (value + prefix_value, selected)
                    incumbent = expanded.get(key)
                    if incumbent is None or candidate[0] > incumbent[0]:
                        expanded[key] = candidate
            if not expanded:
                return None, ()
            frontier = list(expanded.values())
            frontier.sort(
                key=lambda row: (
                    -row[0],
                    len(self._export_records(row[1])),
                    tuple((key, len(value.packets)) for key, value in sorted(row[1].items())),
                )
            )
            best_by_bytes: dict[int, tuple[Decimal, dict[str, PrivateRecord]]] = {}
            for row in frontier:
                used = len(self._export_records(row[1]))
                incumbent = best_by_bytes.get(used)
                if incumbent is None or row[0] > incumbent[0]:
                    best_by_bytes[used] = row
            frontier = list(best_by_bytes.values())
        if not frontier:
            return None, ()
        _value, selected = min(
            frontier,
            key=lambda row: (
                -row[0],
                len(self._export_records(row[1])),
                tuple((key, len(value.packets)) for key, value in sorted(row[1].items())),
            ),
        )
        if protected_handle not in selected:
            return None, ()
        evicted = tuple(handle for handle in handles if handle not in selected)
        return selected, evicted

    def _fit(
        self,
        records: dict[str, PrivateRecord],
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, PrivateRecord] | None, tuple[str, ...]]:
        if self.policy == "size_aware":
            return self._fit_size_aware(records, protected_handle, budget)
        return self._fit_separable(records, protected_handle, budget)

    @staticmethod
    def _decoded_code(record: PrivateRecord) -> NDArray[np.float32]:
        code = _decode_base(record.base_payload)
        for packet in record.packets:
            if len(packet.payload) < _RESIDUAL_HEADER.size:
                raise ValueError("private progressive residual packet is truncated")
            group, start = _RESIDUAL_HEADER.unpack(packet.payload[: _RESIDUAL_HEADER.size])
            if group != packet.group:
                raise ValueError("private progressive residual group mismatch")
            body = packet.payload[_RESIDUAL_HEADER.size :]
            if len(body) % 2:
                raise ValueError("private progressive residual body is truncated")
            values = np.frombuffer(body, dtype="<f2").astype(np.float32)
            if start + len(values) > len(code):
                raise ValueError("private progressive residual exceeds code width")
            code[start : start + len(values)] += values
        return code

    def _sample_sha(self, record: PrivateRecord, prompt_id: str, seed: int) -> tuple[str, str]:
        try:
            code = self._decoded_code(record)
        except ValueError:
            if self._backbone is not None:
                raise
            code_sha = hashlib.sha256(
                record.base_payload + b"".join(packet.payload for packet in record.packets)
            ).hexdigest()
            sample_sha = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_backbone": True,
                        "code_sha256": code_sha,
                        "prompt": prompt_id,
                        "seed": seed,
                    }
                )
            ).hexdigest()
            return code_sha, sample_sha
        code_sha = hashlib.sha256(np.ascontiguousarray(code).tobytes(order="C")).hexdigest()
        if self._backbone is None:
            return code_sha, hashlib.sha256(
                canonical_json_bytes(
                    {
                        "synthetic_no_backbone": True,
                        "code_sha256": code_sha,
                        "prompt": prompt_id,
                        "seed": seed,
                    }
                )
            ).hexdigest()
        contract, _reader = self._require_active()
        self._backbone.install_code(torch.from_numpy(code))
        try:
            sample = self._backbone.generate(
                prompt_id,
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
        if self._last_event_index is not None and event.event_index != self._last_event_index + 1:
            raise ValueError("events must be applied exactly once in trace order")
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
                records[event.handle] = self._record_for_event(
                    event.event_index,
                    event.handle,
                    view.current_index,
                )
                fitted, evicted = self._fit(records, event.handle, contract.byte_budget)
                if fitted is None:
                    records.pop(event.handle)
                    outcome = "rejected"
                    evicted = ()
                else:
                    affected = tuple(sorted(set(fitted) | set(records)))
                    records = fitted
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                incoming = self._record_for_event(
                    event.event_index,
                    event.handle,
                    view.current_index,
                )
                records[event.handle] = replace(
                    incoming,
                    created_event=previous.created_event,
                    last_read_event=previous.last_read_event,
                    reads=previous.reads,
                    update_count=previous.update_count + 1,
                )
                fitted, evicted = self._fit(records, event.handle, contract.byte_budget)
                if fitted is None:
                    records[event.handle] = previous
                    outcome = "rejected"
                    evicted = ()
                else:
                    affected = tuple(sorted(set(fitted) | set(records)))
                    records = fitted
                    outcome = "updated"
        elif isinstance(event, ReadEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                decoded_sha, generated_sha = self._sample_sha(
                    previous,
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
            raise TypeError(f"unsupported private progressive event: {type(event).__name__}")
        self._records = records
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("private progressive adapter exceeded the exact byte budget")
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
        token = hashlib.sha256(b"private-progressive-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    def _records_from_payload(self, payload: bytes) -> dict[str, PrivateRecord]:
        components = decode_state(payload)
        bases = {
            str(row["handle"]): row
            for row in components["base_codes"]
            if isinstance(row, dict)
        }
        usage = {
            str(row["handle"]): row
            for row in components["usage_age"]
            if isinstance(row, dict)
        }
        packet_rows = [row for row in components["packet_payloads"] if isinstance(row, dict)]
        hash_rows = {
            str(row["key"]): row
            for row in components["packet_hashes"]
            if isinstance(row, dict)
        }
        gain_rows = {
            str(row["key"]): row
            for row in components["incidences_gains"]
            if isinstance(row, dict)
        }
        restored: dict[str, PrivateRecord] = {}
        for handle in sorted(bases):
            base = bases[handle].get("data")
            if type(base) is not bytes or handle not in usage:
                raise ValueError("private progressive base or metadata is invalid")
            rows = sorted(
                (row for row in packet_rows if row.get("handle") == handle),
                key=lambda row: int(row["prefix_index"]),
            )
            packets: list[PrivatePacketState] = []
            for expected_index, row in enumerate(rows):
                if int(row["prefix_index"]) != expected_index:
                    raise ValueError("private progressive packet set is not a prefix")
                key = str(row["key"])
                metadata = hash_rows.get(key)
                gain = gain_rows.get(key)
                raw = row.get("data")
                if metadata is None or gain is None or type(raw) is not bytes:
                    raise ValueError("private progressive packet metadata is incomplete")
                packets.append(
                    PrivatePacketState(
                        packet_id=str(row["packet_id"]),
                        payload=raw,
                        dictionary_revision_sha256=str(metadata["dictionary_revision_sha256"]),
                        group=int(metadata["group"]),
                        stage=int(metadata["stage"]),
                        entry=int(metadata["entry"]),
                        gain_q=int(gain["gain_q"]),
                        serialized_key=key,
                    )
                )
            meta = usage[handle]
            restored[handle] = PrivateRecord(
                handle,
                base,
                tuple(packets),
                int(meta["created_event"]),
                int(meta["last_read_event"]),
                int(meta["reads"]),
                int(meta["update_count"]),
            )
        return restored

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("private progressive controller state is invalid")
        if controller[0].get("policy") != self.policy:
            raise ValueError("private progressive policy differs from exported state")
        last_event = controller[0].get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("private progressive last event index is invalid")
        restored = self._records_from_payload(payload)
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export_records(restored) != payload:
            self._last_event_index = prior_event
            raise ValueError("private progressive state does not roundtrip canonically")
        self._records = restored

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract, _reader = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        record = self._records_from_payload(payload).get(probe.handle)
        if record is None:
            raise ValueError("probe handle is absent from the private snapshot")
        _code_sha, sample_sha = self._sample_sha(record, probe.prompt_id, probe.generation_seed)
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

    def inspect_state(self) -> PrivateProgressiveStateView:
        self._require_active()
        return PrivateProgressiveStateView(
            private_packets={
                (handle, index): packet
                for handle, record in self._records.items()
                for index, packet in enumerate(record.packets)
            },
            prefixes={
                handle: tuple(range(len(record.packets)))
                for handle, record in self._records.items()
            },
            records=dict(self._records),
            ledger=self.state_ledger(),
        )

    def close(self) -> None:
        self._records.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "PrivatePacketState",
    "PrivateProgressiveAdapter",
    "PrivateProgressiveStateView",
    "PrivateRecord",
    "RateAllocation",
    "RateChoice",
    "exact_separable_allocation",
]
