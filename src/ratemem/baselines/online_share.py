"""Deterministic online shared-subspace control."""

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

Float32 = NDArray[np.float32]


def canonicalize_basis_signs(basis: Float32) -> Float32:
    if basis.ndim != 2 or basis.dtype != np.float32 or not np.isfinite(basis).all():
        raise ValueError("basis must be a finite float32 matrix")
    result = basis.copy()
    for row in result:
        if not np.any(row):
            continue
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    return result


def update_subspace(
    basis: Float32,
    coefficients: dict[str, Float32],
    incoming: Float32,
    incoming_handle: str,
    rank: int,
) -> tuple[Float32, dict[str, Float32]]:
    """Reproject only reconstructed resident codes plus the current incoming target."""

    if not incoming_handle:
        raise ValueError("incoming handle must be non-empty")
    if type(rank) is not int or rank < 1:
        raise ValueError("online SHARE rank must be positive")
    if basis.ndim != 2 or basis.dtype != np.float32 or not np.isfinite(basis).all():
        raise ValueError("basis must be a finite float32 matrix")
    if incoming.ndim != 1 or incoming.dtype != np.float32 or not np.isfinite(incoming).all():
        raise ValueError("incoming code must be one finite float32 vector")
    if basis.shape[1] != incoming.shape[0]:
        raise ValueError("basis and incoming code dimensions differ")
    reconstructed: dict[str, Float32] = {}
    for handle, coefficient in coefficients.items():
        if not handle or coefficient.dtype != np.float32 or coefficient.shape != (basis.shape[0],):
            raise ValueError("resident online SHARE coefficient has an invalid layout")
        if not np.isfinite(coefficient).all():
            raise ValueError("resident online SHARE coefficient must be finite")
        reconstructed[handle] = (coefficient @ basis).astype(np.float32)
    reconstructed[incoming_handle] = incoming.copy()
    handles = sorted(reconstructed)
    matrix = np.stack([reconstructed[handle] for handle in handles]).astype(np.float64)
    _u, _singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    new_rank = min(rank, len(vh))
    new_basis = canonicalize_basis_signs(vh[:new_rank].astype(np.float32))
    new_coefficients = {
        handle: (reconstructed[handle] @ new_basis.T).astype(np.float32)
        for handle in handles
    }
    return new_basis, new_coefficients


def reconstruction_drift_sha256(
    prior_basis: Float32,
    prior_coefficients: dict[str, Float32],
    new_basis: Float32,
    new_coefficients: dict[str, Float32],
) -> dict[str, str]:
    """Hash per-handle drift without retaining any original target code."""

    import hashlib

    common = sorted(set(prior_coefficients) & set(new_coefficients))
    result: dict[str, str] = {}
    for handle in common:
        before = prior_coefficients[handle] @ prior_basis
        after = new_coefficients[handle] @ new_basis
        drift = np.ascontiguousarray(after - before, dtype=np.float32)
        result[handle] = hashlib.sha256(drift.tobytes(order="C")).hexdigest()
    return result


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
class OnlineHandleRecord:
    handle: str
    created_event: int
    last_read_event: int
    reads: int
    update_count: int


@dataclass(frozen=True, slots=True)
class OnlineShareStateView:
    basis: Float32
    coefficients: dict[str, Float32]
    records: dict[str, OnlineHandleRecord]
    drift_sha256_by_handle: dict[str, str]
    ledger: ExactByteLedger


class OnlineShareAdapter:
    """Causal SHARE-style mutable subspace with no retained target-code archive."""

    method_id = "share_style_online"
    role: Literal["causal"] = "causal"
    shared_trained_bytes = 0
    external_support_bytes = 0

    def __init__(
        self,
        *,
        rank: int,
        shared_inputs: SharedInputReader | None = None,
        backbone: BackboneRunner | None = None,
    ) -> None:
        if type(rank) is not int or not 1 <= rank <= 480:
            raise ValueError("online SHARE rank must be in [1, 480]")
        self.rank = rank
        self._reader = shared_inputs
        self._backbone = backbone
        self._contract: FrozenComparisonContract | None = None
        self._basis: Float32 = np.empty((0, 480), dtype=np.float32)
        self._coefficients: dict[str, Float32] = {}
        self._records: dict[str, OnlineHandleRecord] = {}
        self._drift_sha256_by_handle: dict[str, str] = {}
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
            raise RuntimeError("online SHARE adapter has no shared-input reader")
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
            raise RuntimeError("online SHARE adapter is not active")
        return self._contract, self._reader

    def _components(
        self,
        basis: Float32 | None = None,
        coefficients: dict[str, Float32] | None = None,
        records: dict[str, OnlineHandleRecord] | None = None,
        drift: dict[str, str] | None = None,
    ) -> dict[str, list[object]]:
        selected_basis = self._basis if basis is None else basis
        selected_coefficients = self._coefficients if coefficients is None else coefficients
        selected_records = self._records if records is None else records
        selected_drift = self._drift_sha256_by_handle if drift is None else drift
        components = empty_components()
        components["allocator_state"].append(
            {
                "kind": "online_share_basis",
                "dtype": selected_basis.dtype.str,
                "shape": list(selected_basis.shape),
                "data": selected_basis.tobytes(order="C"),
                "rank_limit": self.rank,
                "drift_sha256_by_handle": {
                    handle: selected_drift[handle] for handle in sorted(selected_drift)
                },
            }
        )
        for handle in sorted(selected_records):
            record = selected_records[handle]
            coefficient = selected_coefficients[handle]
            components["base_codes"].append(
                {
                    "handle": handle,
                    "kind": "subspace_coefficient",
                    "dtype": coefficient.dtype.str,
                    "shape": list(coefficient.shape),
                    "data": coefficient.tobytes(order="C"),
                }
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
                    "sha256": hashlib.sha256(coefficient.tobytes(order="C")).hexdigest(),
                }
            )
        components["controller_state"].append(
            {
                "policy": "lru",
                "rank": self.rank,
                "last_event_index": self._last_event_index,
                "retains_original_targets": False,
            }
        )
        return components

    def _export(
        self,
        basis: Float32,
        coefficients: dict[str, Float32],
        records: dict[str, OnlineHandleRecord],
        drift: dict[str, str],
    ) -> bytes:
        return export_state(self._components(basis, coefficients, records, drift))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export(
            self._basis,
            self._coefficients,
            self._records,
            self._drift_sha256_by_handle,
        )

    def state_ledger(self) -> ExactByteLedger:
        return ledger_from_export(self.export_online_state(), 0, 0)

    def _incoming(self, event_index: int, handle: str, current_index: int) -> Float32:
        _contract, reader = self._require_active()
        loaded = reader.load_event(event_index, current_index)
        if loaded.record.handle != handle:
            raise ValueError("shared-input handle differs from lifecycle event")
        return loaded.target_code.copy()

    def _rebuild(
        self,
        basis: Float32,
        coefficients: dict[str, Float32],
    ) -> tuple[Float32, dict[str, Float32]]:
        if not coefficients:
            return np.empty((0, 480), dtype=np.float32), {}
        reconstructed = {
            handle: (coefficient @ basis).astype(np.float32)
            for handle, coefficient in coefficients.items()
        }
        handles = sorted(reconstructed)
        matrix = np.stack([reconstructed[handle] for handle in handles]).astype(np.float64)
        _u, _values, vh = np.linalg.svd(matrix, full_matrices=False)
        rebuilt_basis = canonicalize_basis_signs(
            vh[: min(self.rank, len(vh))].astype(np.float32)
        )
        rebuilt_coefficients = {
            handle: (reconstructed[handle] @ rebuilt_basis.T).astype(np.float32)
            for handle in handles
        }
        return rebuilt_basis, rebuilt_coefficients

    def _fit_with_eviction(
        self,
        basis: Float32,
        coefficients: dict[str, Float32],
        records: dict[str, OnlineHandleRecord],
        drift: dict[str, str],
        *,
        protected_handle: str | None,
        budget: int,
    ) -> tuple[
        Float32 | None,
        dict[str, Float32],
        dict[str, OnlineHandleRecord],
        dict[str, str],
        tuple[str, ...],
    ]:
        selected_basis = basis
        selected_coefficients = dict(coefficients)
        selected_records = dict(records)
        selected_drift = dict(drift)
        evicted: list[str] = []
        while (
            len(
                self._export(
                    selected_basis,
                    selected_coefficients,
                    selected_records,
                    selected_drift,
                )
            )
            > budget
        ):
            eligible = [
                row
                for handle, row in selected_records.items()
                if handle != protected_handle
            ]
            if not eligible:
                return None, {}, {}, {}, ()
            victim = min(
                eligible,
                key=lambda row: (row.last_read_event, row.created_event, row.handle),
            )
            del selected_records[victim.handle]
            del selected_coefficients[victim.handle]
            selected_drift.pop(victim.handle, None)
            evicted.append(victim.handle)
            selected_basis, selected_coefficients = self._rebuild(
                selected_basis,
                selected_coefficients,
            )
        return (
            selected_basis,
            selected_coefficients,
            selected_records,
            selected_drift,
            tuple(evicted),
        )

    def _sample_sha(self, code: Float32, prompt: str, seed: int) -> tuple[str, str]:
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
        if self._last_event_index is not None and event.event_index != self._last_event_index + 1:
            raise ValueError("events must be applied exactly once in trace order")
        before = self.state_ledger()
        old_basis = self._basis
        old_coefficients = dict(self._coefficients)
        old_records = dict(self._records)
        old_drift = dict(self._drift_sha256_by_handle)
        self._last_event_index = event.event_index
        basis = self._basis
        coefficients = dict(self._coefficients)
        records = dict(self._records)
        drift = dict(self._drift_sha256_by_handle)
        affected: tuple[str, ...] = ()
        evicted: tuple[str, ...] = ()
        decoded_sha: str | None = None
        generated_sha: str | None = None
        outcome: Outcome
        protected: str | None = None
        needs_fit = False
        if isinstance(event, CreateEvent):
            if event.handle in records:
                outcome = "rejected"
            else:
                incoming = self._incoming(event.event_index, event.handle, view.current_index)
                prior_basis = basis
                prior_coefficients = coefficients
                basis, coefficients = update_subspace(
                    basis,
                    coefficients,
                    incoming,
                    event.handle,
                    self.rank,
                )
                drift.update(
                    reconstruction_drift_sha256(
                        prior_basis,
                        prior_coefficients,
                        basis,
                        coefficients,
                    )
                )
                records[event.handle] = OnlineHandleRecord(
                    event.handle,
                    event.event_index,
                    event.event_index,
                    0,
                    0,
                )
                protected = event.handle
                needs_fit = True
                outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                incoming = self._incoming(event.event_index, event.handle, view.current_index)
                prior_basis = basis
                prior_coefficients = coefficients
                basis, coefficients = update_subspace(
                    basis,
                    coefficients,
                    incoming,
                    event.handle,
                    self.rank,
                )
                drift.update(
                    reconstruction_drift_sha256(
                        prior_basis,
                        prior_coefficients,
                        basis,
                        coefficients,
                    )
                )
                records[event.handle] = replace(
                    previous,
                    update_count=previous.update_count + 1,
                )
                protected = event.handle
                needs_fit = True
                outcome = "updated"
        elif isinstance(event, ReadEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                code = (coefficients[event.handle] @ basis).astype(np.float32)
                decoded_sha, generated_sha = self._sample_sha(
                    code,
                    event.prompt_id,
                    event.generation_seed,
                )
                records[event.handle] = replace(
                    previous,
                    last_read_event=event.event_index,
                    reads=previous.reads + 1,
                )
                affected = (event.handle,)
                needs_fit = True
                outcome = "read"
        elif isinstance(event, DeleteEvent):
            if event.handle not in records:
                outcome = "stale_handle"
            else:
                del records[event.handle]
                del coefficients[event.handle]
                drift.pop(event.handle, None)
                basis, coefficients = self._rebuild(basis, coefficients)
                affected = (event.handle,)
                needs_fit = True
                outcome = "deleted"
        else:
            raise TypeError(f"unsupported online SHARE event: {type(event).__name__}")
        if needs_fit:
            fitted_basis, coefficients, records, drift, evicted = self._fit_with_eviction(
                basis,
                coefficients,
                records,
                drift,
                protected_handle=protected,
                budget=contract.byte_budget,
            )
            if fitted_basis is None:
                basis = old_basis
                coefficients = old_coefficients
                records = old_records
                drift = old_drift
                outcome = "rejected"
                evicted = ()
            else:
                basis = fitted_basis
                affected = tuple(sorted(set(old_records) | set(records)))
        self._basis = basis
        self._coefficients = coefficients
        self._records = records
        self._drift_sha256_by_handle = drift
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("online SHARE adapter exceeded the exact byte budget")
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
        token = hashlib.sha256(b"online-share-snapshot-v1\0" + payload).hexdigest()
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
        Float32,
        dict[str, Float32],
        dict[str, OnlineHandleRecord],
        dict[str, str],
        int | None,
    ]:
        components = decode_state(payload)
        allocator = components["allocator_state"]
        if len(allocator) != 1 or not isinstance(allocator[0], dict):
            raise ValueError("online SHARE allocator state is invalid")
        row = allocator[0]
        raw_basis = row.get("data")
        shape = row.get("shape")
        dtype = row.get("dtype")
        if type(raw_basis) is not bytes or not isinstance(shape, list) or type(dtype) is not str:
            raise ValueError("online SHARE basis record is invalid")
        basis = (
            np.frombuffer(raw_basis, dtype=np.dtype(dtype))
            .reshape(tuple(int(value) for value in shape))
            .copy()
        )
        if basis.dtype != np.float32 or basis.ndim != 2 or basis.shape[1] != 480:
            raise ValueError("online SHARE basis layout is invalid")
        drift_raw = row.get("drift_sha256_by_handle")
        if not isinstance(drift_raw, dict):
            raise ValueError("online SHARE drift map is invalid")
        drift = {str(handle): str(value) for handle, value in drift_raw.items()}
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("online SHARE controller state is invalid")
        if controller[0].get("rank") != self.rank:
            raise ValueError("online SHARE rank differs from exported state")
        last_event = controller[0].get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("online SHARE last event index is invalid")
        coefficient_rows = {
            str(item["handle"]): item
            for item in components["base_codes"]
            if isinstance(item, dict)
        }
        usage_rows = {
            str(item["handle"]): item
            for item in components["usage_age"]
            if isinstance(item, dict)
        }
        if set(coefficient_rows) != set(usage_rows):
            raise ValueError("online SHARE coefficient and metadata handles differ")
        coefficients: dict[str, Float32] = {}
        records: dict[str, OnlineHandleRecord] = {}
        for handle in sorted(coefficient_rows):
            coefficient_row = coefficient_rows[handle]
            raw = coefficient_row.get("data")
            coefficient_shape = coefficient_row.get("shape")
            coefficient_dtype = coefficient_row.get("dtype")
            if (
                type(raw) is not bytes
                or not isinstance(coefficient_shape, list)
                or type(coefficient_dtype) is not str
            ):
                raise ValueError("online SHARE coefficient record is invalid")
            coefficient = (
                np.frombuffer(raw, dtype=np.dtype(coefficient_dtype))
                .reshape(tuple(int(value) for value in coefficient_shape))
                .copy()
            )
            if coefficient.dtype != np.float32 or coefficient.shape != (len(basis),):
                raise ValueError("online SHARE coefficient layout is invalid")
            coefficients[handle] = coefficient
            meta = usage_rows[handle]
            records[handle] = OnlineHandleRecord(
                handle,
                int(meta["created_event"]),
                int(meta["last_read_event"]),
                int(meta["reads"]),
                int(meta["update_count"]),
            )
        return basis, coefficients, records, drift, last_event

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        basis, coefficients, records, drift, last_event = self._state_from_payload(payload)
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export(basis, coefficients, records, drift) != payload:
            self._last_event_index = prior_event
            raise ValueError("online SHARE state does not roundtrip canonically")
        self._basis = basis
        self._coefficients = coefficients
        self._records = records
        self._drift_sha256_by_handle = drift

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract, _reader = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        basis, coefficients, _records, _drift, _last = self._state_from_payload(payload)
        coefficient = coefficients.get(probe.handle)
        if coefficient is None:
            raise ValueError("probe handle is absent from online SHARE snapshot")
        code = (coefficient @ basis).astype(np.float32)
        _code_sha, sample_sha = self._sample_sha(
            code,
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

    def inspect_state(self) -> OnlineShareStateView:
        self._require_active()
        return OnlineShareStateView(
            basis=self._basis.copy(),
            coefficients={handle: value.copy() for handle, value in self._coefficients.items()},
            records=dict(self._records),
            drift_sha256_by_handle=dict(self._drift_sha256_by_handle),
            ledger=self.state_ledger(),
        )

    def close(self) -> None:
        self._basis = np.empty((0, 480), dtype=np.float32)
        self._coefficients.clear()
        self._records.clear()
        self._drift_sha256_by_handle.clear()
        self._snapshots.clear()
        self._contract = None
        self._closed = True


__all__ = [
    "OnlineHandleRecord",
    "OnlineShareAdapter",
    "OnlineShareStateView",
    "canonicalize_basis_signs",
    "reconstruction_drift_sha256",
    "update_subspace",
]
