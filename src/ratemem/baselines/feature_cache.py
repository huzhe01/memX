"""DreamCache-style feature-state lifecycle control."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
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

Policy = Literal["fifo", "lru"]
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
class CachedFeature:
    tensor: NDArray[np.generic]
    tap_path: str
    injection_path: str
    encoding_timestep: int
    scale: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tensor, np.ndarray)
            or self.tensor.dtype.hasobject
            or not np.issubdtype(self.tensor.dtype, np.number)
            or not np.isfinite(self.tensor).all()
        ):
            raise ValueError("cached feature must be a finite numeric ndarray")
        if not self.tap_path or not self.injection_path:
            raise ValueError("cached feature paths must be non-empty")
        if type(self.encoding_timestep) is not int or self.encoding_timestep < 0:
            raise ValueError("cached feature timestep must be nonnegative")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("cached feature scale must be finite and positive")
        object.__setattr__(self, "tensor", np.ascontiguousarray(self.tensor).copy())


class FeatureBackend(Protocol):
    backbone_id: str
    source_revision: str
    shared_trained_bytes: int

    def encode_support(
        self,
        support_image_ids: Sequence[str],
        description_id: str,
    ) -> CachedFeature: ...

    def generate(self, feature: CachedFeature, prompt_id: str, seed: int) -> Tensor: ...

    def one_step_latent(
        self,
        feature: CachedFeature,
        prompt_id: str,
        seed: int,
        timestep: int,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    handle: str
    feature: CachedFeature
    description_id: str
    created_event: int
    last_read_event: int
    update_count: int


@dataclass(frozen=True, slots=True)
class FeatureStateView:
    handles: tuple[str, ...]
    cached_features: dict[str, NDArray[np.generic]]
    records: dict[str, FeatureRecord]


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class FeatureCacheAdapter:
    """Stores only canonical features and never retained support-image bytes."""

    role: Literal["causal"] = "causal"

    def __init__(
        self,
        method_id: str,
        backend: FeatureBackend,
        *,
        policy: Policy = "lru",
    ) -> None:
        if method_id != "dreamcache_feature_cache":
            raise ValueError("feature-cache method id must be dreamcache_feature_cache")
        if policy not in {"fifo", "lru"}:
            raise ValueError("feature-cache policy must be fifo or lru")
        if type(backend.shared_trained_bytes) is not int or backend.shared_trained_bytes < 0:
            raise ValueError("feature backend shared trained bytes are invalid")
        self.method_id = method_id
        self.backend = backend
        self.policy = policy
        self.shared_trained_bytes = backend.shared_trained_bytes
        self.external_support_bytes = 0
        self._contract: FrozenComparisonContract | None = None
        self._records: dict[str, FeatureRecord] = {}
        self._last_event_index: int | None = None
        self._snapshots: dict[str, bytes] = {}
        self._closed = False

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("feature adapter cannot be initialized twice or after close")
        if self.backend.backbone_id != contract.backbone_id:
            raise ValueError("feature backend does not use the frozen primary backbone")
        if len(self.backend.source_revision) != 40:
            raise ValueError("feature backend source revision is not immutable")
        self._contract = contract
        if self.state_ledger().online_state_bytes > contract.byte_budget:
            self._contract = None
            raise ValueError("byte budget is smaller than canonical empty state")

    def _require_active(self) -> FrozenComparisonContract:
        if self._contract is None or self._closed:
            raise RuntimeError("feature adapter is not active")
        return self._contract

    @staticmethod
    def _feature_row(record: FeatureRecord) -> dict[str, object]:
        tensor = np.ascontiguousarray(record.feature.tensor)
        return {
            "handle": record.handle,
            "dtype": tensor.dtype.str,
            "shape": list(tensor.shape),
            "data": tensor.tobytes(order="C"),
            "tap_path": record.feature.tap_path,
            "injection_path": record.feature.injection_path,
            "encoding_timestep": record.feature.encoding_timestep,
            "scale": record.feature.scale,
        }

    def _components(
        self,
        records: dict[str, FeatureRecord] | None = None,
    ) -> dict[str, list[object]]:
        selected = self._records if records is None else records
        components = empty_components()
        for handle in sorted(selected):
            record = selected[handle]
            row = self._feature_row(record)
            components["feature_cache"].append(row)
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
                {
                    "handle": handle,
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(record.feature.tensor).tobytes(order="C")
                    ).hexdigest(),
                }
            )
        components["controller_state"].append(
            {"policy": self.policy, "last_event_index": self._last_event_index}
        )
        return components

    def _export_records(self, records: dict[str, FeatureRecord]) -> bytes:
        return export_state(self._components(records))

    def export_online_state(self) -> bytes:
        self._require_active()
        return self._export_records(self._records)

    def state_ledger(self) -> ExactByteLedger:
        payload = self.export_online_state()
        return ledger_from_export(
            payload,
            shared_trained_bytes=self.shared_trained_bytes,
            external_support_bytes=0,
        )

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        components = decode_state(payload)
        controller = components["controller_state"]
        if len(controller) != 1 or not isinstance(controller[0], dict):
            raise ValueError("feature-cache controller state is invalid")
        if controller[0].get("policy") != self.policy:
            raise ValueError("feature-cache policy differs from exported state")
        last_event = controller[0].get("last_event_index")
        if last_event is not None and (type(last_event) is not int or last_event < 0):
            raise ValueError("feature-cache last event index is invalid")
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
        feature_rows = {
            str(row["handle"]): row
            for row in components["feature_cache"]
            if isinstance(row, dict)
        }
        if set(metadata) != set(descriptions) or set(metadata) != set(feature_rows):
            raise ValueError("feature-cache exported handle sets differ")
        restored: dict[str, FeatureRecord] = {}
        for handle in sorted(metadata):
            row = feature_rows[handle]
            raw = row.get("data")
            shape = row.get("shape")
            dtype = row.get("dtype")
            if type(raw) is not bytes or not isinstance(shape, list) or type(dtype) is not str:
                raise ValueError("feature-cache tensor record is invalid")
            array = (
                np.frombuffer(raw, dtype=np.dtype(dtype))
                .reshape(tuple(int(x) for x in shape))
                .copy()
            )
            feature = CachedFeature(
                tensor=array,
                tap_path=str(row["tap_path"]),
                injection_path=str(row["injection_path"]),
                encoding_timestep=int(row["encoding_timestep"]),
                scale=float(row["scale"]),
            )
            meta = metadata[handle]
            restored[handle] = FeatureRecord(
                handle=handle,
                feature=feature,
                description_id=str(descriptions[handle]["description_id"]),
                created_event=int(meta["created_event"]),
                last_read_event=int(meta["last_read_event"]),
                update_count=int(meta["update_count"]),
            )
        prior_event = self._last_event_index
        self._last_event_index = last_event
        if self._export_records(restored) != payload:
            self._last_event_index = prior_event
            raise ValueError("feature-cache state does not roundtrip canonically")
        self._records = restored

    def _fit_with_eviction(
        self,
        records: dict[str, FeatureRecord],
        *,
        protected_handle: str,
        budget: int,
    ) -> tuple[dict[str, FeatureRecord] | None, tuple[str, ...]]:
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
            if self.policy == "fifo":
                victim = min(eligible, key=lambda row: (row.created_event, row.handle))
            else:
                victim = min(
                    eligible,
                    key=lambda row: (row.last_read_event, row.created_event, row.handle),
                )
            del candidate[victim.handle]
            evicted.append(victim.handle)
        return candidate, tuple(evicted)

    def _sample_sha(self, feature: CachedFeature, prompt_id: str, seed: int) -> str:
        return _tensor_sha256(self.backend.generate(feature, prompt_id, seed))

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract = self._require_active()
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
        generated_sha: str | None = None
        decoded_sha: str | None = None
        outcome: Outcome
        if isinstance(event, CreateEvent):
            if event.handle in records:
                outcome = "rejected"
            else:
                feature = self.backend.encode_support(
                    event.support_image_ids,
                    event.description_id,
                )
                records[event.handle] = FeatureRecord(
                    event.handle,
                    feature,
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
                    outcome = "created"
        elif isinstance(event, UpdateEvent):
            previous = records.get(event.handle)
            if previous is None:
                outcome = "stale_handle"
            else:
                feature = self.backend.encode_support(
                    event.support_image_ids,
                    previous.description_id,
                )
                records[event.handle] = replace(
                    previous,
                    feature=feature,
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
                    outcome = "updated"
        elif isinstance(event, ReadEvent):
            record = records.get(event.handle)
            if record is None:
                outcome = "stale_handle"
            else:
                records[event.handle] = replace(record, last_read_event=event.event_index)
                decoded_sha = hashlib.sha256(
                    np.ascontiguousarray(record.feature.tensor).tobytes(order="C")
                ).hexdigest()
                generated_sha = self._sample_sha(
                    record.feature,
                    event.prompt_id,
                    event.generation_seed,
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
            raise TypeError(f"unsupported feature-cache event: {type(event).__name__}")
        self._records = records
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise RuntimeError("feature-cache adapter exceeded the exact byte budget")
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
        token = hashlib.sha256(b"feature-snapshot-v1\0" + payload).hexdigest()
        self._snapshots[token] = payload
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(payload),
            opaque_snapshot_token=token,
        )

    @staticmethod
    def _feature_from_snapshot(payload: bytes, handle: str) -> CachedFeature:
        rows = decode_state(payload)["feature_cache"]
        matches = [row for row in rows if isinstance(row, dict) and row.get("handle") == handle]
        if len(matches) != 1:
            raise ValueError("probe handle is absent from the feature snapshot")
        row = matches[0]
        raw = row.get("data")
        shape = row.get("shape")
        dtype = row.get("dtype")
        if type(raw) is not bytes or not isinstance(shape, list) or type(dtype) is not str:
            raise ValueError("snapshot feature tensor is invalid")
        array = (
            np.frombuffer(raw, dtype=np.dtype(dtype))
            .reshape(tuple(int(x) for x in shape))
            .copy()
        )
        return CachedFeature(
            array,
            str(row["tap_path"]),
            str(row["injection_path"]),
            int(row["encoding_timestep"]),
            float(row["scale"]),
        )

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        payload = self._snapshots.get(snapshot.opaque_snapshot_token)
        if payload is None or hashlib.sha256(payload).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        feature = self._feature_from_snapshot(payload, probe.handle)
        generated_sha = self._sample_sha(feature, probe.prompt_id, probe.generation_seed)
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

    def inspect_state(self) -> FeatureStateView:
        self._require_active()
        return FeatureStateView(
            handles=tuple(sorted(self._records)),
            cached_features={
                handle: record.feature.tensor.copy()
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
    "CachedFeature",
    "FeatureBackend",
    "FeatureCacheAdapter",
    "FeatureRecord",
    "FeatureStateView",
]
