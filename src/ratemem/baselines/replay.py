"""Paired, byte-audited lifecycle replay for every matched comparator."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from ratemem.baselines.protocol import (
    BaselineAdapter,
    CausalEventView,
    EventReceipt,
    FrozenComparisonContract,
    ProbeResult,
)
from ratemem.evaluation.canonical import (
    canonical_json_bytes,
    file_sha256,
    write_json_atomic,
    write_text_atomic,
)
from ratemem.evaluation.traces import ProbeEvent, Trace

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class PairedReplayError(RuntimeError):
    """Raised when a method breaks the locked paired replay contract."""


class AccessAuditRow(BaseModel):
    model_config = _MODEL_CONFIG

    event_index: NonNegativeInt
    current_event: NonNegativeInt
    maximum_visible_event: NonNegativeInt
    access_mode: Literal["causal_prefix", "full_trace_upper_reference"]


class MethodReplayArtifact(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    method_id: str = Field(min_length=1)
    role: Literal["causal", "upper_reference", "latency_control"]
    trace_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_commitment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_stream_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_budget: int = Field(gt=0)
    receipts: tuple[EventReceipt, ...]
    probes: tuple[ProbeResult, ...]
    access_audit: tuple[AccessAuditRow, ...]
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("artifact_sha256")
        return canonical_json_bytes(payload)


class PairedReplay(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    trace_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_commitment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: dict[str, MethodReplayArtifact]
    replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifacts(self) -> PairedReplay:
        if not self.artifacts or tuple(self.artifacts) != tuple(sorted(self.artifacts)):
            raise ValueError("paired replay artifacts must be non-empty and sorted")
        for method_id, artifact in self.artifacts.items():
            if method_id != artifact.method_id:
                raise ValueError("paired replay artifact key differs from method id")
            if artifact.trace_id != self.trace_id:
                raise ValueError("paired replay artifact trace differs")
            if artifact.input_commitment_sha256 != self.input_commitment_sha256:
                raise ValueError("paired replay artifact input commitment differs")
            if hashlib.sha256(artifact.semantic_bytes).hexdigest() != artifact.artifact_sha256:
                raise ValueError("paired replay artifact hash mismatch")
        return self

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("replay_sha256")
        return canonical_json_bytes(payload)


_PAIRED_CONTRACT_FIELDS = (
    "trace_id",
    "dataset_lock_sha256",
    "evaluation_lock_sha256",
    "baseline_requirements_sha256",
    "backbone_id",
    "backbone_revision",
    "adapter_layout_sha256",
    "prompt_pool_sha256",
    "support_pool_sha256",
    "noise_seed_manifest_sha256",
    "sampler_id",
    "scheduler_revision",
    "cfg_scale",
    "resolution",
    "denoising_steps",
    "byte_budget",
    "request_regime",
    "search_budget_sha256",
)


def paired_input_commitment(
    trace: Trace,
    contracts: dict[str, FrozenComparisonContract],
) -> str:
    if type(trace) is not Trace:
        raise TypeError("trace must be an exact Trace")
    if not contracts or tuple(contracts) != tuple(sorted(contracts)):
        raise PairedReplayError("contracts must be non-empty and sorted by method id")
    reference = next(iter(contracts.values()))
    if reference.trace_id != trace.trace_id:
        raise PairedReplayError("comparison contract trace differs from the replay trace")
    for method_id, contract in contracts.items():
        if any(
            getattr(contract, field) != getattr(reference, field)
            for field in _PAIRED_CONTRACT_FIELDS
        ):
            raise PairedReplayError(f"paired contract invariants differ for {method_id}")
    payload = {
        "trace": trace.model_dump(mode="json"),
        "paired_contract": {
            field: reference.model_dump(mode="json")[field]
            for field in _PAIRED_CONTRACT_FIELDS
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_receipt(
    receipt: EventReceipt,
    event_index: int,
    adapter: BaselineAdapter,
    contract: FrozenComparisonContract,
) -> None:
    if receipt.method_id != adapter.method_id or receipt.trace_id != contract.trace_id:
        raise PairedReplayError("event receipt method or trace identity differs")
    if receipt.event_index != event_index:
        raise PairedReplayError("event receipt index differs from the replay event")
    exported = adapter.export_online_state()
    observed_sha = hashlib.sha256(exported).hexdigest()
    if (
        receipt.ledger.online_state_bytes != len(exported)
        or receipt.ledger.online_state_sha256 != observed_sha
        or sum(receipt.ledger.component_bytes.values()) != len(exported)
    ):
        raise PairedReplayError("event receipt ledger differs from host-observed state")
    if len(exported) > contract.byte_budget:
        raise PairedReplayError("method exceeded the exact online byte budget")


def _stderr_sha256(adapter: BaselineAdapter) -> str:
    observer = getattr(adapter, "stderr_sha256", None)
    if observer is None:
        return hashlib.sha256(b"").hexdigest()
    value = observer()
    if type(value) is not str or len(value) != 64:
        raise PairedReplayError("adapter stderr digest is invalid")
    return value


def replay_one(
    trace: Trace,
    adapter: BaselineAdapter,
    contract: FrozenComparisonContract,
    *,
    input_commitment_sha256: str,
) -> MethodReplayArtifact:
    receipts: list[EventReceipt] = []
    probes: list[ProbeResult] = []
    access: list[AccessAuditRow] = []
    snapshots: dict[int, object] = {}
    adapter.initialize(contract)
    try:
        for event in trace.events:
            view = CausalEventView(trace.events, event.event_index)
            access.append(
                AccessAuditRow(
                    event_index=event.event_index,
                    current_event=event.event_index,
                    maximum_visible_event=(
                        len(trace.events) - 1
                        if adapter.role == "upper_reference"
                        else view.current_index
                    ),
                    access_mode=(
                        "full_trace_upper_reference"
                        if adapter.role == "upper_reference"
                        else "causal_prefix"
                    ),
                )
            )
            if isinstance(event, ProbeEvent):
                snapshot = snapshots.get(event.snapshot_event_index)
                if snapshot is None:
                    raise PairedReplayError("probe references a missing operational snapshot")
                before = adapter.export_online_state()
                result = adapter.score_probe(snapshot, event)  # type: ignore[arg-type]
                after = adapter.export_online_state()
                if before != after or result.update_usage is not False:
                    raise PairedReplayError("probe mutated online state or usage")
                if result.probe_event_index != event.event_index:
                    raise PairedReplayError("probe result index differs from the trace")
                probes.append(result)
                continue
            receipt = adapter.apply_event(event, view)
            _validate_receipt(receipt, event.event_index, adapter, contract)
            expected_input = hashlib.sha256(
                canonical_json_bytes(event.model_dump(mode="json"))
            ).hexdigest()
            if receipt.input_commitment_sha256 != expected_input:
                raise PairedReplayError("event receipt input commitment differs")
            receipts.append(receipt)
            snapshot = adapter.copy_snapshot()
            exported = adapter.export_online_state()
            if (
                snapshot.event_index != event.event_index
                or snapshot.state_sha256 != hashlib.sha256(exported).hexdigest()
                or snapshot.online_state_bytes != len(exported)
            ):
                raise PairedReplayError("method snapshot differs from host-observed state")
            snapshots[event.event_index] = snapshot
        stderr_sha = _stderr_sha256(adapter)
        provisional = MethodReplayArtifact(
            method_id=adapter.method_id,
            role=adapter.role,
            trace_id=trace.trace_id,
            input_commitment_sha256=input_commitment_sha256,
            candidate_stream_sha256=contract.candidate_stream_sha256,
            byte_budget=contract.byte_budget,
            receipts=tuple(receipts),
            probes=tuple(probes),
            access_audit=tuple(access),
            stderr_sha256=stderr_sha,
            artifact_sha256="0" * 64,
        )
        return provisional.model_copy(
            update={
                "artifact_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
            }
        )
    finally:
        adapter.close()


def replay_paired(
    trace: Trace,
    adapters: dict[str, BaselineAdapter],
    contracts: dict[str, FrozenComparisonContract],
) -> PairedReplay:
    """Run every method on one frozen trace with identical observable inputs."""

    method_ids = tuple(sorted(adapters))
    if tuple(adapters) != method_ids or tuple(contracts) != method_ids:
        raise PairedReplayError("adapter and contract registries must have identical sorted keys")
    if any(method_id != adapters[method_id].method_id for method_id in method_ids):
        raise PairedReplayError("adapter registry key differs from adapter method id")
    commitment = paired_input_commitment(trace, contracts)
    artifacts = {
        method_id: replay_one(
            trace,
            adapters[method_id],
            contracts[method_id],
            input_commitment_sha256=commitment,
        )
        for method_id in method_ids
    }
    provisional = PairedReplay(
        trace_id=trace.trace_id,
        input_commitment_sha256=commitment,
        artifacts=artifacts,
        replay_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={"replay_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )


def _jsonl(rows: tuple[BaseModel, ...]) -> str:
    return "".join(
        canonical_json_bytes(row.model_dump(mode="json")).decode("utf-8") + "\n"
        for row in rows
    )


def write_paired_replay(root: Path, replay: PairedReplay) -> Path:
    """Publish immutable per-method replay artifacts and a checksummed manifest."""

    if root.exists() or root.is_symlink():
        raise FileExistsError("paired replay output must not already exist")
    root.mkdir(parents=True, exist_ok=False)
    top_files: dict[str, str] = {}
    for method_id, artifact in replay.artifacts.items():
        method_root = root / method_id
        method_root.mkdir()
        write_json_atomic(method_root / "manifest.json", artifact.model_dump(mode="json"))
        write_text_atomic(method_root / "events.jsonl", _jsonl(artifact.receipts))
        write_text_atomic(method_root / "probes.jsonl", _jsonl(artifact.probes))
        write_text_atomic(
            method_root / "ledgers.jsonl",
            _jsonl(tuple(receipt.ledger for receipt in artifact.receipts)),
        )
        write_text_atomic(method_root / "access-audit.jsonl", _jsonl(artifact.access_audit))
        files = {
            path.name: file_sha256(path)
            for path in sorted(method_root.iterdir())
            if path.is_file()
        }
        write_json_atomic(method_root / "checksums.json", files)
        top_files[f"{method_id}/checksums.json"] = file_sha256(
            method_root / "checksums.json"
        )
    manifest = {
        "schema_version": "1.0",
        "trace_id": replay.trace_id,
        "input_commitment_sha256": replay.input_commitment_sha256,
        "replay_sha256": replay.replay_sha256,
        "method_checksums": top_files,
    }
    write_json_atomic(root / "manifest.json", manifest)
    return root / "manifest.json"


__all__ = [
    "AccessAuditRow",
    "MethodReplayArtifact",
    "PairedReplay",
    "PairedReplayError",
    "paired_input_commitment",
    "replay_one",
    "replay_paired",
    "write_paired_replay",
]
