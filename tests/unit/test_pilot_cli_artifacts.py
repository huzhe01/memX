from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

import ratemem.pilot.cli as pilot_cli
from ratemem.adapters.checkpoint import CheckpointFileIdentity
from ratemem.pilot.artifacts import ArtifactWriter
from ratemem.pilot.one_shot import (
    PilotIdentity,
    claim_global_pilot_slot,
    consume_launch_request,
    create_launch_permit,
)
from ratemem.pilot.private_io import (
    canonical_json_bytes,
    read_private_json,
)

ATTEMPT_ID = "019d0000-0000-7000-8000-000000000021"
GIT_COMMIT = "a" * 40
SOURCE_SHA256 = hashlib.sha256(GIT_COMMIT.encode("ascii")).hexdigest()
GIT_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
WORKSPACE = "authorized-workspace"
PROBE_NAMES = (
    "checkpoint_compatibility",
    "dynamic_numerics",
    "gradient_flow",
    "frozen_backbone",
    "peak_memory",
    "one_step_inference",
    "one_timestep_backward",
    "step_timing",
    "held_in_loss",
)
RATES = {
    "gpu_l40s_per_second": "0.000542",
    "cpu_core_per_second": "0.0000131",
    "memory_gib_per_second": "0.00000222",
    "volume_gib_month": "0.09",
}
RATES_SHA256 = hashlib.sha256(canonical_json_bytes(RATES)).hexdigest()


@dataclass(frozen=True, slots=True)
class BuiltArtifact:
    pending: Path
    slot: Path
    permit: Path
    submission_receipt: Path
    request: dict[str, object]


ArtifactMutation = Literal[
    "none",
    "launch",
    "permit",
    "execution_receipt",
    "credential",
    "forensic",
    "forensic_extra",
    "forensic_token",
    "forensic_bad_hash",
    "diagnostics_conflict",
    "metrics_projection",
    "one_timestep_unbound",
    "one_timestep_finite_tamper",
]
ArtifactFactory = Callable[..., BuiltArtifact]


def _attempt(
    *,
    request: dict[str, object],
    config_sha256: str,
    dataset_sha256: str,
    status: Literal["succeeded", "exception"],
    checkpoint: bytes | None,
) -> dict[str, Any]:
    succeeded = status == "succeeded"
    results = {name: {"status": "pass" if succeeded else "not_run"} for name in PROBE_NAMES}
    return {
        "schema_version": "1.0.0",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "attempt_id": ATTEMPT_ID,
        "phase": "first_pilot",
        "status": status,
        "started_at": "2026-08-29T12:00:00+00:00",
        "ended_at": "2026-08-29T12:01:00+00:00",
        "source": {
            "git_commit": GIT_COMMIT,
            "git_diff_sha256": GIT_DIFF_SHA256,
            "config_sha256": config_sha256,
        },
        "software": {
            "python": "3.11.13",
            "torch": "2.13.0",
            "diffusers": "0.40.0",
            "peft": "0.20.0",
            "transformers": "5.16.1",
            "modal": "1.5.4",
            "container_image_id": "im-test",
        },
        "model": {
            "model_id": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            "revision": "b77948f2b4eed5c728e9b828ccff07f7427b43cc",
            "support_model_id": "facebook/dinov2-small",
            "support_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        },
        "dataset": {
            "dataset_id": "Yuanshi/Subjects200K",
            "revision": "0d1cf6536239888f1a8e218790649344810067bc",
            "manifest_sha256": dataset_sha256,
            "row_indices": list(range(8)),
            "held_in": True,
        },
        "runtime": {
            "seed": 20260824,
            "requested_gpu": "L40S",
            "observed_gpu": "NVIDIA L40S",
            "gpu_count": 1,
            "cpu_cores": 4,
            "memory_gib": 32,
            "timeout_seconds": 7200,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
        },
        "modal": {
            "profile": "ratemem-pilot",
            "workspace": WORKSPACE,
            "environment": "main",
            "launch_attempt_id": ATTEMPT_ID,
            "launch_source_sha256": SOURCE_SHA256,
            "pilot_slot_sha256": request["slot_sha256"],
            "submission_receipt_sha256": request["submission_receipt_sha256"],
            "function_call_id": "fc-test",
            "input_id": "in-test",
            "task_id": "ta-test",
            "execution_receipt_count": 1,
            "execution_receipt_semantics": "lower_bound_may_miss_precommit_reschedule",
            "retries": 0,
            "detached": False,
        },
        "cost": {
            "workspace_budget_usd": "28.00",
            "internal_limit_usd": "27.00",
            "known_usage_before_usd": "1.00",
            "pending_worst_case_usd": "10.15",
            "phase_bound_usd": "10.15",
            "estimated_cost_usd": "0.01",
            "reconciliation_status": "pending",
            "reconciled_cost_usd": None,
            "rates_sha256": RATES_SHA256,
        },
        "probes": {
            "allowed_probe_names": list(PROBE_NAMES),
            "results": results,
            "warmup_steps": 10,
            "measured_steps": 20,
            "p50_step_seconds": 1.0 if succeeded else None,
            "p95_step_seconds": 1.2 if succeeded else None,
            "held_in_step_cap": 1 if succeeded else 0,
            "initial_flow_loss": 1.1 if succeeded else None,
            "final_flow_loss": 0.9 if succeeded else None,
            "transformer_passes_per_step": 1,
        },
        "checkpoint": (
            None
            if checkpoint is None
            else {
                "path": "trainable.safetensors",
                "sha256": hashlib.sha256(checkpoint).hexdigest(),
                "bytes": len(checkpoint),
            }
        ),
        "files": {"checksums_sha256": "0" * 64},
        "error": (
            None
            if succeeded
            else {
                "type": "PilotException",
                "message": "engineering pilot execution failed",
            }
        ),
    }


@pytest.fixture
def artifact_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ArtifactFactory:
    def build(
        *,
        status: Literal["succeeded", "exception"] = "succeeded",
        mutation: ArtifactMutation = "none",
    ) -> BuiltArtifact:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)

        state = tmp_path / "state"
        permits = tmp_path / "permits"
        attempts = tmp_path / "attempts"
        state.mkdir(mode=0o700)
        permits.mkdir(mode=0o700)
        attempts.mkdir(mode=0o700)
        slot = state / "modal-pilot-slot.json"
        permit = permits / "launch-permit.json"
        submission_receipt = state / "modal-pilot-submitted.json"

        config = {
            "phase": "first_pilot",
            "scope": "engineering_pilot_only",
            "schema_version": 1,
        }
        config_bytes = canonical_json_bytes(config)
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        identity = PilotIdentity(ATTEMPT_ID, WORKSPACE, SOURCE_SHA256, GIT_COMMIT)
        claim_global_pilot_slot(slot, identity=identity)
        create_launch_permit(
            permit,
            slot=slot,
            receipt=submission_receipt,
            identity=identity,
            known_usage_before_usd="1.00",
            pending_worst_case_usd="10.15",
            phase_bound_usd="10.15",
            rates=RATES,
            rates_sha256=RATES_SHA256,
            git_diff_sha256=GIT_DIFF_SHA256,
            config_sha256=config_sha256,
        )
        request = consume_launch_request(
            permit,
            slot=slot,
            receipt=submission_receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE_SHA256,
        )

        monkeypatch.setattr(pilot_cli, "PERMIT_PATH", permit)
        monkeypatch.setattr(pilot_cli, "GLOBAL_SLOT_PATH", slot)
        monkeypatch.setattr(
            pilot_cli,
            "GLOBAL_SUBMISSION_RECEIPT_PATH",
            submission_receipt,
        )
        monkeypatch.setattr(
            pilot_cli,
            "source_tree_identity",
            lambda: pilot_cli.SourceTreeIdentity(
                git_commit=GIT_COMMIT,
                source_sha256=SOURCE_SHA256,
                git_diff_sha256=GIT_DIFF_SHA256,
            ),
        )
        monkeypatch.setattr(pilot_cli, "pilot_config_sha256", lambda: config_sha256)

        dataset = {
            "dataset_id": "Yuanshi/Subjects200K",
            "held_in": True,
            "revision": "0d1cf6536239888f1a8e218790649344810067bc",
            "row_indices": list(range(8)),
            "scope": "engineering_pilot_only",
        }
        dataset_bytes = canonical_json_bytes(dataset)
        checkpoint = b"realistic-test-checkpoint" if status == "succeeded" else None
        attempt = _attempt(
            request=request,
            config_sha256=config_sha256,
            dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
            status=status,
            checkpoint=checkpoint,
        )
        if mutation == "launch":
            attempt["modal"]["workspace"] = "different-workspace"
        elif mutation == "permit":
            attempt["cost"]["known_usage_before_usd"] = "1.01"

        execution_receipt = {
            **request,
            "function_call_id": "fc-test",
            "input_id": "in-test",
            "task_id": "ta-test",
            "receipt_id": "b" * 64,
            "observed_at": "2026-08-29T12:00:00.000000+00:00",
            "semantics": "lower_bound_may_miss_precommit_reschedule",
        }
        if mutation == "execution_receipt":
            execution_receipt["workspace"] = "different-workspace"

        execution_receipt_bytes = canonical_json_bytes(execution_receipt) + b"\n"
        if mutation.startswith("forensic"):
            marker: dict[str, object] = {
                "attempt_id": ATTEMPT_ID,
                "evidence": "external_forensic_directory",
                "forensic_path": f"execution-receipts/{ATTEMPT_ID}",
                "raw_snapshot_bytes": len(execution_receipt_bytes),
                "raw_snapshot_sha256": hashlib.sha256(execution_receipt_bytes).hexdigest(),
                "scope": "engineering_pilot_only",
                "status": "semantic_invalid",
            }
            if mutation == "forensic_extra":
                marker["raw_receipt"] = "untrusted bytes must not be copied"
            elif mutation == "forensic_token":
                marker["injected_token"] = "ak" + "-" + "FAKE_MARKER_TOKEN"
            elif mutation == "forensic_bad_hash":
                marker["raw_snapshot_sha256"] = "not-a-sha"
            execution_receipt_bytes = canonical_json_bytes(marker) + b"\n"

        result = {
            "status": status,
            "allowed_probe_names": list(PROBE_NAMES),
            "results": attempt["probes"]["results"],
            **{name: attempt["probes"]["results"][name] for name in PROBE_NAMES},
            "warmup_steps": attempt["probes"]["warmup_steps"],
            "measured_steps": attempt["probes"]["measured_steps"],
            "p50_step_seconds": attempt["probes"]["p50_step_seconds"],
            "p95_step_seconds": attempt["probes"]["p95_step_seconds"],
            "held_in_step_cap": attempt["probes"]["held_in_step_cap"],
            "one_timestep_backward_loss": 0.5 if status == "succeeded" else None,
            "initial_flow_loss": attempt["probes"]["initial_flow_loss"],
            "final_flow_loss": attempt["probes"]["final_flow_loss"],
            "transformer_passes_per_step": attempt["probes"]["transformer_passes_per_step"],
            "checkpoint_sha256": (
                None if attempt["checkpoint"] is None else attempt["checkpoint"]["sha256"]
            ),
            "checkpoint_bytes": (
                None if attempt["checkpoint"] is None else attempt["checkpoint"]["bytes"]
            ),
        }
        if mutation == "metrics_projection":
            result["final_flow_loss"] = 99.0
        elif mutation == "one_timestep_unbound":
            result["one_timestep_backward_loss"] = None
        elif mutation == "one_timestep_finite_tamper":
            result["one_timestep_backward_loss"] = 0.7
        diagnostics: dict[str, object] = {
            "scope": "engineering_pilot_only",
            "backend_initialized": status == "succeeded",
            "standalone_backward_loss": 0.5 if status == "succeeded" else None,
            "execution_receipt_semantic_invalid": False,
            "execution_receipt_evidence": "validated_canonical_snapshot",
        }
        if mutation.startswith("forensic"):
            diagnostics.update(
                {
                    "execution_receipt_semantic_invalid": True,
                    "execution_receipt_evidence": "external_forensic_directory",
                }
            )
        if mutation == "credential":
            credential_name = "HF_" + "TOKEN"
            diagnostics["test_only_marker"] = f"{credential_name}=definitely_fake_test_token"
        elif mutation == "diagnostics_conflict":
            diagnostics["execution_receipt_semantic_invalid"] = True
            diagnostics["execution_receipt_evidence"] = "external_forensic_directory"
        metrics = b"".join(
            canonical_json_bytes(row) + b"\n"
            for row in (
                {
                    "scope": "engineering_pilot_only",
                    "request_permit_sha256": request["permit_sha256"],
                    "result": result,
                },
                diagnostics,
            )
        )

        checkpoint_identity = (
            None
            if checkpoint is None
            else CheckpointFileIdentity(
                sha256=hashlib.sha256(checkpoint).hexdigest(),
                byte_count=len(checkpoint),
            )
        )
        checkpoint_source = tmp_path / "trainable-staging.safetensors"
        if checkpoint is not None:
            checkpoint_source.write_bytes(checkpoint)
            checkpoint_source.chmod(0o600)
        with ArtifactWriter(
            attempts / ATTEMPT_ID,
            attempt,
            checkpoint_identity=checkpoint_identity,
        ) as writer:
            writer.write_bytes("config.json", config_bytes)
            writer.write_bytes("dataset-manifest.json", dataset_bytes)
            writer.write_bytes(
                "execution-receipts.jsonl",
                execution_receipt_bytes,
            )
            writer.write_bytes("metrics.jsonl", metrics)
            writer.write_bytes("rates.json", canonical_json_bytes(RATES))
            if checkpoint is not None:
                writer.write_checkpoint(checkpoint_source)
            pending = writer.write_pending()
        return BuiltArtifact(
            pending=pending,
            slot=slot,
            permit=permit,
            submission_receipt=submission_receipt,
            request=request,
        )

    return build


def test_validate_artifact_accepts_success_with_checkpoint(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory()

    validated = pilot_cli._validate_artifact(artifact.pending)

    assert validated["status"] == "succeeded"
    assert validated["checkpoint"]["path"] == "trainable.safetensors"
    assert {path.name for path in artifact.pending.parent.iterdir()} == {
        "attempt.pending.json",
        "checksums.sha256",
        "config.json",
        "dataset-manifest.json",
        "execution-receipts.jsonl",
        "metrics.jsonl",
        "rates.json",
        "trainable.safetensors",
    }


def test_validate_artifact_accepts_early_failure_without_checkpoint(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(status="exception")

    validated = pilot_cli._validate_artifact(artifact.pending)

    assert validated["status"] == "exception"
    assert validated["checkpoint"] is None
    assert all(
        result == {"status": "not_run"} for result in validated["probes"]["results"].values()
    )
    assert {path.name for path in artifact.pending.parent.iterdir()} == {
        "attempt.pending.json",
        "checksums.sha256",
        "config.json",
        "dataset-manifest.json",
        "execution-receipts.jsonl",
        "metrics.jsonl",
        "rates.json",
    }


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_validate_artifact_rejects_nonexact_file_set(
    artifact_factory: ArtifactFactory,
    change: str,
) -> None:
    artifact = artifact_factory()
    if change == "missing":
        (artifact.pending.parent / "rates.json").unlink()
    else:
        unexpected = artifact.pending.parent / "unexpected.txt"
        unexpected.write_bytes(b"unexpected")
        unexpected.chmod(0o600)

    with pytest.raises(ValueError, match="missing or unexpected file"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_checksum_tampering(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory()
    metrics = artifact.pending.parent / "metrics.jsonl"
    metrics.write_bytes(metrics.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact checksum mismatch: metrics.jsonl"):
        pilot_cli._validate_artifact(artifact.pending)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("launch", "artifact launch identity differs"),
        ("permit", "artifact cost, rate, diff, or config evidence differs"),
        ("execution_receipt", "execution receipt request differs"),
    ],
)
def test_validate_artifact_rejects_launch_permit_or_execution_receipt_mismatch(
    artifact_factory: ArtifactFactory,
    mutation: ArtifactMutation,
    message: str,
) -> None:
    artifact = artifact_factory(mutation=mutation)

    with pytest.raises(ValueError, match=message):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_tampered_local_submission_receipt(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory()
    receipt = read_private_json(artifact.submission_receipt)
    receipt["permit_sha256"] = "f" * 64
    artifact.submission_receipt.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(ValueError, match="submission receipt binding"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_credential_finding_in_bound_payload(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(mutation="credential")

    with pytest.raises(ValueError, match="credential material found in 1 artifact files"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_accepts_exact_external_forensic_marker_for_exception(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(status="exception", mutation="forensic")
    validated = pilot_cli._validate_artifact(artifact.pending)
    assert validated["status"] == "exception"
    assert validated["checkpoint"] is None


@pytest.mark.parametrize(
    "mutation",
    ["forensic_extra", "forensic_token", "forensic_bad_hash"],
)
def test_validate_artifact_rejects_nonexact_or_raw_forensic_marker(
    artifact_factory: ArtifactFactory,
    mutation: ArtifactMutation,
) -> None:
    artifact = artifact_factory(status="exception", mutation=mutation)
    with pytest.raises(ValueError, match="receipt|marker"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_never_accepts_forensic_marker_as_success_receipt(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(status="succeeded", mutation="forensic")
    with pytest.raises(ValueError, match="marker"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_normal_receipt_with_forensic_diagnostics(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(mutation="diagnostics_conflict")
    with pytest.raises(ValueError, match="canonical receipt diagnostics"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_metrics_result_projection_mismatch(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(mutation="metrics_projection")
    with pytest.raises(ValueError, match="metrics result differs from attempt probes"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_backward_loss_inconsistent_with_probe_state(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(mutation="one_timestep_unbound")
    with pytest.raises(ValueError, match="one-timestep backward loss"):
        pilot_cli._validate_artifact(artifact.pending)


def test_validate_artifact_rejects_finite_backward_loss_tamper(
    artifact_factory: ArtifactFactory,
) -> None:
    artifact = artifact_factory(mutation="one_timestep_finite_tamper")
    with pytest.raises(ValueError, match="independent trainer observation"):
        pilot_cli._validate_artifact(artifact.pending)


def test_forensic_validator_binds_external_snapshot_to_semantic_invalid_marker(
    artifact_factory: ArtifactFactory,
    tmp_path: Path,
) -> None:
    artifact = artifact_factory(status="exception", mutation="forensic")
    receipt = {
        **artifact.request,
        "function_call_id": "fc-test",
        "input_id": "in-test",
        "task_id": "ta-test",
        "receipt_id": "b" * 64,
        "observed_at": "2026-08-29T12:00:00.000000+00:00",
        "semantics": "lower_bound_may_miss_precommit_reschedule",
    }
    directory = tmp_path / ATTEMPT_ID
    directory.mkdir(mode=0o700)
    raw = directory / f"{'b' * 64}.json"
    raw.write_bytes(canonical_json_bytes(receipt))
    raw.chmod(0o600)

    manifest = pilot_cli._validate_forensic_receipts(artifact.pending, directory)
    assert (
        manifest["raw_snapshot_sha256"]
        == hashlib.sha256(canonical_json_bytes(receipt) + b"\n").hexdigest()
    )
    assert manifest["raw_snapshot_bytes"] == len(canonical_json_bytes(receipt)) + 1

    raw.write_bytes(canonical_json_bytes(receipt) + b" ")
    with pytest.raises(ValueError, match="forensic snapshot.*marker"):
        pilot_cli._validate_forensic_receipts(artifact.pending, directory)


def test_forensic_validator_binds_normal_receipts_to_local_jsonl(
    artifact_factory: ArtifactFactory,
    tmp_path: Path,
) -> None:
    artifact = artifact_factory()
    local = (artifact.pending.parent / "execution-receipts.jsonl").read_bytes()
    directory = tmp_path / ATTEMPT_ID
    directory.mkdir(mode=0o700)
    raw = directory / f"{'b' * 64}.json"
    raw.write_bytes(local.removesuffix(b"\n"))
    raw.chmod(0o600)

    manifest = pilot_cli._validate_forensic_receipts(artifact.pending, directory)
    assert manifest["evidence"] == "validated_canonical_snapshot"
    assert manifest["raw_snapshot_sha256"] == hashlib.sha256(local).hexdigest()


def test_forensic_validator_scans_before_artifact_parsing(
    artifact_factory: ArtifactFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = artifact_factory(status="exception", mutation="forensic")
    directory = tmp_path / ATTEMPT_ID
    directory.mkdir(mode=0o700)
    raw = directory / f"{'b' * 64}.json"
    raw.write_bytes(("ak" + "-" + "X" * 24).encode("ascii"))
    raw.chmod(0o600)
    monkeypatch.setattr(
        pilot_cli,
        "_validate_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("artifact parser must not run before forensic scan")
        ),
    )

    with pytest.raises(ValueError, match="credential material"):
        pilot_cli._validate_forensic_receipts(artifact.pending, directory)
