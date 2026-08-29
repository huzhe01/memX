from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from decimal import Decimal
from pathlib import Path

import pytest
import torch

import ratemem.pilot.runner as runner_module
from ratemem.adapters.checkpoint import CheckpointFileIdentity
from ratemem.pilot.private_io import canonical_json_bytes
from ratemem.pilot.probes import ALLOWED_PROBES, CudaPeak
from ratemem.pilot.runner import (
    PilotLimits,
    PilotRequest,
    PilotRunner,
    RealSanaPilotBackend,
    _validate_modal_ids,
    pilot_config_sha256,
    run_real_pilot,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_request() -> dict[str, object]:
    commit = "1" * 40
    rates = {
        "gpu_l40s_per_second": "0.000542",
        "cpu_core_per_second": "0.0000131",
        "memory_gib_per_second": "0.00000222",
        "volume_gib_month": "0.09",
    }
    return {
        "attempt_id": "019d0000-0000-7000-8000-000000000001",
        "workspace": "authorized-workspace",
        "source_sha256": _sha(commit.encode("ascii")),
        "git_commit": commit,
        "git_diff_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "slot_sha256": "4" * 64,
        "permit_sha256": "5" * 64,
        "submission_receipt_sha256": "6" * 64,
        "known_usage_before_usd": "1.00",
        "pending_worst_case_usd": "10.15",
        "phase_bound_usd": "10.15",
        "rates": rates,
        "rates_sha256": _sha(canonical_json_bytes(rates)),
    }


class FakeBackend:
    def __init__(self, *, losses: tuple[float, float] = (1.2, 0.8)) -> None:
        self.events: list[str] = []
        self.losses = list(losses)
        self.step_times = [99.0] * 10 + [float(value) for value in range(1, 21)]

    def compatibility(self) -> dict[str, object]:
        self.events.append("compatibility")
        return {"status": "pass"}

    def inference(self) -> dict[str, object]:
        self.events.append("inference")
        return {"status": "pass"}

    def backward(self) -> float:
        self.events.append("backward")
        return 1.0

    def training_step(self) -> float:
        self.events.append("training_step")
        return self.step_times.pop(0) if self.step_times else 0.01

    def evaluate_loss(self) -> float:
        self.events.append("evaluate_loss")
        return self.losses.pop(0)

    def save_checkpoint(self, path: Path) -> CheckpointFileIdentity:
        self.events.append("checkpoint")
        path.write_bytes(b"state")
        path.chmod(0o600)
        return CheckpointFileIdentity(sha256=_sha(b"state"), byte_count=5)


class ArtifactBackend(FakeBackend):
    def __init__(
        self,
        *,
        sana_config: object,
        subjects_config: object,
        failure: BaseException | None = None,
        losses: tuple[float, float] = (1.2, 0.8),
    ) -> None:
        super().__init__(losses=losses)
        self.sana_config = sana_config
        self.subjects_config = subjects_config
        self.failure = failure
        self._manifest = {
            "schema_version": "1.0.0",
            "scope": "engineering_pilot_only",
            "publication_eligible": False,
            "identity": "test-cache",
        }

    def compatibility(self) -> dict[str, object]:
        if self.failure is not None:
            raise self.failure
        return super().compatibility()

    @property
    def software(self) -> dict[str, str]:
        return {
            "python": "3.11.13",
            "torch": "2.13.0",
            "diffusers": "0.40.0",
            "peft": "0.20.0",
            "transformers": "5.16.1",
            "modal": "1.5.4",
        }

    @property
    def peak(self) -> CudaPeak:
        return CudaPeak(allocated_bytes=0, reserved_bytes=0)

    @property
    def dataset_manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    @property
    def dataset_manifest_sha256(self) -> str:
        return _sha(canonical_json_bytes(self._manifest))

    def diagnostics(self) -> dict[str, object]:
        return {"scope": "engineering_pilot_only", "fake_backend": True}


def limits(*, held_usd: str = "0.03", rate: str = "1.00") -> PilotLimits:
    return PilotLimits(
        warmup_steps=10,
        measured_steps=20,
        held_in_allocation_usd=Decimal(held_usd),
        resource_usd_per_second=Decimal(rate),
        timeout_seconds=7200,
        shutdown_reserve_seconds=120,
    )


def test_runner_has_one_inference_one_probe_backward_and_exact_timing_counts(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    result = PilotRunner(backend=backend, limits=limits()).run(tmp_path)
    assert backend.events.count("compatibility") == 1
    assert backend.events.count("inference") == 1
    assert backend.events.count("backward") == 1
    assert backend.events.count("training_step") == 30
    assert backend.events.count("evaluate_loss") == 2
    assert backend.events.count("checkpoint") == 1
    assert result["p50_step_seconds"] == 10.0
    assert result["p95_step_seconds"] == 19.0
    assert result["held_in_step_cap"] == 0
    assert result["initial_flow_loss"] == 1.2
    assert result["final_flow_loss"] == 0.8
    assert result["held_in_loss"] == {"status": "pass"}
    assert result["status"] == "succeeded"
    assert result["transformer_passes_per_step"] == 1


def test_runner_cap_adds_only_the_measured_p95_derived_held_in_steps(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    result = PilotRunner(
        backend=backend,
        limits=limits(held_usd="57.00", rate="1.00"),
    ).run(tmp_path)
    assert result["held_in_step_cap"] == 3
    assert backend.events.count("training_step") == 33


def test_nonfalling_loss_is_a_probe_failure_but_still_saves_checkpoint(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(losses=(0.8, 0.8))
    result = PilotRunner(backend=backend, limits=limits()).run(tmp_path)
    assert result["status"] == "probe_failed"
    assert result["held_in_loss"] == {"status": "fail"}
    assert result["checkpoint_sha256"] == _sha(b"state")
    assert (tmp_path / "trainable.safetensors").read_bytes() == b"state"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("warmup_steps", 9, ValueError),
        ("measured_steps", 21, ValueError),
        ("held_in_allocation_usd", Decimal("NaN"), ValueError),
        ("held_in_allocation_usd", Decimal("-1"), ValueError),
        ("resource_usd_per_second", Decimal("0"), ValueError),
        ("timeout_seconds", True, TypeError),
        ("shutdown_reserve_seconds", -1, ValueError),
        ("held_in_allocation_usd", "1.00", TypeError),
    ],
)
def test_limits_lock_counts_and_reject_nonfinite_or_nonexact_values(
    field: str, value: object, error: type[Exception]
) -> None:
    arguments: dict[str, object] = {
        "warmup_steps": 10,
        "measured_steps": 20,
        "held_in_allocation_usd": Decimal("1.00"),
        "resource_usd_per_second": Decimal("0.001"),
        "timeout_seconds": 7200,
        "shutdown_reserve_seconds": 120,
    }
    arguments[field] = value
    with pytest.raises(error):
        PilotLimits(**arguments)  # type: ignore[arg-type]


def test_request_binds_source_rates_costs_and_exact_closed_keys() -> None:
    assert len(valid_request()) == 14
    request = PilotRequest.from_mapping(valid_request())
    assert request.source_sha256 == _sha(request.git_commit.encode("ascii"))
    assert request.phase_bound_usd == Decimal("10.15")
    assert request.resource_usd_per_second == Decimal("0.00066544")

    extra = valid_request() | {"scientific_endpoint": "validation"}
    with pytest.raises(ValueError, match="missing or unexpected"):
        PilotRequest.from_mapping(extra)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_sha256", "a" * 64, "source"),
        ("rates_sha256", "a" * 64, "rates"),
        ("known_usage_before_usd", "NaN", "finite"),
        ("known_usage_before_usd", "1E+0", "fixed-point"),
        ("pending_worst_case_usd", "-1.00", "nonnegative"),
        ("phase_bound_usd", "10.16", "phase"),
        ("pending_worst_case_usd", "26.01", "internal"),
        ("attempt_id", "11111111-1111-4111-8111-111111111111", "UUID"),
        ("attempt_id", 7, "string"),
        ("workspace", "Uppercase", "workspace"),
        ("workspace", "under_score", "workspace"),
        ("workspace", "-leading", "workspace"),
        ("workspace", "trailing-", "workspace"),
        ("workspace", "a" * 65, "workspace"),
    ],
)
def test_request_rejects_unbound_nonfinite_or_over_cap_metadata(
    field: str, value: object, match: str
) -> None:
    payload = valid_request()
    payload[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        PilotRequest.from_mapping(payload)


def test_modal_metadata_requires_a_nonempty_task_identity(tmp_path: Path) -> None:
    request_mapping = valid_request()
    request = PilotRequest.from_mapping(request_mapping)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request_mapping)
    modal_ids = _modal_ids(request_mapping, receipt_path, receipt_directory)
    modal_ids["task_id"] = None
    with pytest.raises(ValueError, match="task_id.*nonempty"):
        _validate_modal_ids(modal_ids, request)


def test_real_backend_source_uses_actual_task_4_to_8_apis_and_no_scientific_endpoint() -> None:
    source = inspect.getsource(RealSanaPilotBackend)
    module_source = inspect.getsource(runner_module)
    required = {
        "hydrate_pinned_snapshots",
        "load_pinned_components",
        "hydrate_locked_examples",
        "build_precomputed_cache",
        "install_sana_dynamic_atoms",
        "OneTimestepFlowTrainer",
        "save_trainable_checkpoint",
        "FlowBatch.from_cache",
        "FlowDraw",
        "timed",
    }
    assert not {name for name in required if name not in source}
    assert "A studio photograph of a red ceramic teapot on a plain gray table." in module_source
    assert "20260824" in module_source and "20260825" in module_source
    assert "num_inference_steps=1" in source
    assert "height=1024" in source and "width=1024" in source
    assert "guidance_scale=4.5" in source
    for forbidden in (
        "identity_score",
        "fid_score",
        "kid_score",
        "composition_score",
        "scientific_endpoint",
    ):
        assert forbidden not in source.lower()


def test_runner_checkpoint_is_owner_only_and_result_identity_matches_bytes(
    tmp_path: Path,
) -> None:
    result = PilotRunner(backend=FakeBackend(), limits=limits()).run(tmp_path)
    checkpoint = tmp_path / "trainable.safetensors"
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    content = checkpoint.read_bytes()
    assert result["checkpoint_sha256"] == _sha(content)
    assert result["checkpoint_bytes"] == len(content)
    json.dumps(result, allow_nan=False)


def _write_execution_receipts(
    root: Path,
    request: dict[str, object],
    *,
    count: int = 1,
) -> tuple[Path, Path]:
    directory = root / "execution-receipts" / str(request["attempt_id"])
    directory.mkdir(mode=0o700, parents=True)
    current: Path | None = None
    for index in range(count):
        receipt_id = _sha(f"receipt-{index}".encode())
        current = directory / f"{receipt_id}.json"
        current.write_bytes(
            canonical_json_bytes(
                request
                | {
                    "function_call_id": "fc-test",
                    "input_id": "in-test",
                    "task_id": "ta-test",
                    "receipt_id": receipt_id,
                    "observed_at": f"2026-08-24T00:00:0{index}.000000+00:00",
                    "semantics": "lower_bound_may_miss_precommit_reschedule",
                }
            )
        )
        current.chmod(0o600)
    assert current is not None
    return current, directory


def _modal_ids(
    request: dict[str, object],
    receipt_path: Path,
    receipt_directory: Path,
    *,
    receipt_count: int = 1,
) -> dict[str, object]:
    receipt_snapshot = b"".join(
        path.read_bytes() + b"\n" for path in sorted(receipt_directory.iterdir())
    )
    modal_ids: dict[str, object] = {
        "profile": "ratemem-pilot",
        "workspace": request["workspace"],
        "environment": "main",
        "launch_attempt_id": request["attempt_id"],
        "launch_source_sha256": request["source_sha256"],
        "pilot_slot_sha256": request["slot_sha256"],
        "submission_receipt_sha256": request["submission_receipt_sha256"],
        "function_call_id": "fc-test",
        "input_id": "in-test",
        "task_id": "ta-test",
        "container_image_id": "im-test",
        "execution_receipt_path": str(receipt_path),
        "execution_receipt_directory": str(receipt_directory),
        "execution_receipt_count": receipt_count,
        "execution_receipts_sha256": _sha(receipt_snapshot),
        "execution_receipt_semantics": "lower_bound_may_miss_precommit_reschedule",
    }
    return modal_ids


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_type"),
    [
        (torch.OutOfMemoryError("secret allocation detail"), "oom", "CudaOutOfMemory"),
        (RuntimeError("secret backend detail"), "exception", "PilotException"),
    ],
)
def test_remote_boundary_publishes_honest_pending_artifact_for_early_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_status: str,
    expected_type: str,
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)

    def backend_factory(**arguments: object) -> ArtifactBackend:
        return ArtifactBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
            failure=failure,
        )

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")

    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )

    assert result["status"] == expected_status
    pending = Path(result["pending_path"])
    assert pending.parent == artifact_root / "attempts" / str(request["attempt_id"])
    attempt = json.loads(pending.read_text(encoding="utf-8"))
    assert attempt["status"] == expected_status
    assert attempt["error"] == {
        "type": expected_type,
        "message": (
            "CUDA allocation failed during the engineering pilot"
            if expected_status == "oom"
            else "engineering pilot execution failed"
        ),
    }
    assert attempt["checkpoint"] is None
    assert attempt["probes"]["p50_step_seconds"] is None
    assert attempt["probes"]["p95_step_seconds"] is None
    assert attempt["probes"]["initial_flow_loss"] is None
    assert attempt["probes"]["final_flow_loss"] is None
    assert attempt["probes"]["held_in_step_cap"] == 0
    assert attempt["probes"]["results"] == {name: {"status": "not_run"} for name in ALLOWED_PROBES}
    assert not (pending.parent / "trainable.safetensors").exists()
    assert "secret" not in pending.read_text(encoding="utf-8")
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600
    assert set(os.listdir(pending.parent)) == {
        "attempt.pending.json",
        "checksums.sha256",
        "config.json",
        "dataset-manifest.json",
        "execution-receipts.jsonl",
        "metrics.jsonl",
        "rates.json",
    }


def test_probe_failure_is_pending_with_checkpoint_and_exact_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request, count=2)

    def backend_factory(**arguments: object) -> ArtifactBackend:
        return ArtifactBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
            losses=(0.8, 0.8),
        )

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")

    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(
            request,
            receipt_path,
            receipt_directory,
            receipt_count=2,
        ),
    )

    assert result["status"] == "probe_failed"
    pending = Path(result["pending_path"])
    assert pending.parent == artifact_root / "attempts" / str(request["attempt_id"])
    attempt = json.loads(pending.read_text(encoding="utf-8"))
    checkpoint = pending.parent / "trainable.safetensors"
    checkpoint_bytes = checkpoint.read_bytes()
    assert attempt["status"] == "probe_failed"
    assert attempt["probes"]["results"]["held_in_loss"] == {"status": "fail"}
    assert attempt["probes"]["initial_flow_loss"] == 0.8
    assert attempt["probes"]["final_flow_loss"] == 0.8
    assert attempt["checkpoint"] == {
        "path": "trainable.safetensors",
        "sha256": _sha(checkpoint_bytes),
        "bytes": len(checkpoint_bytes),
    }
    assert result["checkpoint_sha256"] == _sha(checkpoint_bytes)
    assert result["checkpoint_bytes"] == len(checkpoint_bytes)
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    dataset_manifest = (pending.parent / "dataset-manifest.json").read_bytes()
    assert attempt["dataset"]["manifest_sha256"] == _sha(dataset_manifest)
    assert _sha((pending.parent / "config.json").read_bytes()) == request["config_sha256"]
    assert _sha((pending.parent / "rates.json").read_bytes()) == request["rates_sha256"]
    copied_receipts = (pending.parent / "execution-receipts.jsonl").read_bytes()
    assert len(copied_receipts.splitlines()) == 2
    assert not (artifact_root / "pilot-staging").exists()


def test_receipt_semantic_tamper_is_published_as_exception_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["observed_at"] = "2026-08-24T00:00:00Z"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    receipt_path.chmod(0o600)
    started = False

    def backend_factory(**_arguments: object) -> ArtifactBackend:
        nonlocal started
        started = True
        raise AssertionError("backend must not start")

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    assert result["status"] == "exception"
    assert started is False
    pending = Path(result["pending_path"])
    copied = (pending.parent / "execution-receipts.jsonl").read_bytes()
    assert copied != receipt_path.read_bytes() + b"\n"
    assert json.loads(copied)["evidence"] == "external_forensic_directory"
    metrics = [
        json.loads(line) for line in (pending.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[1]["execution_receipt_semantic_invalid"] is True
    assert metrics[1]["execution_receipt_evidence"] == "external_forensic_directory"


def test_semantically_invalid_receipt_never_copies_injected_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)
    fake_token = "ak-FAKE_INJECTED_TOKEN_ID"
    fake_secret = "as-FAKE_INJECTED_TOKEN_SECRET"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["injected_token_id"] = fake_token
    receipt["injected_token_secret"] = fake_secret
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    receipt_path.chmod(0o600)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )

    pending = Path(result["pending_path"])
    attempt_bytes = b"".join(
        path.read_bytes() for path in sorted(pending.parent.iterdir()) if path.is_file()
    )
    assert fake_token.encode() not in attempt_bytes
    assert fake_secret.encode() not in attempt_bytes
    assert fake_token.encode() in receipt_path.read_bytes()
    marker = json.loads((pending.parent / "execution-receipts.jsonl").read_text(encoding="utf-8"))
    assert marker == {
        "attempt_id": request["attempt_id"],
        "evidence": "external_forensic_directory",
        "forensic_path": f"execution-receipts/{request['attempt_id']}",
        "raw_snapshot_bytes": len(receipt_path.read_bytes()) + 1,
        "raw_snapshot_sha256": _sha(receipt_path.read_bytes() + b"\n"),
        "scope": "engineering_pilot_only",
        "status": "semantic_invalid",
    }


def test_receipt_directory_outside_artifact_root_cannot_self_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(tmp_path / "wrong", request)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    with pytest.raises(ValueError, match="exact attempt directory"):
        run_real_pilot(
            request=request,
            cache_root=cache_root,
            artifact_root=artifact_root,
            modal_ids=_modal_ids(request, receipt_path, receipt_directory),
        )
    assert not (artifact_root / "attempts").exists()
    assert receipt_path.exists()


def test_receipt_snapshot_hash_rejects_same_count_directory_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request, count=2)
    modal_ids = _modal_ids(
        request,
        receipt_path,
        receipt_directory,
        receipt_count=2,
    )

    replacement = tmp_path / "replacement-receipts"
    replacement.mkdir(mode=0o700)
    replacement_current = replacement / receipt_path.name
    replacement_current.write_bytes(receipt_path.read_bytes())
    replacement_current.chmod(0o600)
    forged_id = "f" * 64 if receipt_path.stem != "f" * 64 else "e" * 64
    forged = request | {
        "function_call_id": "WANDB_API_KEY=must-not-enter-artifact",
        "input_id": "in-forged",
        "task_id": "ta-forged",
        "receipt_id": forged_id,
        "observed_at": "2026-08-24T00:00:09.000000+00:00",
        "semantics": "lower_bound_may_miss_precommit_reschedule",
    }
    forged_path = replacement / f"{forged_id}.json"
    forged_path.write_bytes(canonical_json_bytes(forged))
    forged_path.chmod(0o600)

    original_directory = tmp_path / "original-receipts"
    real_assert_ancestors = runner_module._assert_real_path_ancestors

    def exchange_after_ancestor_check(path: Path) -> None:
        real_assert_ancestors(path)
        path.rename(original_directory)
        replacement.rename(path)

    monkeypatch.setattr(
        runner_module,
        "_assert_real_path_ancestors",
        exchange_after_ancestor_check,
    )
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        runner_module._read_execution_receipts(
            modal_ids,
            PilotRequest.from_mapping(request),
            artifact_root,
        )


def test_staging_failure_is_published_but_writer_failure_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)
    cache_root = tmp_path / "cache-file"
    cache_root.write_bytes(b"not-a-directory")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    assert result["status"] == "exception"

    second = valid_request()
    second["attempt_id"] = "019d0000-0000-7000-8000-000000000002"
    second["config_sha256"] = pilot_config_sha256()
    second_receipt, second_directory = _write_execution_receipts(artifact_root, second)
    blocked = artifact_root / "attempts" / str(second["attempt_id"])
    blocked.mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        run_real_pilot(
            request=second,
            cache_root=cache_root,
            artifact_root=artifact_root,
            modal_ids=_modal_ids(second, second_receipt, second_directory),
        )
    assert second_receipt.exists()


def test_over_bound_estimate_is_not_capped_and_forces_exception_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)

    def backend_factory(**arguments: object) -> ArtifactBackend:
        return ArtifactBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
        )

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(runner_module, "_estimated_cost", lambda **_arguments: Decimal("10.150001"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    attempt = json.loads(Path(result["pending_path"]).read_text())
    assert result["status"] == "exception"
    assert attempt["status"] == "exception"
    assert attempt["cost"]["estimated_cost_usd"] == "10.150001"
    assert attempt["checkpoint"] is not None


def test_over_bound_estimate_preserves_cuda_oom_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)

    def backend_factory(**arguments: object) -> ArtifactBackend:
        return ArtifactBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
            failure=torch.OutOfMemoryError("do not serialize this detail"),
        )

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(runner_module, "_estimated_cost", lambda **_arguments: Decimal("10.150001"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    attempt = json.loads(Path(result["pending_path"]).read_text())
    assert result["status"] == "oom"
    assert attempt["status"] == "oom"
    assert attempt["cost"]["estimated_cost_usd"] == "10.150001"
    metrics = [
        json.loads(line)
        for line in (Path(result["pending_path"]).parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[1]["phase_bound_exceeded"] is True


@pytest.mark.parametrize(
    ("fallback", "diagnostic_flag"),
    [
        ("diagnostics", "diagnostics_unavailable"),
        ("software", "software_inventory_unavailable"),
    ],
)
def test_evidence_fallback_failure_preserves_cuda_oom_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
    diagnostic_flag: str,
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)

    class OomBackend(ArtifactBackend):
        def diagnostics(self) -> dict[str, object]:
            if fallback == "diagnostics":
                raise RuntimeError("diagnostics unavailable")
            return super().diagnostics()

    def backend_factory(**arguments: object) -> OomBackend:
        return OomBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
            failure=torch.OutOfMemoryError("do not serialize this detail"),
        )

    def unavailable_software() -> dict[str, str]:
        raise RuntimeError("software inventory unavailable")

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    if fallback == "software":
        monkeypatch.setattr(runner_module, "_software_inventory", unavailable_software)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    attempt = json.loads(Path(result["pending_path"]).read_text())
    metrics = [
        json.loads(line)
        for line in (Path(result["pending_path"]).parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert result["status"] == "oom"
    assert attempt["status"] == "oom"
    assert attempt["error"]["type"] == "CudaOutOfMemory"
    assert metrics[1][diagnostic_flag] is True


def test_dataset_and_peak_fallbacks_are_explicit_and_preserve_base_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)

    class MissingEvidenceBackend(ArtifactBackend):
        @property
        def dataset_manifest(self) -> dict[str, object]:
            raise RuntimeError("dataset evidence unavailable")

        @property
        def peak(self) -> CudaPeak:
            return object()  # type: ignore[return-value]

    def backend_factory(**arguments: object) -> MissingEvidenceBackend:
        return MissingEvidenceBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
        )

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    assert result["status"] == "exception"
    pending = Path(result["pending_path"])
    metrics = [
        json.loads(line) for line in (pending.parent / "metrics.jsonl").read_text().splitlines()
    ]
    diagnostics = metrics[1]
    assert diagnostics["dataset_evidence_unavailable"] is True
    assert diagnostics["peak_unavailable"] is True
    assert diagnostics["execution_receipt_semantic_invalid"] is False
    assert diagnostics["execution_receipt_evidence"] == "validated_canonical_snapshot"
    assert diagnostics["fake_backend"] is True


def test_missing_dataset_evidence_alone_downgrades_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = valid_request()
    request["config_sha256"] = pilot_config_sha256()
    cache_root = tmp_path / "cache"
    artifact_root = tmp_path / "artifacts"
    cache_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    receipt_path, receipt_directory = _write_execution_receipts(artifact_root, request)

    class MissingDatasetBackend(ArtifactBackend):
        @property
        def dataset_manifest(self) -> dict[str, object]:
            raise RuntimeError("dataset evidence unavailable")

    def backend_factory(**arguments: object) -> MissingDatasetBackend:
        return MissingDatasetBackend(
            sana_config=arguments["sana_config"],
            subjects_config=arguments["subjects_config"],
        )

    monkeypatch.setattr(runner_module, "RealSanaPilotBackend", backend_factory)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA L40S")
    result = run_real_pilot(
        request=request,
        cache_root=cache_root,
        artifact_root=artifact_root,
        modal_ids=_modal_ids(request, receipt_path, receipt_directory),
    )
    assert result["status"] == "exception"
    pending = Path(result["pending_path"])
    metrics = [
        json.loads(line) for line in (pending.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[1]["dataset_evidence_unavailable"] is True
    assert metrics[1]["peak_unavailable"] is False
