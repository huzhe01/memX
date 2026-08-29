from __future__ import annotations

import gc
import hashlib
import json
import os
import stat
import weakref
from copy import deepcopy
from pathlib import Path

import pytest

import ratemem.pilot.artifacts as artifacts
from ratemem.adapters.checkpoint import CheckpointFileIdentity
from ratemem.pilot.artifacts import ArtifactWriter, validate_attempt

PROBE_NAMES = [
    "checkpoint_compatibility",
    "dynamic_numerics",
    "gradient_flow",
    "frozen_backbone",
    "peak_memory",
    "one_step_inference",
    "one_timestep_backward",
    "step_timing",
    "held_in_loss",
]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_attempt(checkpoint: bytes = b"checkpoint") -> dict[str, object]:
    commit = "1" * 40
    return {
        "schema_version": "1.0.0",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "attempt_id": "019d0000-0000-7000-8000-000000000001",
        "phase": "first_pilot",
        "status": "succeeded",
        "started_at": "2026-08-24T00:00:00Z",
        "ended_at": "2026-08-24T01:00:00Z",
        "source": {
            "git_commit": commit,
            "git_diff_sha256": "2" * 64,
            "config_sha256": "3" * 64,
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
            "manifest_sha256": "4" * 64,
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
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        "modal": {
            "profile": "ratemem-pilot",
            "workspace": "authorized-workspace",
            "environment": "main",
            "launch_attempt_id": "019d0000-0000-7000-8000-000000000001",
            "launch_source_sha256": _sha(commit.encode()),
            "pilot_slot_sha256": "8" * 64,
            "submission_receipt_sha256": "9" * 64,
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
            "known_usage_before_usd": "0.00",
            "pending_worst_case_usd": "9.50",
            "phase_bound_usd": "9.50",
            "estimated_cost_usd": "1.25",
            "reconciliation_status": "pending",
            "reconciled_cost_usd": None,
            "rates_sha256": "5" * 64,
        },
        "probes": {
            "allowed_probe_names": list(PROBE_NAMES),
            "results": {name: {"status": "pass"} for name in PROBE_NAMES},
            "warmup_steps": 10,
            "measured_steps": 20,
            "p50_step_seconds": 1.0,
            "p95_step_seconds": 1.2,
            "held_in_step_cap": 40,
            "initial_flow_loss": 1.1,
            "final_flow_loss": 0.9,
            "transformer_passes_per_step": 1,
        },
        "checkpoint": {
            "path": "trainable.safetensors",
            "sha256": _sha(checkpoint),
            "bytes": len(checkpoint),
        },
        "files": {"checksums_sha256": "7" * 64},
        "error": None,
    }


def _identity(content: bytes = b"checkpoint") -> CheckpointFileIdentity:
    return CheckpointFileIdentity(sha256=_sha(content), byte_count=len(content))


class _FloatSubclass(float):
    pass


def _writer(tmp_path: Path) -> tuple[ArtifactWriter, Path]:
    checkpoint = tmp_path / "source.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.chmod(0o600)
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    for name in (
        "config.json",
        "rates.json",
        "dataset-manifest.json",
        "execution-receipts.jsonl",
        "metrics.jsonl",
    ):
        writer.write_bytes(name, f"{name}\n".encode())
    writer.write_checkpoint(checkpoint)
    return writer, checkpoint


def test_schema_is_draft_202012_and_rejects_unknowns_at_every_nested_level() -> None:
    validate_attempt(valid_attempt())
    schema = json.loads(
        Path("schemas/ratemem-pilot-attempt-v1.schema.json").read_text()
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    mutations = [
        (("publication_eligible",), True),
        (("scope",), "scientific_evaluation"),
        (("headline_identity_score",), 0.9),
        (("source", "extra"), "x"),
        (("probes", "results", "scientific_comparison"), {"status": "pass"}),
        (("probes", "results", "checkpoint_compatibility", "extra"), 1),
    ]
    for path, value in mutations:
        payload = valid_attempt()
        target = payload
        for key in path[:-1]:
            target = target[key]  # type: ignore[index,assignment]
        target[path[-1]] = value  # type: ignore[index]
        with pytest.raises(ValueError):
            validate_attempt(payload)


def test_success_requires_all_canonical_probes_and_failure_requires_failed_probe_error() -> None:
    missing = valid_attempt()
    missing["probes"]["allowed_probe_names"].pop()  # type: ignore[index,union-attr]
    missing["probes"]["results"].pop("held_in_loss")  # type: ignore[index,union-attr]
    with pytest.raises(ValueError, match="successful|canonical|probe"):
        validate_attempt(missing)

    false_success = valid_attempt()
    false_success["probes"]["results"]["held_in_loss"]["status"] = "fail"  # type: ignore[index]
    with pytest.raises(ValueError, match="successful|pass|probe"):
        validate_attempt(false_success)

    failed = valid_attempt()
    failed["status"] = "probe_failed"
    failed["error"] = {"type": "ProbeFailure", "message": "held-in loss failed"}
    with pytest.raises(ValueError, match="failed probe|fail"):
        validate_attempt(failed)
    failed["probes"]["results"]["held_in_loss"]["status"] = "fail"  # type: ignore[index]
    validate_attempt(failed)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attempt_id",), "019d0000-0000-6000-8000-000000000001"),
        (("started_at",), "2026-08-24 00:00:00"),
        (("ended_at",), "2026-08-23T00:00:00Z"),
        (("probes", "p50_step_seconds"), float("nan")),
        (("probes", "p50_step_seconds"), _FloatSubclass(1.0)),
        (("runtime", "peak_allocated_bytes"), True),
        (("cost", "estimated_cost_usd"), "01.25"),
        (("cost", "known_usage_before_usd"), "26.00"),
        (("cost", "phase_bound_usd"), "9.51"),
        (("modal", "launch_attempt_id"), "019d0000-0000-7000-8000-000000000002"),
        (("modal", "launch_source_sha256"), "a" * 64),
    ],
)
def test_semantic_types_dates_money_and_launch_identity_fail_closed(
    path: tuple[str, ...], value: object
) -> None:
    payload = valid_attempt()
    target = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises((TypeError, ValueError)):
        validate_attempt(payload)


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "nested/file", ".", "attempt.json"]
)
def test_writer_rejects_traversal_reserved_and_non_leaf_paths(
    tmp_path: Path, path: str
) -> None:
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    with pytest.raises((TypeError, ValueError, FileExistsError)):
        writer.write_bytes(path, b"x")


def test_owner_only_create_symlink_hardlink_and_existing_root_fail_closed(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        ArtifactWriter(existing, valid_attempt(), checkpoint_identity=_identity())
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises((FileExistsError, PermissionError, ValueError)):
        ArtifactWriter(link, valid_attempt(), checkpoint_identity=_identity())

    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    assert stat.S_IMODE(writer.root.stat().st_mode) == 0o700
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    os.link(outside, writer.root / "config.json")
    with pytest.raises((FileExistsError, PermissionError, ValueError)):
        writer.write_bytes("config.json", b"safe")


def test_pending_is_immutable_and_finalize_rechecks_every_file_and_receipt(
    tmp_path: Path,
) -> None:
    writer, _source = _writer(tmp_path)
    pending = writer.write_pending()
    pending_before = pending.read_bytes()
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600
    assert stat.S_IMODE((writer.root / "checksums.sha256").stat().st_mode) == 0o600
    with pytest.raises(RuntimeError, match="sealed"):
        writer.write_bytes("metrics.jsonl", b"changed")
    with pytest.raises(RuntimeError, match="sealed"):
        writer.write_pending()

    checkpoint = writer.root / "trainable.safetensors"
    checkpoint.chmod(0o600)
    checkpoint.write_bytes(b"tampered!!")
    with pytest.raises(ValueError, match="checkpoint|checksum"):
        writer.finalize(reconciled_cost_usd="1.31")
    assert pending.read_bytes() == pending_before
    assert not (writer.root / "attempt.json").exists()


def test_finalize_create_only_preserves_pending_and_reconciles_exact_decimal(
    tmp_path: Path,
) -> None:
    writer, _source = _writer(tmp_path)
    pending = writer.write_pending()
    before = pending.read_bytes()
    final_path = writer.finalize(reconciled_cost_usd="1.31")
    final = json.loads(final_path.read_text())
    assert final["cost"]["reconciliation_status"] == "reconciled"
    assert final["cost"]["reconciled_cost_usd"] == "1.31"
    assert pending.read_bytes() == before
    assert stat.S_IMODE(final_path.stat().st_mode) == 0o600
    with pytest.raises((FileExistsError, RuntimeError)):
        writer.finalize(reconciled_cost_usd="1.31")


def test_checkpoint_receipt_and_exact_file_set_are_required(tmp_path: Path) -> None:
    content = b"checkpoint"
    source = tmp_path / "source.safetensors"
    source.write_bytes(content)
    source.chmod(0o600)
    wrong = CheckpointFileIdentity(sha256="0" * 64, byte_count=len(content))
    with pytest.raises(ValueError, match="checkpoint"):
        ArtifactWriter(
            tmp_path / "attempt", valid_attempt(content), checkpoint_identity=wrong
        )

    incomplete = ArtifactWriter(
        tmp_path / "incomplete", valid_attempt(), checkpoint_identity=_identity()
    )
    incomplete.write_checkpoint(source)
    with pytest.raises(ValueError, match="file set"):
        incomplete.write_pending()


def test_attempt_input_and_schema_are_not_mutable_aliases(tmp_path: Path) -> None:
    attempt = valid_attempt()
    original = deepcopy(attempt)
    writer = ArtifactWriter(
        tmp_path / "attempt", attempt, checkpoint_identity=_identity()
    )
    attempt["scope"] = "changed"
    assert writer.attempt == original


def test_atomic_writer_handles_short_os_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    real_write = os.write
    shortened = False

    def short_once(descriptor: int, value: bytes) -> int:
        nonlocal shortened
        if not shortened and len(value) > 1:
            shortened = True
            return real_write(descriptor, value[: len(value) // 2])
        return real_write(descriptor, value)

    monkeypatch.setattr(artifacts.os, "write", short_once)
    writer.write_bytes("config.json", b"0123456789")
    assert (writer.root / "config.json").read_bytes() == b"0123456789"


def test_root_swap_to_symlink_is_rejected_without_writing_attacker_target(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    original = tmp_path / "original-attempt"
    writer.root.rename(original)
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    writer.root.symlink_to(attacker, target_is_directory=True)
    with pytest.raises((PermissionError, RuntimeError)):
        writer.write_bytes("config.json", b"secret")
    assert not (attacker / "config.json").exists()


def test_checkpoint_source_toctou_failure_removes_published_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"checkpoint")
    source.chmod(0o600)
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    real_atomic = writer._atomic

    def mutate_after_copy(path: Path, blocks: object) -> tuple[str, int]:
        result = real_atomic(path, blocks)  # type: ignore[arg-type]
        source.write_bytes(b"changed---")
        source.chmod(0o600)
        return result

    monkeypatch.setattr(writer, "_atomic", mutate_after_copy)
    with pytest.raises(RuntimeError, match="changed during artifact copy"):
        writer.write_checkpoint(source)
    assert not (writer.root / "trainable.safetensors").exists()


def test_constructor_open_failure_leaves_only_an_empty_failed_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "attempt"
    real_open = os.open

    def fail_root_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == root:
            raise OSError("injected root descriptor failure")
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(artifacts.os, "open", fail_root_open)
    with pytest.raises(OSError, match="injected root descriptor failure"):
        ArtifactWriter(root, valid_attempt(), checkpoint_identity=_identity())
    assert root.is_dir()
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("amount", ["9.51", "28.01", "NaN", "1.2"])
def test_finalize_reconciled_decimal_and_bounds_fail_closed(
    tmp_path: Path, amount: str
) -> None:
    writer, _source = _writer(tmp_path)
    writer.write_pending()
    with pytest.raises((TypeError, ValueError)):
        writer.finalize(reconciled_cost_usd=amount)
    assert not (writer.root / "attempt.json").exists()


def test_pending_publication_failure_rolls_back_transaction_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer, _source = _writer(tmp_path)
    real_atomic = writer._atomic

    def fail_pending(path: Path, blocks: object) -> tuple[str, int]:
        if path.name == "attempt.pending.json":
            raise OSError("injected pending publication failure")
        return real_atomic(path, blocks)  # type: ignore[arg-type]

    monkeypatch.setattr(writer, "_atomic", fail_pending)
    with pytest.raises(OSError, match="injected pending publication failure"):
        writer.write_pending()
    assert not (writer.root / "checksums.sha256").exists()
    assert not (writer.root / "attempt.pending.json").exists()
    assert sorted(path.name for path in writer.root.iterdir()) == sorted(
        [
            "config.json",
            "rates.json",
            "dataset-manifest.json",
            "execution-receipts.jsonl",
            "metrics.jsonl",
            "trainable.safetensors",
        ]
    )


def test_pending_and_finalize_detect_payload_mutation_across_marker_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer, _source = _writer(tmp_path)
    real_verify = writer._verify_payload_files
    calls = 0

    def mutate_after_first_snapshot() -> list[str]:
        nonlocal calls
        entries = real_verify()
        calls += 1
        if calls == 1:
            metrics = writer.root / "metrics.jsonl"
            metrics.write_bytes(b"mutated\n")
            metrics.chmod(0o600)
        return entries

    monkeypatch.setattr(writer, "_verify_payload_files", mutate_after_first_snapshot)
    with pytest.raises((RuntimeError, ValueError), match="changed|checksum|payload"):
        writer.write_pending()
    assert not (writer.root / "checksums.sha256").exists()
    assert not (writer.root / "attempt.pending.json").exists()

    finalize_case = tmp_path / "finalize-case"
    finalize_case.mkdir()
    writer2, _source2 = _writer(finalize_case)
    writer2.write_pending()
    real_sealed = writer2._verify_sealed
    sealed_calls = 0

    def mutate_after_sealed_snapshot() -> dict[str, object]:
        nonlocal sealed_calls
        payload = real_sealed()
        sealed_calls += 1
        if sealed_calls == 1:
            metrics = writer2.root / "metrics.jsonl"
            metrics.write_bytes(b"mutated\n")
            metrics.chmod(0o600)
        return payload

    monkeypatch.setattr(writer2, "_verify_sealed", mutate_after_sealed_snapshot)
    with pytest.raises((RuntimeError, ValueError), match="changed|checksum|payload"):
        writer2.finalize(reconciled_cost_usd="1.31")
    assert not (writer2.root / "attempt.json").exists()


def test_atomic_rejects_concurrent_published_inode_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    real_link = os.link

    def link_then_rewrite(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)  # type: ignore[arg-type]
        destination = args[1]
        descriptor = os.open(
            destination,  # type: ignore[arg-type]
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=writer._root_descriptor,
        )
        try:
            os.write(descriptor, b"attacker")
        finally:
            os.close(descriptor)

    monkeypatch.setattr(artifacts.os, "link", link_then_rewrite)
    with pytest.raises((RuntimeError, ValueError), match="changed|digest|published"):
        writer.write_bytes("config.json", b"original")
    assert not (writer.root / "config.json").exists()


def test_constructor_cleanup_never_deletes_replacement_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "attempt"
    moved = tmp_path / "moved-original"
    real_open = os.open

    def replace_then_fail_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        if path == root:
            root.rename(moved)
            root.mkdir(mode=0o700)
            raise RuntimeError("injected constructor replacement")
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(artifacts.os, "open", replace_then_fail_open)
    with pytest.raises(RuntimeError, match="injected constructor replacement"):
        ArtifactWriter(root, valid_attempt(), checkpoint_identity=_identity())
    assert root.is_dir()
    assert moved.is_dir()
    assert root.stat().st_ino != moved.stat().st_ino


def test_constructor_rejects_lstat_to_open_root_replacement_and_closes_fd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "attempt"
    moved = tmp_path / "moved-original"
    real_open = os.open
    opened_descriptors: list[int] = []

    def swap_before_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        if path == root:
            root.rename(moved)
            root.mkdir(mode=0o700)
            descriptor = real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]
            opened_descriptors.append(descriptor)
            return descriptor
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(artifacts.os, "open", swap_before_open)
    with pytest.raises(RuntimeError, match="changed|identity|inode"):
        ArtifactWriter(root, valid_attempt(), checkpoint_identity=_identity())
    assert root.is_dir()
    assert moved.is_dir()
    assert root.stat().st_ino != moved.stat().st_ino
    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


def test_constructor_failure_never_uses_path_rmdir_after_identity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "attempt"
    rmdir_called = False

    def fail_copy(value: object) -> object:
        raise RuntimeError("injected copy failure")

    def swap_during_rmdir(path: Path) -> None:
        nonlocal rmdir_called
        rmdir_called = True
        raise AssertionError(f"unsafe path rmdir attempted for {path}")

    monkeypatch.setattr(artifacts.copy, "deepcopy", fail_copy)
    monkeypatch.setattr(Path, "rmdir", swap_during_rmdir)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        ArtifactWriter(root, valid_attempt(), checkpoint_identity=_identity())
    assert not rmdir_called
    assert not root.exists()


def test_close_context_manager_and_gc_finalizer_own_root_fd(tmp_path: Path) -> None:
    writer = ArtifactWriter(
        tmp_path / "attempt", valid_attempt(), checkpoint_identity=_identity()
    )
    descriptor = writer._root_descriptor
    writer.close()
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = writer.attempt
    with pytest.raises(RuntimeError, match="closed"):
        writer.write_bytes("config.json", b"x")
    with pytest.raises(RuntimeError, match="closed"):
        writer.write_pending()
    with pytest.raises(RuntimeError, match="closed"):
        writer.finalize(reconciled_cost_usd="1.31")
    with pytest.raises(OSError):
        os.fstat(descriptor)

    with ArtifactWriter(
        tmp_path / "context", valid_attempt(), checkpoint_identity=_identity()
    ) as contextual:
        context_descriptor = contextual._root_descriptor
        os.fstat(context_descriptor)
    with pytest.raises(OSError):
        os.fstat(context_descriptor)

    collected = ArtifactWriter(
        tmp_path / "collected", valid_attempt(), checkpoint_identity=_identity()
    )
    collected_descriptor = collected._root_descriptor
    reference = weakref.ref(collected)
    del collected
    gc.collect()
    assert reference() is None
    with pytest.raises(OSError):
        os.fstat(collected_descriptor)
