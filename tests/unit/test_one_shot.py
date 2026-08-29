from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest

import ratemem.pilot.one_shot as one_shot
import ratemem.pilot.private_io as private_io
from ratemem.pilot.one_shot import (
    GLOBAL_LOCK_NAME,
    PilotIdentity,
    claim_global_pilot_slot,
    consume_launch_request,
    create_launch_permit,
    new_attempt_id,
    validate_consumed_launch_evidence,
)
from ratemem.pilot.private_io import (
    canonical_json_bytes,
    file_sha256,
    read_private_bytes,
    read_private_json,
    write_exclusive_private_bytes,
)

ATTEMPT_A = "0198f1bc-0f00-7001-8000-000000000001"
ATTEMPT_B = "0198f1bc-0f00-7001-8000-000000000002"
GIT_COMMIT = "a" * 40
SOURCE = "e33cdf9c7f7120b98e8c78408953e07f2ecd183006b5606df349b4c212acf43e"
DIFF = "2" * 64
CONFIG = "4" * 64
WORKSPACE = "authorized-workspace"
RATES = {
    "gpu_l40s_per_second": "0.000542",
    "cpu_core_per_second": "0.0000131",
    "memory_gib_per_second": "0.00000222",
    "volume_gib_month": "0.09",
}
RATES_SHA = "3c1872e4711ae17c02824da16cc9b7bf503e62b664735485a4cd796a67805640"


def _paths(root: Path) -> tuple[Path, Path, Path]:
    state, permits = root / "state", root / "permits"
    state.mkdir(mode=0o700)
    permits.mkdir(mode=0o700)
    return (
        state / "modal-pilot-slot.json",
        permits / "launch-permit.json",
        state / "modal-pilot-submitted.json",
    )


def _prepare(root: Path) -> tuple[Path, Path, Path]:
    slot, permit, receipt = _paths(root)
    identity = PilotIdentity(ATTEMPT_A, WORKSPACE, SOURCE, GIT_COMMIT)
    claim_global_pilot_slot(slot, identity=identity)
    create_launch_permit(
        permit,
        slot=slot,
        receipt=receipt,
        identity=identity,
        known_usage_before_usd="1.25",
        pending_worst_case_usd="0.60",
        phase_bound_usd="0.50",
        git_diff_sha256=DIFF,
        config_sha256=CONFIG,
        rates=RATES,
        rates_sha256=RATES_SHA,
    )
    return slot, permit, receipt


def _consume_worker(slot: str, permit: str, receipt: str, queue: Any) -> None:
    try:
        consume_launch_request(
            Path(permit),
            slot=Path(slot),
            receipt=Path(receipt),
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )
    except FileExistsError:
        queue.put("blocked")
    else:
        queue.put("submitted")


def test_new_attempt_id_is_canonical_unique_uuid7() -> None:
    values = {new_attempt_id() for _ in range(100)}
    assert len(values) == 100
    for value in values:
        import uuid

        parsed = uuid.UUID(value)
        assert str(parsed) == value and parsed.version == 7 and parsed.variant == uuid.RFC_4122


@pytest.mark.parametrize(
    "attempt", ["11111111-1111-4111-8111-111111111111", ATTEMPT_A.upper(), "bad"]
)
def test_identity_rejects_noncanonical_non_uuid7(attempt: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        PilotIdentity(attempt, WORKSPACE, SOURCE, GIT_COMMIT)


def test_identity_rejects_source_not_bound_to_commit() -> None:
    with pytest.raises(ValueError, match="bind"):
        PilotIdentity(ATTEMPT_A, WORKSPACE, "0" * 64, GIT_COMMIT)


def test_round_trip_is_canonical_create_only_and_bound(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    request = consume_launch_request(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )
    evidence = validate_consumed_launch_evidence(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )
    assert (
        evidence["submission_receipt_sha256"]
        == request["submission_receipt_sha256"]
        == file_sha256(receipt)
    )
    assert evidence["slot_sha256"] == file_sha256(slot)
    assert evidence["permit_sha256"] == file_sha256(permit)
    assert request["git_diff_sha256"] == DIFF
    assert request["config_sha256"] == CONFIG
    assert request["pending_worst_case_usd"] == "0.60"
    assert set(request) == {
        "attempt_id",
        "workspace",
        "source_sha256",
        "git_commit",
        "git_diff_sha256",
        "config_sha256",
        "slot_sha256",
        "permit_sha256",
        "submission_receipt_sha256",
        "known_usage_before_usd",
        "pending_worst_case_usd",
        "phase_bound_usd",
        "rates",
        "rates_sha256",
    }
    from ratemem.pilot.modal_app import _validate_request

    assert _validate_request(request) == request
    for path in (slot, permit, receipt, slot.with_name(GLOBAL_LOCK_NAME)):
        assert path.stat().st_mode & 0o777 == 0o600
        if path.suffix == ".json":
            assert path.read_bytes() == canonical_json_bytes(read_private_json(path))
    with pytest.raises(FileExistsError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )


def test_tamper_and_identity_mismatch_never_create_receipt(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    payload = read_private_json(permit)
    payload["workspace"] = "wrong"
    permit.unlink()
    permit.write_bytes(canonical_json_bytes(payload))
    permit.chmod(0o600)
    with pytest.raises(ValueError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )
    assert not receipt.exists()


@pytest.mark.parametrize("target", ["slot", "permit"])
def test_hardlinks_and_symlinks_are_rejected_without_receipt(tmp_path: Path, target: str) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    path = slot if target == "slot" else permit
    other = tmp_path / f"{target}-link"
    os.link(path, other)
    with pytest.raises(PermissionError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )
    assert not receipt.exists()


def test_noncanonical_or_unknown_permit_is_rejected(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    payload = read_private_json(permit)
    payload["unknown"] = 1
    permit.unlink()
    permit.write_text(json.dumps(payload, indent=2))
    permit.chmod(0o600)
    with pytest.raises(ValueError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )
    assert not receipt.exists()


def test_symlink_permit_is_rejected_without_receipt(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    real = permit.with_name("saved")
    permit.rename(real)
    permit.symlink_to(real)
    with pytest.raises(PermissionError):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )
    assert not receipt.exists()


def test_repeated_validation_does_not_leak_file_descriptors(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(20):
        with pytest.raises(ValueError):
            validate_consumed_launch_evidence(
                permit,
                slot=slot,
                receipt=receipt,
                expected_workspace="wrong",
                current_source_sha256=SOURCE,
            )
    after = len(list(Path("/proc/self/fd").iterdir()))
    assert after <= before + 1


def test_threads_and_spawn_processes_have_exactly_one_winner(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)

    def run() -> str:
        try:
            consume_launch_request(
                permit,
                slot=slot,
                receipt=receipt,
                expected_workspace=WORKSPACE,
                current_source_sha256=SOURCE,
            )
            return "submitted"
        except FileExistsError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sorted(pool.map(lambda _: run(), range(4))) == ["blocked"] * 3 + ["submitted"]

    root2 = tmp_path / "second"
    root2.mkdir()
    slot2, permit2, receipt2 = _prepare(root2)
    ctx = get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_consume_worker, args=(str(slot2), str(permit2), str(receipt2), queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0
    assert sorted(queue.get(timeout=5) for _ in processes) == ["blocked", "submitted"]


def test_abstract_kernel_guard_blocks_directory_aba_second_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    clone = tmp_path / "state-clone"
    clone.mkdir(mode=0o700)
    write_exclusive_private_bytes(clone / slot.name, read_private_bytes(slot))

    first_entered_validation = threading.Event()
    release_first = threading.Event()
    real_validate = one_shot._validate

    def paused_validate(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name == "consumer-a":
            first_entered_validation.set()
            assert release_first.wait(5)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(one_shot, "_validate", paused_validate)
    outcomes: list[tuple[str, str]] = []

    def consume(label: str) -> None:
        try:
            consume_launch_request(
                permit,
                slot=slot,
                receipt=receipt,
                expected_workspace=WORKSPACE,
                current_source_sha256=SOURCE,
            )
        except FileExistsError:
            outcomes.append((label, "blocked"))
        else:
            outcomes.append((label, "returned"))

    first = threading.Thread(target=consume, args=("a",), name="consumer-a")
    first.start()
    assert first_entered_validation.wait(5)

    original = tmp_path / "state-original"
    os.rename(slot.parent, original)
    os.rename(clone, slot.parent)
    second = threading.Thread(target=consume, args=("b",), name="consumer-b")
    second.start()
    second.join(5)
    assert not second.is_alive()

    moved_clone = tmp_path / "state-clone-after"
    os.rename(slot.parent, moved_clone)
    os.rename(original, slot.parent)
    release_first.set()
    first.join(5)
    assert not first.is_alive()

    assert sorted(outcomes) == [("a", "returned"), ("b", "blocked")]
    assert (slot.parent / receipt.name).exists()
    assert not (moved_clone / receipt.name).exists()


def test_consumed_evidence_does_not_expire_after_remote_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Clock(datetime):
        current = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

        @classmethod
        def now(cls, tz: object = None) -> datetime:
            if tz is None:
                return cls.current.replace(tzinfo=None)
            return cls.current

    monkeypatch.setattr(one_shot, "datetime", Clock)
    slot, permit, receipt = _prepare(tmp_path)
    consume_launch_request(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )
    Clock.current += timedelta(hours=3)

    evidence = validate_consumed_launch_evidence(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )

    assert evidence["submission_receipt_sha256"] == file_sha256(receipt)


def test_records_have_distinct_exact_schema_kind_and_timestamps(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    claimed = read_private_json(slot)
    authorized = read_private_json(permit)

    assert type(claimed["schema_version"]) is int and claimed["schema_version"] == 1
    assert claimed["kind"] == "ratemem-pilot-slot"
    assert "claimed_at_utc" in claimed and "authorized_at_utc" not in claimed
    assert type(authorized["schema_version"]) is int and authorized["schema_version"] == 1
    assert authorized["kind"] == "ratemem-pilot-launch-permit"
    assert "authorized_at_utc" in authorized and "claimed_at_utc" not in authorized

    consume_launch_request(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )
    submitted = read_private_json(receipt)
    assert type(submitted["schema_version"]) is int and submitted["schema_version"] == 1
    assert submitted["kind"] == "ratemem-pilot-submission-receipt"
    assert "submitted_at_utc" in submitted
    assert "claimed_at_utc" not in submitted and "authorized_at_utc" not in submitted


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("kind", "forged-kind"),
        ("schema_version", False),
        ("authorized_at_utc", "not-a-timestamp"),
    ],
)
def test_permit_self_description_tamper_never_creates_receipt(
    tmp_path: Path, field: str, replacement: object
) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    payload = read_private_json(permit)
    payload[field] = replacement
    permit.unlink()
    permit.write_bytes(canonical_json_bytes(payload))
    permit.chmod(0o600)

    with pytest.raises((TypeError, ValueError)):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )
    assert not receipt.exists()


def test_receipt_rejects_boolean_schema_version(tmp_path: Path) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    consume_launch_request(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )
    payload = read_private_json(receipt)
    payload["schema_version"] = True
    receipt.unlink()
    receipt.write_bytes(canonical_json_bytes(payload))
    receipt.chmod(0o600)

    with pytest.raises((TypeError, ValueError)):
        validate_consumed_launch_evidence(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )


def test_unlock_failure_after_durable_receipt_is_best_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    real_flock = private_io.fcntl.flock

    def fail_unlock(descriptor: int, operation: int) -> None:
        if operation == private_io.fcntl.LOCK_UN:
            raise OSError("injected unlock failure")
        real_flock(descriptor, operation)

    monkeypatch.setattr(private_io.fcntl, "flock", fail_unlock)
    request = consume_launch_request(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )

    assert request["submission_receipt_sha256"] == file_sha256(receipt)


def test_receipt_fchmod_failure_removes_created_inode_before_returning_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    real_fchmod = one_shot.os.fchmod

    def fail_receipt_fchmod(descriptor: int, mode: int) -> None:
        target = os.readlink(f"/proc/self/fd/{descriptor}")
        if target.endswith(receipt.name):
            raise OSError("injected receipt fchmod failure")
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(one_shot.os, "fchmod", fail_receipt_fchmod)
    with pytest.raises(OSError, match="injected receipt fchmod failure"):
        consume_launch_request(
            permit,
            slot=slot,
            receipt=receipt,
            expected_workspace=WORKSPACE,
            current_source_sha256=SOURCE,
        )

    assert not receipt.exists() and not receipt.is_symlink()


def test_validate_unsubmitted_permit_is_repeatable_and_never_creates_receipt(
    tmp_path: Path,
) -> None:
    slot, permit, receipt = _prepare(tmp_path)

    first = one_shot.validate_unsubmitted_launch_permit(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )
    second = one_shot.validate_unsubmitted_launch_permit(
        permit,
        slot=slot,
        receipt=receipt,
        expected_workspace=WORKSPACE,
        current_source_sha256=SOURCE,
    )

    assert first == second == read_private_json(permit)
    assert not receipt.exists() and not receipt.is_symlink()


def test_concurrent_unsubmitted_validation_has_one_kernel_guard_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    slot, permit, receipt = _prepare(tmp_path)
    first_entered_validation = threading.Event()
    release_first = threading.Event()
    real_validate = one_shot._validate

    def paused_validate(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name == "validator-a":
            first_entered_validation.set()
            assert release_first.wait(5)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(one_shot, "_validate", paused_validate)
    outcomes: list[str] = []

    def validate() -> None:
        try:
            one_shot.validate_unsubmitted_launch_permit(
                permit,
                slot=slot,
                receipt=receipt,
                expected_workspace=WORKSPACE,
                current_source_sha256=SOURCE,
            )
        except FileExistsError:
            outcomes.append("blocked")
        else:
            outcomes.append("validated")

    first = threading.Thread(target=validate, name="validator-a")
    first.start()
    assert first_entered_validation.wait(5)
    second = threading.Thread(target=validate, name="validator-b")
    second.start()
    second.join(5)
    assert not second.is_alive()
    release_first.set()
    first.join(5)
    assert not first.is_alive()

    assert sorted(outcomes) == ["blocked", "validated"]
    assert not receipt.exists() and not receipt.is_symlink()


def test_invalid_budget_and_permissions_fail_before_permit(tmp_path: Path) -> None:
    slot, permit, receipt = _paths(tmp_path)
    identity = PilotIdentity(ATTEMPT_B, WORKSPACE, SOURCE, GIT_COMMIT)
    claim_global_pilot_slot(slot, identity=identity)
    with pytest.raises(ValueError):
        create_launch_permit(
            permit,
            slot=slot,
            receipt=receipt,
            identity=identity,
            known_usage_before_usd="20.00",
            pending_worst_case_usd="8.00",
            phase_bound_usd="8.00",
            git_diff_sha256=DIFF,
            config_sha256=CONFIG,
            rates=RATES,
            rates_sha256=RATES_SHA,
        )
    assert not permit.exists() and not receipt.exists()
    permit.parent.chmod(0o755)
    with pytest.raises(PermissionError):
        create_launch_permit(
            permit,
            slot=slot,
            receipt=receipt,
            identity=identity,
            known_usage_before_usd="1.00",
            pending_worst_case_usd="1.00",
            phase_bound_usd="1.00",
            git_diff_sha256=DIFF,
            config_sha256=CONFIG,
            rates=RATES,
            rates_sha256=RATES_SHA,
        )


@pytest.mark.parametrize(
    ("known", "pending", "phase"),
    [
        ("26.01", "1.00", "1.00"),
        ("1.00", "1.00", "1.01"),
        ("1.00", "1.00", "0.00"),
        ("1", "1.00", "1.00"),
        ("NaN", "1.00", "1.00"),
    ],
)
def test_exact_budget_relations_and_decimal_grammar_fail_before_permit(
    tmp_path: Path, known: str, pending: str, phase: str
) -> None:
    slot, permit, receipt = _paths(tmp_path)
    identity = PilotIdentity(ATTEMPT_B, WORKSPACE, SOURCE, GIT_COMMIT)
    claim_global_pilot_slot(slot, identity=identity)
    with pytest.raises((TypeError, ValueError)):
        create_launch_permit(
            permit,
            slot=slot,
            receipt=receipt,
            identity=identity,
            known_usage_before_usd=known,
            pending_worst_case_usd=pending,
            phase_bound_usd=phase,
            git_diff_sha256=DIFF,
            config_sha256=CONFIG,
            rates=RATES,
            rates_sha256=RATES_SHA,
        )
    assert not permit.exists() and not receipt.exists()
