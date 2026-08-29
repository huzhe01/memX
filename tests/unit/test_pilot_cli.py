from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest
import typer

import ratemem.pilot.cli as pilot_cli
from ratemem.pilot.cli import credential_findings, source_tree_identity, source_tree_sha256
from ratemem.pilot.costs import AttemptCost
from ratemem.pilot.private_io import ensure_private_directory, read_private_json
from ratemem.pilot.workspace import WorkspaceSnapshot

RECONCILE_ATTEMPT_ID = "019d0000-0000-7000-8000-000000000031"


def test_source_identity_requires_clean_tracked_staged_and_untracked_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dirty(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=b"?? untracked.py\0", stderr=b"")

    monkeypatch.setattr(pilot_cli.subprocess, "run", dirty)
    with pytest.raises(ValueError, match="tracked, staged, or untracked"):
        source_tree_sha256()


def test_source_identity_binds_exact_head_and_empty_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = b"1" * 40
    replies = iter((b"", commit + b"\n", b"", b""))

    def clean(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=next(replies), stderr=b"")

    monkeypatch.setattr(pilot_cli.subprocess, "run", clean)
    identity = source_tree_identity()
    assert identity.git_commit == commit.decode()
    assert identity.source_sha256 == hashlib.sha256(commit).hexdigest()
    assert identity.git_diff_sha256 == hashlib.sha256(b"").hexdigest()


def test_scanner_accepts_redacted_metadata_and_flags_secret_assignments_streaming(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text('{"modal_profile":"ratemem-pilot","credential_values":"redacted"}')
    unsafe = tmp_path / "unsafe.env"
    assignment = b"WANDB_API_" + b"KEY" + b"=fixture-value"
    unsafe.write_bytes(b"x" * 65534 + assignment)
    assert credential_findings([safe]) == []
    assert credential_findings([unsafe]) == [unsafe]


def test_operator_evidence_is_canonical_private_and_binds_screenshot(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    screenshot = private / "usage-budget-28.png"
    screenshot.write_bytes(b"private screenshot")
    screenshot.chmod(0o600)
    timestamp = datetime.now(UTC).replace(microsecond=123456)
    os.utime(screenshot, (timestamp.timestamp(), timestamp.timestamp()))
    evidence = private / "operator.json"
    pilot_cli.create_operator_budget_evidence(
        screenshot=screenshot,
        output=evidence,
        workspace="authorized-workspace",
        confirmation_statement=pilot_cli.BUDGET_CONFIRMATION_STATEMENT,
    )
    before = evidence.read_bytes()
    pilot_cli.create_operator_budget_evidence(
        screenshot=screenshot,
        output=evidence,
        workspace="authorized-workspace",
        confirmation_statement=pilot_cli.BUDGET_CONFIRMATION_STATEMENT,
    )
    assert evidence.read_bytes() == before
    payload = read_private_json(evidence)
    assert evidence.read_bytes() == pilot_cli.canonical_json_bytes(payload)
    assert payload == {
        "kind": "operator-dashboard-budget-v1",
        "profile": "ratemem-pilot",
        "workspace": "authorized-workspace",
        "environment": "main",
        "workspace_budget_usd": "28.00",
        "workspace_spend_limit_usd": "0.00",
        "captured_at": datetime.fromtimestamp(screenshot.stat().st_mtime, UTC).isoformat(),
        "dashboard_evidence_path": str(screenshot.absolute()),
        "dashboard_evidence_sha256": hashlib.sha256(b"private screenshot").hexdigest(),
        "confirmation_statement": pilot_cli.BUDGET_CONFIRMATION_STATEMENT,
    }
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o600


def test_modal_helper_has_allowlisted_profile_environment_and_no_secret_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setenv("MODAL_TOKEN_SECRET", "must-not-be-forwarded")
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/attacker/config.toml")
    monkeypatch.setattr(pilot_cli.subprocess, "run", fake_run)
    assert pilot_cli._modal_cli_json(["volume", "list", "--env", "main"]) == []
    assert observed["command"] == ["modal", "volume", "list", "--env", "main", "--json"]
    environment = observed["kwargs"]["env"]  # type: ignore[index]
    assert environment["MODAL_PROFILE"] == "ratemem-pilot"
    assert environment["MODAL_CONFIG_PATH"] == (
        "/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml"
    )
    assert "/attacker/config.toml" not in environment.values()
    assert "MODAL_TOKEN_SECRET" not in environment
    with pytest.raises(ValueError, match="allowlisted"):
        pilot_cli._modal_cli(["token", "set"])


def test_dedicated_modal_config_is_empty_before_auth_and_exact_after_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = tmp_path / "modal"
    config_root.mkdir(mode=0o700)
    config = config_root / "ratemem-pilot.toml"
    config.write_bytes(b"")
    config.chmod(0o600)
    monkeypatch.setattr(pilot_cli, "MODAL_CONFIG_PATH", config)

    pilot_cli._validate_modal_config_state("empty")
    config.write_text(
        '[ratemem-pilot]\ntoken_id = "synthetic-id"\n'
        'token_secret = "synthetic-secret"\nactive = true\n',
        encoding="utf-8",
    )
    pilot_cli._validate_modal_config_state("configured")


@pytest.mark.parametrize(
    "content",
    [
        b"# a nonempty pre-auth file is forbidden\n",
        (
            b'[ratemem-pilot]\ntoken_id = "synthetic-id"\n'
            b'token_secret = "synthetic-secret"\nactive = true\n'
            b'server_url = "https://example.invalid"\n'
        ),
        (
            b'[ratemem-pilot]\ntoken_id = "synthetic-id"\n'
            b'token_secret = "synthetic-secret"\nactive = true\n'
            b'[another-profile]\ntoken_id = "other"\n'
        ),
        (
            b'[ratemem-pilot]\ntoken_id = "synthetic-id"\n'
            b'token_secret = "synthetic-secret"\nactive = false\n'
        ),
    ],
)
def test_dedicated_modal_config_rejects_nonempty_pre_auth_and_unsafe_post_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
) -> None:
    config_root = tmp_path / "modal"
    config_root.mkdir(mode=0o700)
    config = config_root / "ratemem-pilot.toml"
    config.write_bytes(content)
    config.chmod(0o600)
    monkeypatch.setattr(pilot_cli, "MODAL_CONFIG_PATH", config)

    if content.startswith(b"#"):
        with pytest.raises(ValueError, match="exactly empty"):
            pilot_cli._validate_modal_config_state("empty")
    else:
        with pytest.raises(ValueError, match="pilot profile|exactly the pilot profile|activation"):
            pilot_cli._validate_modal_config_state("configured")


def test_strict_json_rejects_duplicate_nonfinite_and_noncanonical_bytes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        pilot_cli._strict_json_bytes(b'{"x":1,"x":2}', label="fixture")
    with pytest.raises(ValueError, match="non-finite"):
        pilot_cli._strict_json_bytes(b'{"x":NaN}', label="fixture")
    with pytest.raises(ValueError, match="canonical"):
        pilot_cli._strict_json_bytes(b'{"x": 1}', label="fixture")


def test_credential_scanner_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("safe")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(PermissionError):
        credential_findings([link])


def _snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        profile="ratemem-pilot",
        workspace="authorized-workspace",
        environment="main",
        workspace_budget_usd="28.00",
        known_metered_usage_usd="1.25",
        verified_at=datetime.now(UTC),
        evidence_path=tmp_path / "unused-evidence.json",
        evidence_sha256="",
        rates={
            "gpu_l40s_per_second": "0.000542",
            "cpu_core_per_second": "0.0000131",
            "memory_gib_per_second": "0.00000222",
            "volume_gib_month": "0.09",
        },
    )


def test_preflight_validates_every_reversible_gate_before_slot_and_binds_exact_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: datetime(2026, 8, 1, tzinfo=UTC))
    source = pilot_cli.SourceTreeIdentity(
        git_commit="1" * 40,
        source_sha256=hashlib.sha256(("1" * 40).encode()).hexdigest(),
        git_diff_sha256=hashlib.sha256(b"").hexdigest(),
    )
    attempt_id = "018f05f2-d680-7a6f-8f5a-9ef777e2b902"
    state = tmp_path / "state"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(pilot_cli, "GLOBAL_SLOT_PATH", state / "modal-pilot-slot.json")
    monkeypatch.setattr(
        pilot_cli,
        "GLOBAL_SUBMISSION_RECEIPT_PATH",
        state / "modal-pilot-submitted.json",
    )
    monkeypatch.setattr(pilot_cli, "PERMIT_PATH", artifacts / "launch-permit.json")
    monkeypatch.setattr(pilot_cli, "LEDGER_PATH", artifacts / "ledger.jsonl")
    monkeypatch.setattr(
        pilot_cli,
        "verify_fresh_attestation_file",
        lambda _path, **_kwargs: _snapshot(tmp_path),
    )
    attempt_ids: list[str] = []

    def one_attempt_id() -> str:
        attempt_ids.append(attempt_id)
        return attempt_id

    monkeypatch.setattr(pilot_cli, "new_attempt_id", one_attempt_id)

    def source_identity() -> pilot_cli.SourceTreeIdentity:
        calls.append("source")
        return source

    monkeypatch.setattr(pilot_cli, "source_tree_identity", source_identity)
    monkeypatch.setattr(
        pilot_cli,
        "pilot_config_sha256",
        lambda: calls.append("config") or "2" * 64,
    )
    monkeypatch.setattr(
        pilot_cli,
        "_modal_cli_json",
        lambda arguments: calls.append(f"modal-list:{arguments}") or [],
    )

    class FakeLedger:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls.append("ledger-init")

        def verify_hash_chain(self) -> None:
            calls.append("ledger-verify")

        def require_pristine(self) -> None:
            calls.append("ledger-pristine")

        def preview_reservation(self, *_args: object, **_kwargs: object) -> Decimal:
            calls.append("ledger-preview")
            return Decimal("0")

        def reserve(self, *_args: object, **_kwargs: object) -> None:
            calls.append("ledger-reserve")

    monkeypatch.setattr(pilot_cli, "CostLedger", FakeLedger)
    captured: dict[str, object] = {}

    def claim(*_args: object, **_kwargs: object) -> None:
        calls.append("slot")

    def permit(*_args: object, **kwargs: object) -> None:
        calls.append("permit")
        captured.update(kwargs)

    monkeypatch.setattr(pilot_cli, "claim_global_pilot_slot", claim)
    monkeypatch.setattr(pilot_cli, "create_launch_permit", permit)

    pilot_cli.preflight(tmp_path / "attestation.json")

    assert calls.index("modal-list:['volume', 'list', '--env', 'main']") < calls.index("slot")
    assert calls.index("ledger-pristine") < calls.index("ledger-preview") < calls.index("slot")
    assert calls.index("slot") < calls.index("ledger-reserve")
    assert calls.index("ledger-reserve") < calls.index("permit")
    assert attempt_ids == [attempt_id]
    assert calls.count("source") == 2 and calls.count("config") == 2
    assert captured["known_usage_before_usd"] == "1.25"
    assert captured["pending_worst_case_usd"] == captured["phase_bound_usd"] == "10.15"
    assert captured["git_diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert captured["config_sha256"] == "2" * 64


def test_preflight_rate_overrun_never_burns_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: datetime(2026, 8, 1, tzinfo=UTC))
    monkeypatch.setattr(
        pilot_cli,
        "verify_fresh_attestation_file",
        lambda _path, **_kwargs: _snapshot(tmp_path),
    )
    monkeypatch.setattr(
        pilot_cli,
        "source_tree_identity",
        lambda: pilot_cli.SourceTreeIdentity(
            "1" * 40,
            hashlib.sha256(("1" * 40).encode()).hexdigest(),
            hashlib.sha256(b"").hexdigest(),
        ),
    )
    monkeypatch.setattr(pilot_cli, "pilot_config_sha256", lambda: "2" * 64)
    monkeypatch.setattr(pilot_cli, "conservative_bound", lambda *_args: Decimal("21.01"))

    def forbidden(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("irreversible slot must not be touched")

    monkeypatch.setattr(pilot_cli, "claim_global_pilot_slot", forbidden)
    with pytest.raises(ValueError, match="21.00"):
        pilot_cli.preflight(tmp_path / "attestation.json")


@pytest.mark.parametrize(
    ("now", "allowed"),
    [
        (datetime(2026, 8, 22, 0, 0, tzinfo=UTC), True),
        (datetime(2026, 8, 22, 0, 0, 0, 1, tzinfo=UTC), False),
        (datetime(2028, 2, 20, 0, 0, tzinfo=UTC), True),
        (datetime(2028, 2, 20, 0, 0, 0, 1, tzinfo=UTC), False),
    ],
)
def test_preflight_calendar_window_requires_ten_full_days_before_next_utc_month(
    now: datetime,
    allowed: bool,
) -> None:
    if allowed:
        pilot_cli._require_calendar_settlement_window(now)
    else:
        with pytest.raises(typer.Exit) as caught:
            pilot_cli._require_calendar_settlement_window(now)
        assert caught.value.exit_code == 3


def test_provision_is_idempotent_and_uses_only_exact_bound_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permit_path = tmp_path / "launch-permit.json"
    permit_path.write_bytes(b"canonical permit fixture")
    permit_path.chmod(0o600)
    monkeypatch.setattr(pilot_cli, "PERMIT_PATH", permit_path)
    monkeypatch.setattr(pilot_cli, "PROVISION_INTENT_PATH", tmp_path / "intent.json")
    monkeypatch.setattr(pilot_cli, "PROVISION_RECEIPT_PATH", tmp_path / "receipt.json")
    monkeypatch.setattr(
        pilot_cli,
        "_verified_unsubmitted_permit",
        lambda _path: {
            "attempt_id": "019d0000-0000-7000-8000-000000000041",
            "workspace": "authorized-workspace",
            "profile": "ratemem-pilot",
            "environment": "main",
            "slot_sha256": "1" * 64,
        },
    )
    lists = iter(
        (
            [],
            [
                {"name": "ratemem-sana-cache"},
                {"name": "ratemem-pilot-artifacts"},
            ],
            [
                {"name": "ratemem-sana-cache"},
                {"name": "ratemem-pilot-artifacts"},
            ],
        )
    )
    queries: list[list[str]] = []
    creates: list[list[str]] = []

    def query(arguments: list[str]) -> object:
        queries.append(arguments)
        return next(lists)

    monkeypatch.setattr(pilot_cli, "_modal_cli_json", query)
    monkeypatch.setattr(pilot_cli, "_modal_cli", lambda arguments: creates.append(arguments))
    pilot_cli.provision_volumes(Path("attestation.json"))
    pilot_cli.provision_volumes(Path("attestation.json"))
    assert queries == [
        ["volume", "list", "--env", "main"],
        ["volume", "list", "--env", "main"],
        ["volume", "list", "--env", "main"],
    ]
    assert creates == [
        ["volume", "create", "--env", "main", "ratemem-pilot-artifacts"],
        ["volume", "create", "--env", "main", "ratemem-sana-cache"],
    ]
    assert pilot_cli.PROVISION_INTENT_PATH.exists()
    assert pilot_cli.PROVISION_RECEIPT_PATH.exists()


@pytest.mark.parametrize("existing_name", ["ratemem-sana-cache", "unrelated-old-volume"])
def test_provision_rejects_any_preexisting_volume_without_attempt_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    permit_path = tmp_path / "launch-permit.json"
    permit_path.write_bytes(b"canonical permit fixture")
    permit_path.chmod(0o600)
    monkeypatch.setattr(pilot_cli, "PERMIT_PATH", permit_path)
    monkeypatch.setattr(pilot_cli, "PROVISION_INTENT_PATH", tmp_path / "intent.json")
    monkeypatch.setattr(pilot_cli, "PROVISION_RECEIPT_PATH", tmp_path / "receipt.json")
    monkeypatch.setattr(
        pilot_cli,
        "_verified_unsubmitted_permit",
        lambda _path: {
            "attempt_id": "019d0000-0000-7000-8000-000000000041",
            "workspace": "authorized-workspace",
            "profile": "ratemem-pilot",
            "environment": "main",
            "slot_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        pilot_cli,
        "_modal_cli_json",
        lambda _arguments: [{"name": existing_name}],
    )

    def forbidden(_arguments: list[str]) -> Never:
        raise AssertionError("an unknown pre-existing volume must never be reused")

    monkeypatch.setattr(pilot_cli, "_modal_cli", forbidden)
    with pytest.raises(ValueError, match="pre-existing Modal volume"):
        pilot_cli.provision_volumes(Path("attestation.json"))
    assert not pilot_cli.PROVISION_INTENT_PATH.exists()
    assert not pilot_cli.PROVISION_RECEIPT_PATH.exists()


def test_provision_rejects_permit_profile_before_any_modal_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_cli,
        "_verified_unsubmitted_permit",
        lambda _path: {"profile": "other", "environment": "main"},
    )

    def forbidden(_arguments: list[str]) -> Never:
        raise AssertionError("Modal CLI must not be reached")

    monkeypatch.setattr(pilot_cli, "_modal_cli_json", forbidden)
    with pytest.raises(ValueError, match="profile"):
        pilot_cli.provision_volumes(Path("attestation.json"))


def test_unsubmitted_permit_validation_binds_fresh_workspace_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    monkeypatch.setattr(pilot_cli, "source_tree_sha256", lambda: "1" * 64)
    observed: dict[str, object] = {}

    def validate(permit: Path, **kwargs: object) -> dict[str, object]:
        observed.update(permit=permit, **kwargs)
        return {"profile": "ratemem-pilot", "environment": "main"}

    monkeypatch.setattr(pilot_cli, "validate_unsubmitted_launch_permit", validate)
    pilot_cli._verified_unsubmitted_permit(tmp_path / "attestation.json")
    assert observed["expected_workspace"] == "authorized-workspace"
    assert observed["current_source_sha256"] == "1" * 64
    assert observed["receipt"] == pilot_cli.GLOBAL_SUBMISSION_RECEIPT_PATH


def test_permit_field_requires_full_unsubmitted_validation_and_consumed_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    monkeypatch.setattr(pilot_cli, "source_tree_sha256", lambda: "1" * 64)
    calls: list[str] = []

    def unsubmitted(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(f"unsubmitted:{kwargs['expected_workspace']}")
        return {"attempt_id": "attempt-a", "workspace": "authorized-workspace"}

    monkeypatch.setattr(pilot_cli, "validate_unsubmitted_launch_permit", unsubmitted)
    pilot_cli.permit_field("attempt_id")
    assert capsys.readouterr().out.strip() == "attempt-a"
    assert calls == ["unsubmitted:authorized-workspace"]

    def submitted(*_args: object, **_kwargs: object) -> Never:
        raise FileExistsError("consumed")

    monkeypatch.setattr(pilot_cli, "validate_unsubmitted_launch_permit", submitted)
    monkeypatch.setattr(
        pilot_cli,
        "validate_consumed_launch_evidence",
        lambda *_args, **_kwargs: {
            "attempt_id": "attempt-a",
            "workspace": "authorized-workspace",
        },
    )
    pilot_cli.permit_field("workspace")
    assert capsys.readouterr().out.strip() == "authorized-workspace"


def _pending_for_reconcile(*, before: str = "1.00", phase: str = "2.00") -> dict[str, object]:
    return {
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "modal": {"workspace": "authorized-workspace"},
        "cost": {
            "known_usage_before_usd": before,
            "phase_bound_usd": phase,
        },
    }


class _ReconcileLedger:
    def __init__(self, record: AttemptCost) -> None:
        self.record = record
        self.reconcile_calls: list[tuple[str, Decimal, Decimal]] = []

    def attempt_cost(self, _attempt_id: str) -> AttemptCost:
        return self.record

    def reconcile(
        self,
        attempt_id: str,
        *,
        reconciled_cost: Decimal,
        known_usage_after: Decimal,
    ) -> None:
        self.reconcile_calls.append((attempt_id, reconciled_cost, known_usage_after))
        self.record = AttemptCost(
            known_usage_before=self.record.known_usage_before,
            phase_bound=self.record.phase_bound,
            reconciled_cost=reconciled_cost,
            known_usage_after=known_usage_after,
        )


def _configure_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    usage: str,
    record: AttemptCost,
    pending: dict[str, object] | None = None,
) -> tuple[Path, _ReconcileLedger]:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    root = tmp_path / RECONCILE_ATTEMPT_ID
    ensure_private_directory(root)
    path = root / "attempt.pending.json"
    path.write_bytes(b"placeholder")
    path.chmod(0o600)
    payload = _pending_for_reconcile() if pending is None else pending
    monkeypatch.setattr(pilot_cli, "_validate_artifact", lambda *_args, **_kwargs: payload)
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot, "known_metered_usage_usd", usage)
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    ledger = _ReconcileLedger(record)
    monkeypatch.setattr(pilot_cli, "CostLedger", lambda *_args, **_kwargs: ledger)
    monkeypatch.setattr(
        pilot_cli,
        "_verified_volume_absence",
        lambda **_kwargs: pilot_cli.VolumeAbsenceEvidence(
            confirmed_at=datetime.now(UTC),
            known_usage=Decimal("1.00"),
            sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(pilot_cli, "_settlement_candidate", lambda **_kwargs: None)
    monkeypatch.setattr(
        pilot_cli,
        "_reconciled_payload",
        lambda _payload, cost: {
            "attempt_id": RECONCILE_ATTEMPT_ID,
            "reconciled": f"{cost:.2f}",
        },
    )
    return path, ledger


def test_reconcile_billing_lag_keeps_open_reservation_and_exits_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="1.00",
        record=AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None),
    )
    with pytest.raises(typer.Exit) as caught:
        pilot_cli.reconcile(path)
    assert caught.value.exit_code == 3
    assert ledger.reconcile_calls == []
    assert not path.with_name("attempt.json").exists()


def test_settlement_requires_equal_observations_four_days_apart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    first = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    minimum_age = timedelta(days=4)
    observations = iter((first, first + minimum_age - timedelta(seconds=1), first + minimum_age))
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: next(observations))
    arguments = {
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "workspace": "authorized-workspace",
        "known_usage_before": Decimal("1.00"),
        "observed_usage": Decimal("2.50"),
        "volume_absence_sha256": "a" * 64,
    }
    with pytest.raises(typer.Exit) as initial:
        pilot_cli._settlement_candidate(**arguments)
    assert initial.value.exit_code == 3
    candidate = tmp_path / "settlement" / f"{RECONCILE_ATTEMPT_ID}.json"
    initial_bytes = candidate.read_bytes()
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o600
    with pytest.raises(typer.Exit) as immature:
        pilot_cli._settlement_candidate(**arguments)
    assert immature.value.exit_code == 3
    assert candidate.read_bytes() == initial_bytes
    pilot_cli._settlement_candidate(**arguments)


def test_settlement_usage_change_restarts_the_stability_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    first = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    minimum_age = timedelta(days=4)
    observations = iter((first, first + minimum_age, first + minimum_age * 2))
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: next(observations))
    common = {
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "workspace": "authorized-workspace",
        "known_usage_before": Decimal("1.00"),
        "volume_absence_sha256": "a" * 64,
    }
    with pytest.raises(typer.Exit):
        pilot_cli._settlement_candidate(observed_usage=Decimal("2.00"), **common)
    with pytest.raises(typer.Exit):
        pilot_cli._settlement_candidate(observed_usage=Decimal("2.10"), **common)
    candidate = read_private_json(tmp_path / "settlement" / f"{RECONCILE_ATTEMPT_ID}.json")
    assert candidate["observed_usage_usd"] == "2.10"
    assert candidate["first_observed_at_utc"] == (first + minimum_age).isoformat(
        timespec="microseconds"
    )
    pilot_cli._settlement_candidate(observed_usage=Decimal("2.10"), **common)


def test_settlement_advances_when_absence_usage_initially_equals_prelaunch_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    first = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    later = first + timedelta(hours=1)
    pilot_cli._ensure_initial_settlement_candidate(
        attempt_id=RECONCILE_ATTEMPT_ID,
        workspace="authorized-workspace",
        known_usage_before=Decimal("1.00"),
        observed_usage=Decimal("1.00"),
        observed_at=first,
        volume_absence_sha256="a" * 64,
    )
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: later)

    with pytest.raises(typer.Exit) as caught:
        pilot_cli._settlement_candidate(
            attempt_id=RECONCILE_ATTEMPT_ID,
            workspace="authorized-workspace",
            known_usage_before=Decimal("1.00"),
            observed_usage=Decimal("2.00"),
            volume_absence_sha256="a" * 64,
        )
    assert caught.value.exit_code == 3
    candidate = read_private_json(tmp_path / "settlement" / f"{RECONCILE_ATTEMPT_ID}.json")
    assert candidate["observed_usage_usd"] == "2.00"
    assert candidate["first_observed_at_utc"] == later.isoformat(timespec="microseconds")


def test_attest_volume_absence_binds_consumed_launch_and_initial_billing_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", Path("artifacts/reconciliation"))
    pending_path = tmp_path / RECONCILE_ATTEMPT_ID / "attempt.pending.json"
    ensure_private_directory(pending_path.parent)
    pending_path.write_bytes(b"fixture")
    pending_path.chmod(0o600)
    pending = _pending_for_reconcile()
    pending["modal"] = {
        "workspace": "authorized-workspace",
        "pilot_slot_sha256": "1" * 64,
        "submission_receipt_sha256": "3" * 64,
    }
    monkeypatch.setattr(pilot_cli, "_validate_artifact", lambda *_args, **_kwargs: pending)
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot, "known_metered_usage_usd", "2.50")
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    monkeypatch.setattr(pilot_cli, "source_tree_sha256", lambda: "4" * 64)
    monkeypatch.setattr(
        pilot_cli,
        "validate_consumed_launch_evidence",
        lambda *_args, **_kwargs: {
            "attempt_id": RECONCILE_ATTEMPT_ID,
            "workspace": "authorized-workspace",
            "permit_sha256": "2" * 64,
            "slot_sha256": "1" * 64,
            "submission_receipt_sha256": "3" * 64,
        },
    )
    queries: list[list[str]] = []
    monkeypatch.setattr(
        pilot_cli,
        "_modal_cli_json",
        lambda arguments: queries.append(arguments) or [],
    )
    confirmed = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: confirmed)

    pilot_cli.attest_volume_absence(pending_path)

    directory = tmp_path / "artifacts/reconciliation"
    evidence_path = directory / f"{RECONCILE_ATTEMPT_ID}.volume-absence.json"
    evidence = read_private_json(evidence_path)
    assert evidence == {
        "schema_version": 1,
        "kind": "ratemem-pilot-volume-absence-v1",
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "workspace": "authorized-workspace",
        "profile": "ratemem-pilot",
        "environment": "main",
        "volume_names": ["ratemem-pilot-artifacts", "ratemem-sana-cache"],
        "permit_sha256": "2" * 64,
        "slot_sha256": "1" * 64,
        "submission_receipt_sha256": "3" * 64,
        "known_metered_usage_usd": "2.50",
        "confirmed_absent_at_utc": confirmed.isoformat(timespec="microseconds"),
    }
    candidate = read_private_json(directory / f"{RECONCILE_ATTEMPT_ID}.json")
    assert candidate["observed_usage_usd"] == "2.50"
    assert candidate["first_observed_at_utc"] == confirmed.isoformat(timespec="microseconds")
    assert candidate["volume_absence_sha256"] == pilot_cli.file_sha256(evidence_path)
    assert stat.S_IMODE(evidence_path.stat().st_mode) == 0o600
    assert queries == [["volume", "list", "--env", "main"]]


def test_attest_volume_absence_rejects_present_required_volume_without_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", Path("settlement"))
    pending_path = tmp_path / RECONCILE_ATTEMPT_ID / "attempt.pending.json"
    ensure_private_directory(pending_path.parent)
    pending_path.write_bytes(b"fixture")
    pending_path.chmod(0o600)
    monkeypatch.setattr(
        pilot_cli,
        "_validate_artifact",
        lambda *_args, **_kwargs: _pending_for_reconcile(),
    )
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: _snapshot(tmp_path)
    )
    monkeypatch.setattr(
        pilot_cli,
        "_modal_cli_json",
        lambda _arguments: [{"name": "ratemem-sana-cache"}],
    )

    with pytest.raises(typer.Exit) as caught:
        pilot_cli.attest_volume_absence(pending_path)
    assert caught.value.exit_code == 3
    assert not (tmp_path / "settlement").exists()


@pytest.mark.parametrize(
    ("reappearance_sha256", "expected_status"),
    [
        (None, "artifact_unavailable"),
        ("f" * 64, "attempt_invalidated"),
    ],
)
def test_record_incident_is_create_only_and_binds_launch_ledger_billing_and_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reappearance_sha256: str | None,
    expected_status: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "INCIDENT_DIRECTORY", Path("artifacts/incidents"))
    permit_path = tmp_path / "launch-permit.json"
    permit = {
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "workspace": "authorized-workspace",
        "known_usage_before_usd": "1.00",
        "phase_bound_usd": "2.00",
        "rates_sha256": "4" * 64,
        "slot_sha256": "1" * 64,
    }
    pilot_cli.write_exclusive_private_json(permit_path, permit)
    monkeypatch.setattr(pilot_cli, "PERMIT_PATH", permit_path)
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot, "known_metered_usage_usd", "1.25")
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    monkeypatch.setattr(pilot_cli, "source_tree_sha256", lambda: "5" * 64)
    monkeypatch.setattr(
        pilot_cli,
        "validate_unsubmitted_launch_permit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError("consumed")),
    )
    monkeypatch.setattr(
        pilot_cli,
        "validate_consumed_launch_evidence",
        lambda *_args, **_kwargs: {
            "attempt_id": RECONCILE_ATTEMPT_ID,
            "workspace": "authorized-workspace",
            "permit_sha256": pilot_cli.file_sha256(permit_path),
            "slot_sha256": "1" * 64,
            "submission_receipt_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        pilot_cli,
        "_modal_cli_json",
        lambda _arguments: [
            {"name": "ratemem-sana-cache"},
            {"name": "ratemem-pilot-artifacts"},
        ],
    )
    ledger = _ReconcileLedger(AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None))
    monkeypatch.setattr(pilot_cli, "CostLedger", lambda *_args, **_kwargs: ledger)
    created = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: created)
    monkeypatch.setattr(
        pilot_cli,
        "_latest_reappearance_sha256",
        lambda _attempt_id: reappearance_sha256,
    )

    pilot_cli.record_incident()
    pilot_cli.record_incident()

    pending = tmp_path / "artifacts/incidents" / RECONCILE_ATTEMPT_ID / "incident.pending.json"
    payload = read_private_json(pending)
    assert payload["attempt_id"] == RECONCILE_ATTEMPT_ID
    assert payload["status"] == expected_status
    assert payload["launch"]["state"] == "consumed"
    assert payload["launch"]["submission_receipt_sha256"] == "3" * 64
    assert payload["cost"]["known_usage_before_usd"] == "1.00"
    assert payload["cost"]["known_usage_at_incident_usd"] == "1.25"
    assert payload["volumes_observed"] == [
        "ratemem-pilot-artifacts",
        "ratemem-sana-cache",
    ]
    assert payload["created_at_utc"] == created.isoformat(timespec="microseconds")
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600


def _incident_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "ratemem-pilot-incident-v1",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "workspace": "authorized-workspace",
        "profile": "ratemem-pilot",
        "environment": "main",
        "status": "artifact_unavailable",
        "reason": "artifact_unavailable_after_launch_command",
        "created_at_utc": "2026-08-01T12:00:00.000000+00:00",
        "launch": {
            "state": "consumed",
            "permit_sha256": "2" * 64,
            "slot_sha256": "1" * 64,
            "submission_receipt_sha256": "3" * 64,
        },
        "cost": {
            "known_usage_before_usd": "1.00",
            "known_usage_at_incident_usd": "1.25",
            "phase_bound_usd": "2.00",
            "rates_sha256": "4" * 64,
            "reconciliation_status": "pending",
            "reconciled_cost_usd": None,
            "hard_budget_violation": None,
        },
        "volumes_observed": ["ratemem-pilot-artifacts", "ratemem-sana-cache"],
    }


def test_incident_absence_and_mature_reconciliation_close_ledger_without_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", Path("settlement"))
    incident_root = tmp_path / RECONCILE_ATTEMPT_ID
    ensure_private_directory(incident_root)
    pending_path = incident_root / "incident.pending.json"
    pilot_cli.write_exclusive_private_json(pending_path, _incident_payload())
    incident = _incident_payload()

    def validate(path: Path, *, allow_final: bool = False) -> dict[str, object]:
        if allow_final:
            return read_private_json(path)
        return incident

    monkeypatch.setattr(pilot_cli, "_validate_incident", validate)
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot, "known_metered_usage_usd", "2.50")
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    monkeypatch.setattr(pilot_cli, "_modal_cli_json", lambda _arguments: [])
    confirmed = datetime(2026, 8, 1, 12, 30, tzinfo=UTC)
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: confirmed)

    pilot_cli.attest_incident_volume_absence(pending_path)
    absence = (
        tmp_path / "settlement" / f"{RECONCILE_ATTEMPT_ID}.incident-volume-absence.initial.json"
    )
    assert absence.exists()
    assert read_private_json(absence)["permit_sha256"] == "2" * 64

    ledger = _ReconcileLedger(AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None))
    monkeypatch.setattr(pilot_cli, "CostLedger", lambda *_args, **_kwargs: ledger)
    monkeypatch.setattr(pilot_cli, "_settlement_candidate", lambda **_kwargs: None)
    pilot_cli.reconcile_incident(pending_path)
    assert ledger.reconcile_calls == [(RECONCILE_ATTEMPT_ID, Decimal("1.50"), Decimal("2.50"))]
    final = read_private_json(incident_root / "incident.json")
    assert final["cost"]["reconciliation_status"] == "reconciled"
    assert final["cost"]["reconciled_cost_usd"] == "1.50"
    assert final["publication_eligible"] is False
    assert not (incident_root / "attempt.json").exists()


def test_incident_zero_delta_matures_and_reconciles_without_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", Path("settlement"))
    incident_root = tmp_path / RECONCILE_ATTEMPT_ID
    ensure_private_directory(incident_root)
    pending_path = incident_root / "incident.pending.json"
    incident = _incident_payload()
    pilot_cli.write_exclusive_private_json(pending_path, incident)

    def validate(path: Path, *, allow_final: bool = False) -> dict[str, object]:
        return read_private_json(path) if allow_final else incident

    monkeypatch.setattr(pilot_cli, "_validate_incident", validate)
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot, "known_metered_usage_usd", "1.00")
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    monkeypatch.setattr(pilot_cli, "_modal_cli_json", lambda _arguments: [])
    confirmed = datetime(2026, 8, 1, 12, 30, tzinfo=UTC)
    current = confirmed
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: current)
    pilot_cli.attest_incident_volume_absence(pending_path)

    ledger = _ReconcileLedger(AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None))
    monkeypatch.setattr(pilot_cli, "CostLedger", lambda *_args, **_kwargs: ledger)
    current = confirmed + timedelta(days=4)
    pilot_cli.reconcile_incident(pending_path)

    assert ledger.reconcile_calls == [(RECONCILE_ATTEMPT_ID, Decimal("0.00"), Decimal("1.00"))]
    assert (
        read_private_json(incident_root / "incident.json")["cost"]["reconciled_cost_usd"] == "0.00"
    )
    assert not (incident_root / "attempt.json").exists()


def test_incident_reappearance_requires_a_new_four_day_absence_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", Path("settlement"))
    incident_root = tmp_path / RECONCILE_ATTEMPT_ID
    ensure_private_directory(incident_root)
    pending_path = incident_root / "incident.pending.json"
    incident = _incident_payload()
    pilot_cli.write_exclusive_private_json(pending_path, incident)
    monkeypatch.setattr(
        pilot_cli,
        "_validate_incident",
        lambda path, *, allow_final=False: (read_private_json(path) if allow_final else incident),
    )
    snapshot = _snapshot(tmp_path)
    object.__setattr__(snapshot, "known_metered_usage_usd", "2.50")
    monkeypatch.setattr(
        pilot_cli, "verify_fresh_attestation_file", lambda _path, **_kwargs: snapshot
    )
    live: list[dict[str, str]] = []
    monkeypatch.setattr(pilot_cli, "_modal_cli_json", lambda _arguments: live)
    current = datetime(2026, 8, 1, 12, 30, tzinfo=UTC)
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: current)
    pilot_cli.attest_incident_volume_absence(pending_path)

    ledger = _ReconcileLedger(AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None))
    monkeypatch.setattr(pilot_cli, "CostLedger", lambda *_args, **_kwargs: ledger)
    current += timedelta(days=4)
    live = [{"name": "ratemem-sana-cache"}]
    with pytest.raises(RuntimeError, match="reappeared"):
        pilot_cli.reconcile_incident(pending_path)
    assert list((tmp_path / "settlement").glob("*.volume-reappearance.*.json"))

    live = []
    with pytest.raises(typer.Exit):
        pilot_cli.reconcile_incident(pending_path)
    current += timedelta(minutes=1)
    pilot_cli.attest_incident_volume_absence(pending_path)
    current += timedelta(days=3, hours=23, minutes=59)
    with pytest.raises(typer.Exit):
        pilot_cli.reconcile_incident(pending_path)
    current += timedelta(minutes=1)
    pilot_cli.reconcile_incident(pending_path)
    assert ledger.reconcile_calls == [(RECONCILE_ATTEMPT_ID, Decimal("1.50"), Decimal("2.50"))]


def test_volume_reappearance_permanently_invalidates_normal_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    ensure_private_directory(tmp_path / "settlement")
    absence = tmp_path / "settlement" / f"{RECONCILE_ATTEMPT_ID}.volume-absence.json"
    pilot_cli.write_exclusive_private_json(absence, {"fixture": True})
    pending = _pending_for_reconcile()
    pending["modal"] = {"workspace": "authorized-workspace"}
    launch = {
        "permit_sha256": "2" * 64,
        "slot_sha256": "1" * 64,
        "submission_receipt_sha256": "3" * 64,
    }
    monkeypatch.setattr(pilot_cli, "_consumed_launch_evidence", lambda _workspace: launch)
    lists = iter(([{"name": "ratemem-sana-cache"}], []))
    monkeypatch.setattr(pilot_cli, "_modal_cli_json", lambda _arguments: next(lists))

    with pytest.raises(RuntimeError, match="reappeared.*permanently invalid"):
        pilot_cli._verified_volume_absence(
            pending=pending,
            workspace="authorized-workspace",
        )
    records = list((tmp_path / "settlement").glob("*.volume-reappearance.*.json"))
    assert len(records) == 1
    assert read_private_json(records[0])["permit_sha256"] == "2" * 64

    monkeypatch.setattr(
        pilot_cli,
        "_read_volume_absence",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalidated absence must never be reused")
        ),
    )
    with pytest.raises(RuntimeError, match="reappearance.*permanently invalid"):
        pilot_cli._verified_volume_absence(
            pending=pending,
            workspace="authorized-workspace",
        )


def test_settlement_rejects_nonfinite_usage_in_durable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    observed_at = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(pilot_cli, "_now_utc", lambda: observed_at)
    arguments = {
        "attempt_id": RECONCILE_ATTEMPT_ID,
        "workspace": "authorized-workspace",
        "known_usage_before": Decimal("1.00"),
        "observed_usage": Decimal("2.50"),
        "volume_absence_sha256": "a" * 64,
    }
    with pytest.raises(typer.Exit):
        pilot_cli._settlement_candidate(**arguments)
    candidate = tmp_path / "settlement" / f"{RECONCILE_ATTEMPT_ID}.json"
    payload = read_private_json(candidate)
    payload["observed_usage_usd"] = "NaN"
    pilot_cli.write_atomic_private_json(candidate, payload)

    with pytest.raises(ValueError, match="observed usage is invalid"):
        pilot_cli._settlement_candidate(**arguments)


def test_reconcile_truthfully_records_realized_cost_above_admission_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="4.00",
        record=AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None),
    )
    pilot_cli.reconcile(path)
    assert ledger.reconcile_calls == [(RECONCILE_ATTEMPT_ID, Decimal("3.00"), Decimal("4.00"))]
    assert read_private_json(path.with_name("attempt.json"))["reconciled"] == "3.00"


def test_reconcile_recovers_idempotently_after_ledger_before_final_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="2.50",
        record=AttemptCost(Decimal("1.00"), Decimal("2.00"), None, None),
    )
    real_write = pilot_cli.write_exclusive_private_bytes
    failed = False

    def fail_once(target: Path, content: bytes) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected final publication crash")
        real_write(target, content)

    monkeypatch.setattr(pilot_cli, "write_exclusive_private_bytes", fail_once)
    with pytest.raises(OSError, match="publication crash"):
        pilot_cli.reconcile(path)
    assert ledger.record.reconciled_cost == Decimal("1.50")
    assert not path.with_name("attempt.json").exists()
    pilot_cli.reconcile(path)
    assert len(ledger.reconcile_calls) == 1
    assert path.with_name("attempt.json").exists()


def test_reconcile_rejects_fresh_usage_drift_after_durable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="2.51",
        record=AttemptCost(Decimal("1.00"), Decimal("2.00"), Decimal("1.50"), Decimal("2.50")),
    )
    with pytest.raises(ValueError, match="differs.*durable"):
        pilot_cli.reconcile(path)
    assert not path.with_name("attempt.json").exists()


@pytest.mark.parametrize("matching", [True, False])
def test_reconcile_existing_final_must_exactly_match_durable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    matching: bool,
) -> None:
    path, ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="2.50",
        record=AttemptCost(Decimal("1.00"), Decimal("2.00"), Decimal("1.50"), Decimal("2.50")),
    )
    expected = {"attempt_id": RECONCILE_ATTEMPT_ID, "reconciled": "1.50"}
    content = pilot_cli.canonical_json_bytes(expected if matching else expected | {"extra": True})
    pilot_cli.write_exclusive_private_bytes(path.with_name("attempt.json"), content)
    if matching:
        pilot_cli.reconcile(path)
    else:
        with pytest.raises(ValueError, match="existing attempt.json differs"):
            pilot_cli.reconcile(path)
    assert ledger.reconcile_calls == []


def test_hard_budget_violation_is_durably_reconciled_without_false_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = _pending_for_reconcile(before="26.00", phase="1.00")
    path, ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="29.00",
        record=AttemptCost(Decimal("26.00"), Decimal("1.00"), None, None),
        pending=pending,
    )
    with pytest.raises(RuntimeError, match="HARD BUDGET VIOLATION"):
        pilot_cli.reconcile(path)
    assert ledger.reconcile_calls == [(RECONCILE_ATTEMPT_ID, Decimal("3.00"), Decimal("29.00"))]
    assert len(list((tmp_path / "settlement").glob("*.hard-budget.*.json"))) == 1
    assert not path.with_name("attempt.json").exists()


def test_hard_budget_violation_before_maturity_is_durable_but_keeps_ledger_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SETTLEMENT_DIRECTORY", tmp_path / "settlement")
    pending = _pending_for_reconcile(before="26.00", phase="1.00")
    path, ledger = _configure_reconcile(
        tmp_path,
        monkeypatch,
        usage="29.00",
        record=AttemptCost(Decimal("26.00"), Decimal("1.00"), None, None),
        pending=pending,
    )

    def immature(**_kwargs: object) -> Never:
        raise typer.Exit(code=3)

    monkeypatch.setattr(pilot_cli, "_settlement_candidate", immature)
    with pytest.raises(RuntimeError, match="HARD BUDGET VIOLATION.*settlement remains open"):
        pilot_cli.reconcile(path)
    assert ledger.reconcile_calls == []
    observations = list((tmp_path / "settlement").glob("*.hard-budget.*.json"))
    assert len(observations) == 1
    observation = read_private_json(observations[0])
    assert observation["attempt_id"] == RECONCILE_ATTEMPT_ID
    assert observation["known_metered_usage_usd"] == "29.00"
    assert observation["hard_budget_usd"] == "28.00"
    assert stat.S_IMODE(observations[0].stat().st_mode) == 0o600


def test_security_scan_includes_changed_untracked_and_explicit_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "_require_repository_cwd", lambda: None)
    changed = tmp_path / "changed-safe.txt"
    untracked = tmp_path / "untracked-safe.txt"
    changed.write_text("safe")
    untracked.write_text("safe")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    unsafe = artifact / "unsafe.env"
    credential_name = "MODAL_TOKEN_" + "SECRET"
    unsafe.write_text(f"{credential_name}=fixture-secret")

    def git(arguments: list[str]) -> bytes:
        if arguments[0] == "diff":
            return b"changed-safe.txt\0"
        return b"untracked-safe.txt\0"

    monkeypatch.setattr(pilot_cli, "_git", git)
    with pytest.raises(typer.Exit) as caught:
        pilot_cli.security_scan([artifact])
    assert caught.value.exit_code == 4


def test_security_scan_skips_generated_source_cache_but_not_attempt_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot_cli, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_cli, "_git", lambda _arguments: b"")
    source_cache = tmp_path / "src" / "__pycache__"
    source_cache.mkdir(parents=True)
    opaque = "ak" + "-" + "SYNTHETIC_CACHE_VALUE_12345"
    (source_cache / "generated.pyc").write_text(opaque)
    (tmp_path / "src" / "safe.py").write_text("value = 'redacted'\n")

    pilot_cli.security_scan([tmp_path / "src"])
    assert "PASS" in capsys.readouterr().out

    attempt = tmp_path / "artifacts" / "pilot" / RECONCILE_ATTEMPT_ID
    attempt_cache = attempt / "__pycache__"
    attempt_cache.mkdir(parents=True)
    (attempt_cache / "untrusted.pyc").write_text(opaque)
    with pytest.raises(typer.Exit) as caught:
        pilot_cli.security_scan([attempt])
    assert caught.value.exit_code == 4
