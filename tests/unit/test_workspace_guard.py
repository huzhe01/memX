from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import ratemem.pilot.workspace as workspace_module
from ratemem.pilot.config import ModalBudgetConfig
from ratemem.pilot.private_io import ensure_private_directory, write_exclusive_private_json
from ratemem.pilot.workspace import (
    WorkspaceSnapshot,
    capture_workspace_snapshot,
    verify_fresh_attestation_file,
    verify_workspace_snapshot,
)

FIXTURES = Path("tests/fixtures/modal")


def _private_evidence(tmp_path: Path) -> Path:
    private = tmp_path / "private"
    ensure_private_directory(private)
    dashboard = private / "usage-budget-28.png"
    dashboard.write_bytes(b"credential-free budget evidence")
    dashboard.chmod(0o600)
    attestation = private / "operator-budget-attestation.json"
    write_exclusive_private_json(
        attestation,
        {
            "kind": "operator-dashboard-budget-v1",
            "profile": "ratemem-pilot",
            "workspace": "authorized-workspace",
            "environment": "main",
            "workspace_budget_usd": "28.00",
            "captured_at": datetime.now(UTC).isoformat(),
            "dashboard_evidence_path": str(dashboard),
            "dashboard_evidence_sha256": hashlib.sha256(dashboard.read_bytes()).hexdigest(),
            "confirmation_statement": (
                "I confirm the Modal dashboard Workspace usage budget is USD 28.00 "
                "before credits."
            ),
        },
    )
    return attestation


def _snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        profile="ratemem-pilot",
        workspace="authorized-workspace",
        environment="main",
        workspace_budget_usd="28.00",
        known_metered_usage_usd="1.25",
        verified_at=datetime.now(UTC),
        evidence_path=_private_evidence(tmp_path),
        evidence_sha256="",
        rates={
            "gpu_l40s_per_second": "0.000542",
            "cpu_core_per_second": "0.0000131",
            "memory_gib_per_second": "0.00000222",
            "volume_gib_month": "0.09",
        },
    ).with_evidence_hash()


def test_committed_budget_is_exact_and_phase_split_is_not_extra_authority() -> None:
    config = ModalBudgetConfig.load(Path("configs/pilot/modal-budget.json"))
    assert config.workspace_budget_usd == Decimal("28.00")
    assert config.internal_limit_usd == Decimal("27.00")
    assert config.first_pilot_allocation_usd == Decimal("21.00")
    assert (
        config.setup_probe_allocation_usd
        + config.timing_probe_allocation_usd
        + config.held_in_pilot_allocation_usd
        == config.first_pilot_allocation_usd
    )
    assert config.first_pilot_allocation_usd + config.unallocated_safety_buffer_usd == Decimal(
        "27.00"
    )
    assert config.retries == 0
    assert config.max_containers == 1
    assert config.detached is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("workspace_budget_usd", "28.01"),
        ("internal_limit_usd", "27"),
        ("gpu_count", True),
        ("retries", 1),
        ("detached", 0),
        ("gpu", "L40S:2"),
    ],
)
def test_budget_config_rejects_value_and_type_confusion(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    payload = json.loads(Path("configs/pilot/modal-budget.json").read_text())
    payload[key] = value
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="locked|canonical|exact"):
        ModalBudgetConfig.load(path)


def test_workspace_snapshot_roundtrip_is_exact_and_verified(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    reconstructed = WorkspaceSnapshot.from_json(snapshot.to_json())
    assert reconstructed == snapshot
    assert (
        verify_workspace_snapshot(
            reconstructed,
            expected_workspace="authorized-workspace",
            max_age_seconds=900,
        )
        == snapshot
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile", "default", "profile"),
        ("workspace", "other", "workspace"),
        ("environment", "dev", "environment"),
        ("workspace_budget_usd", "28.01", "budget"),
        ("known_metered_usage_usd", "-0.01", "metered"),
        ("known_metered_usage_usd", "NaN", "metered"),
        ("evidence_sha256", "f" * 64, "evidence"),
    ],
)
def test_workspace_mismatch_nonfinite_and_changed_evidence_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        candidate = replace(_snapshot(tmp_path), **{field: value})
        verify_workspace_snapshot(
            candidate,
            expected_workspace="authorized-workspace",
            max_age_seconds=900,
        )


@pytest.mark.parametrize("seconds", [901, -1])
def test_stale_or_future_attestation_fails_closed(tmp_path: Path, seconds: int) -> None:
    candidate = replace(
        _snapshot(tmp_path),
        verified_at=datetime.now(UTC) - timedelta(seconds=seconds),
    )
    with pytest.raises(ValueError, match="stale|future"):
        verify_workspace_snapshot(
            candidate,
            expected_workspace="authorized-workspace",
            max_age_seconds=900,
        )


def test_snapshot_parser_rejects_coercion_extra_keys_bad_rates_and_naive_time(
    tmp_path: Path,
) -> None:
    payload = _snapshot(tmp_path).to_json()
    cases: list[dict[str, object]] = []
    cases.append(payload | {"unexpected": True})
    cases.append(payload | {"workspace_budget_usd": 28})
    cases.append(payload | {"verified_at": datetime.now().isoformat()})
    cases.append(payload | {"rates": {"gpu_l40s_per_second": "0.1"}})
    cases.append(payload | {"rates": payload["rates"] | {"volume_gib_month": "Infinity"}})  # type: ignore[operator]
    for candidate in cases:
        with pytest.raises((TypeError, ValueError)):
            WorkspaceSnapshot.from_json(candidate)


def test_capture_uses_active_profile_and_metered_cost_before_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _private_evidence(tmp_path)
    profiles = json.loads((FIXTURES / "profile-list.json").read_text())
    billing = json.loads((FIXTURES / "billing-summary.json").read_text())
    rates = json.loads((FIXTURES / "rates.json").read_text())

    def modal_json(profile: str, arguments: list[str], **_kwargs: object) -> object:
        assert profile == "ratemem-pilot"
        if arguments[:2] == ["profile", "list"]:
            return profiles
        if arguments[:2] == ["billing", "summary"]:
            return billing
        if arguments[:2] == ["billing", "rates"]:
            return rates
        raise AssertionError(arguments)

    monkeypatch.setattr(workspace_module, "_modal_json", modal_json)
    snapshot = capture_workspace_snapshot(
        evidence_path=evidence,
        confirmed_budget="28.00",
        config_path=evidence,
    )
    assert snapshot.workspace == "authorized-workspace"
    assert snapshot.known_metered_usage_usd == "1.25"
    assert snapshot.known_metered_usage_usd != billing["billed_cost"]


@pytest.mark.parametrize(
    "changes",
    [
        {"adjustments": {"credits": "0.01"}},
        {"metered_cost_breakdown": {"compute": "1.24"}},
        {"billed_cost": "0.01"},
        {"billed_cost": "-0.01", "adjustments": {"credits": "-1.26"}},
    ],
)
def test_billing_semantic_contradictions_fail_closed(changes: dict[str, object]) -> None:
    payload = json.loads((FIXTURES / "billing-summary.json").read_text()) | changes
    with pytest.raises(ValueError, match="credits|compute|billed|semantic"):
        workspace_module._validate_billing(payload)


def test_arbitrary_dashboard_evidence_bytes_are_not_an_operator_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    evidence = private / "arbitrary.bin"
    evidence.write_bytes(b"not a structured attestation")
    evidence.chmod(0o600)
    fixtures = {
        ("profile", "list"): json.loads((FIXTURES / "profile-list.json").read_text()),
        ("billing", "summary"): json.loads((FIXTURES / "billing-summary.json").read_text()),
        ("billing", "rates"): json.loads((FIXTURES / "rates.json").read_text()),
    }
    monkeypatch.setattr(
        workspace_module,
        "_modal_json",
        lambda _profile, arguments, **_kwargs: fixtures[tuple(arguments[:2])],
    )
    with pytest.raises(ValueError, match="operator-attested dashboard evidence|JSON"):
        capture_workspace_snapshot(
            evidence_path=evidence,
            confirmed_budget="28.00",
            config_path=evidence,
        )


def test_modal_subprocess_binds_exact_config_and_scrubs_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _private_evidence(tmp_path)
    monkeypatch.setenv("MODAL_TOKEN_ID", "must-not-pass")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "must-not-pass")
    monkeypatch.setenv("MODAL_CONFIG_PATH", "/wrong/config")
    observed: dict[str, str] = {}

    def run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs["env"])  # type: ignore[arg-type]
        return subprocess.CompletedProcess([], 0, stdout="[]", stderr="")

    monkeypatch.setattr(workspace_module.subprocess, "run", run)
    workspace_module._modal_json(
        "ratemem-pilot", ["profile", "list"], config_path=config
    )
    assert observed["MODAL_CONFIG_PATH"] == str(config)
    assert observed["MODAL_PROFILE"] == "ratemem-pilot"
    assert "MODAL_TOKEN_ID" not in observed
    assert "MODAL_TOKEN_SECRET" not in observed
    assert "/wrong/config" not in observed.values()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("modal missing"), "unavailable"),
        (subprocess.CompletedProcess([], 1, stdout="{}", stderr="denied"), "failed"),
        (subprocess.CompletedProcess([], 0, stdout="{}", stderr="warning"), "failed"),
        (subprocess.CompletedProcess([], 0, stdout="not-json", stderr=""), "strict JSON"),
    ],
)
def test_modal_json_command_failures_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | subprocess.CompletedProcess[str],
    message: str,
) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if isinstance(failure, BaseException):
            raise failure
        return failure

    monkeypatch.setattr(workspace_module.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError), match=message):
        workspace_module._modal_json(
            "ratemem-pilot",
            ["billing", "summary", "--for", "this month"],
            config_path=_private_evidence(tmp_path),
        )


def test_modal_json_rejects_duplicate_nonfinite_and_unknown_billing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '{"metered_cost":"1","metered_cost":"2"}',
            '{"value":NaN}',
            '{"metered_cost":"1","billed_cost":"0","adjustments":{"credits":"1"},'
            '"metered_cost_breakdown":{"compute":"1"},"unknown":true}',
        ]
    )

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(workspace_module.subprocess, "run", run)
    config = _private_evidence(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        workspace_module._modal_json(
            "ratemem-pilot", ["billing", "rates"], config_path=config
        )
    with pytest.raises(ValueError, match="non-finite"):
        workspace_module._modal_json(
            "ratemem-pilot", ["billing", "rates"], config_path=config
        )
    with pytest.raises(ValueError, match="unknown"):
        workspace_module._validate_billing(
            workspace_module._modal_json(
                "ratemem-pilot",
                ["billing", "summary", "--for", "this month"],
                config_path=config,
            )
        )


def test_verify_fresh_attestation_requeries_profile_metered_usage_and_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    attestation = snapshot.evidence_path.parent / "attestation.json"
    write_exclusive_private_json(attestation, snapshot.to_json())
    fixtures = {
        ("profile", "list"): json.loads((FIXTURES / "profile-list.json").read_text()),
        ("billing", "summary"): json.loads(
            (FIXTURES / "billing-summary.json").read_text()
        ),
        ("billing", "rates"): json.loads((FIXTURES / "rates.json").read_text()),
    }

    def modal_json(profile: str, arguments: list[str], **_kwargs: object) -> object:
        assert profile == "ratemem-pilot"
        return fixtures[tuple(arguments[:2])]

    monkeypatch.setattr(workspace_module, "_modal_json", modal_json)
    refreshed = verify_fresh_attestation_file(attestation)
    assert refreshed.known_metered_usage_usd == "1.25"
    assert refreshed.rates == fixtures[("billing", "rates")]
