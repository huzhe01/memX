from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from ratemem.evaluation.canonical import write_json_atomic
from ratemem.evaluation.compute import (
    BaselineFidelityBindings,
    BaselineFidelityPhaseRequest,
    ScientificComputeDenied,
    WorkspaceSelection,
    WorkspaceSnapshot,
    authorize_baseline_fidelity,
    load_baseline_fidelity_policy,
    require_baseline_fidelity_permit,
    reserve_baseline_fidelity_cost,
)

POLICY_PATH = Path("configs/scientific/baseline-fidelity-compute-policy.yaml")
NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def selection(*, workspace_id: str = "workspace-a") -> WorkspaceSelection:
    return WorkspaceSelection(
        schema_version="1.0",
        workspace_id=workspace_id,
        explicit_profile="ratemem-scientific-baseline-a",
        selected_at_utc=NOW - timedelta(minutes=1),
        operator_file_sha256="1" * 64,
    )


def snapshot(
    *,
    workspace_id: str = "workspace-a",
    outer_budget_usd: str = "28.00",
    known_usage_usd: str = "10.00",
) -> WorkspaceSnapshot:
    return WorkspaceSnapshot.create(
        workspace_id=workspace_id,
        explicit_profile="ratemem-scientific-baseline-a",
        provider="modal",
        outer_budget_usd=outer_budget_usd,
        known_usage_usd=known_usage_usd,
        budget_evidence_sha256="2" * 64,
        observed_at_utc=NOW,
    )


def phase(**updates: object) -> BaselineFidelityPhaseRequest:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "scope": "baseline_fidelity",
        "phase_id": "dreamcache_sana_fidelity",
        "input_role": "held_in",
        "input_manifest_sha256": "3" * 64,
        "job_spec_sha256": "4" * 64,
        "source_revision": "5" * 40,
        "source_archive_sha256": "6" * 64,
        "git_commit": "7" * 40,
        "clean_diff_sha256": "8" * 64,
        "payload_references": [],
        "selection_fields": [],
        "claim_metric_fields": [],
    }
    payload.update(updates)
    return BaselineFidelityPhaseRequest.model_validate(payload)


def bindings(**updates: object) -> BaselineFidelityBindings:
    payload: dict[str, object] = {
        "dataset_lock_sha256": "9" * 64,
        "baseline_requirements_sha256": "a" * 64,
        "comparator_catalog_sha256": "b" * 64,
        "fidelity_policy_sha256": "c" * 64,
        "source_inventory_sha256": "d" * 64,
        "git_commit": "7" * 40,
        "clean_diff_sha256": "8" * 64,
    }
    payload.update(updates)
    return BaselineFidelityBindings.model_validate(payload)


def authorization():
    return authorize_baseline_fidelity(
        selection(),
        snapshot(),
        phase(),
        bindings(),
        load_baseline_fidelity_policy(POLICY_PATH),
        issued_at_utc=NOW,
    )


def test_authorization_is_bound_to_one_explicit_workspace_and_clean_source() -> None:
    permit = authorization()
    assert permit.scope == "baseline_fidelity"
    assert permit.workspace_id == "workspace-a"
    assert permit.explicit_profile == "ratemem-scientific-baseline-a"
    assert permit.source_revision == "5" * 40
    assert permit.authorization_sha256 == permit.recomputed_sha256


@pytest.mark.parametrize("input_role", ["validation", "final_test"])
def test_validation_and_final_roles_are_rejected(input_role: str) -> None:
    with pytest.raises(ScientificComputeDenied, match="forbidden input role"):
        authorize_baseline_fidelity(
            selection(),
            snapshot(),
            phase(input_role=input_role),
            bindings(),
            load_baseline_fidelity_policy(POLICY_PATH),
            issued_at_utc=NOW,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "payload_references",
            ["configs/scientific/traces/final-test-envelope.json"],
            "final-trace reference",
        ),
        (
            "selection_fields",
            ["best_validation_checkpoint"],
            "model-selection field",
        ),
        (
            "claim_metric_fields",
            ["request_weighted_identity"],
            "claim-quality metric",
        ),
    ],
)
def test_authorization_recursively_rejects_future_or_outcome_material(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ScientificComputeDenied, match=message):
        authorize_baseline_fidelity(
            selection(),
            snapshot(),
            phase(**{field: value}),
            bindings(),
            load_baseline_fidelity_policy(POLICY_PATH),
            issued_at_utc=NOW,
        )


def test_mutable_source_dirty_diff_and_second_workspace_are_rejected() -> None:
    policy = load_baseline_fidelity_policy(POLICY_PATH)
    with pytest.raises(ScientificComputeDenied, match="immutable source revision"):
        authorize_baseline_fidelity(
            selection(),
            snapshot(),
            phase(source_revision="main"),
            bindings(),
            policy,
            issued_at_utc=NOW,
        )
    with pytest.raises(ScientificComputeDenied, match="clean diff mismatch"):
        authorize_baseline_fidelity(
            selection(),
            snapshot(),
            phase(),
            bindings(clean_diff_sha256="f" * 64),
            policy,
            issued_at_utc=NOW,
        )
    with pytest.raises(ScientificComputeDenied, match="explicit workspace mismatch"):
        authorize_baseline_fidelity(
            selection(),
            snapshot(workspace_id="workspace-b"),
            phase(),
            bindings(),
            policy,
            issued_at_utc=NOW,
        )


@pytest.mark.parametrize("outer_cap", ["27.99", "28.01", "100.00"])
def test_outer_workspace_cap_must_be_exactly_28_usd(outer_cap: str) -> None:
    with pytest.raises(ScientificComputeDenied, match="USD 28.00 outer cap"):
        authorize_baseline_fidelity(
            selection(),
            snapshot(outer_budget_usd=outer_cap),
            phase(),
            bindings(),
            load_baseline_fidelity_policy(POLICY_PATH),
        )


def test_reservation_includes_known_pending_and_new_cost() -> None:
    policy = load_baseline_fidelity_policy(POLICY_PATH)
    accepted = reserve_baseline_fidelity_cost(
        authorization(),
        snapshot(),
        pending_worst_case_usd=Decimal("8.00"),
        new_phase_bound_usd=Decimal("9.00"),
        policy=policy,
        reserved_at_utc=NOW,
    )
    assert accepted.reserved_total_usd == Decimal("27.00")
    with pytest.raises(ScientificComputeDenied, match="internal USD 27.00 limit"):
        reserve_baseline_fidelity_cost(
            authorization(),
            snapshot(),
            pending_worst_case_usd=Decimal("8.01"),
            new_phase_bound_usd=Decimal("9.00"),
            policy=policy,
            reserved_at_utc=NOW,
        )


def test_authorization_and_reservation_digests_are_acyclic() -> None:
    policy = load_baseline_fidelity_policy(POLICY_PATH)
    first = authorization()
    first_reservation = reserve_baseline_fidelity_cost(
        first,
        snapshot(),
        pending_worst_case_usd=Decimal("0.00"),
        new_phase_bound_usd=Decimal("1.00"),
        policy=policy,
        reserved_at_utc=NOW,
    )
    changed = first.model_copy(update={"job_spec_sha256": "f" * 64})
    second_reservation = reserve_baseline_fidelity_cost(
        changed,
        snapshot(),
        pending_worst_case_usd=Decimal("0.00"),
        new_phase_bound_usd=Decimal("1.00"),
        policy=policy,
        reserved_at_utc=NOW,
    )
    assert "reservation" not in first.model_dump_json()
    assert first_reservation.authorization_sha256 == first.authorization_sha256
    assert first_reservation.reservation_sha256 != second_reservation.reservation_sha256


def test_permit_consumption_is_one_shot_before_provider_invocation(tmp_path: Path) -> None:
    permit = authorization()
    reservation = reserve_baseline_fidelity_cost(
        permit,
        snapshot(),
        pending_worst_case_usd=Decimal("0.00"),
        new_phase_bound_usd=Decimal("1.00"),
        policy=load_baseline_fidelity_policy(POLICY_PATH),
        reserved_at_utc=NOW,
    )
    authorization_path = tmp_path / "authorization.json"
    reservation_path = tmp_path / "reservation.json"
    launch_receipt = tmp_path / "launch-receipt.json"
    write_json_atomic(authorization_path, permit.model_dump(mode="json"))
    write_json_atomic(reservation_path, reservation.model_dump(mode="json"))

    consumed = require_baseline_fidelity_permit(
        authorization_path,
        reservation_path,
        expected_phase_id="dreamcache_sana_fidelity",
        expected_workspace_id="workspace-a",
        launch_receipt_path=launch_receipt,
        consumed_at_utc=NOW,
    )
    assert consumed.provider_invocations_before_consumption == 0
    with pytest.raises(ScientificComputeDenied, match="already consumed"):
        require_baseline_fidelity_permit(
            authorization_path,
            reservation_path,
            expected_phase_id="dreamcache_sana_fidelity",
            expected_workspace_id="workspace-a",
            launch_receipt_path=launch_receipt,
            consumed_at_utc=NOW,
        )
