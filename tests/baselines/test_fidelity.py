from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ratemem.baselines.catalog import load_catalog
from ratemem.baselines.fidelity import (
    BaselineAuditBlocked,
    FidelityAuditError,
    FidelityMatrix,
    FidelityMeasurement,
    FidelityPolicy,
    FidelityReport,
    SearchBudgetError,
    SearchLedger,
    SearchPolicy,
    SearchTrial,
    audit_fidelity_report,
    audit_search_ledger,
    deterministic_search_cells,
    load_fidelity_policy,
    load_search_policy,
    resolve_primary_backbone_plan,
    seal_fidelity_report,
)
from ratemem.baselines.registry import RuntimeRegistryLock, build_registry
from ratemem.baselines.sources import build_source_inventory
from ratemem.evaluation.canonical import canonical_json_bytes


def _policies() -> tuple[FidelityPolicy, SearchPolicy]:
    return (
        load_fidelity_policy(Path("configs/baselines/fidelity-policy.yaml")),
        load_search_policy(Path("configs/baselines/policy-search.yaml")),
    )


def _registry() -> RuntimeRegistryLock:
    catalog = load_catalog(Path("configs/baselines/literature-classification.yaml"))
    return build_registry(catalog, baseline_lock_id="1" * 64).lock()


def _report(
    method_id: str,
    policy: FidelityPolicy,
    registry: RuntimeRegistryLock,
    *,
    backbone_id: str = "sana_1_5_1_6b",
) -> FidelityReport:
    factory = next(row for row in registry.entries if row.method_id == method_id)
    measurements = tuple(
        FidelityMeasurement(
            case_id=case_id,
            input_sha256=hashlib.sha256(f"input:{case_id}".encode()).hexdigest(),
            raw_output_sha256=hashlib.sha256(f"output:{case_id}".encode()).hexdigest(),
            passed=True,
        )
        for case_id in policy.case_ids_for(method_id)
    )
    provisional = FidelityReport(
        method_id=method_id,
        factory_import_path=factory.import_path,
        concrete_factory_sha256=factory.source_sha256,
        source_record_sha256s=(),
        source_revisions={},
        backbone_id=backbone_id,  # type: ignore[arg-type]
        backbone_revision="4" * 40,
        case_policy_sha256=policy.sha256,
        measurements=measurements,
        state_roundtrip_report_sha256="1" * 64,
        byte_budget_report_sha256="2" * 64,
        causal_access_report_sha256="3" * 64,
        synthetic_provider_contract_report_sha256="4" * 64,
        environment_lock_sha256="5" * 64,
        status="faithful",
        report_sha256="0" * 64,
    )
    return seal_fidelity_report(provisional)


def test_fidelity_report_binds_policy_factory_and_raw_outputs() -> None:
    policy, _search = _policies()
    registry = _registry()
    inventory = build_source_inventory(())
    report = _report("independent_fifo", policy, registry)
    assert audit_fidelity_report(report, policy, inventory, registry) == report
    changed = report.model_copy(
        update={
            "measurements": (
                report.measurements[0].model_copy(update={"raw_output_sha256": "0" * 64}),
                *report.measurements[1:],
            )
        }
    )
    with pytest.raises(FidelityAuditError, match="hash mismatch"):
        audit_fidelity_report(changed, policy, inventory, registry)


def test_primary_gate_never_uses_contextual_evidence_as_a_substitute() -> None:
    policy, _search = _policies()
    registry = _registry()
    reports = tuple(
        _report(method, policy, registry)
        for method in policy.sana_primary_required_methods
    )
    matrix = FidelityMatrix(
        policy_sha256=policy.sha256,
        required_methods=policy.sana_primary_required_methods,
        reports=reports,
    )
    assert resolve_primary_backbone_plan(matrix).primary_backbone == "sana_1_5_1_6b"
    missing_method = "dreamcache_feature_cache"
    contextual = _report(
        missing_method,
        policy,
        registry,
        backbone_id="sdxl_1_0",
    )
    blocked = matrix.model_copy(
        update={
            "reports": tuple(
                row for row in reports if row.method_id != missing_method
            )
            + (contextual,)
        }
    )
    with pytest.raises(
        BaselineAuditBlocked,
        match=f"required_sana_control_unfaithful:{missing_method}",
    ):
        resolve_primary_backbone_plan(blocked)
    with pytest.raises(
        BaselineAuditBlocked,
        match="contextual_backbone_not_primary_eligible:sdxl_1_0",
    ):
        resolve_primary_backbone_plan(matrix).require_primary("sdxl_1_0")


def _search_ledger(
    policy: SearchPolicy,
    *,
    backbone_id: str = "sana_1_5_1_6b",
    status: str = "passed",
) -> SearchLedger:
    configuration = deterministic_search_cells(policy, "independent_fifo")[0]
    trial = SearchTrial(
        attempt_id="attempt-0",
        method_id="independent_fifo",
        backbone_id=backbone_id,  # type: ignore[arg-type]
        backbone_revision="4" * 40,
        configuration=configuration,
        configuration_sha256=hashlib.sha256(
            canonical_json_bytes(configuration)
        ).hexdigest(),
        split="validation",
        shared_input_lock_sha256="6" * 64,
        started_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
        ended_at_utc=datetime(2026, 8, 30, tzinfo=UTC) + timedelta(minutes=5),
        gpu_sku="PPU-ZW810E",
        gpu_hours=0.1,
        exit_status=status,  # type: ignore[arg-type]
        artifact_sha256="7" * 64,
    )
    provisional = SearchLedger(
        policy_sha256=policy.sha256,
        method_id="independent_fifo",
        rows=(trial,),
        ledger_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "ledger_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
        }
    )


def test_search_is_validation_only_equal_budget_and_counts_failure() -> None:
    _fidelity, policy = _policies()
    audit = audit_search_ledger(_search_ledger(policy, status="failed"), policy)
    assert audit.trials == 1
    assert audit.failed_trials == 1
    assert audit.seen_splits == {"validation"}
    with pytest.raises(SearchBudgetError, match="search backbone must be"):
        audit_search_ledger(_search_ledger(policy, backbone_id="sdxl_1_0"), policy)


def test_large_spaces_are_reduced_to_deterministic_balanced_twenty_four_cells() -> None:
    _fidelity, policy = _policies()
    first = deterministic_search_cells(policy, "vb_lora_style_static")
    second = deterministic_search_cells(policy, "vb_lora_style_static")
    assert first == second
    assert len(first) == 24
    assert len({canonical_json_bytes(row) for row in first}) == 24
