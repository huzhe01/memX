from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from ratemem.baselines.catalog import load_catalog
from ratemem.evaluation.baselines import (
    AuditInputs,
    BaselineLockError,
    ComplianceReport,
    FidelityReport,
    SearchPolicyEvidence,
    SourceInventoryRecord,
    SyntheticProviderReport,
    freeze_baseline_lock,
    load_requirements,
    select_strongest_eligible_control,
    validate_prelock_handoff,
)

REQUIREMENTS = Path("configs/scientific/baseline-requirements.yaml")
CATALOG = Path("configs/baselines/literature-classification.yaml")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def faithful_report(method_id: str, *, backbone: str = "sana_1_5_1_6b") -> FidelityReport:
    return FidelityReport.create(
        method_id=method_id,
        backbone=backbone,
        source_revision="1" * 40,
        source_archive_sha256=_sha(f"source:{method_id}"),
        structural_fidelity_passed=True,
        paid_compute=False,
        reconciliation_sha256=None,
    )


def valid_audit_inputs() -> AuditInputs:
    requirements = load_requirements(REQUIREMENTS)
    catalog = load_catalog(CATALOG)
    methods = requirements.runnable_registry
    return AuditInputs(
        requirements=requirements,
        catalog_sha256=catalog.sha256,
        backbone_plan_sha256=_sha("backbone-plan"),
        source_inventory_sha256=_sha("source-inventory"),
        shared_input_schema_sha256=_sha("shared-input-schema"),
        synthetic_provider_report=SyntheticProviderReport(
            schema_version="1.0",
            bundle_kind="synthetic_protocol",
            provider_neutral_schema_passed=True,
            exact_ledger_roundtrip_passed=True,
            outcome_rows=[],
            referenced_splits=["synthetic"],
            learned_dictionary_sha256=None,
            final_trace_sha256=None,
            report_sha256=_sha("synthetic-provider-report"),
        ),
        search_policy=SearchPolicyEvidence(
            schema_version="1.0",
            maximum_trials_per_method=24,
            maximum_gpu_hours_per_method=48.0,
            split="validation",
            frozen_policy_sha256=_sha("search-policy"),
            execution_ledger_sha256=None,
            outcome_rows=[],
        ),
        source_records={
            method_id: SourceInventoryRecord.create(
                method_id=method_id,
                repository=f"https://example.org/{method_id}",
                revision="1" * 40,
                source_archive_sha256=_sha(f"source:{method_id}"),
                license_spdx="Apache-2.0",
                redistribution_allowed=True,
            )
            for method_id in methods
        },
        fidelity_reports={
            method_id: faithful_report(method_id) for method_id in methods
        },
        compliance_reports={
            method_id: ComplianceReport(
                method_id=method_id,
                state_ledger_roundtrip_passed=True,
                causal_access_passed=(method_id != "exact_future_trace_packets"),
                declared_upper_reference=(method_id == "exact_future_trace_packets"),
                report_sha256=_sha(f"compliance:{method_id}"),
            )
            for method_id in methods
        },
    )


def test_prelock_audit_rejects_real_dictionary_or_validation_outcomes() -> None:
    inputs = valid_audit_inputs()
    inputs.synthetic_provider_report.bundle_kind = "ratemem_learned_dictionary"
    inputs.synthetic_provider_report.outcome_rows = [
        {"request_weighted_identity": 0.7}
    ]
    with pytest.raises(BaselineLockError, match="pre-lock evidence boundary"):
        freeze_baseline_lock(inputs)


def test_baseline_lock_requires_source_fidelity_and_frozen_search_policy() -> None:
    inputs = valid_audit_inputs()
    inputs.fidelity_reports.pop("dreamcache_feature_cache")
    with pytest.raises(BaselineLockError, match="fidelity receipt"):
        freeze_baseline_lock(inputs)

    inputs = valid_audit_inputs()
    inputs.search_policy.execution_ledger_sha256 = _sha("forbidden-search-run")
    with pytest.raises(BaselineLockError, match="pre-lock evidence boundary"):
        freeze_baseline_lock(inputs)


def test_sdxl_fidelity_cannot_replace_a_required_sana_comparator() -> None:
    inputs = valid_audit_inputs()
    inputs.fidelity_reports["share_style_online"] = faithful_report(
        "share_style_online",
        backbone="sdxl_1_0",
    )
    with pytest.raises(BaselineLockError, match="required SANA fidelity"):
        freeze_baseline_lock(inputs)


def test_strongest_control_selector_uses_only_postlock_validation_rows() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "method_id": "private_progressive_size_aware",
                "split": "validation",
                "identity": 0.61,
                "prompt": 0.72,
            },
            {
                "method_id": "share_style_online",
                "split": "validation",
                "identity": 0.64,
                "prompt": 0.71,
            },
            {
                "method_id": "dreamcache_feature_cache",
                "split": "validation",
                "identity": 0.66,
                "prompt": 0.60,
            },
            {
                "method_id": "future_leak",
                "split": "final_test",
                "identity": 1.0,
                "prompt": 1.0,
            },
        ]
    )
    selected = select_strongest_eligible_control(
        rows,
        endpoint="identity",
        constraint="prompt",
        minimum_constraint=0.70,
    )
    assert selected == "share_style_online"


def test_baseline_lock_and_audit_receipt_are_content_addressed() -> None:
    inputs = valid_audit_inputs()
    receipt = validate_prelock_handoff(inputs)
    lock = freeze_baseline_lock(
        inputs,
        sealed_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    assert receipt.status == "pass"
    assert receipt.receipt_sha256 == lock.audit_receipt_sha256
    assert hashlib.sha256(lock.semantic_bytes).hexdigest() == lock.lock_id
    assert len(lock.method_entries) == 15


def test_paid_fidelity_requires_reconciliation() -> None:
    inputs = valid_audit_inputs()
    inputs.fidelity_reports["dreamcache_feature_cache"] = FidelityReport.create(
        method_id="dreamcache_feature_cache",
        backbone="sana_1_5_1_6b",
        source_revision="1" * 40,
        source_archive_sha256=_sha("source:dreamcache_feature_cache"),
        structural_fidelity_passed=True,
        paid_compute=True,
        reconciliation_sha256=None,
    )
    with pytest.raises(BaselineLockError, match="paid fidelity reconciliation"):
        freeze_baseline_lock(inputs)
