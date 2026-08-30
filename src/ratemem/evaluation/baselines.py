"""Scientific registry consumer and fail-closed pre-lock baseline audit."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from pydantic import (
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from ratemem.baselines.catalog import REQUIRED_CONTROL_IDS, load_catalog
from ratemem.baselines.protocol import (
    BaselineAdapter as BaselineAdapter,
)
from ratemem.baselines.protocol import (
    CausalEventView as CausalEventView,
)
from ratemem.baselines.protocol import (
    EventReceipt as EventReceipt,
)
from ratemem.baselines.protocol import (
    ExactByteLedger as ExactByteLedger,
)
from ratemem.baselines.protocol import (
    FrozenComparisonContract as FrozenComparisonContract,
)
from ratemem.baselines.protocol import (
    FutureAccessError as FutureAccessError,
)
from ratemem.baselines.protocol import (
    MethodSnapshot as MethodSnapshot,
)
from ratemem.baselines.protocol import (
    ProbeResult as ProbeResult,
)
from ratemem.evaluation.canonical import (
    canonical_json_bytes,
    write_yaml_atomic,
)
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_MUTABLE_CONFIG = ConfigDict(extra="forbid", validate_assignment=False)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


class BaselineLockError(ValueError):
    """Raised when baseline requirements or their pre-lock evidence are invalid."""


class PrimaryRoles(BaseModel):
    model_config = _MODEL_CONFIG

    representation: tuple[str, ...]
    allocator: tuple[str, ...]
    optimization_free_tradeoff: tuple[str, ...]


class BackboneResolution(BaseModel):
    model_config = _MODEL_CONFIG

    sole_primary: Literal["sana_1_5_1_6b"]
    allow_primary_backbone_fallback: Literal[False]
    sdxl_native_evidence: Literal["contextual_only"]
    block_claim_if_required_sana_fidelity_fails: Literal[True]


class PrelockSharedInputPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    permitted_bundle_kind: Literal["synthetic_protocol"]
    require_provider_neutral_schema: Literal[True]
    require_exact_ledger_roundtrip: Literal[True]
    forbid_learned_ratemem_dictionary: Literal[True]
    forbid_validation_metrics: Literal[True]
    forbid_final_trace: Literal[True]


class BaselineSearchPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    maximum_trials_per_method: PositiveInt
    maximum_gpu_hours_per_method: PositiveFloat
    split: Literal["validation"]
    prelock_mode: Literal["frozen_policy_only"]
    require_postlock_receipts_before_selection: Literal[True]

    @model_validator(mode="after")
    def validate_limits(self) -> BaselineSearchPolicy:
        if self.maximum_trials_per_method != 24:
            raise ValueError("baseline search trial limit must be exactly 24")
        if self.maximum_gpu_hours_per_method != 48.0:
            raise ValueError("baseline search GPU-hour limit must be exactly 48.0")
        return self


class LiteratureDisposition(BaseModel):
    model_config = _MODEL_CONFIG

    require_every_catalog_entry_classified: Literal[True]
    contextual_or_incompatible_never_blocks_primary_lock: Literal[True]


class Requirements(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    catalog: Literal["configs/baselines/literature-classification.yaml"]
    runnable_registry: tuple[str, ...]
    postlock_execution_required: tuple[str, ...]
    sana_primary_required: tuple[str, ...]
    primary_roles: PrimaryRoles
    upper_reference_only: tuple[str, ...]
    secondary_only: tuple[str, ...]
    contextual_literature_citation_keys: tuple[str, ...]
    backbone_resolution: BackboneResolution
    prelock_shared_input: PrelockSharedInputPolicy
    search_policy: BaselineSearchPolicy
    literature_disposition: LiteratureDisposition

    @model_validator(mode="after")
    def validate_registry(self) -> Requirements:
        runnable = self.runnable_registry
        if (
            len(runnable) != 15
            or len(runnable) != len(set(runnable))
            or set(runnable) != REQUIRED_CONTROL_IDS
        ):
            raise ValueError("runnable registry must equal the 15 catalog controls")
        if self.postlock_execution_required != runnable:
            raise ValueError("every runnable control requires post-lock execution")
        if not set(self.sana_primary_required) <= set(runnable):
            raise ValueError("SANA primary requirements must be runnable controls")
        role_members = {
            item
            for values in (
                self.primary_roles.representation,
                self.primary_roles.allocator,
                self.primary_roles.optimization_free_tradeoff,
            )
            for item in values
        }
        if not role_members <= set(runnable):
            raise ValueError("primary role registry names an unknown control")
        if set(self.upper_reference_only) != {
            "exact_append_only_quantized",
            "exact_future_trace_packets",
        }:
            raise ValueError("upper-reference registry differs from the policy")
        if set(self.secondary_only) != {"hyperlora_upstream", "stateless_amortizer"}:
            raise ValueError("secondary-only registry differs from the policy")
        if self.contextual_literature_citation_keys != (
            "sinelora_delta_aaai2026",
        ):
            raise ValueError("contextual literature registry differs from the policy")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class SourceInventoryRecord(BaseModel):
    model_config = _MODEL_CONFIG

    method_id: str
    repository: AnyUrl
    revision: GitCommit
    source_archive_sha256: Sha256
    license_spdx: str
    redistribution_allowed: bool
    record_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("record_sha256")
        return canonical_json_bytes(payload)

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        repository: str,
        revision: str,
        source_archive_sha256: str,
        license_spdx: str,
        redistribution_allowed: bool,
    ) -> SourceInventoryRecord:
        provisional = cls.model_validate(
            {
                "method_id": method_id,
                "repository": repository,
                "revision": revision,
                "source_archive_sha256": source_archive_sha256,
                "license_spdx": license_spdx,
                "redistribution_allowed": redistribution_allowed,
                "record_sha256": "0" * 64,
            }
        )
        return provisional.model_copy(
            update={
                "record_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
            }
        )


class FidelityReport(BaseModel):
    model_config = _MUTABLE_CONFIG

    schema_version: Literal["1.0"]
    method_id: str
    backbone: Literal["sana_1_5_1_6b", "sdxl_1_0"]
    source_revision: GitCommit
    source_archive_sha256: Sha256
    structural_fidelity_passed: bool
    paid_compute: bool
    reconciliation_sha256: Sha256 | None
    report_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("report_sha256")
        return canonical_json_bytes(payload)

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        backbone: str,
        source_revision: str,
        source_archive_sha256: str,
        structural_fidelity_passed: bool,
        paid_compute: bool,
        reconciliation_sha256: str | None,
    ) -> FidelityReport:
        provisional = cls(
            schema_version="1.0",
            method_id=method_id,
            backbone=cast(Literal["sana_1_5_1_6b", "sdxl_1_0"], backbone),
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
            structural_fidelity_passed=structural_fidelity_passed,
            paid_compute=paid_compute,
            reconciliation_sha256=reconciliation_sha256,
            report_sha256="0" * 64,
        )
        return provisional.model_copy(
            update={"report_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
        )


class ComplianceReport(BaseModel):
    model_config = _MUTABLE_CONFIG

    method_id: str
    state_ledger_roundtrip_passed: bool
    causal_access_passed: bool
    declared_upper_reference: bool
    report_sha256: Sha256


class SyntheticProviderReport(BaseModel):
    model_config = _MUTABLE_CONFIG

    schema_version: Literal["1.0"]
    bundle_kind: str
    provider_neutral_schema_passed: bool
    exact_ledger_roundtrip_passed: bool
    outcome_rows: list[dict[str, float]]
    referenced_splits: list[str]
    learned_dictionary_sha256: Sha256 | None
    final_trace_sha256: Sha256 | None
    report_sha256: Sha256


class SearchPolicyEvidence(BaseModel):
    model_config = _MUTABLE_CONFIG

    schema_version: Literal["1.0"]
    maximum_trials_per_method: PositiveInt
    maximum_gpu_hours_per_method: PositiveFloat
    split: Literal["validation"]
    frozen_policy_sha256: Sha256
    execution_ledger_sha256: Sha256 | None
    outcome_rows: list[dict[str, float]]


class AuditInputs(BaseModel):
    model_config = _MUTABLE_CONFIG

    requirements: Requirements
    catalog_sha256: Sha256
    backbone_plan_sha256: Sha256
    source_inventory_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report: SyntheticProviderReport
    search_policy: SearchPolicyEvidence
    source_records: dict[str, SourceInventoryRecord]
    fidelity_reports: dict[str, FidelityReport]
    compliance_reports: dict[str, ComplianceReport]


class BaselineLockEntry(BaseModel):
    model_config = _MODEL_CONFIG

    method_id: str
    source_inventory_record_sha256: Sha256
    fidelity_report_sha256: Sha256
    compliance_report_sha256: Sha256
    state_ledger_test_sha256: Sha256
    supported_backbones: frozenset[Literal["sana_1_5_1_6b", "sdxl_1_0"]]
    sana_primary_eligible: bool
    disposition: Literal[
        "faithful",
        "incompatible",
        "secondary_only",
        "upper_reference_only",
    ]


class BaselineAuditReceipt(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    status: Literal["pass"]
    requirements_sha256: Sha256
    catalog_sha256: Sha256
    backbone_plan_sha256: Sha256
    source_inventory_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report_sha256: Sha256
    search_policy_sha256: Sha256
    method_entries: tuple[BaselineLockEntry, ...]
    receipt_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("receipt_sha256")
        return canonical_json_bytes(payload)


class BaselineLock(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    lock_id: Sha256
    sealed_at_utc: AwareDatetime
    primary_backbone: Literal["sana_1_5_1_6b"]
    audit_receipt_sha256: Sha256
    requirements_sha256: Sha256
    catalog_sha256: Sha256
    backbone_plan_sha256: Sha256
    source_inventory_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report_sha256: Sha256
    search_policy_sha256: Sha256
    method_entries: tuple[BaselineLockEntry, ...]

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("lock_id")
        payload.pop("sealed_at_utc")
        return canonical_json_bytes(payload)


def _validate_hashed_evidence(inputs: AuditInputs) -> None:
    for method_id, source in inputs.source_records.items():
        if method_id != source.method_id:
            raise BaselineLockError("source inventory method id mismatch")
        if hashlib.sha256(source.semantic_bytes).hexdigest() != source.record_sha256:
            raise BaselineLockError("source inventory record hash mismatch")
        if not source.license_spdx.strip() or not source.redistribution_allowed:
            raise BaselineLockError("source inventory lacks usable license evidence")
        if _IMMUTABLE_REVISION.fullmatch(source.revision) is None:
            raise BaselineLockError("source inventory revision is mutable")
    for method_id, report in inputs.fidelity_reports.items():
        if method_id != report.method_id:
            raise BaselineLockError("fidelity report method id mismatch")
        if hashlib.sha256(report.semantic_bytes).hexdigest() != report.report_sha256:
            raise BaselineLockError("fidelity report hash mismatch")


def validate_prelock_handoff(inputs: AuditInputs) -> BaselineAuditReceipt:
    """Admit only source/fidelity/protocol evidence with no validation outcomes."""

    if type(inputs) is not AuditInputs:
        raise TypeError("inputs must be an exact AuditInputs")
    requirements = inputs.requirements
    required = set(requirements.runnable_registry)
    report = inputs.synthetic_provider_report
    if (
        report.bundle_kind != "synthetic_protocol"
        or not report.provider_neutral_schema_passed
        or not report.exact_ledger_roundtrip_passed
        or report.outcome_rows
        or report.learned_dictionary_sha256 is not None
        or report.final_trace_sha256 is not None
        or any(split in {"validation", "final_test"} for split in report.referenced_splits)
    ):
        raise BaselineLockError("pre-lock evidence boundary rejected shared-input evidence")
    search = inputs.search_policy
    if (
        search.maximum_trials_per_method != 24
        or search.maximum_gpu_hours_per_method != 48.0
        or search.split != "validation"
        or search.execution_ledger_sha256 is not None
        or search.outcome_rows
    ):
        raise BaselineLockError("pre-lock evidence boundary rejected search outcomes")
    if set(inputs.source_records) != required:
        raise BaselineLockError("source inventory is incomplete")
    if set(inputs.fidelity_reports) != required:
        raise BaselineLockError("fidelity receipt registry is incomplete")
    if set(inputs.compliance_reports) != required:
        raise BaselineLockError("compliance report registry is incomplete")
    _validate_hashed_evidence(inputs)

    entries: list[BaselineLockEntry] = []
    sana_required = set(requirements.sana_primary_required)
    for method_id in requirements.runnable_registry:
        source = inputs.source_records[method_id]
        fidelity = inputs.fidelity_reports[method_id]
        compliance = inputs.compliance_reports[method_id]
        if (
            fidelity.source_revision != source.revision
            or fidelity.source_archive_sha256 != source.source_archive_sha256
            or not fidelity.structural_fidelity_passed
        ):
            raise BaselineLockError(f"fidelity receipt is unfaithful: {method_id}")
        if method_id in sana_required and fidelity.backbone != "sana_1_5_1_6b":
            raise BaselineLockError(f"required SANA fidelity is missing: {method_id}")
        if fidelity.paid_compute and fidelity.reconciliation_sha256 is None:
            raise BaselineLockError(
                f"paid fidelity reconciliation is missing: {method_id}"
            )
        if not compliance.state_ledger_roundtrip_passed:
            raise BaselineLockError(f"state ledger roundtrip failed: {method_id}")
        if method_id == "exact_future_trace_packets":
            if not compliance.declared_upper_reference:
                raise BaselineLockError("future trace comparator must be an upper reference")
        elif not compliance.causal_access_passed:
            raise BaselineLockError(f"causal access failed: {method_id}")
        if method_id in requirements.upper_reference_only:
            disposition: Literal[
                "faithful", "incompatible", "secondary_only", "upper_reference_only"
            ] = "upper_reference_only"
        elif method_id in requirements.secondary_only:
            disposition = "secondary_only"
        else:
            disposition = "faithful"
        entries.append(
            BaselineLockEntry(
                method_id=method_id,
                source_inventory_record_sha256=source.record_sha256,
                fidelity_report_sha256=fidelity.report_sha256,
                compliance_report_sha256=compliance.report_sha256,
                state_ledger_test_sha256=compliance.report_sha256,
                supported_backbones=frozenset({fidelity.backbone}),
                sana_primary_eligible=(
                    method_id in sana_required
                    and fidelity.backbone == "sana_1_5_1_6b"
                ),
                disposition=disposition,
            )
        )
    provisional = BaselineAuditReceipt(
        schema_version="1.0",
        status="pass",
        requirements_sha256=requirements.sha256,
        catalog_sha256=inputs.catalog_sha256,
        backbone_plan_sha256=inputs.backbone_plan_sha256,
        source_inventory_sha256=inputs.source_inventory_sha256,
        shared_input_schema_sha256=inputs.shared_input_schema_sha256,
        synthetic_provider_report_sha256=report.report_sha256,
        search_policy_sha256=search.frozen_policy_sha256,
        method_entries=tuple(entries),
        receipt_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "receipt_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
        }
    )


def freeze_baseline_lock(
    inputs: AuditInputs,
    output: Path | None = None,
    *,
    sealed_at_utc: datetime | None = None,
) -> BaselineLock:
    """Seal a baseline lock only after the complete pre-lock handoff passes."""

    if output is not None and (output.exists() or output.is_symlink()):
        raise BaselineLockError("baseline lock output already exists")
    receipt = validate_prelock_handoff(inputs)
    timestamp = sealed_at_utc if sealed_at_utc is not None else datetime.now(UTC)
    if timestamp.utcoffset() != timedelta(0):
        raise BaselineLockError("sealed_at_utc must be UTC")
    semantic = {
        "schema_version": "1.0",
        "primary_backbone": "sana_1_5_1_6b",
        "audit_receipt_sha256": receipt.receipt_sha256,
        "requirements_sha256": receipt.requirements_sha256,
        "catalog_sha256": receipt.catalog_sha256,
        "backbone_plan_sha256": receipt.backbone_plan_sha256,
        "source_inventory_sha256": receipt.source_inventory_sha256,
        "shared_input_schema_sha256": receipt.shared_input_schema_sha256,
        "synthetic_provider_report_sha256": receipt.synthetic_provider_report_sha256,
        "search_policy_sha256": receipt.search_policy_sha256,
        "method_entries": [entry.model_dump(mode="json") for entry in receipt.method_entries],
    }
    lock = BaselineLock(
        schema_version="1.0",
        lock_id=hashlib.sha256(canonical_json_bytes(semantic)).hexdigest(),
        sealed_at_utc=timestamp,
        primary_backbone="sana_1_5_1_6b",
        audit_receipt_sha256=receipt.receipt_sha256,
        requirements_sha256=receipt.requirements_sha256,
        catalog_sha256=receipt.catalog_sha256,
        backbone_plan_sha256=receipt.backbone_plan_sha256,
        source_inventory_sha256=receipt.source_inventory_sha256,
        shared_input_schema_sha256=receipt.shared_input_schema_sha256,
        synthetic_provider_report_sha256=receipt.synthetic_provider_report_sha256,
        search_policy_sha256=receipt.search_policy_sha256,
        method_entries=receipt.method_entries,
    )
    if output is not None:
        write_yaml_atomic(output, lock.model_dump(mode="json"))
    return lock


def select_strongest_eligible_control(
    rows: pd.DataFrame,
    *,
    endpoint: str,
    constraint: str,
    minimum_constraint: float,
) -> str:
    """Select by frozen validation endpoint after applying the constraint margin."""

    if type(endpoint) is not str or type(constraint) is not str:
        raise TypeError("endpoint and constraint must be exact strings")
    if (
        type(minimum_constraint) is not float
        or not math.isfinite(minimum_constraint)
    ):
        raise TypeError("minimum_constraint must be a finite exact float")
    required_columns = {"method_id", "split", endpoint, constraint}
    if not required_columns <= set(rows.columns):
        raise BaselineLockError("control selection rows lack required columns")
    eligible = rows.loc[
        (rows["split"] == "validation")
        & (rows[constraint] >= minimum_constraint),
        ["method_id", endpoint],
    ].copy()
    if eligible.empty:
        raise BaselineLockError("no control satisfies the frozen constraint")
    if eligible[endpoint].isna().any() or not eligible[endpoint].map(math.isfinite).all():
        raise BaselineLockError("control selection endpoint must be finite")
    ordered = eligible.sort_values(
        by=[endpoint, "method_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(ordered.iloc[0]["method_id"])


def load_requirements(path: Path) -> Requirements:
    """Load baseline requirements and prove exact agreement with the catalog."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        requirements = Requirements.model_validate(payload)
        catalog = load_catalog(Path(requirements.catalog))
    except (OSError, ValueError) as error:
        raise BaselineLockError(f"invalid baseline requirements: {error}") from error
    if set(requirements.runnable_registry) != set(catalog.control_ids):
        raise BaselineLockError("baseline requirements differ from comparator catalog")
    return requirements


__all__ = [
    "AuditInputs",
    "BaselineAdapter",
    "BaselineAuditReceipt",
    "BaselineLock",
    "BaselineLockEntry",
    "BaselineLockError",
    "BaselineSearchPolicy",
    "BackboneResolution",
    "CausalEventView",
    "ComplianceReport",
    "EventReceipt",
    "ExactByteLedger",
    "FrozenComparisonContract",
    "FutureAccessError",
    "FidelityReport",
    "LiteratureDisposition",
    "MethodSnapshot",
    "PrelockSharedInputPolicy",
    "PrimaryRoles",
    "ProbeResult",
    "Requirements",
    "SearchPolicyEvidence",
    "SourceInventoryRecord",
    "SyntheticProviderReport",
    "freeze_baseline_lock",
    "load_requirements",
    "select_strongest_eligible_control",
    "validate_prelock_handoff",
]
