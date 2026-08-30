"""Algorithmic fidelity, equal search-budget, and primary-backbone gates."""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    model_validator,
)

from ratemem.baselines.registry import RuntimeRegistryLock
from ratemem.baselines.sources import SourceInventory
from ratemem.evaluation.canonical import canonical_json_bytes

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PRIMARY_METHODS = (
    "independent_fifo",
    "independent_lru",
    "independent_lrua",
    "private_progressive_size_aware",
    "private_progressive_separable_rate",
    "shared_packet_plain_greedy",
    "cts_style_static",
    "vb_lora_style_static",
    "share_style_online",
    "dreamcache_feature_cache",
    "per_concept_lora",
)
Status = Literal["faithful", "incompatible", "failed"]


class FidelityAuditError(RuntimeError):
    """Raised when fidelity evidence is incomplete or provenance-mismatched."""


class SearchBudgetError(RuntimeError):
    """Raised when search consumes unregistered data, trials, or GPU time."""


class BaselineAuditBlocked(RuntimeError):
    """Raised when a primary scientific comparison is not eligible."""


class FidelityCase(BaseModel):
    model_config = _MODEL_CONFIG

    id: str = Field(min_length=1)
    comparator: Literal[
        "sha256_exact",
        "boolean_true",
        "exception_type_exact",
        "value_exact",
        "float64_allclose",
        "float32_allclose",
        "bf16_allclose",
    ]
    atol: NonNegativeFloat | None = None
    rtol: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_tolerance(self) -> FidelityCase:
        approximate = self.comparator.endswith("_allclose")
        if approximate != (self.atol is not None and self.rtol is not None):
            raise ValueError("only allclose fidelity cases require atol and rtol")
        return self


class FidelityPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    reports_are_algorithmic_contracts_not_published_metric_reproductions: Literal[True]
    common_required_cases: tuple[FidelityCase, ...]
    method_cases: dict[str, tuple[FidelityCase, ...]]
    sana_primary_required_methods: tuple[str, ...]
    allowed_statuses: tuple[Literal["faithful", "incompatible", "failed"], ...]

    @model_validator(mode="after")
    def validate_policy(self) -> FidelityPolicy:
        if self.sana_primary_required_methods != _PRIMARY_METHODS:
            raise ValueError("SANA primary fidelity set differs from the locked controls")
        if self.allowed_statuses != ("faithful", "incompatible", "failed"):
            raise ValueError("fidelity statuses differ from the locked policy")
        common_ids = tuple(row.id for row in self.common_required_cases)
        if not common_ids or len(common_ids) != len(set(common_ids)):
            raise ValueError("common fidelity case ids must be non-empty and unique")
        for method_id, cases in self.method_cases.items():
            ids = tuple(row.id for row in cases)
            if not method_id or not ids or len(ids) != len(set(ids)):
                raise ValueError("method fidelity case ids must be non-empty and unique")
            if set(ids) & set(common_ids):
                raise ValueError("method fidelity cases duplicate a common case")
        if not set(self.sana_primary_required_methods) <= set(self.method_cases):
            raise ValueError("every primary SANA method requires a method-specific case")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()

    def case_ids_for(self, method_id: str) -> tuple[str, ...]:
        try:
            method = self.method_cases[method_id]
        except KeyError as error:
            raise FidelityAuditError(f"method has no fidelity policy: {method_id}") from error
        return tuple(row.id for row in self.common_required_cases) + tuple(
            row.id for row in method
        )


class FidelityMeasurement(BaseModel):
    model_config = _MODEL_CONFIG

    case_id: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool


class IncompatibilityRecord(BaseModel):
    model_config = _MODEL_CONFIG

    failing_case_id: str
    observed_mismatch: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backbone_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    attempted_bridge_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    technical_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FidelityReport(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    method_id: str = Field(min_length=1)
    factory_import_path: str = Field(min_length=1)
    concrete_factory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_sha256s: tuple[str, ...]
    source_revisions: dict[str, str]
    backbone_id: Literal["sana_1_5_1_6b", "sdxl_1_0"]
    backbone_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    case_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measurements: tuple[FidelityMeasurement, ...]
    state_roundtrip_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_budget_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_access_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthetic_provider_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Status
    incompatibility_record: IncompatibilityRecord | None = None
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_shape(self) -> FidelityReport:
        if self.source_record_sha256s != tuple(sorted(set(self.source_record_sha256s))):
            raise ValueError("fidelity source records must be sorted and unique")
        ids = tuple(row.case_id for row in self.measurements)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("fidelity measurements must be non-empty and unique")
        if (self.status == "incompatible") != (self.incompatibility_record is not None):
            raise ValueError("only incompatible fidelity requires a technical record")
        return self

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("report_sha256")
        return canonical_json_bytes(payload)


def load_fidelity_policy(path: Path) -> FidelityPolicy:
    try:
        return FidelityPolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise FidelityAuditError(f"invalid fidelity policy: {error}") from error


def seal_fidelity_report(report: FidelityReport) -> FidelityReport:
    if report.report_sha256 != "0" * 64:
        raise FidelityAuditError("only an unsealed fidelity report can be sealed")
    return report.model_copy(
        update={"report_sha256": hashlib.sha256(report.semantic_bytes).hexdigest()}
    )


def audit_fidelity_report(
    report: FidelityReport,
    policy: FidelityPolicy,
    source_inventory: SourceInventory,
    runtime_registry: RuntimeRegistryLock,
) -> FidelityReport:
    """Validate exact source, factory, case, backbone, and raw-output evidence."""

    if hashlib.sha256(report.semantic_bytes).hexdigest() != report.report_sha256:
        raise FidelityAuditError("fidelity report hash mismatch")
    if report.case_policy_sha256 != policy.sha256:
        raise FidelityAuditError("fidelity report uses a different case policy")
    registry = {row.method_id: row for row in runtime_registry.entries}
    factory = registry.get(report.method_id)
    if factory is None:
        raise FidelityAuditError("fidelity method is absent from the runtime registry")
    if (
        report.factory_import_path != factory.import_path
        or report.concrete_factory_sha256 != factory.source_sha256
    ):
        raise FidelityAuditError("fidelity concrete factory provenance changed")
    source_records = [
        row for row in source_inventory.records if report.method_id in row.methods
    ]
    expected_source_hashes = tuple(sorted(row.record_sha256 for row in source_records))
    if report.source_record_sha256s != expected_source_hashes:
        raise FidelityAuditError("fidelity source inventory provenance changed")
    expected_revisions = {row.source_id: row.source_revision for row in source_records}
    if report.source_revisions != expected_revisions:
        raise FidelityAuditError("fidelity source revisions changed")
    expected_cases = policy.case_ids_for(report.method_id)
    observed_cases = tuple(row.case_id for row in report.measurements)
    if observed_cases != expected_cases:
        raise FidelityAuditError("fidelity case set or order differs from policy")
    failures = tuple(row.case_id for row in report.measurements if not row.passed)
    if report.status == "faithful" and failures:
        raise FidelityAuditError("faithful fidelity report contains a failed case")
    if report.status == "failed" and not failures:
        raise FidelityAuditError("failed fidelity report contains no failed case")
    if report.status == "incompatible":
        record = report.incompatibility_record
        if record is None or record.failing_case_id not in failures:
            raise FidelityAuditError("incompatibility record does not bind a failed case")
        if record.backbone_revision != report.backbone_revision:
            raise FidelityAuditError("incompatibility backbone revision changed")
        if record.source_revision not in set(report.source_revisions.values()):
            raise FidelityAuditError("incompatibility source revision changed")
    return report


class FidelityMatrix(BaseModel):
    model_config = _MODEL_CONFIG

    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_methods: tuple[str, ...]
    reports: tuple[FidelityReport, ...]

    @model_validator(mode="after")
    def validate_reports(self) -> FidelityMatrix:
        identities = tuple((row.method_id, row.backbone_id) for row in self.reports)
        if len(identities) != len(set(identities)):
            raise ValueError("fidelity matrix repeats a method/backbone report")
        return self

    def unfaithful_required(
        self,
        backbone_id: Literal["sana_1_5_1_6b", "sdxl_1_0"],
    ) -> tuple[str, ...]:
        statuses = {
            (row.method_id, row.backbone_id): row.status for row in self.reports
        }
        return tuple(
            method_id
            for method_id in self.required_methods
            if statuses.get((method_id, backbone_id)) != "faithful"
        )


class BackbonePlan(BaseModel):
    model_config = _MODEL_CONFIG

    primary_backbone: Literal["sana_1_5_1_6b"] = "sana_1_5_1_6b"
    contextual_backbones: tuple[Literal["sdxl_1_0"], ...] = ("sdxl_1_0",)
    contextual_promotion_allowed: Literal[False] = False

    def require_primary(self, backbone_id: str) -> Literal["sana_1_5_1_6b"]:
        if backbone_id != self.primary_backbone:
            raise BaselineAuditBlocked(
                f"contextual_backbone_not_primary_eligible:{backbone_id}"
            )
        return self.primary_backbone


def resolve_primary_backbone_plan(matrix: FidelityMatrix) -> BackbonePlan:
    missing = matrix.unfaithful_required("sana_1_5_1_6b")
    if missing:
        raise BaselineAuditBlocked(
            f"required_sana_control_unfaithful:{'.'.join(missing)}"
        )
    return BackbonePlan()


class SearchSelector(BaseModel):
    model_config = _MODEL_CONFIG

    endpoint: Literal["request_weighted_identity"]
    prompt_constraint_source: Literal["evaluation_lock"]
    tie_break: tuple[
        Literal["lower_online_bytes", "lower_insert_latency", "method_id"], ...
    ]


class SearchPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    split: Literal["validation"]
    primary_backbone: Literal["sana_1_5_1_6b"]
    contextual_backbones_are_not_search_candidates: Literal[True]
    maximum_trials_per_method: Literal[24]
    maximum_gpu_hours_per_method: float
    failed_trials_consume_budget: Literal[True]
    selector: SearchSelector
    spaces: dict[str, dict[str, tuple[Any, ...]]]

    @model_validator(mode="after")
    def validate_search(self) -> SearchPolicy:
        if self.maximum_gpu_hours_per_method != 48.0:
            raise ValueError("search GPU-hour limit must be exactly 48")
        if self.selector.tie_break != (
            "lower_online_bytes",
            "lower_insert_latency",
            "method_id",
        ):
            raise ValueError("search tie break differs from the lock")
        if set(self.spaces) != set(_PRIMARY_METHODS):
            raise ValueError("search spaces differ from primary SANA controls")
        for method_id, dimensions in self.spaces.items():
            if not dimensions or any(not values for values in dimensions.values()):
                raise ValueError(f"search space is empty: {method_id}")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


def load_search_policy(path: Path) -> SearchPolicy:
    try:
        return SearchPolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SearchBudgetError(f"invalid search policy: {error}") from error


def deterministic_search_cells(
    policy: SearchPolicy,
    method_id: str,
) -> tuple[dict[str, Any], ...]:
    """Return all small grids or a deterministic balanced 24-cell categorical design."""

    try:
        dimensions = policy.spaces[method_id]
    except KeyError as error:
        raise SearchBudgetError(f"method has no search space: {method_id}") from error
    names = tuple(sorted(dimensions))
    values = tuple(dimensions[name] for name in names)
    all_cells = tuple(
        {name: value for name, value in zip(names, row, strict=True)}
        for row in itertools.product(*values)
    )
    maximum = policy.maximum_trials_per_method
    if len(all_cells) <= maximum:
        return all_cells
    ranked = sorted(
        all_cells,
        key=lambda cell: hashlib.sha256(
            policy.sha256.encode("ascii")
            + b"\0"
            + method_id.encode("utf-8")
            + b"\0"
            + canonical_json_bytes(cell)
        ).digest(),
    )
    selected: list[dict[str, Any]] = []
    remaining = list(ranked)
    while remaining and len(selected) < maximum:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        def diversity(cell: dict[str, Any]) -> tuple[int, bytes]:
            minimum = min(
                sum(cell[name] != prior[name] for name in names) for prior in selected
            )
            digest = hashlib.sha256(canonical_json_bytes(cell)).digest()
            return minimum, bytes(255 - value for value in digest)

        chosen = max(remaining, key=diversity)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


class SearchTrial(BaseModel):
    model_config = _MODEL_CONFIG

    attempt_id: str = Field(min_length=1)
    method_id: str
    backbone_id: Literal["sana_1_5_1_6b", "sdxl_1_0"]
    backbone_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    configuration: dict[str, Any]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: Literal["train", "validation", "final_test"]
    shared_input_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at_utc: AwareDatetime
    ended_at_utc: AwareDatetime
    gpu_sku: str = Field(min_length=1)
    gpu_hours: NonNegativeFloat
    exit_status: Literal["passed", "failed"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_trial(self) -> SearchTrial:
        if self.ended_at_utc <= self.started_at_utc:
            raise ValueError("search trial end must follow start")
        expected = hashlib.sha256(canonical_json_bytes(self.configuration)).hexdigest()
        if expected != self.configuration_sha256:
            raise ValueError("search trial configuration hash mismatch")
        return self


class SearchLedger(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_id: str
    rows: tuple[SearchTrial, ...]
    ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("ledger_sha256")
        return canonical_json_bytes(payload)


class SearchAudit(BaseModel):
    model_config = _MODEL_CONFIG

    method_id: str
    trials: PositiveInt
    gpu_hours: NonNegativeFloat
    failed_trials: int = Field(ge=0)
    seen_splits: frozenset[str]


def audit_search_ledger(ledger: SearchLedger, policy: SearchPolicy) -> SearchAudit:
    if hashlib.sha256(ledger.semantic_bytes).hexdigest() != ledger.ledger_sha256:
        raise SearchBudgetError("search ledger hash mismatch")
    if ledger.policy_sha256 != policy.sha256:
        raise SearchBudgetError("search ledger policy hash mismatch")
    if not ledger.rows:
        raise SearchBudgetError("search ledger contains no attempted trials")
    if len(ledger.rows) > policy.maximum_trials_per_method:
        raise SearchBudgetError("search trial limit exceeded")
    attempts = tuple(row.attempt_id for row in ledger.rows)
    if len(attempts) != len(set(attempts)):
        raise SearchBudgetError("search ledger repeats an attempt id")
    for row in ledger.rows:
        if row.method_id != ledger.method_id:
            raise SearchBudgetError("search trial method differs from its ledger")
        if row.backbone_id != policy.primary_backbone:
            raise SearchBudgetError("search backbone must be sana_1_5_1_6b")
        if row.split != policy.split:
            raise SearchBudgetError("search may consume validation only")
        if row.configuration not in deterministic_search_cells(policy, ledger.method_id):
            raise SearchBudgetError("search trial configuration is outside the frozen design")
    gpu_hours = math.fsum(float(row.gpu_hours) for row in ledger.rows)
    if gpu_hours > policy.maximum_gpu_hours_per_method + 1e-12:
        raise SearchBudgetError("search GPU-hour limit exceeded")
    return SearchAudit(
        method_id=ledger.method_id,
        trials=len(ledger.rows),
        gpu_hours=gpu_hours,
        failed_trials=sum(row.exit_status == "failed" for row in ledger.rows),
        seen_splits=frozenset(row.split for row in ledger.rows),
    )


class PrimaryMethodEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    method_id: str
    backbone_id: Literal["sana_1_5_1_6b"]
    backbone_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    factory_importable: Literal[True]
    concrete_factory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_sha256s: tuple[str, ...]
    fidelity_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_roundtrip_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_budget_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_access_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthetic_provider_contract_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineFidelityAuditReceipt(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass"] = "pass"
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_method_ids: tuple[str, ...]
    primary_methods: tuple[PrimaryMethodEvidence, ...]
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("receipt_sha256")
        return canonical_json_bytes(payload)


def build_fidelity_audit_receipt(
    policy: FidelityPolicy,
    matrix: FidelityMatrix,
    registry: RuntimeRegistryLock,
) -> BaselineFidelityAuditReceipt:
    resolve_primary_backbone_plan(matrix)
    by_method = {
        row.method_id: row
        for row in matrix.reports
        if row.backbone_id == "sana_1_5_1_6b"
    }
    factory_by_method = {row.method_id: row for row in registry.entries}
    evidence: list[PrimaryMethodEvidence] = []
    for method_id in policy.sana_primary_required_methods:
        report = by_method[method_id]
        factory = factory_by_method[method_id]
        evidence.append(
            PrimaryMethodEvidence(
                method_id=method_id,
                backbone_id="sana_1_5_1_6b",
                backbone_revision=report.backbone_revision,
                factory_importable=True,
                concrete_factory_sha256=factory.source_sha256,
                source_record_sha256s=report.source_record_sha256s,
                fidelity_report_sha256=report.report_sha256,
                state_roundtrip_report_sha256=report.state_roundtrip_report_sha256,
                byte_budget_report_sha256=report.byte_budget_report_sha256,
                causal_access_report_sha256=report.causal_access_report_sha256,
                synthetic_provider_contract_report_sha256=(
                    report.synthetic_provider_contract_report_sha256
                ),
            )
        )
    provisional = BaselineFidelityAuditReceipt(
        policy_sha256=policy.sha256,
        runtime_registry_sha256=registry.registry_sha256,
        primary_method_ids=policy.sana_primary_required_methods,
        primary_methods=tuple(evidence),
        receipt_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={"receipt_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )


__all__ = [
    "BackbonePlan",
    "BaselineAuditBlocked",
    "BaselineFidelityAuditReceipt",
    "FidelityAuditError",
    "FidelityCase",
    "FidelityMatrix",
    "FidelityMeasurement",
    "FidelityPolicy",
    "FidelityReport",
    "IncompatibilityRecord",
    "PrimaryMethodEvidence",
    "SearchAudit",
    "SearchBudgetError",
    "SearchLedger",
    "SearchPolicy",
    "SearchSelector",
    "SearchTrial",
    "audit_fidelity_report",
    "audit_search_ledger",
    "build_fidelity_audit_receipt",
    "deterministic_search_cells",
    "load_fidelity_policy",
    "load_search_policy",
    "resolve_primary_backbone_plan",
    "seal_fidelity_report",
]
