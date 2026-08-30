"""Strict, content-addressed freeze for the scientific evaluation protocol."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import canonical_json_bytes, semantic_sha256
from ratemem.evaluation.dataset_lock import DatasetLock
from ratemem.evaluation.types import Sha256

_LOCK_CONFIG = ConfigDict(extra="forbid", frozen=True)
_DRAFT_CONFIG = ConfigDict(extra="forbid", validate_assignment=True)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset({"train", "validation", "final_test"})
_CLAIMS = frozenset(
    {
        "shared_packet_representation",
        "causal_packet_allocator",
        "allocator_guarantee",
        "optimization_free_tradeoff",
        "autonomous_lookup",
    }
)
_METRIC_FORMULAS = {
    "identity": "identity_mean_v1",
    "prompt": "prompt_mean_v1",
    "request_weighted_identity": "request_weighted_identity_v1",
    "request_weighted_utility": "equal_weight_identity_prompt_v1",
    "retention_auc": "normalized_event_trapezoid_v1",
    "active_state_drift": "acquisition_delta_v1",
    "maximum_active_degradation": "maximum_acquisition_drop_v1",
    "oracle_regret": "future_oracle_utility_gap_v1",
    "lookup_aurc": "lookup_risk_coverage_v1",
    "diversity": "thresholded_conditional_diversity_v1",
}
_POSTLOCK_RECEIPTS = (
    "ratemem_shared_input_bundle",
    "method_train_receipts",
    "method_search_ledgers",
    "search_budget_compliance",
)


class EvaluationLockError(ValueError):
    """Raised when a protocol draft is incomplete, mutable, or inconsistent."""


def _identifier(value: str, field: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical identifier")
    return value


class AllocatorBoundaryLock(BaseModel):
    model_config = _LOCK_CONFIG

    fixture_id: Literal["four_concepts_eight_packets_each_v1"]
    proposal_count: Literal[32]
    prescreen_input_count: Literal[32]
    allocator_input_count: Literal[24]
    deterministic_tie_break: Literal[
        "lexicographically_larger_packet_id_wins"
    ]


class ClaimLock(BaseModel):
    model_config = _LOCK_CONFIG

    primary_endpoint: str
    inference_unit: str
    required_controls: tuple[str, ...]
    pass_rule: str
    constraint_metric: str | None = None
    ground_set_scope: Literal[
        "causal_singleton_density_prescreen_C_t_max24"
    ] | None = None
    allocator_boundary: AllocatorBoundaryLock | None = None

    @field_validator(
        "primary_endpoint",
        "inference_unit",
        "pass_rule",
        "constraint_metric",
    )
    @classmethod
    def validate_identifiers(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "claim field")
        return _identifier(value, str(field_name))

    @field_validator("required_controls")
    @classmethod
    def validate_controls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("required controls must be non-empty and unique")
        for item in value:
            _identifier(item, "required control")
        return value


EvaluatorRole = Literal[
    "training_loss",
    "filtering",
    "headline_identity",
    "headline_prompt",
    "diversity",
]


class EvaluatorLock(BaseModel):
    model_config = _LOCK_CONFIG

    evaluator_id: str
    repository: AnyUrl
    revision: str
    weights_sha256: Sha256
    preprocessing_id: str
    preprocessing_sha256: Sha256
    roles: frozenset[EvaluatorRole]

    @field_validator("evaluator_id", "preprocessing_id")
    @classmethod
    def validate_id(cls, value: str, info: object) -> str:
        return _identifier(value, str(getattr(info, "field_name", "identifier")))

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if type(value) is not str or _IMMUTABLE_REVISION.fullmatch(value) is None:
            raise ValueError("evaluator revision must be immutable")
        return value

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: frozenset[EvaluatorRole]) -> frozenset[EvaluatorRole]:
        if not value:
            raise ValueError("evaluator must have at least one role")
        return value


class MarginLock(BaseModel):
    model_config = _LOCK_CONFIG

    claim_id: str
    metric_id: str
    value: NonNegativeFloat
    direction: Literal["higher", "lower"]
    source_kind: Literal["published_reliability", "separate_calibration"]
    source_reference: str
    calibration_pool_sha256: Sha256
    calibration_artifact_sha256: Sha256
    used_for_model_selection: Literal[False]

    @field_validator("claim_id", "metric_id", "source_reference")
    @classmethod
    def validate_id(cls, value: str, info: object) -> str:
        return _identifier(value, str(getattr(info, "field_name", "margin field")))


class BudgetCell(BaseModel):
    model_config = _LOCK_CONFIG

    label: Literal["25pct", "50pct", "75pct"]
    fraction: float
    bytes: PositiveInt
    active_set_size: PositiveInt
    independent_cache_ledger_sha256: Sha256

    @field_validator("fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if type(value) is not float or value not in {0.25, 0.5, 0.75}:
            raise ValueError("budget fraction must be exactly 0.25, 0.50, or 0.75")
        return value


class WorkloadLock(BaseModel):
    model_config = _LOCK_CONFIG

    workload_id: str
    request_regime: Literal["uniform", "zipf"]
    exponent: PositiveFloat | None
    update_delete_rate_multiplier: PositiveFloat

    @field_validator("workload_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "workload_id")

    @model_validator(mode="after")
    def validate_exponent(self) -> WorkloadLock:
        if (self.request_regime == "zipf") != (self.exponent is not None):
            raise ValueError("only a Zipf workload requires an exponent")
        return self


class GenerationLock(BaseModel):
    model_config = _LOCK_CONFIG

    backbone_id: Literal["sana_1_5_1_6b"]
    resolution: Literal[1024]
    sampler_id: str
    steps: PositiveInt
    guidance_scale: PositiveFloat
    prompt_seed_pairing: Literal["strict"]
    noise_seed_pairing: Literal["strict"]
    generation_seeds: tuple[NonNegativeInt, ...]

    @field_validator("sampler_id")
    @classmethod
    def validate_sampler(cls, value: str) -> str:
        return _identifier(value, "sampler_id")

    @field_validator("generation_seeds")
    @classmethod
    def validate_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("generation seeds must be non-empty, sorted, and unique")
        return value


class LatencyLock(BaseModel):
    model_config = _LOCK_CONFIG

    hardware_id: str
    device_count: PositiveInt
    warmup_requests: NonNegativeInt
    measured_requests: PositiveInt
    batch_size: Literal[1]
    resolution: Literal[1024]
    sampler_id: str
    steps: PositiveInt

    @field_validator("hardware_id", "sampler_id")
    @classmethod
    def validate_id(cls, value: str, info: object) -> str:
        return _identifier(value, str(getattr(info, "field_name", "latency field")))


class HumanStudyLock(BaseModel):
    model_config = _LOCK_CONFIG

    enabled: bool
    blinded: Literal[True]
    randomized_side: Literal[True]
    minimum_raters: PositiveInt
    attention_checks: Literal[True]


class BootstrapLock(BaseModel):
    model_config = _LOCK_CONFIG

    alpha: float = Field(gt=0.0, lt=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    resamples: PositiveInt
    seed: NonNegativeInt
    multiplicity_method: Literal["holm"]
    minimum_training_seeds: PositiveInt
    prompt_noise_pairing: Literal["strict"]
    strongest_control_selector: Literal["validation_primary_endpoint_v1"]


class PowerLock(BaseModel):
    model_config = _LOCK_CONFIG

    required_units_record_sha256: Sha256
    target_power: float = Field(gt=0.0, lt=1.0)
    two_sided_alpha: float = Field(gt=0.0, lt=1.0)
    minimum_detectable_effect: PositiveFloat
    required_deployment_episodes: PositiveInt


class ApprovalLock(BaseModel):
    model_config = _LOCK_CONFIG

    approval_id: str
    approver_id: str
    role: Literal["protocol_owner", "independent_reviewer"]
    approved_at_utc: AwareDatetime
    approved_policy_sha256: Sha256
    signed_record_sha256: Sha256

    @field_validator("approval_id", "approver_id")
    @classmethod
    def validate_id(cls, value: str, info: object) -> str:
        return _identifier(value, str(getattr(info, "field_name", "approval field")))

    @field_validator("approved_at_utc")
    @classmethod
    def validate_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("approval timestamp must be UTC")
        return value


class EvaluationLockDraft(BaseModel):
    """Mutable compilation record; only freeze_evaluation_lock can seal it."""

    model_config = _DRAFT_CONFIG

    schema_version: Literal["1.0"]
    policy_sha256: Sha256
    dataset_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    baseline_audit_receipt_sha256: Sha256
    comparator_catalog_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report_sha256: Sha256
    search_policy_sha256: Sha256
    trace_manifest_sha256: dict[Literal["train", "validation", "final_test"], Sha256]
    evaluators: list[EvaluatorLock]
    metric_formulas: dict[str, str]
    margins: list[MarginLock]
    budget_cells: list[BudgetCell]
    workload_distributions: list[WorkloadLock]
    generation: GenerationLock
    latency: LatencyLock
    human_study: HumanStudyLock
    claims: dict[str, ClaimLock]
    bootstrap: BootstrapLock
    power: PowerLock
    required_postlock_receipts: list[str]
    approvals: list[ApprovalLock]


class EvaluationLock(BaseModel):
    """Immutable, schema-versioned evaluation protocol and all its dependencies."""

    model_config = _LOCK_CONFIG

    schema_version: Literal["1.0"]
    lock_id: Sha256
    sealed_at_utc: AwareDatetime
    policy_sha256: Sha256
    dataset_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    baseline_audit_receipt_sha256: Sha256
    comparator_catalog_sha256: Sha256
    shared_input_schema_sha256: Sha256
    synthetic_provider_report_sha256: Sha256
    search_policy_sha256: Sha256
    trace_manifest_sha256: dict[Literal["train", "validation", "final_test"], Sha256]
    evaluators: tuple[EvaluatorLock, ...]
    metric_formulas: dict[str, str]
    margins: tuple[MarginLock, ...]
    budget_cells: tuple[BudgetCell, ...]
    workload_distributions: tuple[WorkloadLock, ...]
    generation: GenerationLock
    latency: LatencyLock
    human_study: HumanStudyLock
    claims: dict[str, ClaimLock]
    bootstrap: BootstrapLock
    power: PowerLock
    required_postlock_receipts: tuple[str, ...]
    approvals: tuple[ApprovalLock, ...]

    @field_validator("sealed_at_utc")
    @classmethod
    def validate_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("sealed_at_utc must be UTC")
        return value

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("lock_id")
        payload.pop("sealed_at_utc")
        return canonical_json_bytes(payload)

    @classmethod
    def load(cls, path: Path) -> EvaluationLock:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            lock = cls.model_validate(payload)
        except (OSError, ValidationError, ValueError) as error:
            raise EvaluationLockError(f"invalid evaluation lock: {error}") from error
        if hashlib.sha256(lock.semantic_bytes).hexdigest() != lock.lock_id:
            raise EvaluationLockError("evaluation lock content hash changed")
        return lock


def derive_budget_cells(
    *,
    reference_total_bytes: int,
    active_set_size: int,
    fractions: tuple[float, ...],
    ledger_sha256: str,
) -> tuple[BudgetCell, ...]:
    """Derive exact cache budgets with decimal half-up rounding and ledger binding."""

    if type(reference_total_bytes) is not int or reference_total_bytes <= 0:
        raise TypeError("reference_total_bytes must be a positive exact int")
    if type(active_set_size) is not int or active_set_size <= 0:
        raise TypeError("active_set_size must be a positive exact int")
    if type(ledger_sha256) is not str or _SHA256.fullmatch(ledger_sha256) is None:
        raise ValueError("ledger_sha256 must be lowercase SHA-256")
    if type(fractions) is not tuple:
        raise TypeError("fractions must be an exact tuple")
    labels = {
        Decimal("0.25"): "25pct",
        Decimal("0.5"): "50pct",
        Decimal("0.75"): "75pct",
    }
    decimals = tuple(Decimal(str(value)) for value in fractions)
    if decimals != tuple(sorted(labels)):
        raise ValueError("budget fractions must be exactly 0.25, 0.50, and 0.75")
    return tuple(
        BudgetCell(
            label=cast(Literal["25pct", "50pct", "75pct"], labels[fraction]),
            fraction=float(fraction),
            bytes=int(
                (Decimal(reference_total_bytes) * fraction).quantize(
                    Decimal("1"),
                    rounding=ROUND_HALF_UP,
                )
            ),
            active_set_size=active_set_size,
            independent_cache_ledger_sha256=ledger_sha256,
        )
        for fraction in decimals
    )


def _validate_raw_boundaries(draft: EvaluationLockDraft) -> None:
    guarantee = draft.claims.get("allocator_guarantee")
    if guarantee is None or (
        guarantee.ground_set_scope
        != "causal_singleton_density_prescreen_C_t_max24"
    ):
        raise EvaluationLockError(
            "allocator guarantee ground-set scope must be the locked reduced set"
        )
    boundary = guarantee.allocator_boundary
    if boundary is None or (
        boundary.fixture_id != "four_concepts_eight_packets_each_v1"
        or boundary.proposal_count != 32
        or boundary.prescreen_input_count != 32
        or boundary.allocator_input_count != 24
        or boundary.deterministic_tie_break
        != "lexicographically_larger_packet_id_wins"
    ):
        raise EvaluationLockError("allocator boundary differs from the locked 32-to-24 fixture")
    if "exact_reduced_set_optimum" not in guarantee.required_controls:
        raise EvaluationLockError(
            "allocator guarantee requires exact_reduced_set_optimum"
        )
    if any(
        claim_id != "allocator_guarantee" and claim.allocator_boundary is not None
        for claim_id, claim in draft.claims.items()
    ):
        raise EvaluationLockError("only allocator_guarantee may define an allocator boundary")


def _validate_budget_cells(cells: list[BudgetCell]) -> None:
    expected = (("25pct", 0.25), ("50pct", 0.5), ("75pct", 0.75))
    observed = tuple((cell.label, float(cell.fraction)) for cell in cells)
    if observed != expected:
        raise EvaluationLockError("evaluation budgets must contain ordered 25/50/75 cells")
    if len({cell.active_set_size for cell in cells}) != 1 or len(
        {cell.independent_cache_ledger_sha256 for cell in cells}
    ) != 1:
        raise EvaluationLockError("budget cells must bind one active set and byte ledger")
    inferred_totals = {
        Decimal(cell.bytes) / Decimal(str(cell.fraction)) for cell in cells
    }
    if len(inferred_totals) != 1:
        raise EvaluationLockError("budget bytes are not derived from one reference ledger total")


def freeze_evaluation_lock(
    draft: EvaluationLockDraft,
    *,
    sealed_at_utc: datetime | None = None,
) -> EvaluationLock:
    """Validate every prespecified invariant and content-address a protocol draft."""

    if type(draft) is not EvaluationLockDraft:
        raise TypeError("draft must be an exact EvaluationLockDraft")
    _validate_raw_boundaries(draft)
    for evaluator in draft.evaluators:
        if _IMMUTABLE_REVISION.fullmatch(evaluator.revision) is None:
            raise EvaluationLockError("evaluator revision must be immutable")
    for margin in draft.margins:
        artifact = margin.calibration_artifact_sha256
        if type(artifact) is not str or _SHA256.fullmatch(artifact) is None:
            raise EvaluationLockError("every margin requires a calibration artifact")
        if margin.used_for_model_selection is not False:
            raise EvaluationLockError("margin calibration cannot be used for model selection")
    try:
        clean = EvaluationLockDraft.model_validate(draft.model_dump(mode="python"))
    except (ValidationError, ValueError) as error:
        raise EvaluationLockError(f"invalid evaluation-lock draft: {error}") from error

    if set(clean.trace_manifest_sha256) != _SPLITS:
        raise EvaluationLockError("trace commitments must cover train, validation, and final_test")
    if set(clean.claims) != _CLAIMS:
        raise EvaluationLockError("evaluation lock must contain the exact claim registry")
    if clean.metric_formulas != _METRIC_FORMULAS:
        raise EvaluationLockError("metric formula registry differs from the prespecification")
    independent_identity = any(
        "headline_identity" in evaluator.roles
        and "training_loss" not in evaluator.roles
        for evaluator in clean.evaluators
    )
    if not independent_identity:
        raise EvaluationLockError("an independent headline identity evaluator is required")
    all_roles = {role for evaluator in clean.evaluators for role in evaluator.roles}
    if not {"headline_prompt", "diversity"} <= all_roles:
        raise EvaluationLockError("independent prompt and diversity evaluators are required")
    evaluator_ids = [evaluator.evaluator_id for evaluator in clean.evaluators]
    if len(evaluator_ids) != len(set(evaluator_ids)):
        raise EvaluationLockError("evaluator ids must be unique")
    margin_claims = {margin.claim_id for margin in clean.margins}
    if not clean.margins or not margin_claims <= set(clean.claims):
        raise EvaluationLockError("margin claims must reference the frozen claim registry")
    _validate_budget_cells(clean.budget_cells)

    regimes = {workload.request_regime: workload for workload in clean.workload_distributions}
    if set(regimes) != {"uniform", "zipf"} or len(clean.workload_distributions) != 2:
        raise EvaluationLockError("workloads must contain exactly uniform and Zipf regimes")
    if regimes["uniform"].exponent is not None or regimes["zipf"].exponent is None:
        raise EvaluationLockError("request-regime exponents are invalid")
    if (
        clean.latency.resolution != clean.generation.resolution
        or clean.latency.sampler_id != clean.generation.sampler_id
        or clean.latency.steps != clean.generation.steps
        or clean.latency.batch_size != 1
    ):
        raise EvaluationLockError("latency workload differs from locked generation settings")
    if clean.bootstrap.minimum_training_seeds < 3:
        raise EvaluationLockError("at least three training seeds are required")
    if abs(clean.bootstrap.confidence_level - (1.0 - clean.bootstrap.alpha)) > 1e-12:
        raise EvaluationLockError("bootstrap confidence level and alpha are inconsistent")
    if (
        clean.power.target_power != 0.80
        or clean.power.two_sided_alpha != 0.05
        or clean.power.minimum_detectable_effect != 0.03
    ):
        raise EvaluationLockError("power targets differ from the prespecification")
    if tuple(clean.required_postlock_receipts) != _POSTLOCK_RECEIPTS:
        raise EvaluationLockError("post-lock receipt requirements differ from the policy")
    approval_roles = {approval.role for approval in clean.approvals}
    if approval_roles != {"protocol_owner", "independent_reviewer"} or len(
        clean.approvals
    ) != 2:
        raise EvaluationLockError("two independent evaluation approvals are required")
    if len({approval.approver_id for approval in clean.approvals}) != 2:
        raise EvaluationLockError("evaluation approvals must have distinct approvers")
    if any(
        approval.approved_policy_sha256 != clean.policy_sha256
        for approval in clean.approvals
    ):
        raise EvaluationLockError("approval does not bind the selected policy")

    timestamp = sealed_at_utc if sealed_at_utc is not None else datetime.now(UTC)
    if not isinstance(timestamp, datetime) or timestamp.utcoffset() != timedelta(0):
        raise EvaluationLockError("sealed_at_utc must be an aware UTC datetime")
    semantic_payload = clean.model_dump(mode="json")
    lock_id = hashlib.sha256(canonical_json_bytes(semantic_payload)).hexdigest()
    return EvaluationLock(
        lock_id=lock_id,
        sealed_at_utc=timestamp,
        **clean.model_dump(mode="python"),
    )


def require_scientific_training_lock(
    dataset_lock: Path,
    evaluation_lock: Path,
    requested_split: str,
) -> None:
    """Fail closed unless scientific training uses the train split and matching locks."""

    if type(requested_split) is not str or requested_split != "train":
        raise EvaluationLockError("scientific training accepts only the train split")
    try:
        dataset_payload = yaml.safe_load(dataset_lock.read_text(encoding="utf-8"))
        locked_dataset = DatasetLock.model_validate(dataset_payload)
    except (OSError, ValidationError, ValueError) as error:
        raise EvaluationLockError(f"invalid dataset lock: {error}") from error
    if semantic_sha256(locked_dataset.model_dump(mode="json")) != locked_dataset.lock_id:
        raise EvaluationLockError("dataset lock content hash changed")
    locked_evaluation = EvaluationLock.load(evaluation_lock)
    if locked_evaluation.dataset_lock_sha256 != locked_dataset.lock_id:
        raise EvaluationLockError("evaluation lock does not bind the selected dataset lock")
    if len(locked_evaluation.approvals) != 2:
        raise EvaluationLockError("evaluation lock is unsigned")


__all__ = [
    "AllocatorBoundaryLock",
    "ApprovalLock",
    "BootstrapLock",
    "BudgetCell",
    "ClaimLock",
    "EvaluationLock",
    "EvaluationLockDraft",
    "EvaluationLockError",
    "EvaluatorLock",
    "GenerationLock",
    "HumanStudyLock",
    "LatencyLock",
    "MarginLock",
    "PowerLock",
    "WorkloadLock",
    "derive_budget_cells",
    "freeze_evaluation_lock",
    "require_scientific_training_lock",
]
