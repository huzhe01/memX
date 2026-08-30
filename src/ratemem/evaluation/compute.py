"""Fail-closed authorization and cost primitives for scientific compute."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, cast

import yaml  # type: ignore[import-untyped]
from filelock import FileLock
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256
from ratemem.evaluation.types import GitCommit, PhaseId, ScientificProfile, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_MUTABLE_CONFIG = ConfigDict(extra="forbid", validate_assignment=False)
_WORKSPACE_ID = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"),
]
_MONEY = Annotated[
    Decimal,
    Field(ge=Decimal("0.00"), decimal_places=2),
]
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_AUTHORIZATION_TTL = timedelta(minutes=15)


class ScientificComputeDenied(PermissionError):
    """Raised before a provider invocation when scientific authority is absent."""


class BaselineFidelityPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    authorization_scope: Literal["baseline_fidelity"]
    provider: Literal["modal"]
    allowed_input_roles: tuple[Literal["held_in", "dedicated_calibration"], ...]
    forbidden_input_roles: tuple[Literal["validation", "final_test"], ...]
    forbid_model_selection: Literal[True]
    forbid_claim_quality_metrics: Literal[True]
    forbid_learned_ratemem_dictionary: Literal[True]
    require_method_cpu_gate: Literal[False]
    require_dataset_lock: Literal[True]
    require_baseline_requirements: Literal[True]
    require_comparator_catalog: Literal[True]
    require_fidelity_policy: Literal[True]
    require_source_inventory: Literal[True]
    require_clean_commit_and_diff: Literal[True]
    workspace_selection: Literal["explicit_operator_file"]
    profile_prefix: Literal["ratemem-scientific-"]
    automatic_workspace_discovery: Literal[False]
    automatic_workspace_reuse: Literal[False]
    automatic_workspace_rotation: Literal[False]
    automatic_workspace_fallback: Literal[False]
    outer_workspace_usage_budget_usd: _MONEY
    internal_reservation_limit_usd: _MONEY
    aggregate_ledger: Path
    reservation_formula: Literal[
        "known_usage_plus_all_pending_worst_case_plus_new_phase_bound"
    ]
    one_phase_per_authorization: Literal[True]
    one_launch_per_reservation: Literal[True]
    require_reconciliation_before_next_phase: Literal[True]

    @model_validator(mode="after")
    def validate_exact_policy(self) -> BaselineFidelityPolicy:
        if self.allowed_input_roles != ("held_in", "dedicated_calibration"):
            raise ValueError("baseline fidelity input roles differ from policy")
        if self.forbidden_input_roles != ("validation", "final_test"):
            raise ValueError("baseline fidelity forbidden roles differ from policy")
        if self.outer_workspace_usage_budget_usd != Decimal("28.00"):
            raise ValueError("outer workspace budget must be exactly USD 28.00")
        if self.internal_reservation_limit_usd != Decimal("27.00"):
            raise ValueError("internal reservation limit must be exactly USD 27.00")
        return self


class WorkspaceSelection(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    workspace_id: _WORKSPACE_ID
    explicit_profile: ScientificProfile
    selected_at_utc: AwareDatetime
    operator_file_sha256: Sha256
    declared_outer_budget_usd: _MONEY | None = None
    declared_known_usage_usd: _MONEY | None = None


class WorkspaceSnapshot(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    workspace_id: _WORKSPACE_ID
    explicit_profile: ScientificProfile
    provider: Literal["modal"]
    outer_budget_usd: _MONEY
    known_usage_usd: _MONEY
    budget_evidence_sha256: Sha256
    observed_at_utc: AwareDatetime
    snapshot_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("snapshot_sha256")
        return canonical_json_bytes(payload)

    @classmethod
    def create(
        cls,
        *,
        workspace_id: str,
        explicit_profile: str,
        provider: str,
        outer_budget_usd: str,
        known_usage_usd: str,
        budget_evidence_sha256: str,
        observed_at_utc: datetime,
    ) -> WorkspaceSnapshot:
        provisional = cls.model_validate(
            {
                "schema_version": "1.0",
                "workspace_id": workspace_id,
                "explicit_profile": explicit_profile,
                "provider": provider,
                "outer_budget_usd": outer_budget_usd,
                "known_usage_usd": known_usage_usd,
                "budget_evidence_sha256": budget_evidence_sha256,
                "observed_at_utc": observed_at_utc,
                "snapshot_sha256": "0" * 64,
            }
        )
        return provisional.model_copy(
            update={
                "snapshot_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
            }
        )


class BaselineFidelityPhaseRequest(BaseModel):
    model_config = _MUTABLE_CONFIG

    schema_version: Literal["1.0"]
    scope: str
    phase_id: PhaseId
    input_role: str
    input_manifest_sha256: Sha256
    job_spec_sha256: Sha256
    source_revision: str
    source_archive_sha256: Sha256
    git_commit: GitCommit
    clean_diff_sha256: Sha256
    payload_references: list[str]
    selection_fields: list[str]
    claim_metric_fields: list[str]


class BaselineFidelityBindings(BaseModel):
    model_config = _MODEL_CONFIG

    dataset_lock_sha256: Sha256
    baseline_requirements_sha256: Sha256
    comparator_catalog_sha256: Sha256
    fidelity_policy_sha256: Sha256
    source_inventory_sha256: Sha256
    git_commit: GitCommit
    clean_diff_sha256: Sha256


class BaselineFidelityAuthorization(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    scope: Literal["baseline_fidelity"]
    phase_id: PhaseId
    workspace_id: _WORKSPACE_ID
    explicit_profile: ScientificProfile
    dataset_lock_sha256: Sha256
    baseline_requirements_sha256: Sha256
    comparator_catalog_sha256: Sha256
    fidelity_policy_sha256: Sha256
    source_inventory_sha256: Sha256
    source_revision: GitCommit
    source_archive_sha256: Sha256
    git_commit: GitCommit
    clean_diff_sha256: Sha256
    input_role: Literal["held_in", "dedicated_calibration"]
    input_manifest_sha256: Sha256
    job_spec_sha256: Sha256
    workspace_snapshot_sha256: Sha256
    issued_at_utc: AwareDatetime
    expires_at_utc: AwareDatetime

    @property
    def authorization_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()

    @property
    def recomputed_sha256(self) -> str:
        return self.authorization_sha256


class BaselineFidelityCostReservation(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    scope: Literal["baseline_fidelity"]
    phase_id: PhaseId
    workspace_id: _WORKSPACE_ID
    authorization_sha256: Sha256
    workspace_snapshot_sha256: Sha256
    known_usage_usd: _MONEY
    pending_worst_case_usd: _MONEY
    new_phase_bound_usd: _MONEY
    reserved_total_usd: _MONEY
    status: Literal["pending"]
    reserved_at_utc: AwareDatetime
    expires_at_utc: AwareDatetime
    reservation_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("reservation_sha256")
        return canonical_json_bytes(payload)


class ConsumedPermit(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    scope: Literal["baseline_fidelity"]
    phase_id: PhaseId
    workspace_id: _WORKSPACE_ID
    authorization_sha256: Sha256
    reservation_sha256: Sha256
    consumed_at_utc: AwareDatetime
    provider_invocations_before_consumption: Literal[0]
    launch_receipt_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("launch_receipt_sha256")
        return canonical_json_bytes(payload)


class BaselineFidelityPhaseCostBound(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    scope: Literal["baseline_fidelity"]
    phase_id: PhaseId
    workspace_id: _WORKSPACE_ID
    worst_case_usd: _MONEY
    estimator_revision: GitCommit
    bound_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("bound_sha256")
        return canonical_json_bytes(payload)


class BaselineFidelityReconciliation(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    scope: Literal["baseline_fidelity"]
    phase_id: PhaseId
    workspace_id: _WORKSPACE_ID
    authorization_sha256: Sha256
    reservation_sha256: Sha256
    launch_receipt_sha256: Sha256
    prior_known_usage_usd: _MONEY
    observed_known_usage_usd: _MONEY
    metered_delta_usd: _MONEY
    pending_remaining_usd: _MONEY
    budget_evidence_sha256: Sha256
    reconciled_at_utc: AwareDatetime
    reconciliation_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("reconciliation_sha256")
        return canonical_json_bytes(payload)


def load_baseline_fidelity_policy(path: Path) -> BaselineFidelityPolicy:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return BaselineFidelityPolicy.model_validate(payload)
    except (OSError, ValueError) as error:
        raise ScientificComputeDenied(f"invalid baseline-fidelity policy: {error}") from error


def attest_scientific_workspace(
    selection: WorkspaceSelection,
    budget_evidence: Path,
    policy: BaselineFidelityPolicy,
    *,
    observed_at_utc: datetime | None = None,
) -> WorkspaceSnapshot:
    """Bind operator-declared usage and cap to one immutable evidence file."""

    if (
        selection.declared_outer_budget_usd is None
        or selection.declared_known_usage_usd is None
    ):
        raise ScientificComputeDenied(
            "workspace selection lacks operator-declared cap or known usage"
        )
    if selection.declared_outer_budget_usd != policy.outer_workspace_usage_budget_usd:
        raise ScientificComputeDenied("workspace lacks the exact USD 28.00 outer cap")
    if not budget_evidence.is_file():
        raise ScientificComputeDenied("workspace budget evidence is missing")
    observed = observed_at_utc if observed_at_utc is not None else datetime.now(UTC)
    return WorkspaceSnapshot.create(
        workspace_id=selection.workspace_id,
        explicit_profile=selection.explicit_profile,
        provider=policy.provider,
        outer_budget_usd=str(selection.declared_outer_budget_usd),
        known_usage_usd=str(selection.declared_known_usage_usd),
        budget_evidence_sha256=file_sha256(budget_evidence),
        observed_at_utc=observed,
    )


def _validate_workspace(
    selection: WorkspaceSelection,
    snapshot: WorkspaceSnapshot,
    policy: BaselineFidelityPolicy,
    issued_at_utc: datetime,
) -> None:
    if (
        selection.workspace_id != snapshot.workspace_id
        or selection.explicit_profile != snapshot.explicit_profile
    ):
        raise ScientificComputeDenied("explicit workspace mismatch")
    if not selection.explicit_profile.startswith(policy.profile_prefix):
        raise ScientificComputeDenied("explicit scientific profile prefix mismatch")
    if snapshot.provider != policy.provider:
        raise ScientificComputeDenied("workspace provider mismatch")
    if snapshot.outer_budget_usd != Decimal("28.00"):
        raise ScientificComputeDenied("workspace lacks the exact USD 28.00 outer cap")
    if hashlib.sha256(snapshot.semantic_bytes).hexdigest() != snapshot.snapshot_sha256:
        raise ScientificComputeDenied("workspace snapshot hash mismatch")
    if issued_at_utc - snapshot.observed_at_utc > _AUTHORIZATION_TTL:
        raise ScientificComputeDenied("workspace snapshot is stale")
    if selection.selected_at_utc > issued_at_utc or snapshot.observed_at_utc > issued_at_utc:
        raise ScientificComputeDenied("workspace evidence is from the future")


def authorize_baseline_fidelity(
    selection: WorkspaceSelection,
    snapshot: WorkspaceSnapshot,
    phase: BaselineFidelityPhaseRequest,
    bindings: BaselineFidelityBindings,
    policy: BaselineFidelityPolicy,
    *,
    issued_at_utc: datetime | None = None,
) -> BaselineFidelityAuthorization:
    """Authorize one held-in fidelity phase without reserving or invoking compute."""

    issued = issued_at_utc if issued_at_utc is not None else datetime.now(UTC)
    if issued.utcoffset() != timedelta(0):
        raise ScientificComputeDenied("authorization timestamp must be UTC")
    if phase.scope != "baseline_fidelity":
        raise ScientificComputeDenied("engineering-pilot authorization is forbidden")
    _validate_workspace(selection, snapshot, policy, issued)
    if phase.input_role not in policy.allowed_input_roles:
        raise ScientificComputeDenied("forbidden input role for baseline fidelity")
    lowered_references = "\n".join(phase.payload_references).lower()
    if any(
        token in lowered_references
        for token in ("final_test", "final-test", "final_trace", "final-trace")
    ):
        raise ScientificComputeDenied("final-trace reference is forbidden")
    if "learned" in lowered_references and "dictionary" in lowered_references:
        raise ScientificComputeDenied("learned RateMem dictionary is forbidden")
    if phase.selection_fields:
        raise ScientificComputeDenied("model-selection field is forbidden")
    if phase.claim_metric_fields:
        raise ScientificComputeDenied("claim-quality metric is forbidden")
    if _IMMUTABLE_REVISION.fullmatch(phase.source_revision) is None:
        raise ScientificComputeDenied("immutable source revision is required")
    if phase.git_commit != bindings.git_commit:
        raise ScientificComputeDenied("git commit mismatch")
    if phase.clean_diff_sha256 != bindings.clean_diff_sha256:
        raise ScientificComputeDenied("clean diff mismatch")
    return BaselineFidelityAuthorization(
        schema_version="1.0",
        scope="baseline_fidelity",
        phase_id=phase.phase_id,
        workspace_id=selection.workspace_id,
        explicit_profile=selection.explicit_profile,
        dataset_lock_sha256=bindings.dataset_lock_sha256,
        baseline_requirements_sha256=bindings.baseline_requirements_sha256,
        comparator_catalog_sha256=bindings.comparator_catalog_sha256,
        fidelity_policy_sha256=bindings.fidelity_policy_sha256,
        source_inventory_sha256=bindings.source_inventory_sha256,
        source_revision=phase.source_revision,
        source_archive_sha256=phase.source_archive_sha256,
        git_commit=phase.git_commit,
        clean_diff_sha256=phase.clean_diff_sha256,
        input_role=cast(
            Literal["held_in", "dedicated_calibration"],
            phase.input_role,
        ),
        input_manifest_sha256=phase.input_manifest_sha256,
        job_spec_sha256=phase.job_spec_sha256,
        workspace_snapshot_sha256=snapshot.snapshot_sha256,
        issued_at_utc=issued,
        expires_at_utc=issued + _AUTHORIZATION_TTL,
    )


def _money(value: Decimal, field: str) -> Decimal:
    if type(value) is not Decimal or value < Decimal("0.00"):
        raise ScientificComputeDenied(f"{field} must be a nonnegative Decimal")
    if value.quantize(Decimal("0.01")) != value:
        raise ScientificComputeDenied(f"{field} must have at most two decimal places")
    return value


def reserve_baseline_fidelity_cost(
    authorization: BaselineFidelityAuthorization,
    snapshot: WorkspaceSnapshot,
    *,
    pending_worst_case_usd: Decimal,
    new_phase_bound_usd: Decimal,
    policy: BaselineFidelityPolicy,
    reserved_at_utc: datetime | None = None,
) -> BaselineFidelityCostReservation:
    """Reserve known plus pending plus new cost under the internal USD 27 limit."""

    reserved = reserved_at_utc if reserved_at_utc is not None else datetime.now(UTC)
    if authorization.workspace_id != snapshot.workspace_id:
        raise ScientificComputeDenied("reservation workspace mismatch")
    if authorization.workspace_snapshot_sha256 != snapshot.snapshot_sha256:
        raise ScientificComputeDenied("reservation workspace snapshot mismatch")
    if reserved > authorization.expires_at_utc:
        raise ScientificComputeDenied("baseline-fidelity authorization expired")
    known = _money(snapshot.known_usage_usd, "known usage")
    pending = _money(pending_worst_case_usd, "pending worst case")
    new = _money(new_phase_bound_usd, "new phase bound")
    if new == Decimal("0.00"):
        raise ScientificComputeDenied("new phase bound must be positive")
    total = known + pending + new
    if total > policy.internal_reservation_limit_usd:
        raise ScientificComputeDenied("internal USD 27.00 limit would be exceeded")
    provisional = BaselineFidelityCostReservation(
        schema_version="1.0",
        scope="baseline_fidelity",
        phase_id=authorization.phase_id,
        workspace_id=authorization.workspace_id,
        authorization_sha256=authorization.authorization_sha256,
        workspace_snapshot_sha256=snapshot.snapshot_sha256,
        known_usage_usd=known,
        pending_worst_case_usd=pending,
        new_phase_bound_usd=new,
        reserved_total_usd=total,
        status="pending",
        reserved_at_utc=reserved,
        expires_at_utc=min(
            authorization.expires_at_utc,
            reserved + _AUTHORIZATION_TTL,
        ),
        reservation_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={
            "reservation_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
        }
    )


def _read_cost_ledger(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if type(payload) is not dict:
                raise ValueError("ledger row is not an object")
            rows.append(payload)
        return rows
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ScientificComputeDenied(f"scientific cost ledger is invalid: {error}") from error


def _pending_reservations(
    records: list[dict[str, object]],
    workspace_id: str,
) -> tuple[Decimal, set[str], set[str]]:
    reservations: dict[str, Decimal] = {}
    authorization_hashes: set[str] = set()
    reconciled: set[str] = set()
    for record in records:
        kind = record.get("record_type")
        body = record.get("payload")
        if type(body) is not dict:
            raise ScientificComputeDenied("scientific cost ledger payload is invalid")
        if kind == "reservation":
            reservation = BaselineFidelityCostReservation.model_validate(body)
            if reservation.workspace_id == workspace_id:
                reservations[reservation.reservation_sha256] = (
                    reservation.new_phase_bound_usd
                )
                authorization_hashes.add(reservation.authorization_sha256)
        elif kind == "reconciliation":
            reconciliation = BaselineFidelityReconciliation.model_validate(body)
            if reconciliation.workspace_id == workspace_id:
                reconciled.add(reconciliation.reservation_sha256)
        else:
            raise ScientificComputeDenied("scientific cost ledger record type is invalid")
    pending = sum(
        (
            bound
            for reservation_sha, bound in reservations.items()
            if reservation_sha not in reconciled
        ),
        Decimal("0.00"),
    )
    return pending, authorization_hashes, reconciled


def _append_cost_ledger(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    stream = os.fdopen(descriptor, "ab")
    with stream:
        stream.write(canonical_json_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


def reserve_baseline_fidelity_in_ledger(
    authorization: BaselineFidelityAuthorization,
    snapshot: WorkspaceSnapshot,
    phase_bound: BaselineFidelityPhaseCostBound,
    *,
    policy: BaselineFidelityPolicy,
    ledger_path: Path,
    output_path: Path,
    reserved_at_utc: datetime | None = None,
) -> BaselineFidelityCostReservation:
    """Atomically account for every pending reservation in the shared ledger."""

    if (
        hashlib.sha256(phase_bound.semantic_bytes).hexdigest()
        != phase_bound.bound_sha256
    ):
        raise ScientificComputeDenied("phase cost bound hash mismatch")
    if (
        phase_bound.phase_id != authorization.phase_id
        or phase_bound.workspace_id != authorization.workspace_id
    ):
        raise ScientificComputeDenied("phase cost bound authorization mismatch")
    lock_path = Path(f"{ledger_path}.lock")
    with FileLock(lock_path):
        records = _read_cost_ledger(ledger_path)
        pending, authorization_hashes, _reconciled = _pending_reservations(
            records,
            authorization.workspace_id,
        )
        if authorization.authorization_sha256 in authorization_hashes:
            raise ScientificComputeDenied("authorization already has a reservation")
        reservation = reserve_baseline_fidelity_cost(
            authorization,
            snapshot,
            pending_worst_case_usd=pending,
            new_phase_bound_usd=phase_bound.worst_case_usd,
            policy=policy,
            reserved_at_utc=reserved_at_utc,
        )
        _write_launch_receipt_exclusive(
            output_path,
            canonical_json_bytes(reservation.model_dump(mode="json")) + b"\n",
        )
        try:
            _append_cost_ledger(
                ledger_path,
                {
                    "record_type": "reservation",
                    "payload": reservation.model_dump(mode="json"),
                },
            )
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
        return reservation


def reconcile_baseline_fidelity_in_ledger(
    authorization: BaselineFidelityAuthorization,
    reservation: BaselineFidelityCostReservation,
    launch_receipt: ConsumedPermit,
    current_snapshot: WorkspaceSnapshot,
    *,
    ledger_path: Path,
    output_path: Path,
    reconciled_at_utc: datetime | None = None,
) -> BaselineFidelityReconciliation:
    """Close one pending reservation against newly attested metered usage."""

    if reservation.authorization_sha256 != authorization.authorization_sha256:
        raise ScientificComputeDenied("reconciliation authorization mismatch")
    if launch_receipt.reservation_sha256 != reservation.reservation_sha256:
        raise ScientificComputeDenied("reconciliation launch receipt mismatch")
    if current_snapshot.workspace_id != reservation.workspace_id:
        raise ScientificComputeDenied("reconciliation workspace mismatch")
    if current_snapshot.outer_budget_usd != Decimal("28.00"):
        raise ScientificComputeDenied("workspace lacks the exact USD 28.00 outer cap")
    if current_snapshot.known_usage_usd < reservation.known_usage_usd:
        raise ScientificComputeDenied("reconciled usage cannot move backwards")
    reconciled = (
        reconciled_at_utc if reconciled_at_utc is not None else datetime.now(UTC)
    )
    lock_path = Path(f"{ledger_path}.lock")
    with FileLock(lock_path):
        records = _read_cost_ledger(ledger_path)
        pending, _authorizations, reconciled_hashes = _pending_reservations(
            records,
            reservation.workspace_id,
        )
        if reservation.reservation_sha256 in reconciled_hashes:
            raise ScientificComputeDenied("reservation already reconciled")
        reservation_present = False
        for record in records:
            payload = record.get("payload")
            if (
                record.get("record_type") == "reservation"
                and isinstance(payload, dict)
                and payload.get("reservation_sha256")
                == reservation.reservation_sha256
            ):
                reservation_present = True
                break
        if not reservation_present:
            raise ScientificComputeDenied("reservation is absent from the cost ledger")
        pending_remaining = pending - reservation.new_phase_bound_usd
        provisional = BaselineFidelityReconciliation(
            schema_version="1.0",
            scope="baseline_fidelity",
            phase_id=reservation.phase_id,
            workspace_id=reservation.workspace_id,
            authorization_sha256=authorization.authorization_sha256,
            reservation_sha256=reservation.reservation_sha256,
            launch_receipt_sha256=launch_receipt.launch_receipt_sha256,
            prior_known_usage_usd=reservation.known_usage_usd,
            observed_known_usage_usd=current_snapshot.known_usage_usd,
            metered_delta_usd=(
                current_snapshot.known_usage_usd - reservation.known_usage_usd
            ),
            pending_remaining_usd=max(pending_remaining, Decimal("0.00")),
            budget_evidence_sha256=current_snapshot.budget_evidence_sha256,
            reconciled_at_utc=reconciled,
            reconciliation_sha256="0" * 64,
        )
        result = provisional.model_copy(
            update={
                "reconciliation_sha256": hashlib.sha256(
                    provisional.semantic_bytes
                ).hexdigest()
            }
        )
        _write_launch_receipt_exclusive(
            output_path,
            canonical_json_bytes(result.model_dump(mode="json")) + b"\n",
        )
        try:
            _append_cost_ledger(
                ledger_path,
                {
                    "record_type": "reconciliation",
                    "payload": result.model_dump(mode="json"),
                },
            )
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise
        return result


def _write_launch_receipt_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ScientificComputeDenied("baseline-fidelity permit already consumed") from error
    stream = os.fdopen(descriptor, "wb")
    with stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def require_baseline_fidelity_permit(
    authorization_path: Path,
    reservation_path: Path,
    expected_phase_id: str,
    expected_workspace_id: str,
    launch_receipt_path: Path,
    *,
    consumed_at_utc: datetime | None = None,
) -> ConsumedPermit:
    """Consume authorization and reservation once before any provider invocation."""

    try:
        authorization = BaselineFidelityAuthorization.model_validate_json(
            authorization_path.read_text(encoding="utf-8")
        )
        reservation = BaselineFidelityCostReservation.model_validate_json(
            reservation_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ScientificComputeDenied(f"invalid baseline-fidelity permit: {error}") from error
    if (
        authorization.phase_id != expected_phase_id
        or reservation.phase_id != expected_phase_id
    ):
        raise ScientificComputeDenied("baseline-fidelity phase mismatch")
    if (
        authorization.workspace_id != expected_workspace_id
        or reservation.workspace_id != expected_workspace_id
    ):
        raise ScientificComputeDenied("baseline-fidelity workspace mismatch")
    if reservation.authorization_sha256 != authorization.authorization_sha256:
        raise ScientificComputeDenied("reservation authorization hash mismatch")
    if hashlib.sha256(reservation.semantic_bytes).hexdigest() != reservation.reservation_sha256:
        raise ScientificComputeDenied("reservation content hash mismatch")
    consumed = consumed_at_utc if consumed_at_utc is not None else datetime.now(UTC)
    if consumed > authorization.expires_at_utc or consumed > reservation.expires_at_utc:
        raise ScientificComputeDenied("baseline-fidelity permit expired")
    provisional = ConsumedPermit(
        schema_version="1.0",
        scope="baseline_fidelity",
        phase_id=authorization.phase_id,
        workspace_id=authorization.workspace_id,
        authorization_sha256=authorization.authorization_sha256,
        reservation_sha256=reservation.reservation_sha256,
        consumed_at_utc=consumed,
        provider_invocations_before_consumption=0,
        launch_receipt_sha256="0" * 64,
    )
    receipt = provisional.model_copy(
        update={
            "launch_receipt_sha256": hashlib.sha256(
                provisional.semantic_bytes
            ).hexdigest()
        }
    )
    _write_launch_receipt_exclusive(
        launch_receipt_path,
        canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n",
    )
    return receipt


__all__ = [
    "BaselineFidelityAuthorization",
    "BaselineFidelityBindings",
    "BaselineFidelityCostReservation",
    "BaselineFidelityPhaseRequest",
    "BaselineFidelityPhaseCostBound",
    "BaselineFidelityPolicy",
    "BaselineFidelityReconciliation",
    "ConsumedPermit",
    "ScientificComputeDenied",
    "WorkspaceSelection",
    "WorkspaceSnapshot",
    "attest_scientific_workspace",
    "authorize_baseline_fidelity",
    "load_baseline_fidelity_policy",
    "require_baseline_fidelity_permit",
    "reserve_baseline_fidelity_cost",
    "reserve_baseline_fidelity_in_ledger",
    "reconcile_baseline_fidelity_in_ledger",
]
