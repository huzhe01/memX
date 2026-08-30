"""Command-line entry point for locked scientific evaluation."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal, Never, cast

import typer
import yaml  # type: ignore[import-untyped]
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

from ratemem.baselines.catalog import load_catalog
from ratemem.evaluation.baselines import BaselineLock, load_requirements
from ratemem.evaluation.canonical import (
    canonical_json_bytes,
    file_sha256,
    semantic_sha256,
    write_json_atomic,
)
from ratemem.evaluation.compute import (
    BaselineFidelityAuthorization,
    BaselineFidelityBindings,
    BaselineFidelityCostReservation,
    BaselineFidelityPhaseCostBound,
    BaselineFidelityPhaseRequest,
    ConsumedPermit,
    ScientificComputeDenied,
    WorkspaceSelection,
    WorkspaceSnapshot,
    attest_scientific_workspace,
    authorize_baseline_fidelity,
    load_baseline_fidelity_policy,
    reconcile_baseline_fidelity_in_ledger,
    reserve_baseline_fidelity_in_ledger,
)
from ratemem.evaluation.dataset_lock import (
    DatasetLock,
    DatasetLockError,
    load_inventory,
    seal_dataset_lock,
    write_dataset_lock_and_card,
)
from ratemem.evaluation.evaluation_lock import EvaluationLock
from ratemem.evaluation.final_trace import (
    FinalEvaluationPermit,
    FinalTraceEnvelope,
    FinalTracePublicManifest,
    generate_x25519_keypair,
    seal_final_trace,
)
from ratemem.evaluation.leakage import (
    DuplicateAuditReport,
    load_feature_encoder_inventory,
    lock_feature_encoder,
)
from ratemem.evaluation.pools import (
    PoolLeakageError,
    PoolManifestLine,
    build_pools_from_catalogs,
)
from ratemem.evaluation.statistics import (
    CalibrationRecord,
    RequiredUnits,
    plan_required_units,
)
from ratemem.evaluation.traces import (
    AllPools,
    TraceManifest,
    TracePolicy,
    build_trace_set,
    write_trace_set,
)

app = typer.Typer(
    no_args_is_help=True,
    help="RateMem-DiT locked scientific evaluation.",
)
data_app = typer.Typer(no_args_is_help=True, help="Audit and seal dataset inventories.")
stats_app = typer.Typer(no_args_is_help=True, help="Freeze calibration and sample-size records.")
traces_app = typer.Typer(no_args_is_help=True, help="Build and verify lifecycle traces.")
lock_app = typer.Typer(no_args_is_help=True, help="Compile immutable scientific locks.")
baselines_app = typer.Typer(no_args_is_help=True, help="Verify baseline audit handoffs.")
compute_app = typer.Typer(no_args_is_help=True, help="Authorize scientific compute phases.")
app.add_typer(data_app, name="data")
app.add_typer(stats_app, name="stats")
app.add_typer(traces_app, name="traces")
app.add_typer(lock_app, name="lock")
app.add_typer(baselines_app, name="baselines")
app.add_typer(compute_app, name="compute")


@app.callback()
def root() -> None:
    """Validate, freeze, replay, and publish scientific artifacts."""


def _blocked(message: str) -> Never:
    typer.echo(f"BLOCKED dataset-lock: {message}", err=True)
    raise typer.Exit(code=2)


def _pools_blocked(message: str) -> Never:
    typer.echo(f"BLOCKED pools: {message}", err=True)
    raise typer.Exit(code=2)


def _duplicate_blocked(message: str) -> Never:
    typer.echo(f"BLOCKED duplicate-audit: {message}", err=True)
    raise typer.Exit(code=2)


def _final_trace_blocked(message: str) -> Never:
    typer.echo(f"BLOCKED final-trace: {message}", err=True)
    raise typer.Exit(code=2)


def _evaluation_lock_blocked(message: str) -> Never:
    typer.echo(f"BLOCKED evaluation-lock: {message}", err=True)
    raise typer.Exit(code=2)


def _compute_blocked(message: str) -> Never:
    typer.echo(f"BLOCKED baseline-fidelity: {message}", err=True)
    raise typer.Exit(code=2)


def _write_new_file(path: Path, payload: bytes, mode: int) -> None:
    """Create one immutable artifact without following or replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, mode)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@baselines_app.command("schema")
def baseline_lock_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the exact scientific baseline-lock JSON Schema."""

    write_json_atomic(output, BaselineLock.model_json_schema())
    typer.echo(f"PASS baseline-lock schema: {output}")


@compute_app.command("schema-baseline-fidelity")
def baseline_fidelity_authorization_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the narrow baseline-fidelity authorization JSON Schema."""

    write_json_atomic(output, BaselineFidelityAuthorization.model_json_schema())
    typer.echo(f"PASS baseline-fidelity authorization schema: {output}")


@compute_app.command("attest-workspace")
def attest_workspace_command(
    policy: Annotated[Path, typer.Option("--policy")],
    workspace_selection: Annotated[
        Path,
        typer.Option("--workspace-selection"),
    ],
    budget_evidence: Annotated[Path, typer.Option("--budget-evidence")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Attest one explicitly selected workspace and its operator cap evidence."""

    if output.exists() or output.is_symlink():
        _compute_blocked("output_exists")
    try:
        compute_policy = load_baseline_fidelity_policy(policy)
        selection = WorkspaceSelection.model_validate_json(
            workspace_selection.read_text(encoding="utf-8")
        )
        snapshot = attest_scientific_workspace(
            selection,
            budget_evidence,
            compute_policy,
        )
        _write_new_file(
            output,
            canonical_json_bytes(snapshot.model_dump(mode="json")) + b"\n",
            0o600,
        )
    except (OSError, ScientificComputeDenied, TypeError, ValueError) as error:
        _compute_blocked(str(error))
    typer.echo(
        "PASS scientific-workspace attestation: "
        f"workspace={snapshot.workspace_id} outer_cap={snapshot.outer_budget_usd:.2f} "
        f"known_usage={snapshot.known_usage_usd:.2f}"
    )


@compute_app.command("authorize-baseline-fidelity")
def authorize_baseline_fidelity_command(
    policy: Annotated[Path, typer.Option("--policy")],
    workspace_selection: Annotated[Path, typer.Option("--workspace-selection")],
    workspace_snapshot: Annotated[Path, typer.Option("--workspace-snapshot")],
    phase_request: Annotated[Path, typer.Option("--phase-request")],
    dataset_lock: Annotated[Path, typer.Option("--dataset-lock")],
    requirements: Annotated[Path, typer.Option("--requirements")],
    catalog: Annotated[Path, typer.Option("--catalog")],
    fidelity_policy: Annotated[Path, typer.Option("--fidelity-policy")],
    source_inventory: Annotated[Path, typer.Option("--source-inventory")],
    clean_diff_receipt: Annotated[Path, typer.Option("--clean-diff-receipt")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Authorize one pre-lock source-fidelity job without reserving cost."""

    if output.exists() or output.is_symlink():
        _compute_blocked("output_exists")
    try:
        compute_policy = load_baseline_fidelity_policy(policy)
        selection = WorkspaceSelection.model_validate_json(
            workspace_selection.read_text(encoding="utf-8")
        )
        snapshot = WorkspaceSnapshot.model_validate_json(
            workspace_snapshot.read_text(encoding="utf-8")
        )
        phase = BaselineFidelityPhaseRequest.model_validate_json(
            phase_request.read_text(encoding="utf-8")
        )
        dataset_payload = yaml.safe_load(dataset_lock.read_text(encoding="utf-8"))
        locked_dataset = DatasetLock.model_validate(dataset_payload)
        if semantic_sha256(locked_dataset.model_dump(mode="json")) != locked_dataset.lock_id:
            raise ScientificComputeDenied("dataset_lock_hash_mismatch")
        locked_requirements = load_requirements(requirements)
        locked_catalog = load_catalog(catalog)
        clean_diff_payload = yaml.safe_load(
            clean_diff_receipt.read_text(encoding="utf-8")
        )
        if type(clean_diff_payload) is not dict:
            raise ScientificComputeDenied("clean_diff_receipt_invalid")
        bindings = BaselineFidelityBindings(
            dataset_lock_sha256=locked_dataset.lock_id,
            baseline_requirements_sha256=locked_requirements.sha256,
            comparator_catalog_sha256=locked_catalog.sha256,
            fidelity_policy_sha256=file_sha256(fidelity_policy),
            source_inventory_sha256=file_sha256(source_inventory),
            git_commit=str(clean_diff_payload.get("git_commit", "")),
            clean_diff_sha256=str(
                clean_diff_payload.get("clean_diff_sha256", "")
            ),
        )
        authorization = authorize_baseline_fidelity(
            selection,
            snapshot,
            phase,
            bindings,
            compute_policy,
        )
        _write_new_file(
            output,
            canonical_json_bytes(authorization.model_dump(mode="json")) + b"\n",
            0o600,
        )
    except (OSError, ScientificComputeDenied, TypeError, ValueError) as error:
        _compute_blocked(str(error))
    typer.echo(
        "PASS baseline-fidelity authorization: "
        f"phase={authorization.phase_id} workspace={authorization.workspace_id} "
        f"authorization={authorization.authorization_sha256}"
    )


@compute_app.command("reserve-baseline-fidelity")
def reserve_baseline_fidelity_command(
    policy: Annotated[Path, typer.Option("--policy")],
    authorization: Annotated[Path, typer.Option("--authorization")],
    phase_bound: Annotated[Path, typer.Option("--phase-bound")],
    workspace_snapshot: Annotated[Path, typer.Option("--workspace-snapshot")],
    ledger: Annotated[Path, typer.Option("--ledger")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Reserve a fidelity phase against all pending shared-ledger cost."""

    try:
        permit = BaselineFidelityAuthorization.model_validate_json(
            authorization.read_text(encoding="utf-8")
        )
        bound = BaselineFidelityPhaseCostBound.model_validate_json(
            phase_bound.read_text(encoding="utf-8")
        )
        snapshot = WorkspaceSnapshot.model_validate_json(
            workspace_snapshot.read_text(encoding="utf-8")
        )
        compute_policy = load_baseline_fidelity_policy(policy)
        reservation = reserve_baseline_fidelity_in_ledger(
            permit,
            snapshot,
            bound,
            policy=compute_policy,
            ledger_path=ledger,
            output_path=output,
        )
    except (OSError, ScientificComputeDenied, TypeError, ValueError) as error:
        _compute_blocked(str(error))
    typer.echo(
        "PASS baseline-fidelity reservation: "
        f"known={reservation.known_usage_usd:.2f} "
        f"pending={reservation.pending_worst_case_usd:.2f} "
        f"new={reservation.new_phase_bound_usd:.2f} "
        f"total={reservation.reserved_total_usd:.2f} <= 27.00"
    )


@compute_app.command("reconcile-baseline-fidelity")
def reconcile_baseline_fidelity_command(
    policy: Annotated[Path, typer.Option("--policy")],
    authorization: Annotated[Path, typer.Option("--authorization")],
    reservation: Annotated[Path, typer.Option("--reservation")],
    launch_receipt: Annotated[Path, typer.Option("--launch-receipt")],
    workspace_selection: Annotated[Path, typer.Option("--workspace-selection")],
    budget_evidence: Annotated[Path, typer.Option("--budget-evidence")],
    ledger: Annotated[Path, typer.Option("--ledger")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Reconcile one consumed fidelity reservation against current usage."""

    try:
        permit = BaselineFidelityAuthorization.model_validate_json(
            authorization.read_text(encoding="utf-8")
        )
        reserved = BaselineFidelityCostReservation.model_validate_json(
            reservation.read_text(encoding="utf-8")
        )
        launch = ConsumedPermit.model_validate_json(
            launch_receipt.read_text(encoding="utf-8")
        )
        selection = WorkspaceSelection.model_validate_json(
            workspace_selection.read_text(encoding="utf-8")
        )
        compute_policy = load_baseline_fidelity_policy(policy)
        current_snapshot = attest_scientific_workspace(
            selection,
            budget_evidence,
            compute_policy,
        )
        reconciliation = reconcile_baseline_fidelity_in_ledger(
            permit,
            reserved,
            launch,
            current_snapshot,
            ledger_path=ledger,
            output_path=output,
        )
    except (OSError, ScientificComputeDenied, TypeError, ValueError) as error:
        _compute_blocked(str(error))
    typer.echo(
        "PASS baseline-fidelity reconciliation: "
        f"workspace={reconciliation.workspace_id} "
        f"metered_delta={reconciliation.metered_delta_usd:.2f} "
        f"pending_remaining={reconciliation.pending_remaining_usd:.2f}"
    )


@lock_app.command("schema")
def lock_schema(
    kind: Annotated[Literal["evaluation"], typer.Option("--kind")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the exact schema for one scientific lock kind."""

    if kind != "evaluation":
        _evaluation_lock_blocked("unsupported lock kind")
    write_json_atomic(output, EvaluationLock.model_json_schema())
    typer.echo(f"PASS evaluation-lock schema: {output}")


@lock_app.command("evaluation")
def compile_evaluation_lock(
    policy: Annotated[Path, typer.Option("--policy")],
    dataset_lock: Annotated[Path, typer.Option("--dataset-lock")],
    baseline_lock: Annotated[Path, typer.Option("--baseline-lock")],
    baseline_audit_receipt: Annotated[
        Path,
        typer.Option("--baseline-audit-receipt"),
    ],
    trace_dir: Annotated[Path, typer.Option("--trace-dir")],
    evaluator_inventory: Annotated[Path, typer.Option("--evaluator-inventory")],
    byte_ledger: Annotated[Path, typer.Option("--byte-ledger")],
    margin_record: Annotated[Path, typer.Option("--margin-record")],
    power_record: Annotated[Path, typer.Option("--power-record")],
    approvals: Annotated[Path, typer.Option("--approvals")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Compile an evaluation lock after the companion baseline audit exists."""

    # Task 8 owns the baseline schema and audit compiler.  This check intentionally
    # runs before reading any other input so the pre-lock boundary fails closed.
    if not baseline_lock.is_file():
        _evaluation_lock_blocked("baseline lock is missing")
    _ = (
        policy,
        dataset_lock,
        baseline_audit_receipt,
        trace_dir,
        evaluator_inventory,
        byte_ledger,
        margin_record,
        power_record,
        approvals,
        output,
    )
    _evaluation_lock_blocked("baseline lock compiler is not installed yet")


@data_app.command("schema")
def dataset_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the exact JSON Schema for a sealed dataset lock."""

    write_json_atomic(output, DatasetLock.model_json_schema())
    typer.echo(f"PASS dataset-lock schema: {output}")


@data_app.command("seal")
def dataset_seal(
    inventory: Annotated[Path, typer.Option("--inventory")],
    policy: Annotated[Path, typer.Option("--policy")],
    lock_output: Annotated[Path, typer.Option("--lock-output")],
    card_output: Annotated[Path, typer.Option("--card-output")],
    mode: Annotated[Literal["synthetic", "scientific"], typer.Option("--mode")],
) -> None:
    """Seal one audited inventory without creating partial outputs on failure."""

    if not inventory.is_file():
        if mode == "scientific":
            _blocked("audited source inventory is missing")
        _blocked("source inventory is missing")
    if lock_output.exists() or card_output.exists():
        _blocked("output path already exists")
    try:
        source_inventory = load_inventory(inventory)
        lock = seal_dataset_lock(source_inventory, policy_path=policy, mode=mode)
        write_dataset_lock_and_card(lock, lock_output, card_output)
    except (DatasetLockError, OSError, TypeError, ValueError) as error:
        if lock_output.exists():
            lock_output.unlink()
        if card_output.exists():
            card_output.unlink()
        _blocked(str(error))
    typer.echo(f"PASS dataset-lock sealed: {lock.lock_id}")


@data_app.command("pool-schema")
def pool_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the schema shared by every public pool JSONL line."""

    write_json_atomic(output, PoolManifestLine.model_json_schema())
    typer.echo(f"PASS pool-manifest schema: {output}")


@data_app.command("build-pools")
def build_pools(
    source_catalog: Annotated[Path, typer.Option("--source-catalog")],
    prompt_catalog: Annotated[Path, typer.Option("--prompt-catalog")],
    split_assignments: Annotated[Path, typer.Option("--split-assignments")],
    split_seed: Annotated[int, typer.Option("--split-seed")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Build anonymous, concept-disjoint pools from explicit split assignments."""

    try:
        build_pools_from_catalogs(
            source_catalog=source_catalog,
            prompt_catalog=prompt_catalog,
            split_assignments=split_assignments,
            split_seed=split_seed,
            output_dir=output,
        )
    except (FileExistsError, OSError, PoolLeakageError, TypeError, ValueError) as error:
        _pools_blocked(str(error))
    typer.echo(
        "PASS pools: train/validation/final_test concept pools and prompt namespaces "
        "are disjoint"
    )


@data_app.command("duplicate-schema")
def duplicate_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the strict duplicate-audit report schema."""

    write_json_atomic(output, DuplicateAuditReport.model_json_schema())
    typer.echo(f"PASS scientific-duplicate-report schema: {output}")


@data_app.command("lock-feature-encoder")
def feature_encoder_lock(
    model_id: Annotated[str, typer.Option("--model-id")],
    model_inventory: Annotated[Path, typer.Option("--model-inventory")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Bind one immutable encoder revision to locally observed weight bytes."""

    if output.exists() or output.is_symlink():
        _duplicate_blocked("feature encoder lock output already exists")
    try:
        inventory = load_feature_encoder_inventory(model_inventory)
        lock = lock_feature_encoder(
            model_id=model_id,
            inventory_entries=inventory.models,
            preprocessing="resize_shorter_518_center_crop_rgb_v1",
        )
        write_json_atomic(output, lock.model_dump(mode="json"))
    except (OSError, TypeError, ValueError) as error:
        if output.exists():
            output.unlink()
        _duplicate_blocked(str(error))
    typer.echo(
        f"PASS duplicate-feature-encoder lock: revision={lock.immutable_revision} "
        f"weights={lock.weights_sha256}"
    )


@stats_app.command("schema-calibration")
def calibration_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the strict calibration-record schema."""

    write_json_atomic(output, CalibrationRecord.model_json_schema())
    typer.echo(f"PASS calibration-record schema: {output}")


@stats_app.command("plan-units")
def plan_units(
    calibration_record: Annotated[Path, typer.Option("--calibration-record")],
    maximum_half_width: Annotated[float, typer.Option("--maximum-half-width")],
    minimum_effect: Annotated[float, typer.Option("--minimum-effect")],
    alpha: Annotated[float, typer.Option("--alpha")],
    power: Annotated[float, typer.Option("--power")],
    minimum_units: Annotated[int, typer.Option("--minimum-units")],
    simulation_seed: Annotated[int, typer.Option("--simulation-seed")],
    output: Annotated[Path, typer.Option("--output")],
    monte_carlo_draws: Annotated[int, typer.Option("--monte-carlo-draws")] = 2048,
) -> None:
    """Plan deployment episodes from a calibration-only paired record."""

    if output.exists() or output.is_symlink():
        typer.echo("BLOCKED power-plan: output already exists", err=True)
        raise typer.Exit(code=2)
    try:
        record = CalibrationRecord.model_validate_json(
            calibration_record.read_text(encoding="utf-8")
        )
        required = plan_required_units(
            record,
            maximum_half_width,
            minimum_effect,
            alpha,
            power,
            minimum_units,
            simulation_seed,
            monte_carlo_draws=monte_carlo_draws,
        )
        write_json_atomic(output, required.model_dump(mode="json"))
    except (OSError, TypeError, ValueError) as error:
        if output.exists():
            output.unlink()
        typer.echo(f"BLOCKED power-plan: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        f"PASS power-plan: final deployment episodes={required.required_units}; "
        f"target_half_width={maximum_half_width:.2f}; power={power:.2f}"
    )


@traces_app.command("schema")
def trace_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the strict visible trace-manifest schema."""

    write_json_atomic(output, TraceManifest.model_json_schema())
    typer.echo(f"PASS trace-manifest schema: {output}")


@traces_app.command("envelope-schema")
def final_trace_envelope_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the exact schema for the public encrypted final-trace envelope."""

    write_json_atomic(output, FinalTraceEnvelope.model_json_schema())
    typer.echo(f"PASS final-trace envelope schema: {output}")


@traces_app.command("freeze-schema")
def final_evaluation_freeze_schema(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the exact schema for a signed final-evaluation permit."""

    write_json_atomic(output, FinalEvaluationPermit.model_json_schema())
    typer.echo(f"PASS final-evaluation freeze schema: {output}")


@traces_app.command("keygen")
def final_trace_keygen(
    private_key: Annotated[Path, typer.Option("--private-key")],
    public_key: Annotated[Path, typer.Option("--public-key")],
) -> None:
    """Generate a final-trace recipient keypair without printing key material."""

    if private_key == public_key:
        _final_trace_blocked("private and public key paths must differ")
    if any(path.exists() or path.is_symlink() for path in (private_key, public_key)):
        _final_trace_blocked("key output path already exists")
    created: list[Path] = []
    try:
        recipient_private, recipient_public = generate_x25519_keypair()
        private_payload = recipient_private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_payload = recipient_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        _write_new_file(private_key, private_payload, 0o600)
        created.append(private_key)
        _write_new_file(public_key, public_payload, 0o644)
        created.append(public_key)
    except (OSError, TypeError, ValueError) as error:
        for path in created:
            path.unlink(missing_ok=True)
        _final_trace_blocked(str(error))
    typer.echo("PASS final-trace keypair generated: private-key-disclosed=false")


@traces_app.command("seal-final")
def seal_final_trace_command(
    dataset_lock: Annotated[Path, typer.Option("--dataset-lock")],
    policy: Annotated[Path, typer.Option("--policy")],
    power_record: Annotated[Path, typer.Option("--power-record")],
    recipient: Annotated[Path, typer.Option("--recipient")],
    manifest_output: Annotated[Path, typer.Option("--manifest-output")],
    envelope_output: Annotated[Path, typer.Option("--envelope-output")],
    concept_pools: Annotated[
        Path,
        typer.Option("--concept-pools"),
    ] = Path("artifacts/scientific/dataset-audit/concept-pools.json"),
) -> None:
    """Build a final trace in memory and publish only its manifest and envelope."""

    outputs = (manifest_output, envelope_output)
    if manifest_output == envelope_output:
        _final_trace_blocked("manifest and envelope output paths must differ")
    if any(path.exists() or path.is_symlink() for path in outputs):
        _final_trace_blocked("output path already exists")
    created: list[Path] = []
    try:
        lock_payload = yaml.safe_load(dataset_lock.read_text(encoding="utf-8"))
        locked_dataset = DatasetLock.model_validate(lock_payload)
        if semantic_sha256(locked_dataset.model_dump(mode="json")) != locked_dataset.lock_id:
            raise ValueError("dataset lock content hash changed")
        required = RequiredUnits.model_validate_json(
            power_record.read_text(encoding="utf-8")
        )
        if hashlib.sha256(required.semantic_bytes).hexdigest() != required.record_sha256:
            raise ValueError("required-unit record content hash changed")
        all_pools = AllPools.load(concept_pools)
        if all_pools.dataset_lock_id != locked_dataset.lock_id:
            raise ValueError("concept pools do not bind the selected dataset lock")
        trace_policy = TracePolicy.load(policy)
        loaded_recipient = serialization.load_pem_public_key(recipient.read_bytes())
        if not isinstance(loaded_recipient, X25519PublicKey):
            raise TypeError("recipient must contain an X25519 public key")

        count = required.required_units
        final_set = build_trace_set(
            all_pools,
            trace_policy,
            counts={"train": count, "validation": count, "final_test": count},
            event_count=trace_policy.events_per_deployment_episode,
        )["final_test"]
        plaintext = b"".join(
            canonical_json_bytes(
                {
                    "trace_id": trace.trace_id,
                    "event": event.model_dump(mode="json"),
                }
            )
            + b"\n"
            for trace in sorted(final_set.traces, key=lambda item: item.trace_id)
            for event in trace.events
        )
        final_pool = all_pools.for_split("final_test")
        public_manifest = FinalTracePublicManifest(
            dataset_lock_id=locked_dataset.lock_id,
            trace_builder_revision=trace_policy.builder_revision,
            trace_policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
            power_record_sha256=required.record_sha256,
            concept_pool_sha256=final_pool.concept_pool_sha256,
            prompt_pool_sha256=final_pool.prompt_pool_sha256,
            trace_ids=final_set.trace_ids,
            generation_seeds=final_set.generation_seeds,
            trace_count=len(final_set.traces),
            event_count=len(plaintext.splitlines()),
            plaintext_sha256=hashlib.sha256(plaintext).hexdigest(),
        )
        manifest_bytes = (
            canonical_json_bytes(public_manifest.model_dump(mode="json")) + b"\n"
        )
        envelope = seal_final_trace(
            plaintext,
            loaded_recipient,
            associated_manifest=manifest_bytes,
        )
        _write_new_file(manifest_output, manifest_bytes, 0o644)
        created.append(manifest_output)
        _write_new_file(
            envelope_output,
            canonical_json_bytes(envelope.model_dump(mode="json")) + b"\n",
            0o644,
        )
        created.append(envelope_output)
    except (OSError, TypeError, ValueError) as error:
        for path in created:
            path.unlink(missing_ok=True)
        _final_trace_blocked(str(error))
    typer.echo(
        f"PASS final-trace sealed: plaintext retained=false envelope={envelope.sha256}"
    )


@traces_app.command("build-visible")
def build_visible_traces(
    dataset_lock: Annotated[Path, typer.Option("--dataset-lock")],
    policy: Annotated[Path, typer.Option("--policy")],
    power_record: Annotated[Path, typer.Option("--power-record")],
    splits: Annotated[str, typer.Option("--splits")],
    output: Annotated[Path, typer.Option("--output")],
    concept_pools: Annotated[
        Path,
        typer.Option("--concept-pools"),
    ] = Path("artifacts/scientific/dataset-audit/concept-pools.json"),
) -> None:
    """Build only development-visible trace payloads from frozen inputs."""

    raw_requested = tuple(part.strip() for part in splits.split(",") if part.strip())
    if (
        not raw_requested
        or len(raw_requested) != len(set(raw_requested))
        or any(split not in {"train", "validation"} for split in raw_requested)
    ):
        typer.echo(
            "BLOCKED traces: visible builds accept unique train and validation splits only",
            err=True,
        )
        raise typer.Exit(code=2)
    requested = cast(tuple[Literal["train", "validation"], ...], raw_requested)
    if output.exists() or output.is_symlink():
        typer.echo("BLOCKED traces: output already exists", err=True)
        raise typer.Exit(code=2)

    staging: Path | None = None
    try:
        lock_payload = yaml.safe_load(dataset_lock.read_text(encoding="utf-8"))
        locked_dataset = DatasetLock.model_validate(lock_payload)
        if semantic_sha256(locked_dataset.model_dump(mode="json")) != locked_dataset.lock_id:
            raise ValueError("dataset lock content hash changed")
        required = RequiredUnits.model_validate_json(
            power_record.read_text(encoding="utf-8")
        )
        if hashlib.sha256(required.semantic_bytes).hexdigest() != required.record_sha256:
            raise ValueError("required-unit record content hash changed")
        all_pools = AllPools.load(concept_pools)
        if all_pools.dataset_lock_id != locked_dataset.lock_id:
            raise ValueError("concept pools do not bind the selected dataset lock")
        trace_policy = TracePolicy.load(policy)
        count = required.required_units
        trace_sets = build_trace_set(
            all_pools,
            trace_policy,
            counts={"train": count, "validation": count, "final_test": count},
            event_count=trace_policy.events_per_deployment_episode,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                dir=output.parent,
                prefix=f".{output.name}.staging-",
            )
        )
        for split in requested:
            write_trace_set(trace_sets[split], staging / split)
        os.chmod(staging, 0o755)
        os.replace(staging, output)
        staging = None
    except (OSError, TypeError, ValueError) as error:
        typer.echo(f"BLOCKED traces: {error}", err=True)
        raise typer.Exit(code=2) from error
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    labels = " and ".join(requested)
    typer.echo(
        f"PASS traces: {labels} manifests have disjoint concepts, ids, and seeds"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
