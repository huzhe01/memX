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

from ratemem.evaluation.canonical import semantic_sha256, write_json_atomic
from ratemem.evaluation.dataset_lock import (
    DatasetLock,
    DatasetLockError,
    load_inventory,
    seal_dataset_lock,
    write_dataset_lock_and_card,
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
app.add_typer(data_app, name="data")
app.add_typer(stats_app, name="stats")
app.add_typer(traces_app, name="traces")


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
