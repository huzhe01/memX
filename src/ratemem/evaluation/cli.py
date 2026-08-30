"""Command-line entry point for locked scientific evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Never

import typer

from ratemem.evaluation.canonical import write_json_atomic
from ratemem.evaluation.dataset_lock import (
    DatasetLock,
    DatasetLockError,
    load_inventory,
    seal_dataset_lock,
    write_dataset_lock_and_card,
)
from ratemem.evaluation.pools import (
    PoolLeakageError,
    PoolManifestLine,
    build_pools_from_catalogs,
)

app = typer.Typer(
    no_args_is_help=True,
    help="RateMem-DiT locked scientific evaluation.",
)
data_app = typer.Typer(no_args_is_help=True, help="Audit and seal dataset inventories.")
app.add_typer(data_app, name="data")


@app.callback()
def root() -> None:
    """Validate, freeze, replay, and publish scientific artifacts."""


def _blocked(message: str) -> Never:
    typer.echo(f"BLOCKED dataset-lock: {message}", err=True)
    raise typer.Exit(code=2)


def _pools_blocked(message: str) -> Never:
    typer.echo(f"BLOCKED pools: {message}", err=True)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
