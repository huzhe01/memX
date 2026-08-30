"""Command-line surface for learned RateMem artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ratemem.evaluation.canonical import write_json_atomic, write_yaml_atomic
from ratemem.method.config import MethodLockInputs, MethodTrainingLock, freeze_method_lock

app = typer.Typer(no_args_is_help=True, help="RateMem learned-method controls.")


@app.command("schema")
def schema(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the strict learned-method lock schema."""

    write_json_atomic(output, MethodTrainingLock.model_json_schema())
    typer.echo(f"PASS method-lock schema: {output}")


@app.command("lock")
def lock(
    policy: Annotated[Path, typer.Option("--policy")],
    dataset_lock: Annotated[Path, typer.Option("--dataset-lock")],
    evaluation_lock: Annotated[Path, typer.Option("--evaluation-lock")],
    baseline_lock: Annotated[Path, typer.Option("--baseline-lock")],
    visible_trace: Annotated[list[Path], typer.Option("--visible-trace")],
    dataset_sha256: Annotated[str, typer.Option("--dataset-sha256")],
    evaluation_sha256: Annotated[str, typer.Option("--evaluation-sha256")],
    baseline_sha256: Annotated[str, typer.Option("--baseline-sha256")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Seal a training lock from explicit approved hashes and visible traces."""

    if output.exists() or output.is_symlink():
        raise typer.BadParameter("method lock output already exists")
    frozen = freeze_method_lock(
        MethodLockInputs(
            policy_path=policy,
            dataset_lock_path=dataset_lock,
            evaluation_lock_path=evaluation_lock,
            baseline_lock_path=baseline_lock,
            visible_trace_manifest_paths=tuple(visible_trace),
            expected_dataset_lock_sha256=dataset_sha256,
            expected_evaluation_lock_sha256=evaluation_sha256,
            expected_baseline_lock_sha256=baseline_sha256,
        )
    )
    write_yaml_atomic(output, frozen.model_dump(mode="json"))
    typer.echo(f"PASS method-lock: {frozen.lock_id}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
