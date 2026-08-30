"""Command-line tools for matched baseline contracts and audit artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ratemem.baselines.catalog import BaselineCatalog
from ratemem.baselines.protocol import EventReceipt, ExactByteLedger
from ratemem.evaluation.canonical import write_json_atomic

app = typer.Typer(no_args_is_help=True, help="RateMem matched-baseline tooling.")
catalog_app = typer.Typer(no_args_is_help=True, help="Comparator catalog tools.")
schema_app = typer.Typer(no_args_is_help=True, help="Write matched-baseline schemas.")
app.add_typer(catalog_app, name="catalog")
app.add_typer(schema_app, name="schema")


@catalog_app.command("schema")
def catalog_schema(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the exact baseline catalog JSON Schema."""

    write_json_atomic(output, BaselineCatalog.model_json_schema())
    typer.echo(f"PASS baseline-catalog schema: {output}")


@schema_app.command("catalog")
def schema_catalog(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the comparator-catalog schema."""

    write_json_atomic(output, BaselineCatalog.model_json_schema())
    typer.echo(f"PASS baseline-catalog schema: {output}")


@schema_app.command("receipt")
def schema_receipt(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the immutable event-receipt schema."""

    write_json_atomic(output, EventReceipt.model_json_schema())
    typer.echo(f"PASS baseline-event-receipt schema: {output}")


@schema_app.command("ledger")
def schema_ledger(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the host-computed exact-byte ledger schema."""

    write_json_atomic(output, ExactByteLedger.model_json_schema())
    typer.echo(f"PASS baseline-ledger schema: {output}")


@schema_app.command("shared-input")
def schema_shared_input(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the provider-neutral shared-input manifest schema."""

    from ratemem.baselines.shared_inputs import SharedInputManifest

    write_json_atomic(output, SharedInputManifest.model_json_schema())
    typer.echo(f"PASS shared-input schema: {output}")


@schema_app.command("static-codebook")
def schema_static_codebook(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Write the frozen static-codebook artifact schema."""

    from ratemem.baselines.static_shared import StaticCodebookArtifact

    write_json_atomic(output, StaticCodebookArtifact.model_json_schema())
    typer.echo(f"PASS static-codebook schema: {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
