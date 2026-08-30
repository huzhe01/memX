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
source_app = typer.Typer(no_args_is_help=True, help="Resolve and verify source inventories.")
app.add_typer(catalog_app, name="catalog")
app.add_typer(schema_app, name="schema")
app.add_typer(source_app, name="sources")


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


@schema_app.command("oracle-certificate")
def schema_oracle_certificate(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the exact future-oracle certificate schema."""

    from ratemem.baselines.oracles import OracleCertificate

    write_json_atomic(output, OracleCertificate.model_json_schema())
    typer.echo(f"PASS oracle-certificate schema: {output}")


@schema_app.command("external-message")
def schema_external_message(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the strict external worker request/response schema."""

    from ratemem.baselines.external_jsonl import external_message_schema

    write_json_atomic(output, external_message_schema())
    typer.echo(f"PASS external-message schema: {output}")


@schema_app.command("paired-replay")
def schema_paired_replay(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the paired lifecycle replay schema."""

    from ratemem.baselines.replay import PairedReplay

    write_json_atomic(output, PairedReplay.model_json_schema())
    typer.echo(f"PASS paired-replay schema: {output}")


@schema_app.command("runtime-registry")
def schema_runtime_registry(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the source-hashed runtime factory registry schema."""

    from ratemem.baselines.registry import RuntimeRegistryLock

    write_json_atomic(output, RuntimeRegistryLock.model_json_schema())
    typer.echo(f"PASS runtime-registry schema: {output}")


@schema_app.command("source-inventory")
def schema_source_inventory(
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Write the sealed source inventory schema."""

    from ratemem.baselines.sources import SourceInventory

    write_json_atomic(output, SourceInventory.model_json_schema())
    typer.echo(f"PASS source-inventory schema: {output}")


@source_app.command("resolve")
def resolve_sources(
    registry: Annotated[Path, typer.Option("--registry")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Resolve, archive, license-audit, and seal all registered sources."""

    from ratemem.baselines.sources import (
        build_source_inventory,
        inventory_source,
        load_source_registry,
    )

    if output.exists() or output.is_symlink():
        raise typer.BadParameter("source inventory output already exists")
    configured = load_source_registry(registry)
    records = tuple(inventory_source(entry, cache_dir=cache_dir) for entry in configured.sources)
    inventory = build_source_inventory(records)
    write_json_atomic(output, inventory.model_dump(mode="json"))
    typer.echo(f"PASS source inventory: {output} {inventory.inventory_sha256}")


@source_app.command("verify")
def verify_sources(
    inventory: Annotated[Path, typer.Option("--inventory")],
) -> None:
    """Verify sealed source artifacts without performing network access."""

    from ratemem.baselines.sources import load_source_inventory, verify_source_record

    loaded = load_source_inventory(inventory)
    for record in loaded.records:
        verify_source_record(record)
    typer.echo(f"PASS source inventory: {loaded.inventory_sha256}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
