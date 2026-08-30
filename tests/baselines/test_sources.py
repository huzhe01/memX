from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ratemem.baselines.sources import (
    GitSource,
    SourceAuditError,
    build_source_inventory,
    inventory_source,
    load_source_registry,
    verify_source_record,
)


def _run(arguments: list[str], cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True, capture_output=True)


def _repository(root: Path, *, licensed: bool) -> Path:
    root.mkdir()
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.name", "Fixture"], root)
    _run(["git", "config", "user.email", "fixture@example.invalid"], root)
    (root / "algorithm.py").write_text("VALUE = 1\n", encoding="utf-8")
    if licensed:
        (root / "LICENSE.spdx").write_text(
            "SPDX-License-Identifier: Apache-2.0\n",
            encoding="utf-8",
        )
    _run(["git", "add", "."], root)
    _run(["git", "commit", "-m", "fixture"], root)
    return root


def _entry(repository: Path, *, required: bool = True) -> GitSource:
    return GitSource(
        source_id="fixture_source",
        methods=("independent_fifo",),
        kind="git",
        repository_url=str(repository),
        license_required_for_execution=required,
    )


def test_inventory_resolves_commit_archive_and_license_hash(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "upstream", licensed=True)
    record = inventory_source(
        _entry(repository),
        cache_dir=tmp_path / "cache",
        resolved_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    assert len(record.source_revision) == 40
    assert len(record.source_archive_sha256) == 64
    assert record.license_expression == "Apache-2.0"
    assert len(record.license_files) == 1
    verify_source_record(record)
    inventory = build_source_inventory((record,))
    assert len(inventory.inventory_sha256) == 64


def test_required_external_source_without_license_is_not_executable(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "upstream", licensed=False)
    with pytest.raises(
        SourceAuditError,
        match="required external source has no auditable license",
    ):
        inventory_source(_entry(repository), cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD", ""])
def test_sealed_source_record_rejects_mutable_revision(
    revision: str,
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "upstream", licensed=True)
    record = inventory_source(_entry(repository), cache_dir=tmp_path / "cache")
    changed = record.model_copy(update={"source_revision": revision})
    with pytest.raises(ValueError, match="source revision must be"):
        changed.validate_sealed()


def test_committed_registry_matches_locked_python_dependencies() -> None:
    registry = load_source_registry(Path("configs/baselines/source-registry.yaml"))
    by_id = {row.source_id: row for row in registry.sources}
    assert by_id["diffusers_upstream"].locked_version == "0.40.0"  # type: ignore[union-attr]
    assert by_id["peft_upstream"].locked_version == "0.20.0"  # type: ignore[union-attr]
