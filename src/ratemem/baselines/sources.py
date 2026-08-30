"""Immutable source, archive, and license inventory for comparator code."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPDX = re.compile(r"SPDX-License-Identifier:\s*([^\s]+)")
_MUTABLE_REVISIONS = frozenset({"", "head", "main", "master", "latest"})


class SourceAuditError(RuntimeError):
    """Raised when source identity or execution-license evidence is insufficient."""


class _SourceBase(BaseModel):
    model_config = _MODEL_CONFIG

    source_id: str
    methods: tuple[str, ...]
    license_required_for_execution: bool

    @model_validator(mode="after")
    def validate_identity(self) -> _SourceBase:
        values = (self.source_id, *self.methods)
        if any(_IDENTIFIER.fullmatch(value) is None for value in values):
            raise ValueError("source and method ids must be canonical")
        if not self.methods or self.methods != tuple(sorted(set(self.methods))):
            raise ValueError("source methods must be sorted, non-empty, and unique")
        return self


class GitSource(_SourceBase):
    kind: Literal["git"]
    repository_url: str

    @field_validator("repository_url")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"}:
            if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment:
                raise ValueError("git repository URL must be credential-free HTTPS")
        elif parsed.scheme == "file":
            if parsed.username or parsed.password or parsed.fragment:
                raise ValueError("local git repository URL is invalid")
        elif parsed.scheme or not Path(value).is_absolute():
            raise ValueError("git repository must be HTTPS, file URL, or absolute local path")
        return value


class PaperSource(_SourceBase):
    kind: Literal["paper"]
    canonical_url: str

    @field_validator("canonical_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.fragment:
            raise ValueError("paper URL must be canonical credential-free HTTPS")
        return value


class InstalledDistributionSource(_SourceBase):
    kind: Literal["installed_distribution"]
    distribution: str
    locked_version: str

    @model_validator(mode="after")
    def validate_distribution(self) -> InstalledDistributionSource:
        if _IDENTIFIER.fullmatch(self.distribution) is None or not self.locked_version:
            raise ValueError("installed distribution identity is invalid")
        return self


SourceEntry = Annotated[
    GitSource | PaperSource | InstalledDistributionSource,
    Field(discriminator="kind"),
]
_SOURCE_ADAPTER: TypeAdapter[SourceEntry] = TypeAdapter(SourceEntry)


class SourceRegistry(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    sources: tuple[SourceEntry, ...]

    @model_validator(mode="after")
    def validate_sources(self) -> SourceRegistry:
        ids = tuple(row.source_id for row in self.sources)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("source registry ids must be non-empty and unique")
        covered = [method for row in self.sources for method in row.methods]
        if not covered:
            raise ValueError("source registry covers no methods")
        return self


class LicenseFile(BaseModel):
    model_config = _MODEL_CONFIG

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceRecord(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    source_id: str
    methods: tuple[str, ...]
    kind: Literal["git", "paper", "installed_distribution"]
    canonical_location: str
    source_revision: str
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_artifact: str
    license_expression: str
    license_files: tuple[LicenseFile, ...]
    license_required_for_execution: bool
    executable: bool
    resolver_versions: dict[str, str]
    resolved_at_utc: AwareDatetime
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_sealed(self) -> SourceRecord:
        revision = self.source_revision.lower()
        if revision in _MUTABLE_REVISIONS or not (
            _COMMIT.fullmatch(revision) or _SHA256.fullmatch(revision)
        ):
            raise ValueError("source revision must be a resolved commit or artifact SHA-256")
        if self.resolved_at_utc.utcoffset() is None:
            raise ValueError("source resolution timestamp must be timezone-aware")
        if self.license_required_for_execution and (
            self.license_expression == "NOASSERTION" or not self.executable
        ):
            raise ValueError("execution-required source lacks an auditable license")
        if self.executable != (self.license_expression != "NOASSERTION"):
            raise ValueError("source executable flag differs from license evidence")
        return self

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("record_sha256")
        return canonical_json_bytes(payload)


class SourceInventory(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    records: tuple[SourceRecord, ...]
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_records(self) -> SourceInventory:
        ids = tuple(row.source_id for row in self.records)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("source inventory records must be sorted and unique")
        for record in self.records:
            if hashlib.sha256(record.semantic_bytes).hexdigest() != record.record_sha256:
                raise ValueError("source record hash mismatch")
        return self

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("inventory_sha256")
        return canonical_json_bytes(payload)


def load_source_registry(path: Path) -> SourceRegistry:
    try:
        return SourceRegistry.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SourceAuditError(f"invalid source registry: {error}") from error


def _run(arguments: list[str], *, cwd: Path | None = None) -> bytes:
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceAuditError(f"source command failed: {arguments[0]}") from error
    return result.stdout


def _git_version() -> str:
    return _run(["git", "--version"]).decode("utf-8").strip()


def _resolve_git_head(repository_url: str) -> str:
    output = _run(["git", "ls-remote", repository_url, "HEAD"]).decode("ascii")
    rows = [line.split() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != "HEAD":
        raise SourceAuditError("git HEAD resolution returned an ambiguous result")
    revision = rows[0][0]
    if _COMMIT.fullmatch(revision) is None:
        raise SourceAuditError("git HEAD did not resolve to one full commit")
    return revision


def _ensure_checkout(entry: GitSource, cache_dir: Path, revision: str) -> Path:
    checkout = cache_dir / "checkouts" / entry.source_id / revision
    if checkout.exists():
        if checkout.is_symlink() or not (checkout / ".git").is_dir():
            raise SourceAuditError("source checkout cache is invalid")
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                entry.repository_url,
                str(checkout),
            ]
        )
    _run(["git", "-C", str(checkout), "fetch", "--no-tags", "origin", revision])
    resolved = _run(["git", "-C", str(checkout), "rev-parse", "FETCH_HEAD^{commit}"])
    if resolved.decode("ascii").strip() != revision:
        raise SourceAuditError("fetched git object differs from resolved HEAD")
    return checkout


def _archive_commit(checkout: Path, revision: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(checkout), "archive", "--format=tar", revision],
                check=True,
                stdout=output,
                stderr=subprocess.PIPE,
            )
            output.flush()
            os.fsync(output.fileno())
        digest = file_sha256(temporary)
        if destination.exists():
            if file_sha256(destination) != digest:
                raise SourceAuditError("cached source archive hash changed")
        else:
            os.replace(temporary, destination)
        return digest
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceAuditError("git archive failed") from error
    finally:
        temporary.unlink(missing_ok=True)


def _git_files(checkout: Path, revision: str) -> tuple[str, ...]:
    output = _run(
        ["git", "-C", str(checkout), "ls-tree", "-r", "--name-only", revision]
    )
    return tuple(row for row in output.decode("utf-8").splitlines() if row)


def _git_blob(checkout: Path, revision: str, path: str) -> bytes:
    return _run(["git", "-C", str(checkout), "show", f"{revision}:{path}"])


def _license_evidence(
    checkout: Path,
    revision: str,
) -> tuple[str, tuple[LicenseFile, ...]]:
    paths = _git_files(checkout, revision)
    candidates = tuple(
        path
        for path in paths
        if Path(path).name.lower().startswith(("license", "copying", "notice"))
        or Path(path).name in {"pyproject.toml", "package.json"}
    )
    expression = "NOASSERTION"
    records: list[LicenseFile] = []
    for path in sorted(candidates):
        raw = _git_blob(checkout, revision, path)
        records.append(LicenseFile(path=path, sha256=hashlib.sha256(raw).hexdigest()))
        text = raw.decode("utf-8", errors="replace")
        match = _SPDX.search(text)
        if match is not None:
            expression = match.group(1)
        elif Path(path).name == "pyproject.toml":
            try:
                project = tomllib.loads(text).get("project", {})
                license_value = project.get("license") if isinstance(project, dict) else None
                if isinstance(license_value, str) and license_value:
                    expression = license_value
            except tomllib.TOMLDecodeError:
                pass
        elif Path(path).name == "package.json":
            try:
                import json

                license_value = json.loads(text).get("license")
                if isinstance(license_value, str) and license_value:
                    expression = license_value
            except json.JSONDecodeError:
                pass
    return expression, tuple(records)


def _seal_record(**values: object) -> SourceRecord:
    provisional = SourceRecord.model_validate({**values, "record_sha256": "0" * 64})
    return provisional.model_copy(
        update={"record_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )


def _inventory_git(entry: GitSource, cache_dir: Path, timestamp: datetime) -> SourceRecord:
    revision = _resolve_git_head(entry.repository_url)
    checkout = _ensure_checkout(entry, cache_dir, revision)
    archive = cache_dir / "archives" / f"{entry.source_id}-{revision}.tar"
    archive_sha = _archive_commit(checkout, revision, archive)
    license_expression, license_files = _license_evidence(checkout, revision)
    executable = license_expression != "NOASSERTION"
    if entry.license_required_for_execution and not executable:
        raise SourceAuditError("required external source has no auditable license")
    return _seal_record(
        source_id=entry.source_id,
        methods=entry.methods,
        kind=entry.kind,
        canonical_location=entry.repository_url,
        source_revision=revision,
        source_archive_sha256=archive_sha,
        local_artifact=str(archive),
        license_expression=license_expression,
        license_files=license_files,
        license_required_for_execution=entry.license_required_for_execution,
        executable=executable,
        resolver_versions={"git": _git_version()},
        resolved_at_utc=timestamp,
    )


def _inventory_distribution(
    entry: InstalledDistributionSource,
    timestamp: datetime,
) -> SourceRecord:
    try:
        distribution = importlib.metadata.distribution(entry.distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise SourceAuditError(
            f"installed distribution is missing: {entry.distribution}"
        ) from error
    if distribution.version != entry.locked_version:
        raise SourceAuditError("installed distribution version differs from the source registry")
    files = sorted(str(path) for path in (distribution.files or ()))
    manifest = canonical_json_bytes(
        {
            "distribution": entry.distribution,
            "version": distribution.version,
            "files": files,
        }
    )
    artifact_sha = hashlib.sha256(manifest).hexdigest()
    license_values = distribution.metadata.get_all(
        "License-Expression"
    ) or distribution.metadata.get_all("License")
    license_expression = license_values[0] if license_values else None
    if not license_expression or license_expression.strip().upper() in {"UNKNOWN", "N/A"}:
        license_expression = "NOASSERTION"
    executable = license_expression != "NOASSERTION"
    if entry.license_required_for_execution and not executable:
        raise SourceAuditError("required installed source has no auditable license")
    return _seal_record(
        source_id=entry.source_id,
        methods=entry.methods,
        kind=entry.kind,
        canonical_location=f"distribution:{entry.distribution}=={entry.locked_version}",
        source_revision=artifact_sha,
        source_archive_sha256=artifact_sha,
        local_artifact="installed-distribution-manifest",
        license_expression=license_expression,
        license_files=(),
        license_required_for_execution=entry.license_required_for_execution,
        executable=executable,
        resolver_versions={"python-importlib-metadata": "stdlib"},
        resolved_at_utc=timestamp,
    )


def _inventory_paper(entry: PaperSource, timestamp: datetime) -> SourceRecord:
    identity = hashlib.sha256(entry.canonical_url.encode("utf-8")).hexdigest()
    return _seal_record(
        source_id=entry.source_id,
        methods=entry.methods,
        kind=entry.kind,
        canonical_location=entry.canonical_url,
        source_revision=identity,
        source_archive_sha256=identity,
        local_artifact="canonical-paper-url",
        license_expression="NOASSERTION",
        license_files=(),
        license_required_for_execution=entry.license_required_for_execution,
        executable=False,
        resolver_versions={"resolver": "canonical-url-v1"},
        resolved_at_utc=timestamp,
    )


def inventory_source(
    entry: SourceEntry,
    *,
    cache_dir: Path,
    resolved_at_utc: datetime | None = None,
) -> SourceRecord:
    """Resolve one source once and seal its exact executable evidence."""

    checked = _SOURCE_ADAPTER.validate_python(entry)
    timestamp = resolved_at_utc or datetime.now(UTC)
    if timestamp.utcoffset() is None:
        raise SourceAuditError("source resolution timestamp must be timezone-aware")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(checked, GitSource):
        return _inventory_git(checked, cache_dir, timestamp)
    if isinstance(checked, InstalledDistributionSource):
        return _inventory_distribution(checked, timestamp)
    return _inventory_paper(checked, timestamp)


def build_source_inventory(records: tuple[SourceRecord, ...]) -> SourceInventory:
    ordered = tuple(sorted(records, key=lambda row: row.source_id))
    provisional = SourceInventory(records=ordered, inventory_sha256="0" * 64)
    return provisional.model_copy(
        update={
            "inventory_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
        }
    )


def load_source_inventory(path: Path) -> SourceInventory:
    try:
        inventory = SourceInventory.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SourceAuditError(f"invalid source inventory: {error}") from error
    if hashlib.sha256(inventory.semantic_bytes).hexdigest() != inventory.inventory_sha256:
        raise SourceAuditError("source inventory hash mismatch")
    return inventory


def verify_source_record(record: SourceRecord) -> None:
    """Verify a sealed local artifact without network access or mutation."""

    if hashlib.sha256(record.semantic_bytes).hexdigest() != record.record_sha256:
        raise SourceAuditError("source record hash changed")
    if record.kind == "git":
        artifact = Path(record.local_artifact)
        if artifact.is_symlink() or not artifact.is_file():
            raise SourceAuditError("sealed git archive is missing")
        if file_sha256(artifact) != record.source_archive_sha256:
            raise SourceAuditError("sealed git archive hash changed")


__all__ = [
    "GitSource",
    "InstalledDistributionSource",
    "LicenseFile",
    "PaperSource",
    "SourceAuditError",
    "SourceEntry",
    "SourceInventory",
    "SourceRecord",
    "SourceRegistry",
    "build_source_inventory",
    "inventory_source",
    "load_source_inventory",
    "load_source_registry",
    "verify_source_record",
]
