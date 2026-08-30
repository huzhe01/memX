"""Canonical serialization and atomic publication for scientific records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml  # type: ignore[import-untyped]

MUTABLE_VALUES = frozenset(
    {
        "",
        "latest",
        "main",
        "master",
        "unknown",
        "unresolved",
        "to be determined",
        "not set",
    }
)
SEAL_METADATA = frozenset({"lock_id", "sealed_at_utc"})
_PATH_TYPE = type(Path())


class MutableLockValueError(ValueError):
    """Raised when a scientific lock contains an unresolved identity."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value to its one canonical byte representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Mapping[str, object]) -> str:
    """Hash record semantics while excluding self-referential seal metadata."""

    payload = {key: item for key, item in value.items() if key not in SEAL_METADATA}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading an unbounded artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_immutable_value(field: str, value: str) -> str:
    """Reject known mutable sentinels and non-canonical surrounding whitespace."""

    if type(field) is not str or not field or field != field.strip():
        raise TypeError("field must be a non-empty canonical exact str")
    if type(value) is not str:
        raise TypeError(f"{field} must be an exact str")
    if value.strip().lower() in MUTABLE_VALUES:
        raise MutableLockValueError(f"{field} must be an immutable resolved value")
    if value != value.strip():
        raise ValueError(f"{field} must be canonical text without surrounding whitespace")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    if type(path) is not _PATH_TYPE:
        raise TypeError("path must be an exact pathlib.Path")
    if type(payload) is not bytes:
        raise TypeError("payload must be exact bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically publish canonical JSON terminated by one newline."""

    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def write_yaml_atomic(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish stable, sorted, UTF-8 YAML."""

    rendered = yaml.safe_dump(
        dict(value),
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    _atomic_write_bytes(path, rendered)


__all__ = [
    "MutableLockValueError",
    "canonical_json_bytes",
    "file_sha256",
    "require_immutable_value",
    "semantic_sha256",
    "write_json_atomic",
    "write_yaml_atomic",
]
