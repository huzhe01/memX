from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ratemem.evaluation.canonical import (
    MutableLockValueError,
    canonical_json_bytes,
    file_sha256,
    require_immutable_value,
    semantic_sha256,
    write_json_atomic,
    write_yaml_atomic,
)
from ratemem.evaluation.types import (
    ConceptToken,
    GitCommit,
    PhaseId,
    ScientificProfile,
    Sha256,
)


def test_canonical_hash_ignores_mapping_order_and_only_seal_metadata(
    tmp_path: Path,
) -> None:
    left = {
        "b": 2,
        "a": 1,
        "lock_id": "old",
        "sealed_at_utc": "2026-01-01T00:00:00Z",
    }
    right = {
        "a": 1,
        "b": 2,
        "lock_id": "new",
        "sealed_at_utc": "2026-08-24T00:00:00Z",
    }

    assert canonical_json_bytes(left) != canonical_json_bytes(right)
    assert semantic_sha256(left) == semantic_sha256(right)
    assert semantic_sha256({**right, "b": 3}) != semantic_sha256(right)

    output = tmp_path / "nested" / "lock.yaml"
    write_yaml_atomic(output, {"b": 2, "a": 1})
    assert output.read_text(encoding="utf-8") == "a: 1\nb: 2\n"


@pytest.mark.parametrize(
    "value",
    ["latest", "main", "master", "unknown", "", "unresolved", " not set "],
)
def test_mutable_lock_values_are_rejected(value: str) -> None:
    with pytest.raises(MutableLockValueError, match="revision"):
        require_immutable_value("revision", value)


def test_immutable_value_preserves_the_exact_resolved_identity() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    assert require_immutable_value("revision", revision) == revision

    with pytest.raises(ValueError, match="canonical"):
        require_immutable_value("revision", f" {revision}")
    with pytest.raises(TypeError, match="exact str"):
        require_immutable_value("revision", 7)  # type: ignore[arg-type]


def test_named_string_aliases_enforce_exact_patterns() -> None:
    assert TypeAdapter(Sha256).validate_python("a" * 64) == "a" * 64
    assert TypeAdapter(GitCommit).validate_python("b" * 40) == "b" * 40
    assert TypeAdapter(ConceptToken).validate_python("<concept_000123>") == "<concept_000123>"
    assert (
        TypeAdapter(ScientificProfile).validate_python("ratemem-scientific-study-a")
        == "ratemem-scientific-study-a"
    )
    assert TypeAdapter(PhaseId).validate_python("meta_train_seed_0") == "meta_train_seed_0"

    for adapter, invalid in (
        (TypeAdapter(Sha256), "A" * 64),
        (TypeAdapter(GitCommit), "b" * 39),
        (TypeAdapter(ConceptToken), "person-name"),
        (TypeAdapter(ScientificProfile), "study-a"),
        (TypeAdapter(PhaseId), "Meta Train"),
    ):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)


def test_canonical_json_rejects_nonfinite_and_unsupported_values() -> None:
    with pytest.raises(ValueError, match="JSON"):
        canonical_json_bytes({"score": float("nan")})
    with pytest.raises(TypeError, match="JSON"):
        canonical_json_bytes({"path": Path("not-json")})


def test_atomic_json_and_file_hash_are_exact(tmp_path: Path) -> None:
    output = tmp_path / "records" / "lock.json"
    write_json_atomic(output, {"z": [2, 1], "a": "概念"})

    expected = b'{"a":"\xe6\xa6\x82\xe5\xbf\xb5","z":[2,1]}\n'
    assert output.read_bytes() == expected
    assert file_sha256(output) == hashlib.sha256(expected).hexdigest()
    assert not list(output.parent.glob(f".{output.name}.*"))
