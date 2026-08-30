from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ratemem.data.manifest import DatasetManifest


def manifest_payload(
    *,
    revision: str = "71ef669fa76b5e5fecdb0500d48c4bbd685a7668",
    train: list[str] | None = None,
    validation: list[str] | None = None,
    test: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "memx-dataset-v1",
        "name": "memx-smoke",
        "revision": revision,
        "license_spdx": "CC0-1.0",
        "profile": "smoke",
        "source": {
            "kind": "generated",
            "identifier": "memx-smoke-rgb-v1",
        },
        "splits": [
            {"name": "train", "concepts": train or ["amber-cube", "blue-ring"]},
            {"name": "validation", "concepts": validation or ["green-star"]},
            {"name": "test", "concepts": test or ["violet-cone"]},
        ],
    }


def write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_rejects_mutable_revision(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "dataset.yaml", manifest_payload(revision="main"))

    with pytest.raises(ValidationError, match="immutable 40-character"):
        DatasetManifest.load(path)


def test_manifest_rejects_split_overlap(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path / "dataset.yaml",
        manifest_payload(train=["same-concept"], validation=["same-concept"]),
    )

    with pytest.raises(ValidationError, match="globally disjoint"):
        DatasetManifest.load(path)


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = manifest_payload()
    payload["download_without_hash_check"] = True
    path = write_manifest(tmp_path / "dataset.yaml", payload)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DatasetManifest.load(path)


def test_manifest_hash_is_independent_of_yaml_key_order(tmp_path: Path) -> None:
    payload = manifest_payload()
    reversed_payload = dict(reversed(tuple(payload.items())))

    first = DatasetManifest.load(write_manifest(tmp_path / "first.yaml", payload))
    second = DatasetManifest.load(
        write_manifest(tmp_path / "second.yaml", reversed_payload)
    )

    assert first.sha256 == second.sha256
    assert first.canonical_bytes() == second.canonical_bytes()


def test_committed_smoke_manifest_is_locked_and_disjoint() -> None:
    manifest = DatasetManifest.load(Path("configs/data/smoke.yaml"))

    assert manifest.profile == "smoke"
    assert manifest.source.kind == "generated"
    assert manifest.concepts_for("train") == (
        "amber-cube",
        "blue-ring",
        "coral-arch",
        "silver-kite",
    )
    assert manifest.concepts_for("validation") == ("green-star", "red-spiral")
    assert manifest.concepts_for("test") == ("violet-cone", "yellow-prism")
