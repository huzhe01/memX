from __future__ import annotations

from pathlib import Path

import pytest

from ratemem.data.manifest import DatasetManifest
from ratemem.data.prepare import PreparedDataset, prepare_dataset


def smoke_manifest() -> DatasetManifest:
    return DatasetManifest.load(Path("configs/data/smoke.yaml"))


def test_independent_smoke_preparations_are_byte_identical(tmp_path: Path) -> None:
    first = prepare_dataset(smoke_manifest(), tmp_path / "first")
    second = prepare_dataset(smoke_manifest(), tmp_path / "second")

    assert first.content_sha256 == second.content_sha256
    assert first.index_sha256 == second.index_sha256
    assert tuple(episode.as_dict() for episode in first.episodes) == tuple(
        episode.as_dict() for episode in second.episodes
    )
    assert len(first.episodes) == 8


def test_preparation_is_idempotent_after_validation(tmp_path: Path) -> None:
    first = prepare_dataset(smoke_manifest(), tmp_path)
    manifest_mtime = (first.root / "prepared-manifest.json").stat().st_mtime_ns

    second = prepare_dataset(smoke_manifest(), tmp_path)

    assert second == first
    assert (second.root / "prepared-manifest.json").stat().st_mtime_ns == manifest_mtime


def test_prepared_split_inventory_is_exact(tmp_path: Path) -> None:
    prepared = prepare_dataset(smoke_manifest(), tmp_path)

    assert tuple(episode.split for episode in prepared.episodes) == (
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    )
    for episode in prepared.episodes:
        assert (prepared.root / episode.support_path).is_file()
        assert (prepared.root / episode.query_path).is_file()


def test_tampered_prepared_image_is_rejected_instead_of_replaced(tmp_path: Path) -> None:
    prepared = prepare_dataset(smoke_manifest(), tmp_path)
    target = prepared.root / prepared.episodes[0].query_path
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="prepared file hash changed"):
        prepare_dataset(smoke_manifest(), tmp_path)


def test_loader_rejects_wrong_expected_manifest_identity(tmp_path: Path) -> None:
    prepared = prepare_dataset(smoke_manifest(), tmp_path)

    with pytest.raises(ValueError, match="dataset manifest hash"):
        PreparedDataset.load(prepared.root, expected_manifest_sha256="0" * 64)


def test_episode_paths_are_relative_and_confined(tmp_path: Path) -> None:
    prepared = prepare_dataset(smoke_manifest(), tmp_path)

    for episode in prepared.episodes:
        for relative in (episode.support_path, episode.query_path):
            assert not relative.is_absolute()
            assert ".." not in relative.parts
            assert (prepared.root / relative).resolve().is_relative_to(
                prepared.root.resolve()
            )
