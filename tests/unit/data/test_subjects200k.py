from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ratemem.data import load_data_manifest
from ratemem.data.subjects200k import (
    CompositePairPolicy,
    ConceptPartitionPolicy,
    PreparedSubjects200KSnapshot,
    Subjects200KManifest,
    SubjectsShard,
    prepare_subjects200k_snapshot,
)


def _payload(index: int) -> bytes:
    return f"test-parquet-shard-{index:05d}".encode()


def _manifest() -> Subjects200KManifest:
    shards = tuple(
        SubjectsShard(
            path=f"data/train-{index:05d}-of-00032.parquet",
            sha256=hashlib.sha256(_payload(index)).hexdigest(),
            size_bytes=len(_payload(index)),
        )
        for index in range(32)
    )
    return Subjects200KManifest(
        schema_version="memx-subjects200k-snapshot-v1",
        name="subjects200k",
        profile="training",
        repository_id="Yuanshi/Subjects200K",
        revision="0d1cf6536239888f1a8e218790649344810067bc",
        config_name="default",
        split="train",
        license_spdx="Apache-2.0",
        expected_total_bytes=sum(shard.size_bytes for shard in shards),
        shards=shards,
        composite_pair=CompositePairPolicy(
            mode="RGB",
            width=1056,
            height=528,
            image_size=512,
            support_crop=(8, 8, 520, 520),
            query_crop=(528, 8, 1040, 520),
            concept_field="item",
            support_prompt_field="description_0",
            query_prompt_field="description_1",
            validity_field="description_valid",
        ),
        partition=ConceptPartitionPolicy(
            algorithm="sha256_concept_identity_mod_10000",
            seed=20260830,
            train_upper_bound=9000,
            validation_upper_bound=10000,
        ),
    )


def test_committed_subjects200k_manifest_locks_every_upstream_shard() -> None:
    manifest = load_data_manifest(Path("configs/data/subjects200k.yaml"))

    assert isinstance(manifest, Subjects200KManifest)
    assert manifest.revision == "0d1cf6536239888f1a8e218790649344810067bc"
    assert len(manifest.shards) == 32
    assert manifest.expected_total_bytes == 10_553_550_156
    assert manifest.shards[0].sha256 == (
        "3d696ccbdfc736961e75e5b7ce33adae40cd70ffb69cdc27020a25d643971903"
    )


def test_subjects200k_snapshot_download_is_verified_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        local_dir: Path,
        local_files_only: bool,
        endpoint: str | None,
    ) -> str:
        del repo_id, repo_type, revision, local_files_only, endpoint
        calls.append(filename)
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        index = int(filename.split("-")[1])
        path.write_bytes(_payload(index))
        return str(path)

    monkeypatch.setattr("ratemem.data.subjects200k.hf_hub_download", fake_download)
    manifest = _manifest()

    first = prepare_subjects200k_snapshot(manifest, tmp_path)
    second = prepare_subjects200k_snapshot(manifest, tmp_path)

    assert first == second
    assert first.shard_count == 32
    assert calls == [shard.path for shard in manifest.shards]


def test_subjects200k_snapshot_rejects_tampered_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        local_dir = kwargs["local_dir"]
        assert isinstance(local_dir, Path)
        path = local_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        index = int(filename.split("-")[1])
        path.write_bytes(_payload(index))
        return str(path)

    monkeypatch.setattr("ratemem.data.subjects200k.hf_hub_download", fake_download)
    manifest = _manifest()
    prepared = prepare_subjects200k_snapshot(manifest, tmp_path)
    (prepared.root / manifest.shards[4].path).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash or size changed"):
        PreparedSubjects200KSnapshot.load(prepared.root, manifest)


def test_subjects200k_manifest_rejects_missing_shard() -> None:
    payload = _manifest().model_dump()
    payload["shards"] = payload["shards"][:-1]
    payload["expected_total_bytes"] = sum(row["size_bytes"] for row in payload["shards"])

    with pytest.raises(ValueError, match="exactly 32"):
        Subjects200KManifest.model_validate(payload)
