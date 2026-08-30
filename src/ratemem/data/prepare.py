from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ratemem.data.fixture import write_fixture_image
from ratemem.data.manifest import DatasetManifest

_HEX64 = frozenset("0123456789abcdef")
_PREPARED_SCHEMA = "memx-prepared-dataset-v1"
_PATH_TYPE = type(Path())


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX64 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object, name: str) -> Path:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact str")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{name} must be a confined relative POSIX path")
    return Path(*pure.parts)


def _write_synced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class PreparedEpisode:
    episode_id: str
    concept_id: str
    split: str
    prompt: str
    support_path: Path
    query_path: Path
    support_sha256: str
    query_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.episode_id, "episode_id")
        if type(self.concept_id) is not str or not self.concept_id:
            raise TypeError("concept_id must be a non-empty exact str")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("episode split is invalid")
        if type(self.prompt) is not str or not self.prompt:
            raise TypeError("episode prompt must be a non-empty exact str")
        for path, name in (
            (self.support_path, "support_path"),
            (self.query_path, "query_path"),
        ):
            if type(path) is not _PATH_TYPE:
                raise TypeError(f"{name} must be an exact Path")
            _safe_relative(path.as_posix(), name)
        _require_sha256(self.support_sha256, "support_sha256")
        _require_sha256(self.query_sha256, "query_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "concept_id": self.concept_id,
            "episode_id": self.episode_id,
            "prompt": self.prompt,
            "query_path": self.query_path.as_posix(),
            "query_sha256": self.query_sha256,
            "split": self.split,
            "support_path": self.support_path.as_posix(),
            "support_sha256": self.support_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PreparedEpisode:
        if type(value) is not dict:
            raise TypeError("prepared episode must be an exact mapping")
        payload = cast(dict[str, object], value)
        expected = {
            "concept_id",
            "episode_id",
            "prompt",
            "query_path",
            "query_sha256",
            "split",
            "support_path",
            "support_sha256",
        }
        if set(payload) != expected:
            raise ValueError("prepared episode fields changed")
        for name in ("concept_id", "episode_id", "prompt", "split"):
            if type(payload[name]) is not str:
                raise TypeError(f"prepared episode {name} must be an exact str")
        return cls(
            episode_id=cast(str, payload["episode_id"]),
            concept_id=cast(str, payload["concept_id"]),
            split=cast(str, payload["split"]),
            prompt=cast(str, payload["prompt"]),
            support_path=_safe_relative(payload["support_path"], "support_path"),
            query_path=_safe_relative(payload["query_path"], "query_path"),
            support_sha256=_require_sha256(payload["support_sha256"], "support_sha256"),
            query_sha256=_require_sha256(payload["query_sha256"], "query_sha256"),
        )


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    root: Path
    manifest_sha256: str
    index_sha256: str
    content_sha256: str
    episodes: tuple[PreparedEpisode, ...]

    def __post_init__(self) -> None:
        if type(self.root) is not _PATH_TYPE:
            raise TypeError("prepared root must be an exact Path")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.index_sha256, "index_sha256")
        _require_sha256(self.content_sha256, "content_sha256")
        if type(self.episodes) is not tuple or not self.episodes:
            raise ValueError("prepared dataset must contain episodes")
        if any(type(episode) is not PreparedEpisode for episode in self.episodes):
            raise TypeError("prepared episodes must contain exact PreparedEpisode values")

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        expected_manifest_sha256: str,
    ) -> PreparedDataset:
        if type(root) is not _PATH_TYPE:
            raise TypeError("prepared root must be an exact Path")
        expected_hash = _require_sha256(
            expected_manifest_sha256, "expected dataset manifest hash"
        )
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
            raise ValueError("prepared dataset root must be a real directory")
        manifest_path = root / "prepared-manifest.json"
        try:
            payload: Any = json.loads(manifest_path.read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("prepared dataset manifest is unreadable") from error
        if type(payload) is not dict:
            raise TypeError("prepared dataset manifest must be an exact mapping")
        manifest = cast(dict[str, object], payload)
        expected_fields = {
            "schema_version",
            "dataset_manifest_sha256",
            "episode_count",
            "episode_index_sha256",
            "content_sha256",
            "files",
        }
        if set(manifest) != expected_fields or manifest["schema_version"] != _PREPARED_SCHEMA:
            raise ValueError("prepared dataset manifest fields changed")
        observed_manifest_hash = _require_sha256(
            manifest["dataset_manifest_sha256"], "dataset manifest hash"
        )
        if observed_manifest_hash != expected_hash:
            raise ValueError(
                "prepared dataset manifest hash differs from expected dataset manifest hash"
            )
        files_value = manifest["files"]
        if type(files_value) is not list or not files_value:
            raise TypeError("prepared file inventory must be a non-empty exact list")
        normalized_files: list[dict[str, object]] = []
        for item in files_value:
            if type(item) is not dict or set(item) != {"path", "sha256", "size"}:
                raise ValueError("prepared file inventory entry changed")
            entry = cast(dict[str, object], item)
            relative = _safe_relative(entry["path"], "prepared file path")
            expected_file_hash = _require_sha256(entry["sha256"], "prepared file hash")
            if type(entry["size"]) is not int or entry["size"] < 0:
                raise TypeError("prepared file size must be a nonnegative exact int")
            path = root / relative
            file_metadata = path.lstat()
            if not stat.S_ISREG(file_metadata.st_mode) or path.is_symlink():
                raise ValueError("prepared file must be a real regular file")
            observed_file_hash, observed_size = _sha256_file(path)
            if observed_file_hash != expected_file_hash or observed_size != entry["size"]:
                raise ValueError(f"prepared file hash changed: {relative.as_posix()}")
            normalized_files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": expected_file_hash,
                    "size": observed_size,
                }
            )

        index_path = root / "episodes.jsonl"
        index_bytes = index_path.read_bytes()
        index_hash = _sha256_bytes(index_bytes)
        if index_hash != _require_sha256(manifest["episode_index_sha256"], "episode index hash"):
            raise ValueError("prepared episode index hash changed")
        episodes: list[PreparedEpisode] = []
        for line in index_bytes.splitlines():
            try:
                decoded: Any = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("prepared episode index is invalid JSONL") from error
            episodes.append(PreparedEpisode.from_dict(decoded))
        if type(manifest["episode_count"]) is not int or manifest["episode_count"] != len(episodes):
            raise ValueError("prepared episode count changed")
        content_payload = {
            "dataset_manifest_sha256": observed_manifest_hash,
            "episode_count": len(episodes),
            "episode_index_sha256": index_hash,
            "files": normalized_files,
        }
        content_hash = _sha256_bytes(_canonical_json(content_payload))
        if content_hash != _require_sha256(manifest["content_sha256"], "content hash"):
            raise ValueError("prepared dataset content hash changed")
        return cls(
            root=root,
            manifest_sha256=observed_manifest_hash,
            index_sha256=index_hash,
            content_sha256=content_hash,
            episodes=tuple(episodes),
        )


def _episode(
    staging: Path,
    *,
    source_identifier: str,
    split: str,
    concept: str,
) -> PreparedEpisode:
    image_root = Path("images") / split
    support_relative = image_root / f"{concept}-support.png"
    query_relative = image_root / f"{concept}-query.png"
    write_fixture_image(
        staging / support_relative,
        identity=f"{source_identifier}\0{split}\0{concept}\0support",
    )
    write_fixture_image(
        staging / query_relative,
        identity=f"{source_identifier}\0{split}\0{concept}\0query",
    )
    support_hash, _support_size = _sha256_file(staging / support_relative)
    query_hash, _query_size = _sha256_file(staging / query_relative)
    identity = _canonical_json(
        {
            "concept_id": concept,
            "query_sha256": query_hash,
            "split": split,
            "support_sha256": support_hash,
        }
    )
    return PreparedEpisode(
        episode_id=_sha256_bytes(identity),
        concept_id=concept,
        split=split,
        prompt=f"A studio image of the {concept} fixture.",
        support_path=support_relative,
        query_path=query_relative,
        support_sha256=support_hash,
        query_sha256=query_hash,
    )


def _inventory(staging: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "prepared-manifest.json":
            relative = path.relative_to(staging).as_posix()
            digest, size = _sha256_file(path)
            result.append({"path": relative, "sha256": digest, "size": size})
    return result


def prepare_dataset(manifest: DatasetManifest, root: Path) -> PreparedDataset:
    if type(manifest) is not DatasetManifest:
        raise TypeError("manifest must be an exact DatasetManifest")
    if type(root) is not _PATH_TYPE:
        raise TypeError("data root must be an exact Path")
    if manifest.source.kind != "generated":
        raise ValueError("release-one preparation supports only generated smoke data")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{manifest.name}-{manifest.sha256[:16]}"
    if destination.exists() or destination.is_symlink():
        return PreparedDataset.load(
            destination, expected_manifest_sha256=manifest.sha256
        )

    staging = root / f".staging-{manifest.name}-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    episodes = tuple(
        _episode(
            staging,
            source_identifier=manifest.source.identifier,
            split=split.name,
            concept=concept,
        )
        for split in manifest.splits
        for concept in split.concepts
    )
    index_bytes = b"".join(_canonical_json(episode.as_dict()) + b"\n" for episode in episodes)
    _write_synced(staging / "episodes.jsonl", index_bytes)
    files = _inventory(staging)
    content_payload = {
        "dataset_manifest_sha256": manifest.sha256,
        "episode_count": len(episodes),
        "episode_index_sha256": _sha256_bytes(index_bytes),
        "files": files,
    }
    prepared_manifest = {
        "schema_version": _PREPARED_SCHEMA,
        **content_payload,
        "content_sha256": _sha256_bytes(_canonical_json(content_payload)),
    }
    _write_synced(staging / "prepared-manifest.json", _canonical_json(prepared_manifest))
    _sync_directory(staging)
    try:
        staging.rename(destination)
    except FileExistsError:
        return PreparedDataset.load(
            destination, expected_manifest_sha256=manifest.sha256
        )
    _sync_directory(root)
    return PreparedDataset.load(destination, expected_manifest_sha256=manifest.sha256)
