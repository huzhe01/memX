"""Immutable dataset manifests, raw snapshots, and prepared episode stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import yaml  # type: ignore[import-untyped]

from ratemem.data.manifest import DatasetManifest, DatasetSource, DatasetSplit
from ratemem.data.prepare import PreparedDataset, PreparedEpisode, prepare_dataset
from ratemem.data.subjects200k import (
    PreparedSubjects200KSnapshot,
    Subjects200KManifest,
    prepare_subjects200k_snapshot,
)

DataManifest: TypeAlias = DatasetManifest | Subjects200KManifest


def load_data_manifest(path: Path) -> DataManifest:
    """Dispatch a strict data manifest by its explicit schema version."""

    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"data manifest is unreadable: {path}") from error
    if type(payload) is not dict:
        raise TypeError("data manifest root must be an exact mapping")
    schema_version = payload.get("schema_version")
    if schema_version == "memx-dataset-v1":
        return DatasetManifest.model_validate(payload)
    if schema_version == "memx-subjects200k-snapshot-v1":
        return Subjects200KManifest.model_validate(payload)
    raise ValueError(f"unsupported data manifest schema: {schema_version!r}")

__all__ = [
    "DatasetManifest",
    "DatasetSource",
    "DatasetSplit",
    "DataManifest",
    "PreparedDataset",
    "PreparedEpisode",
    "PreparedSubjects200KSnapshot",
    "Subjects200KManifest",
    "load_data_manifest",
    "prepare_dataset",
    "prepare_subjects200k_snapshot",
]
