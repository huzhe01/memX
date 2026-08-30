from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ratemem.training.segments import (
    FrozenTrainingEvent,
    FrozenVisibleTrace,
    SegmentPolicy,
    load_visible_trace,
    segment_trace,
)


def _trace() -> FrozenVisibleTrace:
    events = (
        FrozenTrainingEvent(0, "create", "a"),
        FrozenTrainingEvent(1, "read", "a", prompt_id="p", has_training_query=True),
        FrozenTrainingEvent(
            2,
            "update",
            "a",
            support_image_ids=("s",),
            has_training_query=True,
        ),
        FrozenTrainingEvent(3, "delete", "a"),
    )
    return FrozenVisibleTrace("a" * 64, "train", "b" * 64, "c" * 64, events)


def test_segment_builder_is_deterministic_and_caps_events_and_queries() -> None:
    policy = SegmentPolicy(length=2, maximum_queries=2)
    first = segment_trace(_trace(), policy)
    second = segment_trace(_trace(), policy)
    assert first == second
    assert all(len(segment.events) <= 2 for segment in first)
    assert all(
        sum(event.has_training_query for event in segment.events) <= 2
        for segment in first
    )


@pytest.mark.parametrize("split", ["final_test", "test"])
def test_training_loader_rejects_nonvisible_split(
    tmp_path: Path,
    split: str,
) -> None:
    manifest = tmp_path / f"{split}-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "trace_id": "a" * 64,
                "split": split,
                "payload_path": "trace.jsonl",
                "payload_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="manifest path|train or validation"):
        load_visible_trace(manifest, expected_manifest_sha256=expected)


def test_visible_loader_verifies_manifest_and_payload_hashes(tmp_path: Path) -> None:
    payload = tmp_path / "trace.jsonl"
    payload.write_text(
        json.dumps({"event_index": 0, "kind": "create", "handle": "a"}) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "train-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "trace_id": "a" * 64,
                "split": "train",
                "payload_path": payload.name,
                "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    trace = load_visible_trace(manifest, expected)
    assert trace.split == "train"
    assert len(trace.events) == 1
    payload.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash"):
        load_visible_trace(manifest, expected)
