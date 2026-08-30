"""Hash-checked visible-trace loading and deterministic bounded segmentation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PATH_TYPE = type(Path())


@dataclass(frozen=True, slots=True)
class SegmentPolicy:
    length: Literal[2]
    maximum_queries: Literal[2]


@dataclass(frozen=True, slots=True)
class FrozenTrainingEvent:
    event_index: int
    kind: Literal["create", "update", "read", "delete"]
    handle: str
    support_image_ids: tuple[str, ...] = ()
    description_id: str | None = None
    prompt_id: str | None = None
    generation_seed: int | None = None
    has_training_query: bool = False

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("training event index must be a nonnegative exact integer")
        if type(self.handle) is not str or not self.handle:
            raise ValueError("training event handle must be a nonempty exact string")
        if type(self.support_image_ids) is not tuple:
            raise TypeError("support_image_ids must be an exact tuple")
        expected_query = self.kind in {"read", "update"}
        if self.has_training_query is not expected_query:
            raise ValueError("training query marker differs from the event kind")


@dataclass(frozen=True, slots=True)
class FrozenVisibleTrace:
    trace_id: str
    split: Literal["train", "validation"]
    manifest_sha256: str
    payload_sha256: str
    events: tuple[FrozenTrainingEvent, ...]


@dataclass(frozen=True, slots=True)
class TrainingSegment:
    trace_id: str
    segment_index: int
    events: tuple[FrozenTrainingEvent, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _payload_path(manifest_path: Path, value: object) -> Path:
    if type(value) is not str or not value:
        raise ValueError("visible trace payload path must be nonempty text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError("visible trace payload path must be confined")
    return manifest_path.parent / Path(*pure.parts)


def _event(value: object) -> FrozenTrainingEvent | None:
    if type(value) is not dict:
        raise TypeError("visible trace event must be an exact object")
    row = cast(dict[str, Any], value)
    kind = row.get("kind")
    if kind == "probe":
        return None
    if kind not in {"create", "update", "read", "delete"}:
        raise ValueError("visible trace contains an unsupported event kind")
    support = row.get("support_image_ids", [])
    if type(support) is not list or any(type(item) is not str for item in support):
        raise TypeError("visible trace support ids must be a string list")
    event_index = row.get("event_index")
    handle = row.get("handle")
    if type(event_index) is not int or event_index < 0:
        raise ValueError("visible trace event_index must be a nonnegative exact integer")
    if type(handle) is not str or not handle:
        raise ValueError("visible trace handle must be a nonempty exact string")
    return FrozenTrainingEvent(
        event_index=event_index,
        kind=kind,
        handle=handle,
        support_image_ids=tuple(support),
        description_id=row.get("description_id"),
        prompt_id=row.get("prompt_id"),
        generation_seed=row.get("generation_seed"),
        has_training_query=kind in {"read", "update"},
    )


def load_visible_trace(
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> FrozenVisibleTrace:
    if type(manifest_path) is not _PATH_TYPE:
        raise TypeError("visible trace manifest path must be an exact Path")
    expected = _require_hash(expected_manifest_sha256, "expected manifest hash")
    if "final" in manifest_path.name.lower() or _sha256(manifest_path) != expected:
        raise ValueError("visible trace manifest path or hash is not approved")
    try:
        raw_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("visible trace manifest is unreadable") from error
    if type(raw_value) is not dict:
        raise TypeError("visible trace manifest must be an exact object")
    raw = cast(dict[str, object], raw_value)
    if raw.get("split") not in {"train", "validation"}:
        raise ValueError("training accepts only train or validation traces")
    payload_hash = _require_hash(raw.get("payload_sha256"), "payload hash")
    payload_path = _payload_path(manifest_path, raw.get("payload_path"))
    if _sha256(payload_path) != payload_hash:
        raise ValueError("visible trace payload hash mismatch")
    events: list[FrozenTrainingEvent] = []
    for line in payload_path.read_bytes().splitlines():
        try:
            decoded: object = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("visible trace payload contains invalid JSONL") from error
        event = _event(decoded)
        if event is not None:
            events.append(event)
    if not events:
        raise ValueError("visible trace contains no operational training events")
    if tuple(event.event_index for event in events) != tuple(
        sorted(event.event_index for event in events)
    ):
        raise ValueError("visible training events must remain in source order")
    trace_id = _require_hash(raw.get("trace_id"), "trace id")
    return FrozenVisibleTrace(
        trace_id=trace_id,
        split=cast(Literal["train", "validation"], raw["split"]),
        manifest_sha256=expected,
        payload_sha256=payload_hash,
        events=tuple(events),
    )


def segment_trace(
    trace: FrozenVisibleTrace,
    policy: SegmentPolicy,
) -> tuple[TrainingSegment, ...]:
    if type(trace) is not FrozenVisibleTrace or type(policy) is not SegmentPolicy:
        raise TypeError("segment_trace requires exact trace and policy values")
    output: list[TrainingSegment] = []
    cursor = 0
    while cursor < len(trace.events):
        rows: list[FrozenTrainingEvent] = []
        query_count = 0
        while cursor < len(trace.events) and len(rows) < policy.length:
            event = trace.events[cursor]
            if event.has_training_query and query_count == policy.maximum_queries:
                break
            rows.append(event)
            query_count += int(event.has_training_query)
            cursor += 1
        if not rows:
            raise RuntimeError("segment policy made no progress")
        output.append(
            TrainingSegment(trace.trace_id, len(output), tuple(rows))
        )
    return tuple(output)
