"""Local-only Subjects200K row validation, splitting, and distributed streaming."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from datasets import load_dataset  # type: ignore[import-untyped]
from PIL import Image

from ratemem.data.subjects200k import (
    ConceptPartitionPolicy,
    PreparedSubjects200KSnapshot,
    Subjects200KManifest,
)
from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.pilot.data import rgb_content_sha256


@dataclass(frozen=True, slots=True)
class SubjectsPair:
    concept_token: str
    partition: Literal["train", "validation"]
    support: Image.Image
    query: Image.Image
    support_prompt: str
    query_prompt: str
    category: str
    composite_sha256: str
    support_sha256: str
    query_sha256: str
    row_sha256: str


def _canonical_concept(value: object) -> str:
    if type(value) is not str:
        raise TypeError("Subjects200K concept field must be an exact str")
    normalized = " ".join(value.strip().lower().split())
    if not normalized:
        raise ValueError("Subjects200K concept field must be non-empty")
    return normalized


def concept_partition(
    concept: str,
    policy: ConceptPartitionPolicy,
) -> Literal["train", "validation"]:
    """Assign all rows for the same normalized concept to one immutable split."""

    normalized = _canonical_concept(concept)
    if type(policy) is not ConceptPartitionPolicy:
        raise TypeError("partition policy must be an exact ConceptPartitionPolicy")
    digest = hashlib.sha256(
        policy.seed.to_bytes(8, "big") + b"\0" + normalized.encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % policy.validation_upper_bound
    return "train" if bucket < policy.train_upper_bound else "validation"


def build_subjects_pair(
    row: Mapping[str, object],
    manifest: Subjects200KManifest,
) -> SubjectsPair | None:
    """Validate one decoded composite row and return a pseudonymous pair."""

    if not isinstance(row, Mapping):
        raise TypeError("Subjects200K row must be a mapping")
    if type(manifest) is not Subjects200KManifest:
        raise TypeError("manifest must be an exact Subjects200KManifest")
    if set(row) != {"image", "collection", "quality_assessment", "description"}:
        raise ValueError("Subjects200K row fields changed")
    image = row["image"]
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise TypeError("Subjects200K image must be a decoded RGB PIL image")
    pair = manifest.composite_pair
    if image.size != (pair.width, pair.height):
        raise ValueError("Subjects200K composite geometry changed")
    description_value = row["description"]
    if type(description_value) is not dict:
        raise TypeError("Subjects200K description must be an exact mapping")
    description = cast(dict[str, object], description_value)
    expected_description = {
        pair.concept_field,
        pair.support_prompt_field,
        pair.query_prompt_field,
        "category",
        pair.validity_field,
    }
    if set(description) != expected_description:
        raise ValueError("Subjects200K description fields changed")
    valid = description[pair.validity_field]
    if type(valid) is not bool:
        raise TypeError("Subjects200K description validity must be an exact bool")
    if not valid:
        return None
    concept = _canonical_concept(description[pair.concept_field])
    prompts: list[str] = []
    for name in (pair.support_prompt_field, pair.query_prompt_field, "category"):
        value = description[name]
        if type(value) is not str or not value.strip():
            raise TypeError(f"Subjects200K description {name} must be non-empty text")
        prompts.append(" ".join(value.strip().split()))
    support = image.crop(pair.support_crop)
    query = image.crop(pair.query_crop)
    expected_size = (pair.image_size, pair.image_size)
    if support.size != expected_size or query.size != expected_size:
        raise RuntimeError("Subjects200K crop policy produced the wrong image size")
    composite_hash = rgb_content_sha256(image)
    support_hash = rgb_content_sha256(support)
    query_hash = rgb_content_sha256(query)
    concept_hash = hashlib.sha256(concept.encode("utf-8")).hexdigest()
    concept_token = f"concept_{concept_hash[:24]}"
    partition = concept_partition(concept, manifest.partition)
    identity = {
        "schema_version": "memx-subjects-pair-v1",
        "dataset_revision": manifest.revision,
        "concept_token": concept_token,
        "partition": partition,
        "composite_sha256": composite_hash,
        "support_sha256": support_hash,
        "query_sha256": query_hash,
        "support_prompt": prompts[0],
        "query_prompt": prompts[1],
        "category": prompts[2],
    }
    return SubjectsPair(
        concept_token=concept_token,
        partition=partition,
        support=support,
        query=query,
        support_prompt=prompts[0],
        query_prompt=prompts[1],
        category=prompts[2],
        composite_sha256=composite_hash,
        support_sha256=support_hash,
        query_sha256=query_hash,
        row_sha256=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
    )


def iter_subjects_pairs(
    snapshot: PreparedSubjects200KSnapshot,
    manifest: Subjects200KManifest,
    *,
    partition: Literal["train", "validation"],
    seed: int,
    rank: int,
    world_size: int,
    shuffle_buffer: int,
) -> Iterator[SubjectsPair]:
    """Stream verified local parquet only, with deterministic shuffle and rank sharding."""

    if type(snapshot) is not PreparedSubjects200KSnapshot:
        raise TypeError("snapshot must be an exact PreparedSubjects200KSnapshot")
    if snapshot.manifest_sha256 != manifest.sha256:
        raise ValueError("Subjects200K snapshot and manifest identities differ")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("stream seed must be a nonnegative signed 64-bit integer")
    if (
        type(rank) is not int
        or type(world_size) is not int
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("stream rank must be within a positive world size")
    if type(shuffle_buffer) is not int or shuffle_buffer < 1:
        raise ValueError("shuffle buffer must be a positive exact integer")
    files = [str(snapshot.root / shard.path) for shard in manifest.shards]
    dataset: Any = load_dataset(
        "parquet",
        data_files={manifest.split: files},
        split=manifest.split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=False)
    for raw in dataset:
        pair = build_subjects_pair(raw, manifest)
        if pair is not None and pair.partition == partition:
            yield pair


__all__ = [
    "SubjectsPair",
    "build_subjects_pair",
    "concept_partition",
    "iter_subjects_pairs",
]
