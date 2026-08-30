"""Deterministic anonymous support, query, and prompt pool construction."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    RootModel,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import (
    canonical_json_bytes,
    file_sha256,
    write_text_atomic,
)
from ratemem.evaluation.types import ConceptToken, Sha256

DatasetSplit: TypeAlias = Literal["train", "validation", "final_test"]
PoolKind: TypeAlias = Literal["support", "query", "prompt"]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_SOURCE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_TEMPLATE_PLACEHOLDER = "{concept}"
_IMAGE_POOL_KINDS: tuple[Literal["support", "query"], ...] = ("support", "query")


class PoolLeakageError(ValueError):
    """Raised before publication when a pool boundary is not disjoint."""


def _identifier(value: str, name: str, *, source: bool = False) -> str:
    pattern = _SOURCE_IDENTIFIER if source else _IDENTIFIER
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


class ImageRecord(BaseModel):
    model_config = _MODEL_CONFIG

    image_id: str
    source_id: str
    concept_id: str
    content_sha256: Sha256
    decoded_pixel_sha256: Sha256
    width: PositiveInt
    height: PositiveInt
    caption_sha256: Sha256 | None = None
    mask_sha256: Sha256 | None = None
    derivative_of: str | None = None
    capture_group: str | None = None
    split: DatasetSplit
    eligible_for_support: bool
    eligible_for_query: bool

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        return _identifier(value, "image_id")

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _identifier(value, "source_id", source=True)

    @field_validator("concept_id")
    @classmethod
    def validate_concept_id(cls, value: str) -> str:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError("concept_id must be non-empty canonical text")
        return value

    @field_validator("derivative_of", "capture_group")
    @classmethod
    def validate_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _identifier(value, "optional image lineage identifier")


class PromptTemplate(BaseModel):
    model_config = _MODEL_CONFIG

    template_id: str
    split: DatasetSplit
    template_text: str
    template_sha256: Sha256

    @field_validator("template_id")
    @classmethod
    def validate_template_id(cls, value: str) -> str:
        return _identifier(value, "template_id")

    @field_validator("template_text")
    @classmethod
    def validate_template_text(cls, value: str) -> str:
        if type(value) is not str or value.count(_TEMPLATE_PLACEHOLDER) != 1:
            raise ValueError("prompt template must contain exactly one {concept} placeholder")
        if value != value.strip() or not value:
            raise ValueError("prompt template must be canonical non-empty text")
        return value

    @model_validator(mode="after")
    def validate_template_hash(self) -> PromptTemplate:
        observed = hashlib.sha256(self.template_text.encode("utf-8")).hexdigest()
        if observed != self.template_sha256:
            raise ValueError("template_sha256 does not match template_text")
        return self


class PoolManifestHeader(BaseModel):
    model_config = _MODEL_CONFIG

    record_type: Literal["header"] = "header"
    schema_version: Literal["1.0"] = "1.0"
    source_id: str
    split: DatasetSplit
    pool_kind: PoolKind
    split_seed_sha256: Sha256
    record_count: int
    records_sha256: Sha256


class PublicImageRecord(BaseModel):
    model_config = _MODEL_CONFIG

    record_type: Literal["image"] = "image"
    image_id: str
    source_id: str
    anonymous_concept_token: ConceptToken
    content_sha256: Sha256
    decoded_pixel_sha256: Sha256
    width: PositiveInt
    height: PositiveInt
    caption_sha256: Sha256 | None
    mask_sha256: Sha256 | None
    derivative_of: str | None
    capture_group_sha256: Sha256 | None


class PromptRecord(BaseModel):
    model_config = _MODEL_CONFIG

    record_type: Literal["prompt"] = "prompt"
    prompt_id: str
    source_id: str
    template_id: str
    split: DatasetSplit
    template_sha256: Sha256
    anonymous_concept_token: ConceptToken
    rendered_prompt_sha256: Sha256


ManifestPayload = Annotated[
    PoolManifestHeader | PublicImageRecord | PromptRecord,
    Field(discriminator="record_type"),
]


class PoolManifestLine(RootModel[ManifestPayload]):
    """Schema wrapper for every canonical JSONL pool line."""


class SplitAssignment(BaseModel):
    model_config = _MODEL_CONFIG

    image_id: str
    split: DatasetSplit

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        return _identifier(value, "image_id")


@dataclass(frozen=True, slots=True)
class PoolBuildResult:
    output_dir: Path
    manifest_paths: tuple[Path, ...]
    private_map_path: Path
    support_image_ids: tuple[str, ...]
    query_image_ids: tuple[str, ...]
    concept_tokens: tuple[str, ...]
    rendered_prompts: tuple[str, ...]
    manifest_sha256: str


def _strict_json_object(line: str, *, path: Path, line_number: int) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            line,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}:{line_number} is not strict JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{path}:{line_number} must contain a JSON object")
    return value


def _read_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read JSONL catalog {path}: {error}") from error
    if not lines or any(not line for line in lines):
        raise ValueError(f"JSONL catalog {path} must contain non-empty lines")
    return tuple(
        _strict_json_object(line, path=path, line_number=index)
        for index, line in enumerate(lines, start=1)
    )


def read_image_records(path: Path) -> tuple[ImageRecord, ...]:
    """Read strict JSONL image records in their declared form."""

    return tuple(ImageRecord.model_validate(value) for value in _read_jsonl_objects(path))


def read_prompt_templates(path: Path) -> tuple[PromptTemplate, ...]:
    """Read strict JSONL prompt templates and verify every content hash."""

    return tuple(PromptTemplate.model_validate(value) for value in _read_jsonl_objects(path))


def read_split_assignments(path: Path) -> tuple[SplitAssignment, ...]:
    values = _read_catalog_objects(path)
    assignments = tuple(SplitAssignment.model_validate(value) for value in values)
    image_ids = tuple(assignment.image_id for assignment in assignments)
    if len(image_ids) != len(set(image_ids)):
        raise PoolLeakageError("split assignments contain duplicate image ids")
    return assignments


def _read_catalog_objects(path: Path) -> tuple[dict[str, Any], ...]:
    if path.suffix == ".jsonl":
        return _read_jsonl_objects(path)
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError("Parquet catalogs require the science dependency extra") from error
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read Parquet catalog {path}: {error}") from error
        frame = frame.astype(object).where(frame.notna(), None)
        return tuple(dict(value) for value in frame.to_dict(orient="records"))
    raise ValueError("catalog must use .jsonl or .parquet")


def _token_map(concepts: Sequence[str], split_seed: int) -> dict[str, str]:
    if type(split_seed) is not int or not 0 <= split_seed < 2**63:
        raise ValueError("split_seed must be a nonnegative signed 64-bit exact int")
    key = split_seed.to_bytes(8, byteorder="big", signed=False)
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for concept in sorted(set(concepts)):
        digest = hmac.new(key, concept.encode("utf-8"), hashlib.sha256).digest()
        number = int.from_bytes(digest[:8], byteorder="big") % 1_000_000
        token = f"<concept_{number:06d}>"
        previous = reverse.get(token)
        if previous is not None and previous != concept:
            raise PoolLeakageError("anonymous concept token collision; choose a new split seed")
        mapping[concept] = token
        reverse[token] = concept
    return mapping


def _pseudonymous_image_map(
    records: Sequence[ImageRecord],
    split_seed: int,
) -> dict[str, str]:
    key = split_seed.to_bytes(8, byteorder="big", signed=False)
    mapping: dict[str, str] = {}
    reverse: set[str] = set()
    for image_id in sorted(record.image_id for record in records):
        digest = hmac.new(
            key,
            f"image\0{image_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        public_id = f"image_{digest[:24]}"
        if public_id in reverse:
            raise PoolLeakageError("anonymous image id collision; choose a new split seed")
        mapping[image_id] = public_id
        reverse.add(public_id)
    return mapping


def _pseudonymous_template_map(
    prompts: Sequence[PromptTemplate],
    split_seed: int,
) -> dict[str, str]:
    key = split_seed.to_bytes(8, byteorder="big", signed=False)
    mapping: dict[str, str] = {}
    reverse: set[str] = set()
    for template_id in sorted(prompt.template_id for prompt in prompts):
        digest = hmac.new(
            key,
            f"template\0{template_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        public_id = f"template_{digest[:24]}"
        if public_id in reverse:
            raise PoolLeakageError("anonymous template id collision; choose a new split seed")
        mapping[template_id] = public_id
        reverse.add(public_id)
    return mapping


def _validate_disjointness(
    records: tuple[ImageRecord, ...],
    prompts: tuple[PromptTemplate, ...],
) -> None:
    by_image: dict[str, ImageRecord] = {}
    for record in records:
        if record.image_id in by_image:
            raise PoolLeakageError("image ids must be globally unique")
        by_image[record.image_id] = record
        if record.eligible_for_support and record.eligible_for_query:
            raise PoolLeakageError(f"{record.image_id} is eligible for both support and query")
        if not record.eligible_for_support and not record.eligible_for_query:
            raise PoolLeakageError(f"{record.image_id} is not assigned to a public pool")

    concept_splits: dict[str, set[str]] = {}
    for record in records:
        concept_splits.setdefault(record.concept_id, set()).add(record.split)
    if any(len(splits) != 1 for splits in concept_splits.values()):
        raise PoolLeakageError("concept ancestry crosses splits")

    template_splits: dict[str, set[str]] = {}
    private_concepts = tuple(concept.casefold() for concept in concept_splits)
    for prompt in prompts:
        template_splits.setdefault(prompt.template_id, set()).add(prompt.split)
    if any(len(splits) != 1 for splits in template_splits.values()):
        raise PoolLeakageError("template id crosses splits")
    template_ids = tuple(prompt.template_id for prompt in prompts)
    if len(template_ids) != len(set(template_ids)):
        raise PoolLeakageError("template ids must be globally unique")
    record_splits = {record.split for record in records}
    prompt_splits = {prompt.split for prompt in prompts}
    if record_splits != prompt_splits:
        raise PoolLeakageError("prompt templates must exactly cover image splits")
    for prompt in prompts:
        if any(concept in prompt.template_text.casefold() for concept in private_concepts):
            raise PoolLeakageError("prompt template contains a private concept identity")
        observed = hashlib.sha256(prompt.template_text.encode("utf-8")).hexdigest()
        if observed != prompt.template_sha256:
            raise PoolLeakageError("prompt template content hash changed")
    pool_by_image = {
        record.image_id: "support" if record.eligible_for_support else "query"
        for record in records
    }
    for record in records:
        visited = {record.image_id}
        parent_id = record.derivative_of
        while parent_id is not None:
            if parent_id in visited:
                raise PoolLeakageError("derivative ancestry contains a cycle")
            visited.add(parent_id)
            parent = by_image.get(parent_id)
            if parent is None:
                raise PoolLeakageError("derivative ancestry references a missing image")
            if parent.split != record.split:
                raise PoolLeakageError("derivative ancestry crosses splits")
            if parent.source_id != record.source_id:
                raise PoolLeakageError("derivative ancestry crosses sources")
            if pool_by_image[parent.image_id] != pool_by_image[record.image_id]:
                raise PoolLeakageError("derivative ancestry crosses support/query pools")
            parent_id = parent.derivative_of

    groups: dict[tuple[str, str, str], list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(
            (record.source_id, record.split, record.concept_id), []
        ).append(record)
    for group_records in groups.values():
        if not any(record.eligible_for_support for record in group_records) or not any(
            record.eligible_for_query for record in group_records
        ):
            raise PoolLeakageError("every concept requires disjoint support and query records")
        support_groups = {
            record.capture_group
            for record in group_records
            if record.eligible_for_support and record.capture_group is not None
        }
        query_groups = {
            record.capture_group
            for record in group_records
            if record.eligible_for_query and record.capture_group is not None
        }
        if support_groups & query_groups:
            raise PoolLeakageError("capture groups cross support/query pools")


def _public_image(
    record: ImageRecord,
    token: str,
    image_ids: Mapping[str, str],
) -> PublicImageRecord:
    capture_hash = (
        None
        if record.capture_group is None
        else hashlib.sha256(record.capture_group.encode("utf-8")).hexdigest()
    )
    return PublicImageRecord(
        image_id=image_ids[record.image_id],
        source_id=record.source_id,
        anonymous_concept_token=token,
        content_sha256=record.content_sha256,
        decoded_pixel_sha256=record.decoded_pixel_sha256,
        width=record.width,
        height=record.height,
        caption_sha256=record.caption_sha256,
        mask_sha256=record.mask_sha256,
        derivative_of=(
            None if record.derivative_of is None else image_ids[record.derivative_of]
        ),
        capture_group_sha256=capture_hash,
    )


def _manifest_bytes(
    *,
    source_id: str,
    split: DatasetSplit,
    kind: PoolKind,
    split_seed_hash: str,
    records: Sequence[PublicImageRecord | PromptRecord],
) -> bytes:
    record_lines = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        for record in records
    )
    header = PoolManifestHeader(
        source_id=source_id,
        split=split,
        pool_kind=kind,
        split_seed_sha256=split_seed_hash,
        record_count=len(records),
        records_sha256=hashlib.sha256(record_lines).hexdigest(),
    )
    return canonical_json_bytes(header.model_dump(mode="json")) + b"\n" + record_lines


def _write_private_map(
    path: Path,
    *,
    concept_mapping: Mapping[str, str],
    image_mapping: Mapping[str, str],
    template_mapping: Mapping[str, str],
) -> None:
    payload = (
        canonical_json_bytes(
            {
                "concept_to_token": dict(sorted(concept_mapping.items())),
                "image_to_public_id": dict(sorted(image_mapping.items())),
                "template_to_public_id": dict(sorted(template_mapping.items())),
            }
        )
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_locked_pools(
    records: Sequence[ImageRecord],
    prompts: Sequence[PromptTemplate],
    *,
    split_seed: int,
    output_dir: Path,
) -> PoolBuildResult:
    """Validate all boundaries, then atomically publish deterministic pool manifests."""

    checked_records = tuple(records)
    checked_prompts = tuple(prompts)
    if not checked_records or not checked_prompts:
        raise ValueError("image records and prompt templates must be non-empty")
    if any(type(record) is not ImageRecord for record in checked_records):
        raise TypeError("records must contain exact ImageRecord values")
    if any(type(prompt) is not PromptTemplate for prompt in checked_prompts):
        raise TypeError("prompts must contain exact PromptTemplate values")
    if type(output_dir) is not type(Path()):
        raise TypeError("output_dir must be an exact concrete Path")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"pool output already exists: {output_dir}")

    _validate_disjointness(checked_records, checked_prompts)
    concepts = tuple(record.concept_id for record in checked_records)
    tokens = _token_map(concepts, split_seed)
    image_ids = _pseudonymous_image_map(checked_records, split_seed)
    template_ids = _pseudonymous_template_map(checked_prompts, split_seed)
    split_seed_hash = hashlib.sha256(str(split_seed).encode("ascii")).hexdigest()

    sorted_records = tuple(
        sorted(
            checked_records,
            key=lambda record: (
                record.split,
                record.source_id,
                record.concept_id,
                record.image_id,
            ),
        )
    )
    sources_by_split: dict[tuple[str, DatasetSplit], set[str]] = {}
    for record in sorted_records:
        sources_by_split.setdefault((record.source_id, record.split), set()).add(
            record.concept_id
        )

    payloads: dict[str, bytes] = {}
    support_ids: list[str] = []
    query_ids: list[str] = []
    rendered_prompts: list[str] = []
    for (source_id, split), source_concepts in sorted(sources_by_split.items()):
        matching = tuple(
            record
            for record in sorted_records
            if record.source_id == source_id and record.split == split
        )
        for kind in _IMAGE_POOL_KINDS:
            selected = tuple(
                record
                for record in matching
                if (
                    record.eligible_for_support
                    if kind == "support"
                    else record.eligible_for_query
                )
            )
            public_records = tuple(
                _public_image(record, tokens[record.concept_id], image_ids)
                for record in selected
            )
            filename = f"{source_id}--{split}--{kind}.jsonl"
            payloads[filename] = _manifest_bytes(
                source_id=source_id,
                split=split,
                kind=kind,
                split_seed_hash=split_seed_hash,
                records=public_records,
            )
            target_ids = support_ids if kind == "support" else query_ids
            target_ids.extend(image_ids[record.image_id] for record in selected)

        prompt_records: list[PromptRecord] = []
        for prompt in sorted(
            (item for item in checked_prompts if item.split == split),
            key=lambda item: item.template_id,
        ):
            for concept in sorted(source_concepts):
                token = tokens[concept]
                rendered = prompt.template_text.replace(_TEMPLATE_PLACEHOLDER, token)
                rendered_prompts.append(rendered)
                prompt_id = hashlib.sha256(
                    f"{source_id}\0{prompt.template_id}\0{token}".encode()
                ).hexdigest()
                prompt_records.append(
                    PromptRecord(
                        prompt_id=prompt_id,
                        source_id=source_id,
                        template_id=template_ids[prompt.template_id],
                        split=split,
                        template_sha256=prompt.template_sha256,
                        anonymous_concept_token=token,
                        rendered_prompt_sha256=hashlib.sha256(
                            rendered.encode("utf-8")
                        ).hexdigest(),
                    )
                )
        filename = f"{source_id}--{split}--prompt.jsonl"
        payloads[filename] = _manifest_bytes(
            source_id=source_id,
            split=split,
            kind="prompt",
            split_seed_hash=split_seed_hash,
            records=tuple(prompt_records),
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging-",
        )
    )
    try:
        for filename, payload in sorted(payloads.items()):
            public_path = staging / filename
            write_text_atomic(public_path, payload.decode("utf-8"))
            os.chmod(public_path, 0o644)
        private_map = staging / "private-concept-map.json"
        _write_private_map(
            private_map,
            concept_mapping=tokens,
            image_mapping=image_ids,
            template_mapping=template_ids,
        )
        os.chmod(staging, 0o755)
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    manifest_paths = tuple(output_dir / name for name in sorted(payloads))
    private_map_path = output_dir / "private-concept-map.json"
    identity = {
        path.name: file_sha256(path)
        for path in (*manifest_paths, private_map_path)
    }
    return PoolBuildResult(
        output_dir=output_dir,
        manifest_paths=manifest_paths,
        private_map_path=private_map_path,
        support_image_ids=tuple(sorted(support_ids)),
        query_image_ids=tuple(sorted(query_ids)),
        concept_tokens=tuple(sorted(tokens.values())),
        rendered_prompts=tuple(sorted(rendered_prompts)),
        manifest_sha256=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
    )


def build_pools_from_catalogs(
    *,
    source_catalog: Path,
    prompt_catalog: Path,
    split_assignments: Path,
    split_seed: int,
    output_dir: Path,
) -> PoolBuildResult:
    """Join explicit assignments to JSONL/Parquet catalogs and build locked pools."""

    source_values = _read_catalog_objects(source_catalog)
    assignments = read_split_assignments(split_assignments)
    assignment_map = {assignment.image_id: assignment.split for assignment in assignments}
    source_ids = tuple(value.get("image_id") for value in source_values)
    if any(type(image_id) is not str for image_id in source_ids):
        raise PoolLeakageError("source catalog contains a missing image_id")
    if set(source_ids) != set(assignment_map):
        raise PoolLeakageError("split assignments must exactly cover the source catalog")
    joined: list[ImageRecord] = []
    for value in source_values:
        image_id = value["image_id"]
        if type(image_id) is not str:
            raise PoolLeakageError("source catalog image_id must be an exact str")
        declared = value.get("split")
        assigned = assignment_map[image_id]
        if declared is not None and declared != assigned:
            raise PoolLeakageError("source catalog split differs from explicit assignment")
        joined.append(ImageRecord.model_validate(value | {"split": assigned}))
    return build_locked_pools(
        joined,
        read_prompt_templates(prompt_catalog),
        split_seed=split_seed,
        output_dir=output_dir,
    )


__all__ = [
    "ImageRecord",
    "PoolBuildResult",
    "PoolLeakageError",
    "PoolManifestLine",
    "PromptRecord",
    "PromptTemplate",
    "build_locked_pools",
    "build_pools_from_catalogs",
    "read_image_records",
    "read_prompt_templates",
    "read_split_assignments",
]
