"""Deterministic, concept-disjoint lifecycle trace construction and sealing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import (
    canonical_json_bytes,
    file_sha256,
    write_json_atomic,
    write_text_atomic,
)
from ratemem.evaluation.types import ConceptToken, Sha256

Split: TypeAlias = Literal["train", "validation", "final_test"]
ProtocolName: TypeAlias = Literal[
    "no_pressure",
    "budget_pressure",
    "autonomous_lookup",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_HANDLE = re.compile(r"^h_[0-9a-f]{12}_[0-9]{6}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_SPLITS: tuple[Split, ...] = ("train", "validation", "final_test")


class TraceHashMismatch(ValueError):
    """Raised before replay when a trace payload no longer matches its manifest."""


def _canonical_identifier(value: str, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


class ConceptRecord(BaseModel):
    model_config = _MODEL_CONFIG

    concept_token: ConceptToken
    description_id: str
    support_image_ids: tuple[str, ...]
    prompt_ids: tuple[str, ...]

    @field_validator("description_id")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _canonical_identifier(value, "description_id")

    @field_validator("support_image_ids", "prompt_ids")
    @classmethod
    def validate_pool_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("concept pool ids must be non-empty and unique")
        for item in value:
            _canonical_identifier(item, "concept pool id")
        return value


class ConceptPoolPayload(BaseModel):
    model_config = _MODEL_CONFIG

    concepts: tuple[ConceptRecord, ...]

    @model_validator(mode="after")
    def validate_unique_concepts(self) -> ConceptPoolPayload:
        tokens = tuple(concept.concept_token for concept in self.concepts)
        if not tokens or len(tokens) != len(set(tokens)):
            raise ValueError("concept pool tokens must be non-empty and unique")
        return self


class ConceptPools(BaseModel):
    model_config = _MODEL_CONFIG

    dataset_lock_id: Sha256
    split: Split
    concepts: tuple[ConceptRecord, ...]

    @property
    def concept_pool_sha256(self) -> str:
        payload = [
            {
                "concept_token": concept.concept_token,
                "description_id": concept.description_id,
                "support_image_ids": list(concept.support_image_ids),
            }
            for concept in self.concepts
        ]
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @property
    def prompt_pool_sha256(self) -> str:
        payload = [
            {
                "concept_token": concept.concept_token,
                "prompt_ids": list(concept.prompt_ids),
            }
            for concept in self.concepts
        ]
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class AllPools(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    dataset_lock_id: Sha256
    splits: dict[Split, ConceptPoolPayload]

    @model_validator(mode="after")
    def validate_disjoint_splits(self) -> AllPools:
        if set(self.splits) != set(_SPLITS):
            raise ValueError("concept pools must contain train, validation, and final_test")
        collections: list[set[str]] = []
        for split in _SPLITS:
            pool = self.splits[split]
            values = {
                value
                for concept in pool.concepts
                for value in (
                    concept.concept_token,
                    *concept.support_image_ids,
                    *concept.prompt_ids,
                )
            }
            collections.append(values)
        for index, left in enumerate(collections):
            if any(left & right for right in collections[index + 1 :]):
                raise ValueError("concept, image, and prompt pools must be split-disjoint")
        return self

    @classmethod
    def load(cls, path: Path) -> AllPools:
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid concept pools: {error}") from error

    def for_split(self, split: Split) -> ConceptPools:
        payload = self.splits[split]
        return ConceptPools(
            dataset_lock_id=self.dataset_lock_id,
            split=split,
            concepts=payload.concepts,
        )


class RequestRegime(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["uniform", "zipf"]
    exponent: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_exponent(self) -> RequestRegime:
        if (self.kind == "zipf") != (self.exponent is not None):
            raise ValueError("only a zipf request regime requires an exponent")
        return self


class TracePolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    builder_revision: Literal["lifecycle_trace_v1"]
    seed_namespaces: dict[Split, str]
    event_probabilities: dict[Literal["create", "update", "read", "delete"], float]
    support_shots: tuple[Literal[1, 3, 5], ...]
    maximum_update_support: PositiveInt
    locked_active_set_size: PositiveInt
    events_per_deployment_episode: PositiveInt
    request_regimes: dict[str, RequestRegime]
    protocols: tuple[ProtocolName, ...]
    prompt_seed_pairing: Literal["strict"]
    probe_update_usage: Literal[False]
    handle_format: Literal["h_<trace-prefix>_<event-index>"]
    final_payload_visibility: Literal["encrypted_until_signed_freeze"]

    @model_validator(mode="after")
    def validate_policy(self) -> TracePolicy:
        if set(self.seed_namespaces) != set(_SPLITS) or len(
            set(self.seed_namespaces.values())
        ) != len(_SPLITS):
            raise ValueError("seed namespaces must be complete and unique")
        if set(self.event_probabilities) != {"create", "update", "read", "delete"}:
            raise ValueError("event probabilities must cover all operational events")
        probabilities = tuple(self.event_probabilities.values())
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in probabilities
        ) or abs(sum(probabilities) - 1.0) > 1e-9:
            raise ValueError("event probabilities must be finite, positive, and sum to one")
        if self.support_shots != tuple(sorted(set(self.support_shots))):
            raise ValueError("support_shots must be sorted and unique")
        if not self.request_regimes or not self.protocols:
            raise ValueError("request regimes and protocols must be non-empty")
        return self

    @classmethod
    def load(cls, path: Path) -> TracePolicy:
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"invalid trace policy: {error}") from error


class CreateEvent(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["create"] = "create"
    event_index: NonNegativeInt
    handle: str
    concept_token: ConceptToken
    support_image_ids: tuple[str, ...]
    description_id: str


class UpdateEvent(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["update"] = "update"
    event_index: NonNegativeInt
    handle: str
    support_image_ids: tuple[str, ...]


class ReadEvent(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["read"] = "read"
    event_index: NonNegativeInt
    handle: str
    prompt_id: str
    generation_seed: NonNegativeInt
    update_usage: Literal[True] = True


class DeleteEvent(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["delete"] = "delete"
    event_index: NonNegativeInt
    handle: str


class ProbeEvent(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["probe"] = "probe"
    event_index: NonNegativeInt
    snapshot_event_index: NonNegativeInt
    handle: str
    prompt_id: str
    generation_seed: NonNegativeInt
    update_usage: Literal[False] = False


LifecycleEvent = Annotated[
    CreateEvent | UpdateEvent | ReadEvent | DeleteEvent | ProbeEvent,
    Field(discriminator="kind"),
]
_EVENT_ADAPTER: TypeAdapter[LifecycleEvent] = TypeAdapter(LifecycleEvent)


class Trace(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    trace_id: Sha256
    split: Split
    dataset_lock_id: Sha256
    trace_builder_revision: Literal["lifecycle_trace_v1"]
    trace_seed: NonNegativeInt
    seed_namespace: str
    request_regime: str
    protocol: ProtocolName
    concept_pool_sha256: Sha256
    prompt_pool_sha256: Sha256
    concept_ids: tuple[ConceptToken, ...]
    generation_seeds: tuple[NonNegativeInt, ...]
    events: tuple[LifecycleEvent, ...]

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Trace:
        event_indices = tuple(event.event_index for event in self.events)
        if event_indices != tuple(range(len(self.events))):
            raise ValueError("trace event indices must be contiguous from zero")
        active: set[str] = set()
        retired: set[str] = set()
        observed_seeds: list[int] = []
        observed_concepts: set[str] = set()
        for event in self.events:
            if _HANDLE.fullmatch(event.handle) is None:
                raise ValueError("trace handle format is invalid")
            if event.kind == "create":
                if event.handle in active or event.handle in retired:
                    raise ValueError("create reuses a live or retired handle")
                active.add(event.handle)
                observed_concepts.add(event.concept_token)
            elif event.kind == "delete":
                if event.handle not in active:
                    raise ValueError("delete references an inactive handle")
                active.remove(event.handle)
                retired.add(event.handle)
            elif event.handle not in active:
                raise ValueError(f"{event.kind} references an inactive handle")
            if isinstance(event, ReadEvent | ProbeEvent):
                observed_seeds.append(event.generation_seed)
                if isinstance(event, ProbeEvent) and (
                    event.snapshot_event_index >= event.event_index
                ):
                    raise ValueError("probe snapshot must precede the probe event")
        if tuple(sorted(set(observed_seeds))) != self.generation_seeds:
            raise ValueError("generation seed commitment differs from trace events")
        if tuple(sorted(observed_concepts)) != self.concept_ids:
            raise ValueError("concept commitment differs from create events")
        return self


class TraceSet(BaseModel):
    model_config = _MODEL_CONFIG

    split: Split
    traces: tuple[Trace, ...]
    concept_ids: tuple[ConceptToken, ...]
    trace_ids: tuple[Sha256, ...]
    generation_seeds: tuple[NonNegativeInt, ...]


class TraceManifest(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    trace_id: Sha256
    split: Split
    dataset_lock_id: Sha256
    trace_builder_revision: Literal["lifecycle_trace_v1"]
    trace_seed: NonNegativeInt
    seed_namespace: str
    request_regime: str
    protocol: ProtocolName
    concept_pool_sha256: Sha256
    prompt_pool_sha256: Sha256
    payload_path: str
    payload_sha256: Sha256
    event_counts: dict[str, NonNegativeInt]
    concept_ids: tuple[ConceptToken, ...]
    trace_set_trace_ids: tuple[Sha256, ...]
    trace_set_generation_seeds: tuple[NonNegativeInt, ...]


def _trace_randomness(
    pools: ConceptPools,
    policy: TracePolicy,
    trace_index: int,
) -> tuple[str, int, np.random.Generator, np.random.Generator]:
    if type(trace_index) is not int or trace_index < 0:
        raise ValueError("trace_index must be a nonnegative exact int")
    namespace = policy.seed_namespaces[pools.split]
    namespace_hash = hashlib.sha256(namespace.encode()).digest()
    dataset_prefix = int(pools.dataset_lock_id[:16], 16)
    entropy = [
        dataset_prefix & 0xFFFFFFFF,
        dataset_prefix >> 32,
        int.from_bytes(namespace_hash[:4], "big"),
        int.from_bytes(namespace_hash[4:8], "big"),
        trace_index,
    ]
    sequence = np.random.SeedSequence(entropy)
    operational_sequence, probe_sequence = sequence.spawn(2)
    trace_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    trace_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "dataset_lock_id": pools.dataset_lock_id,
                "namespace": namespace,
                "trace_index": trace_index,
                "trace_seed": trace_seed,
            }
        )
    ).hexdigest()
    return (
        trace_id,
        trace_seed,
        np.random.default_rng(operational_sequence),
        np.random.default_rng(probe_sequence),
    )


def _generation_seed(
    generator: np.random.Generator,
    namespace: str,
    observed: set[int],
) -> int:
    namespace_tag = int(hashlib.sha256(namespace.encode()).hexdigest()[:6], 16)
    while True:
        suffix = int(generator.integers(0, 2**40, endpoint=False))
        value = (namespace_tag << 40) | suffix
        if value not in observed:
            observed.add(value)
            return value


def _choose_active_handle(
    active: dict[str, ConceptRecord],
    regime: RequestRegime,
    generator: np.random.Generator,
) -> str:
    handles = tuple(sorted(active))
    if regime.kind == "uniform":
        return handles[int(generator.integers(0, len(handles)))]
    exponent = cast(float, regime.exponent)
    ranks = np.arange(1, len(handles) + 1, dtype=np.float64)
    probabilities = ranks ** (-exponent)
    probabilities /= probabilities.sum()
    return handles[int(generator.choice(len(handles), p=probabilities))]


def build_trace(
    *,
    split: Split,
    trace_index: int,
    pools: ConceptPools,
    policy: TracePolicy,
    event_count: int,
) -> Trace:
    """Build one legal lifecycle after sampling operational event kinds once."""

    if split != pools.split:
        raise ValueError("requested split differs from the concept pool")
    if type(event_count) is not int or event_count < 1:
        raise ValueError("event_count must be a positive exact int")
    trace_id, trace_seed, generator, probe_generator = _trace_randomness(
        pools,
        policy,
        trace_index,
    )
    regime_name = tuple(sorted(policy.request_regimes))[trace_index % len(policy.request_regimes)]
    regime = policy.request_regimes[regime_name]
    protocol = policy.protocols[trace_index % len(policy.protocols)]
    kind_names: tuple[Literal["create", "update", "read", "delete"], ...] = (
        "create",
        "update",
        "read",
        "delete",
    )
    probabilities = np.asarray(
        [policy.event_probabilities[name] for name in kind_names],
        dtype=np.float64,
    )
    desired_kinds = tuple(
        str(value)
        for value in generator.choice(kind_names, size=event_count, p=probabilities)
    )

    active: dict[str, ConceptRecord] = {}
    events: list[LifecycleEvent] = []
    observed_seeds: set[int] = set()
    concepts = tuple(sorted(pools.concepts, key=lambda concept: concept.concept_token))
    for operation_index, desired_kind in enumerate(desired_kinds):
        inactive = tuple(
            concept
            for concept in concepts
            if concept.concept_token
            not in {active_concept.concept_token for active_concept in active.values()}
        )
        kind = desired_kind
        if not active:
            kind = "create"
        elif kind == "create" and (
            not inactive or len(active) >= policy.locked_active_set_size
        ):
            kind = "read"

        event_index = len(events)
        if kind == "create":
            concept = inactive[int(generator.integers(0, len(inactive)))]
            handle = f"h_{trace_id[:12]}_{event_index:06d}"
            eligible_shots = tuple(
                shot
                for shot in policy.support_shots
                if shot <= len(concept.support_image_ids)
            )
            shot = eligible_shots[int(generator.integers(0, len(eligible_shots)))]
            indices = generator.choice(
                len(concept.support_image_ids),
                size=shot,
                replace=False,
            )
            support_ids = tuple(
                sorted(concept.support_image_ids[int(index)] for index in indices)
            )
            event: LifecycleEvent = CreateEvent(
                event_index=event_index,
                handle=handle,
                concept_token=concept.concept_token,
                support_image_ids=support_ids,
                description_id=concept.description_id,
            )
            active[handle] = concept
        else:
            handle = _choose_active_handle(active, regime, generator)
            concept = active[handle]
            if kind == "update":
                maximum = min(
                    policy.maximum_update_support,
                    len(concept.support_image_ids),
                )
                count = int(generator.integers(1, maximum + 1))
                indices = generator.choice(
                    len(concept.support_image_ids),
                    size=count,
                    replace=False,
                )
                event = UpdateEvent(
                    event_index=event_index,
                    handle=handle,
                    support_image_ids=tuple(
                        sorted(
                            concept.support_image_ids[int(index)] for index in indices
                        )
                    ),
                )
            elif kind == "delete":
                event = DeleteEvent(event_index=event_index, handle=handle)
                del active[handle]
            else:
                prompt_id = concept.prompt_ids[
                    int(generator.integers(0, len(concept.prompt_ids)))
                ]
                event = ReadEvent(
                    event_index=event_index,
                    handle=handle,
                    prompt_id=prompt_id,
                    generation_seed=_generation_seed(
                        generator,
                        policy.seed_namespaces[split],
                        observed_seeds,
                    ),
                )
        events.append(event)

        if (operation_index + 1) % 10 == 0 and active:
            probe_handle = tuple(sorted(active))[
                int(probe_generator.integers(0, len(active)))
            ]
            probe_concept = active[probe_handle]
            events.append(
                ProbeEvent(
                    event_index=len(events),
                    snapshot_event_index=event.event_index,
                    handle=probe_handle,
                    prompt_id=probe_concept.prompt_ids[
                        int(probe_generator.integers(0, len(probe_concept.prompt_ids)))
                    ],
                    generation_seed=_generation_seed(
                        probe_generator,
                        f"{policy.seed_namespaces[split]}:probe",
                        observed_seeds,
                    ),
                )
            )

    observed_concepts = tuple(
        sorted(
            {
                event.concept_token
                for event in events
                if isinstance(event, CreateEvent)
            }
        )
    )
    return Trace(
        schema_version="1.0",
        trace_id=trace_id,
        split=split,
        dataset_lock_id=pools.dataset_lock_id,
        trace_builder_revision=policy.builder_revision,
        trace_seed=trace_seed,
        seed_namespace=policy.seed_namespaces[split],
        request_regime=regime_name,
        protocol=protocol,
        concept_pool_sha256=pools.concept_pool_sha256,
        prompt_pool_sha256=pools.prompt_pool_sha256,
        concept_ids=observed_concepts,
        generation_seeds=tuple(sorted(observed_seeds)),
        events=tuple(events),
    )


def build_trace_set(
    pools: AllPools,
    policy: TracePolicy,
    *,
    counts: Mapping[Split, int],
    event_count: int,
) -> dict[Split, TraceSet]:
    """Build all namespaces and prove cross-split commitments are disjoint."""

    if set(counts) != set(_SPLITS):
        raise ValueError("trace counts must cover train, validation, and final_test")
    result: dict[Split, TraceSet] = {}
    for split in _SPLITS:
        count = counts[split]
        if type(count) is not int or count < 1:
            raise ValueError("each trace count must be a positive exact int")
        traces = tuple(
            build_trace(
                split=split,
                trace_index=index,
                pools=pools.for_split(split),
                policy=policy,
                event_count=event_count,
            )
            for index in range(count)
        )
        result[split] = TraceSet(
            split=split,
            traces=traces,
            concept_ids=tuple(
                sorted({concept for trace in traces for concept in trace.concept_ids})
            ),
            trace_ids=tuple(sorted(trace.trace_id for trace in traces)),
            generation_seeds=tuple(
                sorted({seed for trace in traces for seed in trace.generation_seeds})
            ),
        )
    for attribute in ("concept_ids", "trace_ids", "generation_seeds"):
        commitments = [set(getattr(result[split], attribute)) for split in _SPLITS]
        for index, left in enumerate(commitments):
            if any(left & right for right in commitments[index + 1 :]):
                raise RuntimeError(f"trace {attribute} commitments cross split namespaces")
    return result


def write_trace_set(trace_set: TraceSet, output_dir: Path) -> tuple[Path, ...]:
    """Atomically write visible trace payloads and their hash-bound manifests."""

    if trace_set.split == "final_test":
        raise PermissionError("final_test trace payload must remain encrypted")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"trace output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging-",
        )
    )
    manifest_names: list[str] = []
    try:
        for trace in sorted(trace_set.traces, key=lambda item: item.trace_id):
            payload_name = f"{trace.split}-{trace.trace_id[:16]}.events.jsonl"
            payload = b"".join(
                canonical_json_bytes(event.model_dump(mode="json")) + b"\n"
                for event in trace.events
            )
            payload_path = staging / payload_name
            write_text_atomic(payload_path, payload.decode("utf-8"))
            os.chmod(payload_path, 0o644)
            manifest = TraceManifest(
                schema_version="1.0",
                trace_id=trace.trace_id,
                split=trace.split,
                dataset_lock_id=trace.dataset_lock_id,
                trace_builder_revision=trace.trace_builder_revision,
                trace_seed=trace.trace_seed,
                seed_namespace=trace.seed_namespace,
                request_regime=trace.request_regime,
                protocol=trace.protocol,
                concept_pool_sha256=trace.concept_pool_sha256,
                prompt_pool_sha256=trace.prompt_pool_sha256,
                payload_path=payload_name,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                event_counts=dict(sorted(Counter(event.kind for event in trace.events).items())),
                concept_ids=trace.concept_ids,
                trace_set_trace_ids=trace_set.trace_ids,
                trace_set_generation_seeds=trace_set.generation_seeds,
            )
            manifest_name = f"{trace.split}-{trace.trace_id[:16]}.manifest.json"
            manifest_path = staging / manifest_name
            write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
            os.chmod(manifest_path, 0o644)
            manifest_names.append(manifest_name)
        os.chmod(staging, 0o755)
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return tuple(output_dir / name for name in sorted(manifest_names))


def verify_trace_manifest(manifest_path: Path) -> TraceManifest:
    """Verify payload bytes, typed events, event counts, and seed commitments."""

    try:
        manifest = TraceManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise TraceHashMismatch(f"invalid trace manifest: {error}") from error
    payload_path = manifest_path.parent / manifest.payload_path
    if payload_path.parent != manifest_path.parent or not payload_path.is_file():
        raise TraceHashMismatch("trace payload path is invalid")
    if file_sha256(payload_path) != manifest.payload_sha256:
        raise TraceHashMismatch("trace payload SHA-256 differs from manifest")
    try:
        lines = payload_path.read_text(encoding="utf-8").splitlines()
        events = tuple(
            _EVENT_ADAPTER.validate_python(json.loads(line)) for line in lines
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise TraceHashMismatch(f"trace payload is not typed canonical JSONL: {error}") from error
    if dict(sorted(Counter(event.kind for event in events).items())) != manifest.event_counts:
        raise TraceHashMismatch("trace event counts differ from manifest")
    observed_seeds = tuple(
        sorted(
            {
                event.generation_seed
                for event in events
                if isinstance(event, ReadEvent | ProbeEvent)
            }
        )
    )
    if not set(observed_seeds) <= set(manifest.trace_set_generation_seeds):
        raise TraceHashMismatch("trace generation seeds differ from manifest commitment")
    return manifest


__all__ = [
    "AllPools",
    "ConceptPools",
    "CreateEvent",
    "DeleteEvent",
    "LifecycleEvent",
    "ProbeEvent",
    "ReadEvent",
    "Trace",
    "TraceHashMismatch",
    "TraceManifest",
    "TracePolicy",
    "TraceSet",
    "UpdateEvent",
    "build_trace",
    "build_trace_set",
    "verify_trace_manifest",
    "write_trace_set",
]
