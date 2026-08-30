"""Global duplicate evidence, adjudication, and component-safe split allocation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

import imagehash
import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image, ImageOps
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256, semantic_sha256
from ratemem.evaluation.pools import ImageRecord
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_SPLIT_ORDER = ("train", "validation", "final_test")


class CrossSplitComponentError(ValueError):
    """Raised when linked images were already assigned to different splits."""


class PairDecision(str, Enum):
    LINK = "link"
    REVIEW = "review"
    DISTINCT = "distinct"


class FeatureEncoderPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    name: str
    immutable_lock_path: str
    preprocessing: Literal["resize_shorter_518_center_crop_rgb_v1"]


class CropVerificationPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    detector: Literal["opencv_sift_homography_v1"]
    minimum_inliers: PositiveInt
    minimum_inlier_ratio: float = Field(gt=0.0, le=1.0)


class BurstNeighborPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    maximum_seconds: PositiveFloat
    feature_cosine_min: float = Field(ge=-1.0, le=1.0)


class AmbiguousReviewPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    independent_reviewers: Literal[2]
    disagreement_resolution: Literal["third_reviewer_majority"]


class DuplicatePolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    decoded_pixel_hash: Literal["sha256_exif_transposed_rgb_v1"]
    perceptual_hash: Literal["phash_64_v1"]
    phash_max_hamming: NonNegativeInt
    feature_encoder: FeatureEncoderPolicy
    feature_cosine_min: float = Field(ge=-1.0, le=1.0)
    crop_verification: CropVerificationPolicy
    burst_neighbor: BurstNeighborPolicy
    ambiguous_pair_review: AmbiguousReviewPolicy
    cluster_assignment: Literal["connected_components_v1"]

    @classmethod
    def load(cls, path: Path) -> DuplicatePolicy:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"invalid duplicate policy: {error}") from error

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


@dataclass(frozen=True, slots=True)
class PairEvidence:
    left_id: str
    right_id: str
    exact_sha256: bool
    decoded_pixel_equal: bool
    phash_hamming: int | None
    feature_cosine: float | None
    sift_inliers: int | None
    sift_inlier_ratio: float | None
    capture_delta_seconds: float | None
    same_capture_group: bool = False

    def __post_init__(self) -> None:
        for name, identifier in (("left_id", self.left_id), ("right_id", self.right_id)):
            if type(identifier) is not str or _IDENTIFIER.fullmatch(identifier) is None:
                raise ValueError(f"{name} must be a canonical identifier")
        if self.left_id >= self.right_id:
            raise ValueError("pair ids must be strictly canonical and increasing")
        if type(self.exact_sha256) is not bool or type(self.decoded_pixel_equal) is not bool:
            raise TypeError("exact evidence fields must be exact bool values")
        if type(self.same_capture_group) is not bool:
            raise TypeError("same_capture_group must be an exact bool")
        if self.phash_hamming is not None and (
            type(self.phash_hamming) is not int or not 0 <= self.phash_hamming <= 64
        ):
            raise ValueError("phash_hamming must be between zero and 64")
        if self.sift_inliers is not None and (
            type(self.sift_inliers) is not int or self.sift_inliers < 0
        ):
            raise ValueError("sift_inliers must be nonnegative")
        for name, numeric, minimum, maximum in (
            ("feature_cosine", self.feature_cosine, -1.0, 1.0),
            ("sift_inlier_ratio", self.sift_inlier_ratio, 0.0, 1.0),
            ("capture_delta_seconds", self.capture_delta_seconds, 0.0, math.inf),
        ):
            if numeric is not None and (
                type(numeric) is not float
                or not math.isfinite(numeric)
                or not minimum <= numeric <= maximum
            ):
                raise ValueError(f"{name} is outside its finite range")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


def classify_pair(evidence: PairEvidence, policy: DuplicatePolicy) -> PairDecision:
    """Classify one candidate from frozen thresholds without mutable heuristics."""

    if type(evidence) is not PairEvidence or type(policy) is not DuplicatePolicy:
        raise TypeError("classify_pair requires exact evidence and policy values")
    if evidence.exact_sha256 or evidence.decoded_pixel_equal:
        return PairDecision.LINK
    if (
        evidence.phash_hamming is not None
        and evidence.phash_hamming <= policy.phash_max_hamming
    ):
        return PairDecision.LINK
    if (
        evidence.feature_cosine is not None
        and evidence.feature_cosine >= policy.feature_cosine_min
    ):
        return PairDecision.LINK
    if (
        evidence.sift_inliers is not None
        and evidence.sift_inlier_ratio is not None
        and evidence.sift_inliers >= policy.crop_verification.minimum_inliers
        and evidence.sift_inlier_ratio >= policy.crop_verification.minimum_inlier_ratio
    ):
        return PairDecision.LINK
    if (
        evidence.same_capture_group
        and evidence.capture_delta_seconds is not None
        and evidence.capture_delta_seconds <= policy.burst_neighbor.maximum_seconds
        and evidence.feature_cosine is not None
        and evidence.feature_cosine >= policy.burst_neighbor.feature_cosine_min
    ):
        return PairDecision.LINK

    if (
        (
            evidence.phash_hamming is not None
            and evidence.phash_hamming <= policy.phash_max_hamming + 4
        )
        or (
            evidence.feature_cosine is not None
            and evidence.feature_cosine >= policy.burst_neighbor.feature_cosine_min - 0.02
        )
        or (
            evidence.sift_inliers is not None
            and evidence.sift_inliers >= policy.crop_verification.minimum_inliers // 2
        )
    ):
        return PairDecision.REVIEW
    return PairDecision.DISTINCT


class Review(BaseModel):
    model_config = _MODEL_CONFIG

    left_id: str
    right_id: str
    reviewer_id: str
    decision: Literal["link", "distinct"]
    evidence_sha256: Sha256

    @field_validator("left_id", "right_id", "reviewer_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("review identifiers must be canonical")
        return value

    @model_validator(mode="after")
    def validate_pair_order(self) -> Review:
        if self.left_id >= self.right_id:
            raise ValueError("review pair ids must be strictly increasing")
        return self


def validate_adjudications(
    candidates: Sequence[PairEvidence],
    reviews: Sequence[Review],
) -> list[tuple[str, str]]:
    """Validate two-reviewer agreement or a three-reviewer majority."""

    candidate_map: dict[tuple[str, str], PairEvidence] = {}
    for candidate in candidates:
        pair = (candidate.left_id, candidate.right_id)
        if pair in candidate_map:
            raise ValueError("adjudication candidates must be unique")
        candidate_map[pair] = candidate
    grouped: dict[tuple[str, str], list[Review]] = {}
    for review in reviews:
        pair = (review.left_id, review.right_id)
        matched_candidate = candidate_map.get(pair)
        if matched_candidate is None:
            raise ValueError("review does not correspond to a candidate pair")
        if review.evidence_sha256 != matched_candidate.sha256:
            raise ValueError("review evidence hash differs from candidate evidence")
        grouped.setdefault(pair, []).append(review)
    if set(grouped) != set(candidate_map):
        raise ValueError("every ambiguous candidate requires adjudication")

    linked: list[tuple[str, str]] = []
    for pair, pair_reviews in sorted(grouped.items()):
        reviewer_ids = tuple(review.reviewer_id for review in pair_reviews)
        if len(reviewer_ids) not in {2, 3} or len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError("adjudication requires independent reviewers")
        decisions = tuple(review.decision for review in pair_reviews)
        if len(decisions) == 2 and len(set(decisions)) != 1:
            raise ValueError("reviewer disagreement requires a third reviewer")
        link_votes = decisions.count("link")
        if link_votes > len(decisions) // 2:
            linked.append(pair)
    return linked


class DuplicateComponent(BaseModel):
    model_config = _MODEL_CONFIG

    component_id: str
    image_ids: tuple[str, ...]
    concept_ids: tuple[str, ...]
    split: Literal["train", "validation", "final_test"]


class ComponentReport(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    components: tuple[DuplicateComponent, ...]
    component_sha256: Sha256


def assign_components(
    records: Sequence[ImageRecord],
    linked_pairs: Sequence[tuple[str, str]],
) -> ComponentReport:
    """Build deterministic global components and reject preassigned split crossings."""

    record_map: dict[str, ImageRecord] = {}
    for record in records:
        if record.image_id in record_map:
            raise ValueError("component records require globally unique image ids")
        record_map[record.image_id] = record
    parent = {image_id: image_id for image_id in record_map}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    seen_pairs: set[tuple[str, str]] = set()
    for raw_left, raw_right in linked_pairs:
        left, right = sorted((raw_left, raw_right))
        if left == right or left not in record_map or right not in record_map:
            raise ValueError("linked pair references an invalid image id")
        pair = (left, right)
        if pair in seen_pairs:
            raise ValueError("linked pairs must be unique")
        seen_pairs.add(pair)
        union(left, right)

    groups: dict[str, list[ImageRecord]] = {}
    for image_id, record in record_map.items():
        groups.setdefault(find(image_id), []).append(record)
    components: list[DuplicateComponent] = []
    for group_records in groups.values():
        image_ids = tuple(sorted(record.image_id for record in group_records))
        splits = {record.split for record in group_records}
        if len(splits) != 1:
            raise CrossSplitComponentError(
                f"duplicate component {' '.join(image_ids)} crosses preassigned splits"
            )
        component_id = hashlib.sha256("\0".join(image_ids).encode()).hexdigest()
        components.append(
            DuplicateComponent(
                component_id=component_id,
                image_ids=image_ids,
                concept_ids=tuple(sorted({record.concept_id for record in group_records})),
                split=next(iter(splits)),
            )
        )
    components.sort(key=lambda component: component.component_id)
    payload = [component.model_dump(mode="json") for component in components]
    return ComponentReport(
        components=tuple(components),
        component_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )


def allocate_component_splits(
    concept_components: Sequence[set[str]],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole connected concept components using an order-independent hash."""

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit exact int")
    if set(ratios) != set(_SPLIT_ORDER):
        raise ValueError("ratios must contain train, validation, and final_test")
    checked_ratios: list[float] = []
    for split in _SPLIT_ORDER:
        value = ratios[split]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError("split ratios must be finite and nonnegative")
        checked_ratios.append(float(value))
    if abs(sum(checked_ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")

    normalized: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for component in concept_components:
        if type(component) is not set or not component:
            raise TypeError("concept components must be non-empty exact sets")
        values = tuple(sorted(component))
        if any(type(value) is not str or not value for value in values):
            raise ValueError("concept identifiers must be non-empty strings")
        if seen & set(values):
            raise ValueError("concept components must be disjoint")
        seen.update(values)
        normalized.append(values)

    result: dict[str, str] = {}
    cumulative = np.cumsum(np.asarray(checked_ratios, dtype=np.float64))
    for normalized_component in sorted(normalized):
        payload = f"{seed}\0{'|'.join(normalized_component)}".encode()
        score = int.from_bytes(hashlib.sha256(payload).digest(), "big") / 2**256
        index = int(np.searchsorted(cumulative, score, side="right"))
        index = min(index, len(_SPLIT_ORDER) - 1)
        for concept in normalized_component:
            result[concept] = _SPLIT_ORDER[index]
    return result


class FrozenFeatureEncoder(Protocol):
    @property
    def repository_revision(self) -> str: ...

    @property
    def weights_sha256(self) -> str: ...

    def encode(self, image: Image.Image) -> NDArray[np.float32]: ...


class ImageFingerprint(BaseModel):
    model_config = _MODEL_CONFIG

    image_id: str
    content_sha256: Sha256
    decoded_pixel_sha256: Sha256
    width: PositiveInt
    height: PositiveInt
    phash_64: NonNegativeInt
    normalized_feature: tuple[float, ...]
    encoder_revision: GitCommit
    encoder_weights_sha256: Sha256

    @model_validator(mode="after")
    def validate_fingerprint(self) -> ImageFingerprint:
        if self.phash_64 >= 2**64:
            raise ValueError("phash_64 must fit exactly in 64 bits")
        feature = np.asarray(self.normalized_feature, dtype=np.float64)
        if feature.size < 1 or not np.isfinite(feature).all():
            raise ValueError("normalized feature must be finite and non-empty")
        if abs(float(np.linalg.norm(feature)) - 1.0) > 1e-5:
            raise ValueError("normalized feature must have unit L2 norm")
        return self


def fingerprint_image(path: Path, encoder: FrozenFeatureEncoder) -> ImageFingerprint:
    """Fingerprint exact bytes, decoded RGB pixels, pHash, and a frozen feature vector."""

    metadata = path.lstat()
    if not os.path.isfile(path) or os.path.islink(path) or metadata.st_nlink != 1:
        raise ValueError("fingerprint input must be a regular single-link image")
    with Image.open(path) as opened:
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        rgb.load()
    decoded_payload = struct.pack(">II", rgb.width, rgb.height) + rgb.tobytes()
    feature = np.asarray(encoder.encode(rgb), dtype=np.float64).reshape(-1)
    if feature.size < 1 or not np.isfinite(feature).all():
        raise ValueError("feature encoder returned an invalid vector")
    norm = float(np.linalg.norm(feature))
    if norm <= 0.0:
        raise ValueError("feature encoder returned a zero vector")
    normalized = feature / norm
    return ImageFingerprint(
        image_id=path.name,
        content_sha256=file_sha256(path),
        decoded_pixel_sha256=hashlib.sha256(decoded_payload).hexdigest(),
        width=rgb.width,
        height=rgb.height,
        phash_64=int(str(imagehash.phash(rgb, hash_size=8)), 16),
        normalized_feature=tuple(float(value) for value in normalized),
        encoder_revision=encoder.repository_revision,
        encoder_weights_sha256=encoder.weights_sha256,
    )


def _sift_evidence(left_path: Path, right_path: Path) -> tuple[int | None, float | None]:
    try:
        import cv2
    except ImportError:
        return None, None
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError("OpenCV could not decode a fingerprinted image")
    detector = cv2.SIFT_create()  # type: ignore[attr-defined]
    left_points, left_descriptors = detector.detectAndCompute(left, None)
    right_points, right_descriptors = detector.detectAndCompute(right, None)
    if left_descriptors is None or right_descriptors is None:
        return 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 4:
        return 0, 0.0
    left_locations = np.asarray(
        [left_points[match.queryIdx].pt for match in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    right_locations = np.asarray(
        [right_points[match.trainIdx].pt for match in good],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    _homography, inlier_mask = cv2.findHomography(
        left_locations,
        right_locations,
        cv2.RANSAC,
        5.0,
    )
    if inlier_mask is None:
        return 0, 0.0
    inliers = int(inlier_mask.ravel().sum())
    return inliers, float(inliers / len(good))


def build_pair_evidence(
    left_path: Path,
    right_path: Path,
    left: ImageFingerprint,
    right: ImageFingerprint,
    *,
    capture_delta_seconds: float | None = None,
    same_capture_group: bool = False,
) -> PairEvidence:
    """Compute stable pair scores from already locked fingerprints."""

    if left.encoder_revision != right.encoder_revision or (
        left.encoder_weights_sha256 != right.encoder_weights_sha256
    ):
        raise ValueError("pair fingerprints use different feature encoders")
    left_id, right_id = sorted((left.image_id, right.image_id))
    if left_id == right_id:
        raise ValueError("pair evidence requires distinct image ids")
    cosine = float(
        np.dot(
            np.asarray(left.normalized_feature, dtype=np.float64),
            np.asarray(right.normalized_feature, dtype=np.float64),
        )
    )
    sift_inliers, sift_ratio = _sift_evidence(left_path, right_path)
    return PairEvidence(
        left_id=left_id,
        right_id=right_id,
        exact_sha256=left.content_sha256 == right.content_sha256,
        decoded_pixel_equal=left.decoded_pixel_sha256 == right.decoded_pixel_sha256,
        phash_hamming=(left.phash_64 ^ right.phash_64).bit_count(),
        feature_cosine=max(-1.0, min(1.0, cosine)),
        sift_inliers=sift_inliers,
        sift_inlier_ratio=sift_ratio,
        capture_delta_seconds=capture_delta_seconds,
        same_capture_group=same_capture_group,
    )


class DuplicateAuditReport(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    report_sha256: Sha256
    policy_sha256: Sha256
    encoder_lock_sha256: Sha256
    encoder_revision: GitCommit
    encoder_weights_sha256: Sha256
    image_fingerprints: tuple[ImageFingerprint, ...]
    candidate_evidence: tuple[AuditedPairEvidence, ...]
    component_membership: tuple[DuplicateComponent, ...]
    unresolved_review_count: NonNegativeInt

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("report_sha256")
        return canonical_json_bytes(payload)

    @classmethod
    def synthetic(
        cls,
        *,
        policy: DuplicatePolicy,
        encoder_revision: str,
        encoder_weights_sha256: str,
        image_fingerprints: tuple[ImageFingerprint, ...],
    ) -> DuplicateAuditReport:
        provisional = cls(
            schema_version="1.0",
            report_sha256="0" * 64,
            policy_sha256=policy.sha256,
            encoder_lock_sha256=hashlib.sha256(
                canonical_json_bytes(
                    {
                        "revision": encoder_revision,
                        "weights_sha256": encoder_weights_sha256,
                    }
                )
            ).hexdigest(),
            encoder_revision=encoder_revision,
            encoder_weights_sha256=encoder_weights_sha256,
            image_fingerprints=tuple(
                sorted(image_fingerprints, key=lambda fingerprint: fingerprint.image_id)
            ),
            candidate_evidence=(),
            component_membership=(),
            unresolved_review_count=0,
        )
        return provisional.model_copy(
            update={"report_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
        )


class AuditedPairEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    left_id: str
    right_id: str
    evidence_sha256: Sha256
    decision: Literal["link", "review", "distinct"]
    candidate_rules: tuple[str, ...]
    exact_sha256: bool
    decoded_pixel_equal: bool
    phash_hamming: int | None
    feature_cosine: float | None
    sift_inliers: int | None
    sift_inlier_ratio: float | None
    capture_delta_seconds: float | None
    same_capture_group: bool
    adjudication_sha256: Sha256 | None


DuplicateAuditReport.model_rebuild()


def _candidate_rules(
    evidence: PairEvidence,
    policy: DuplicatePolicy,
) -> tuple[str, ...]:
    rules: list[str] = []
    if evidence.exact_sha256:
        rules.append("exact_content_sha256")
    if evidence.decoded_pixel_equal:
        rules.append("decoded_pixel_sha256")
    if evidence.phash_hamming is not None:
        if evidence.phash_hamming <= policy.phash_max_hamming:
            rules.append("phash_link")
        elif evidence.phash_hamming <= policy.phash_max_hamming + 4:
            rules.append("phash_review")
    if evidence.feature_cosine is not None:
        if evidence.feature_cosine >= policy.feature_cosine_min:
            rules.append("feature_link")
        elif evidence.feature_cosine >= policy.burst_neighbor.feature_cosine_min - 0.02:
            rules.append("feature_review")
    if evidence.sift_inliers is not None and evidence.sift_inlier_ratio is not None:
        if (
            evidence.sift_inliers >= policy.crop_verification.minimum_inliers
            and evidence.sift_inlier_ratio
            >= policy.crop_verification.minimum_inlier_ratio
        ):
            rules.append("sift_crop_link")
        elif evidence.sift_inliers >= policy.crop_verification.minimum_inliers // 2:
            rules.append("sift_crop_review")
    if (
        evidence.same_capture_group
        and evidence.capture_delta_seconds is not None
        and evidence.capture_delta_seconds <= policy.burst_neighbor.maximum_seconds
    ):
        rules.append("burst_neighbor_candidate")
    return tuple(rules) or ("no_duplicate_rule",)


def build_duplicate_audit_report(
    *,
    policy: DuplicatePolicy,
    encoder_lock_sha256: str,
    records: Sequence[ImageRecord],
    image_fingerprints: Sequence[ImageFingerprint],
    candidates: Sequence[PairEvidence],
    reviews: Sequence[Review],
) -> DuplicateAuditReport:
    """Resolve candidates, connect all accepted pairs, and seal a full audit report."""

    fingerprints = tuple(
        sorted(image_fingerprints, key=lambda fingerprint: fingerprint.image_id)
    )
    record_ids = {record.image_id for record in records}
    fingerprint_ids = {fingerprint.image_id for fingerprint in fingerprints}
    if not fingerprints or record_ids != fingerprint_ids:
        raise ValueError("fingerprints must exactly cover component records")
    encoder_revisions = {fingerprint.encoder_revision for fingerprint in fingerprints}
    encoder_weights = {fingerprint.encoder_weights_sha256 for fingerprint in fingerprints}
    if len(encoder_revisions) != 1 or len(encoder_weights) != 1:
        raise ValueError("all fingerprints must use one frozen feature encoder")

    decisions = {candidate.sha256: classify_pair(candidate, policy) for candidate in candidates}
    review_candidates = tuple(
        candidate
        for candidate in candidates
        if decisions[candidate.sha256] is PairDecision.REVIEW
    )
    adjudicated_links: list[tuple[str, str]] = []
    if review_candidates:
        adjudicated_links = validate_adjudications(review_candidates, reviews)
    elif reviews:
        raise ValueError("reviews were supplied without ambiguous candidate pairs")
    review_map: dict[tuple[str, str], list[Review]] = {}
    for review in reviews:
        review_map.setdefault((review.left_id, review.right_id), []).append(review)

    direct_links = [
        (candidate.left_id, candidate.right_id)
        for candidate in candidates
        if decisions[candidate.sha256] is PairDecision.LINK
    ]
    component_report = assign_components(records, (*direct_links, *adjudicated_links))
    audited: list[AuditedPairEvidence] = []
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in sorted(candidates, key=lambda item: (item.left_id, item.right_id)):
        pair = (candidate.left_id, candidate.right_id)
        if pair in seen_pairs:
            raise ValueError("candidate evidence pairs must be unique")
        seen_pairs.add(pair)
        pair_reviews = review_map.get(pair, [])
        adjudication_sha = (
            None
            if not pair_reviews
            else hashlib.sha256(
                canonical_json_bytes(
                    [
                        review.model_dump(mode="json")
                        for review in sorted(
                            pair_reviews,
                            key=lambda item: item.reviewer_id,
                        )
                    ]
                )
            ).hexdigest()
        )
        audited.append(
            AuditedPairEvidence(
                **asdict(candidate),
                evidence_sha256=candidate.sha256,
                decision=decisions[candidate.sha256].value,
                candidate_rules=_candidate_rules(candidate, policy),
                adjudication_sha256=adjudication_sha,
            )
        )
    provisional = DuplicateAuditReport(
        schema_version="1.0",
        report_sha256="0" * 64,
        policy_sha256=policy.sha256,
        encoder_lock_sha256=encoder_lock_sha256,
        encoder_revision=next(iter(encoder_revisions)),
        encoder_weights_sha256=next(iter(encoder_weights)),
        image_fingerprints=fingerprints,
        candidate_evidence=tuple(audited),
        component_membership=component_report.components,
        unresolved_review_count=0,
    )
    return provisional.model_copy(
        update={"report_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )


class FeatureEncoderInventoryEntry(BaseModel):
    model_config = _MODEL_CONFIG

    model_id: str
    repository_uri: str
    immutable_revision: str
    weights_path: str
    weights_sha256: Sha256


class FeatureEncoderInventory(BaseModel):
    model_config = _MODEL_CONFIG

    models: tuple[FeatureEncoderInventoryEntry, ...]

    @model_validator(mode="after")
    def validate_unique_model_ids(self) -> FeatureEncoderInventory:
        model_ids = tuple(model.model_id for model in self.models)
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("feature encoder inventory model ids must be non-empty and unique")
        return self


class FeatureEncoderLock(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    lock_id: Sha256
    model_id: str
    repository_uri: str
    immutable_revision: str
    weights_sha256: Sha256
    preprocessing: Literal["resize_shorter_518_center_crop_rgb_v1"]


def lock_feature_encoder(
    *,
    model_id: str,
    inventory_entries: Sequence[FeatureEncoderInventoryEntry],
    preprocessing: Literal["resize_shorter_518_center_crop_rgb_v1"],
) -> FeatureEncoderLock:
    """Bind an immutable repository revision to locally observed weight bytes."""

    matches = [entry for entry in inventory_entries if entry.model_id == model_id]
    if len(matches) != 1:
        raise ValueError("model inventory must contain exactly one requested encoder")
    entry = matches[0]
    if re.fullmatch(r"[0-9a-f]{40,64}", entry.immutable_revision) is None:
        raise ValueError("feature encoder revision must be immutable lowercase hex")
    weight_path = Path(entry.weights_path)
    if not weight_path.is_file() or file_sha256(weight_path) != entry.weights_sha256:
        raise ValueError("feature encoder weight bytes do not match the inventory")
    provisional = FeatureEncoderLock(
        schema_version="1.0",
        lock_id="0" * 64,
        model_id=entry.model_id,
        repository_uri=entry.repository_uri,
        immutable_revision=entry.immutable_revision,
        weights_sha256=entry.weights_sha256,
        preprocessing=preprocessing,
    )
    return provisional.model_copy(
        update={"lock_id": semantic_sha256(provisional.model_dump(mode="json"))}
    )


def load_feature_encoder_inventory(path: Path) -> FeatureEncoderInventory:
    """Load a strict feature-encoder model inventory."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FeatureEncoderInventory.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid feature encoder inventory: {error}") from error


__all__ = [
    "AuditedPairEvidence",
    "ComponentReport",
    "CrossSplitComponentError",
    "DuplicateAuditReport",
    "DuplicatePolicy",
    "FeatureEncoderInventory",
    "FeatureEncoderInventoryEntry",
    "FeatureEncoderLock",
    "FrozenFeatureEncoder",
    "ImageFingerprint",
    "PairDecision",
    "PairEvidence",
    "Review",
    "allocate_component_splits",
    "assign_components",
    "build_pair_evidence",
    "build_duplicate_audit_report",
    "classify_pair",
    "fingerprint_image",
    "lock_feature_encoder",
    "load_feature_encoder_inventory",
    "validate_adjudications",
]
