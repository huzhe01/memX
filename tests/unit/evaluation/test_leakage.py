from __future__ import annotations

from pathlib import Path

import pytest

from ratemem.evaluation.leakage import (
    CrossSplitComponentError,
    DuplicatePolicy,
    FeatureEncoderInventoryEntry,
    PairDecision,
    PairEvidence,
    Review,
    allocate_component_splits,
    assign_components,
    classify_pair,
    lock_feature_encoder,
    validate_adjudications,
)
from ratemem.evaluation.pools import ImageRecord

POLICY = Path("configs/scientific/duplicate-policy.yaml")


def _record(image_id: str, concept_id: str, split: str) -> ImageRecord:
    return ImageRecord.model_validate(
        {
            "image_id": image_id,
            "source_id": "synthetic",
            "concept_id": concept_id,
            "content_sha256": "1" * 64,
            "decoded_pixel_sha256": "2" * 64,
            "width": 64,
            "height": 64,
            "caption_sha256": None,
            "mask_sha256": None,
            "derivative_of": None,
            "capture_group": None,
            "split": split,
            "eligible_for_support": True,
            "eligible_for_query": False,
        }
    )


def _evidence(**updates: object) -> PairEvidence:
    values: dict[str, object] = {
        "left_id": "a",
        "right_id": "b",
        "exact_sha256": False,
        "decoded_pixel_equal": False,
        "phash_hamming": None,
        "feature_cosine": None,
        "sift_inliers": None,
        "sift_inlier_ratio": None,
        "capture_delta_seconds": None,
        "same_capture_group": False,
    }
    values.update(updates)
    return PairEvidence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "evidence",
    [
        _evidence(exact_sha256=True),
        _evidence(decoded_pixel_equal=True),
        _evidence(phash_hamming=3),
        _evidence(feature_cosine=0.99),
        _evidence(sift_inliers=15, sift_inlier_ratio=0.5),
        _evidence(
            feature_cosine=0.975,
            capture_delta_seconds=1.0,
            same_capture_group=True,
        ),
    ],
)
def test_any_strong_duplicate_evidence_links_records(evidence: PairEvidence) -> None:
    assert classify_pair(evidence, DuplicatePolicy.load(POLICY)) == PairDecision.LINK


def test_borderline_evidence_requires_review_and_weak_evidence_is_distinct() -> None:
    policy = DuplicatePolicy.load(POLICY)
    assert classify_pair(_evidence(phash_hamming=6), policy) == PairDecision.REVIEW
    assert classify_pair(_evidence(feature_cosine=0.96), policy) == PairDecision.REVIEW
    assert classify_pair(_evidence(phash_hamming=20, feature_cosine=0.4), policy) == (
        PairDecision.DISTINCT
    )


def test_component_crossing_preassigned_splits_is_rejected() -> None:
    records = [
        _record("a", "concept-a", "train"),
        _record("b", "concept-b", "final_test"),
    ]
    with pytest.raises(CrossSplitComponentError, match="a.*b"):
        assign_components(records, linked_pairs=[("a", "b")])


def test_concepts_connected_by_duplicate_images_share_one_split() -> None:
    assignment = allocate_component_splits(
        concept_components=[{"c1", "c2"}, {"c3"}, {"c4"}],
        ratios={"train": 0.5, "validation": 0.25, "final_test": 0.25},
        seed=87321,
    )
    repeated = allocate_component_splits(
        concept_components=[{"c4"}, {"c3"}, {"c2", "c1"}],
        ratios={"final_test": 0.25, "validation": 0.25, "train": 0.5},
        seed=87321,
    )
    assert assignment == repeated
    assert assignment["c1"] == assignment["c2"]
    assert set(assignment) == {"c1", "c2", "c3", "c4"}


def test_adjudication_requires_independent_majority_and_binds_evidence() -> None:
    candidate = _evidence(phash_hamming=6)
    evidence_sha = candidate.sha256
    reviews = [
        Review(
            left_id="a",
            right_id="b",
            reviewer_id="reviewer_1",
            decision="link",
            evidence_sha256=evidence_sha,
        ),
        Review(
            left_id="a",
            right_id="b",
            reviewer_id="reviewer_2",
            decision="link",
            evidence_sha256=evidence_sha,
        ),
    ]
    assert validate_adjudications([candidate], reviews) == [("a", "b")]

    with pytest.raises(ValueError, match="independent reviewers"):
        validate_adjudications([candidate], [reviews[0], reviews[0]])


def test_feature_encoder_lock_binds_revision_and_observed_weight_bytes(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "dinov2.safetensors"
    weights.write_bytes(b"frozen-feature-encoder")
    entry = FeatureEncoderInventoryEntry(
        model_id="dinov2_large_duplicate_audit",
        repository_uri="https://github.com/facebookresearch/dinov2",
        immutable_revision="7" * 40,
        weights_path=str(weights),
        weights_sha256="4a8acb5238bc94ae73b8470a30d1e76bfcc5c76c8a2a1be16060b0138ac23368",
    )
    lock = lock_feature_encoder(
        model_id=entry.model_id,
        inventory_entries=[entry],
        preprocessing="resize_shorter_518_center_crop_rgb_v1",
    )
    assert lock.immutable_revision == "7" * 40
    assert lock.weights_sha256 == entry.weights_sha256

    changed = entry.model_copy(update={"immutable_revision": "main"})
    with pytest.raises(ValueError, match="immutable lowercase hex"):
        lock_feature_encoder(
            model_id=entry.model_id,
            inventory_entries=[changed],
            preprocessing="resize_shorter_518_center_crop_rgb_v1",
        )
