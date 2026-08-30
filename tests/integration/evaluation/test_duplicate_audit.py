from __future__ import annotations

import hashlib
from pathlib import Path

import jsonschema
import numpy as np
from PIL import Image, ImageDraw

from ratemem.evaluation.leakage import (
    DuplicateAuditReport,
    DuplicatePolicy,
    FrozenFeatureEncoder,
    PairDecision,
    build_duplicate_audit_report,
    build_pair_evidence,
    classify_pair,
    fingerprint_image,
)
from ratemem.evaluation.pools import ImageRecord


class TinyColorEncoder(FrozenFeatureEncoder):
    @property
    def repository_revision(self) -> str:
        return "7" * 40

    @property
    def weights_sha256(self) -> str:
        return "8" * 64

    def encode(self, image: Image.Image) -> np.ndarray:
        pixels = np.asarray(image.resize((16, 16)), dtype=np.float32) / 255.0
        means = pixels.reshape(-1, 3).mean(axis=0)
        histogram, _ = np.histogram(pixels, bins=8, range=(0.0, 1.0))
        return np.concatenate((means, histogram.astype(np.float32)))


def _write_images(root: Path) -> tuple[Path, Path, Path, Path]:
    original = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(original)
    for index in range(40):
        x = 8 + (index * 47) % 224
        y = 8 + (index * 83) % 224
        radius = 4 + index % 9
        color = (
            (index * 53) % 256,
            (index * 97) % 256,
            (index * 193) % 256,
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.line((x, 0, 255 - x, 255), fill=color, width=2 + index % 3)
    draw.rectangle((28, 31, 221, 226), outline=(10, 10, 10), width=5)

    original_path = root / "original.png"
    jpeg_path = root / "recompressed.jpg"
    crop_path = root / "crop.png"
    different_path = root / "different.png"
    original.save(original_path)
    original.save(jpeg_path, format="JPEG", quality=88)
    original.crop((8, 8, 248, 248)).resize((256, 256)).save(crop_path)

    different = Image.new("RGB", (256, 256), "navy")
    different_draw = ImageDraw.Draw(different)
    for offset in range(0, 256, 16):
        different_draw.rectangle(
            (offset, 0, min(offset + 7, 255), 255),
            fill="yellow",
        )
    different.save(different_path)
    return original_path, jpeg_path, crop_path, different_path


def test_tiny_image_audit_links_recompression_and_crop_but_not_different(
    tmp_path: Path,
) -> None:
    original, recompressed, crop, different = _write_images(tmp_path)
    encoder = TinyColorEncoder()
    policy = DuplicatePolicy.load(Path("configs/scientific/duplicate-policy.yaml"))
    fingerprints = {
        path: fingerprint_image(path, encoder)
        for path in (original, recompressed, crop, different)
    }

    evidence_rows = []
    for candidate in (recompressed, crop):
        evidence = build_pair_evidence(
            original,
            candidate,
            fingerprints[original],
            fingerprints[candidate],
        )
        assert classify_pair(evidence, policy) == PairDecision.LINK, (
            f"phash={evidence.phash_hamming} feature={evidence.feature_cosine} "
            f"sift={evidence.sift_inliers}/{evidence.sift_inlier_ratio}"
        )
        evidence_rows.append(evidence)
    distinct = build_pair_evidence(
        original,
        different,
        fingerprints[original],
        fingerprints[different],
    )
    assert classify_pair(distinct, policy) == PairDecision.DISTINCT
    evidence_rows.append(distinct)

    records = tuple(
        ImageRecord.model_validate(
            {
                "image_id": path.name,
                "source_id": "synthetic",
                "concept_id": f"concept-{index}",
                "content_sha256": fingerprints[path].content_sha256,
                "decoded_pixel_sha256": fingerprints[path].decoded_pixel_sha256,
                "width": fingerprints[path].width,
                "height": fingerprints[path].height,
                "caption_sha256": None,
                "mask_sha256": None,
                "derivative_of": None,
                "capture_group": None,
                "split": "train",
                "eligible_for_support": True,
                "eligible_for_query": False,
            }
        )
        for index, path in enumerate((original, recompressed, crop, different))
    )
    report = build_duplicate_audit_report(
        policy=policy,
        encoder_lock_sha256="9" * 64,
        records=records,
        image_fingerprints=tuple(fingerprints.values()),
        candidates=tuple(evidence_rows),
        reviews=(),
    )
    component_sets = {component.image_ids for component in report.component_membership}
    assert tuple(sorted((original.name, recompressed.name, crop.name))) in component_sets
    assert (different.name,) in component_sets

    schema_fixture = DuplicateAuditReport.synthetic(
        policy=policy,
        encoder_revision=encoder.repository_revision,
        encoder_weights_sha256=encoder.weights_sha256,
        image_fingerprints=tuple(fingerprints.values()),
    )
    schema = DuplicateAuditReport.model_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(report.model_dump(mode="json"), schema)
    assert report.report_sha256 == hashlib.sha256(report.semantic_bytes).hexdigest()
    assert schema_fixture.report_sha256 == hashlib.sha256(
        schema_fixture.semantic_bytes
    ).hexdigest()
