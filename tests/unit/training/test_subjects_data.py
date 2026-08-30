from __future__ import annotations

from pathlib import Path

from PIL import Image

from ratemem.data.subjects200k import Subjects200KManifest
from ratemem.training.subjects_data import build_subjects_pair, concept_partition


def _manifest() -> Subjects200KManifest:
    return Subjects200KManifest.load(Path("configs/data/subjects200k.yaml"))


def _row(*, concept: str = "Red Ceramic Teapot", valid: bool = True) -> dict[str, object]:
    return {
        "image": Image.new("RGB", (1056, 528), (71, 92, 113)),
        "collection": "synthetic-test",
        "quality_assessment": {
            "compositeStructure": 5,
            "objectConsistency": 5,
            "imageQuality": 5,
        },
        "description": {
            "item": concept,
            "description_0": "A red ceramic teapot.",
            "description_1": "The teapot on a wooden table.",
            "category": "object",
            "description_valid": valid,
        },
    }


def test_subject_pair_is_cropped_hashed_and_pseudonymous() -> None:
    manifest = _manifest()
    pair = build_subjects_pair(_row(), manifest)

    assert pair is not None
    assert pair.support.size == (512, 512)
    assert pair.query.size == (512, 512)
    assert pair.concept_token.startswith("concept_")
    assert "teapot" not in pair.concept_token
    assert len(pair.row_sha256) == 64


def test_concept_partition_is_case_and_whitespace_stable() -> None:
    manifest = _manifest()

    assert concept_partition(" Red   Ceramic Teapot ", manifest.partition) == (
        concept_partition("red ceramic teapot", manifest.partition)
    )


def test_invalid_description_row_is_skipped() -> None:
    manifest: Subjects200KManifest = _manifest()

    assert build_subjects_pair(_row(valid=False), manifest) is None
