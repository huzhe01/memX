from __future__ import annotations

import os
from pathlib import Path

import pytest

from ratemem.pilot.config import SubjectsPilotConfig
from ratemem.pilot.data import hydrate_locked_examples, rgb_content_sha256

CONFIG_PATH = Path("configs/pilot/subjects200k-held-in.json")

EXPECTED_QUALITY = (
    (5, 5, 5),
    None,
    (5, 5, 5),
    (0, 1, 5),
    (5, 5, 5),
    (5, 5, 5),
    (0, 4, 5),
    (5, 5, 5),
)


@pytest.mark.skipif(
    os.environ.get("RATEMEM_RUN_REAL_SUBJECTS") != "1",
    reason="explicit real-Subjects200K network opt-in is required",
)
def test_fixed_real_subjects_rows_match_locked_schema_geometry_and_identity() -> None:
    """This opts into dataset hydration only; it never loads a model or invokes Modal."""

    config = SubjectsPilotConfig.load(CONFIG_PATH)
    cache_dir_text = os.environ.get("RATEMEM_REAL_SUBJECTS_HF_CACHE")
    if not cache_dir_text:
        pytest.fail("RATEMEM_REAL_SUBJECTS_HF_CACHE must name an explicit cache directory")
    examples = hydrate_locked_examples(config, cache_dir=Path(cache_dir_text))

    # source_file_sha256 binds the pinned revision's published LFS identity. Streaming
    # eight rows does not claim to have downloaded and hashed all 429,744,278 shard bytes.
    assert (
        config.source_file_sha256
        == "3d696ccbdfc736961e75e5b7ce33adae40cd70ffb69cdc27020a25d643971903"
    )
    assert tuple(example.row_index for example in examples) == tuple(range(8))
    assert tuple(example.quality_assessment for example in examples) == EXPECTED_QUALITY
    assert {example.collection for example in examples} == {"collection_1"}
    assert {example.item for example in examples} == {"Eames Lounge Chair"}
    assert all(example.description_valid for example in examples)
    assert len({example.row_sha256 for example in examples}) == 8

    for example in examples:
        example.validate(config)
        support = example.support_image()
        query = example.query_image()
        assert support.mode == query.mode == "RGB"
        assert support.size == query.size == (512, 512)
        assert rgb_content_sha256(support) == example.support_sha256
        assert rgb_content_sha256(query) == example.query_sha256
