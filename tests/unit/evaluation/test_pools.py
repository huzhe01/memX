from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

from ratemem.evaluation.pools import (
    PoolLeakageError,
    build_locked_pools,
    read_image_records,
    read_prompt_templates,
)

IMAGES = Path("tests/fixtures/scientific/images.jsonl")
PROMPTS = Path("tests/fixtures/scientific/prompts.jsonl")


def test_pool_builder_anonymizes_names_and_keeps_support_query_disjoint(
    tmp_path: Path,
) -> None:
    records = read_image_records(IMAGES)
    prompts = read_prompt_templates(PROMPTS)
    output = tmp_path / "pools"
    result = build_locked_pools(
        records,
        prompts,
        split_seed=87321,
        output_dir=output,
    )

    assert set(result.support_image_ids).isdisjoint(result.query_image_ids)
    assert all(
        re.fullmatch(r"<concept_[0-9]{6}>", token)
        for token in result.concept_tokens
    )
    assert not any("Ada" in prompt for prompt in result.rendered_prompts)
    public_bytes = b"".join(
        path.read_bytes()
        for path in result.manifest_paths
        if path.name != "private-concept-map.json"
    )
    lowered_public = public_bytes.lower()
    assert b"ada" not in lowered_public
    assert b"grace" not in lowered_public
    assert b"katherine" not in lowered_public
    assert stat.S_IMODE(output.stat().st_mode) == 0o755
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o644 for path in result.manifest_paths)
    assert stat.S_IMODE(result.private_map_path.stat().st_mode) == 0o600


def test_derivative_of_query_cannot_enter_training_pool(tmp_path: Path) -> None:
    records = list(read_image_records(IMAGES))
    records[0] = records[0].model_copy(
        update={"derivative_of": records[-1].image_id}
    )

    with pytest.raises(PoolLeakageError, match="derivative ancestry crosses splits"):
        build_locked_pools(
            records,
            read_prompt_templates(PROMPTS),
            split_seed=87321,
            output_dir=tmp_path / "blocked",
        )
    assert not (tmp_path / "blocked").exists()

    records = list(read_image_records(IMAGES))
    records[1] = records[1].model_copy(
        update={"derivative_of": records[0].image_id}
    )
    with pytest.raises(
        PoolLeakageError,
        match="derivative ancestry crosses support/query pools",
    ):
        build_locked_pools(
            records,
            read_prompt_templates(PROMPTS),
            split_seed=87321,
            output_dir=tmp_path / "cross-pool",
        )


def test_pool_builder_rejects_dual_eligibility_and_prompt_identity_leakage(
    tmp_path: Path,
) -> None:
    records = list(read_image_records(IMAGES))
    records[0] = records[0].model_copy(update={"eligible_for_query": True})
    with pytest.raises(PoolLeakageError, match="both support and query"):
        build_locked_pools(
            records,
            read_prompt_templates(PROMPTS),
            split_seed=87321,
            output_dir=tmp_path / "dual",
        )

    prompts = list(read_prompt_templates(PROMPTS))
    prompts[0] = prompts[0].model_copy(
        update={"template_text": "a portrait of Ada Lovelace and {concept}"}
    )
    with pytest.raises(PoolLeakageError, match="private concept identity"):
        build_locked_pools(
            read_image_records(IMAGES),
            prompts,
            split_seed=87321,
            output_dir=tmp_path / "prompt-leak",
        )


def test_concepts_and_prompt_templates_are_disjoint_across_splits(tmp_path: Path) -> None:
    records = list(read_image_records(IMAGES))
    records[-1] = records[-1].model_copy(update={"concept_id": "Ada Lovelace"})
    with pytest.raises(PoolLeakageError, match="concept ancestry crosses splits"):
        build_locked_pools(
            records,
            read_prompt_templates(PROMPTS),
            split_seed=87321,
            output_dir=tmp_path / "concept-crossing",
        )

    prompts = list(read_prompt_templates(PROMPTS))
    prompts[-1] = prompts[-1].model_copy(update={"template_id": prompts[0].template_id})
    with pytest.raises(PoolLeakageError, match="template id crosses splits"):
        build_locked_pools(
            read_image_records(IMAGES),
            prompts,
            split_seed=87321,
            output_dir=tmp_path / "template-crossing",
        )


def test_prompt_templates_exactly_cover_splits_and_have_unique_ids(tmp_path: Path) -> None:
    prompts = read_prompt_templates(PROMPTS)
    with pytest.raises(PoolLeakageError, match="exactly cover image splits"):
        build_locked_pools(
            read_image_records(IMAGES),
            prompts[:-1],
            split_seed=87321,
            output_dir=tmp_path / "missing-prompts",
        )

    with pytest.raises(PoolLeakageError, match="template ids must be globally unique"):
        build_locked_pools(
            read_image_records(IMAGES),
            (*prompts, prompts[0]),
            split_seed=87321,
            output_dir=tmp_path / "duplicate-prompts",
        )
