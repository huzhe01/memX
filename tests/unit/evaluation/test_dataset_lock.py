from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratemem.evaluation.canonical import semantic_sha256
from ratemem.evaluation.dataset_lock import (
    DatasetLockError,
    SourceInventory,
    load_inventory,
    render_data_card,
    seal_dataset_lock,
    write_dataset_lock_and_card,
)

FIXTURE = Path("tests/fixtures/scientific/source-inventory.json")
POLICY = Path("configs/scientific/dataset-policy.yaml")


def _payload() -> dict[str, object]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_seal_dataset_lock_binds_every_pool_and_renders_card(tmp_path: Path) -> None:
    inventory = load_inventory(FIXTURE)
    lock = seal_dataset_lock(inventory, policy_path=POLICY, mode="synthetic")

    assert lock.lock_id == semantic_sha256(lock.model_dump(mode="json"))
    assert {pool.kind for source in lock.sources for pool in source.pools} >= {
        "support",
        "query",
    }
    assert len(lock.sources) == 6
    assert all(source.immutable_revision != "main" for source in lock.sources)

    card = render_data_card(lock)
    assert "## Licenses and allowed uses" in card
    assert "## Duplicate, derivative, and contamination audit" in card
    assert "synthetic-primary" in card
    assert "reference_prompt_only" in card

    lock_path = tmp_path / "dataset-lock.yaml"
    card_path = tmp_path / "data-card.md"
    write_dataset_lock_and_card(lock, lock_path, card_path)
    assert lock_path.is_file()
    assert card_path.read_text(encoding="utf-8") == card


def test_scientific_mode_rejects_the_synthetic_inventory() -> None:
    inventory = load_inventory(FIXTURE)

    with pytest.raises(DatasetLockError, match="synthetic inventory"):
        seal_dataset_lock(inventory, policy_path=POLICY, mode="scientific")


def test_scientific_lock_rejects_missing_post_checkpoint_source() -> None:
    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, list)
    payload["sources"] = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("role") != "controlled_post_checkpoint_evaluation"
    ]
    inventory = SourceInventory.model_validate(payload)

    with pytest.raises(DatasetLockError, match="controlled_post_checkpoint_evaluation"):
        seal_dataset_lock(inventory, policy_path=POLICY, mode="synthetic")


def test_lock_rejects_mutable_revision_and_support_query_overlap() -> None:
    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["immutable_revision"] = "main"
    mutable = SourceInventory.model_validate(payload)
    with pytest.raises(DatasetLockError, match="immutable_revision"):
        seal_dataset_lock(mutable, policy_path=POLICY, mode="synthetic")

    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    pools = sources[0]["pools"]
    assert isinstance(pools, list) and len(pools) >= 2
    assert isinstance(pools[0], dict) and isinstance(pools[1], dict)
    pools[1]["record_ids"] = pools[0]["record_ids"]
    overlap = SourceInventory.model_validate(payload)
    with pytest.raises(DatasetLockError, match="support/query records must be disjoint"):
        seal_dataset_lock(overlap, policy_path=POLICY, mode="synthetic")


def test_lock_rejects_identity_names_and_wrong_dreambench_semantics() -> None:
    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["concept_tokens"] = ["Alice"]
    named = SourceInventory.model_validate(payload)
    with pytest.raises(DatasetLockError, match="anonymous concept token"):
        seal_dataset_lock(named, policy_path=POLICY, mode="synthetic")

    payload = _payload()
    sources = payload["sources"]
    assert isinstance(sources, list)
    dreambench = next(
        source
        for source in sources
        if isinstance(source, dict) and source.get("source_id") == "dreambench_plus_plus"
    )
    assert isinstance(dreambench, dict)
    dreambench["evaluation_pool_semantics"] = "held_out_query"
    wrong_semantics = SourceInventory.model_validate(payload)
    with pytest.raises(DatasetLockError, match="reference_prompt_only"):
        seal_dataset_lock(wrong_semantics, policy_path=POLICY, mode="synthetic")


def test_customconcept_shots_are_derived_and_input_order_does_not_change_lock() -> None:
    inventory = load_inventory(FIXTURE)
    first = seal_dataset_lock(inventory, policy_path=POLICY, mode="synthetic")
    reversed_inventory = inventory.model_copy(
        update={"sources": tuple(reversed(inventory.sources))}
    )
    second = seal_dataset_lock(reversed_inventory, policy_path=POLICY, mode="synthetic")

    assert first.lock_id == second.lock_id
    custom = next(
        source
        for source in first.sources
        if source.source_id == "customconcept101_eligible"
    )
    assert custom.eligible_support_shots == (1, 3, 5)
