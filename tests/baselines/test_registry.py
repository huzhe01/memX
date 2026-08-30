from __future__ import annotations

from pathlib import Path

import pytest

from ratemem.baselines.catalog import REQUIRED_CONTROL_IDS, load_catalog
from ratemem.baselines.independent import IndependentCodeCacheAdapter
from ratemem.baselines.registry import RegistryError, build_registry
from ratemem.baselines.shared_inputs import SharedInputReader, materialize_fixture_bundle


def test_every_control_resolves_to_one_source_hashed_factory(tmp_path: Path) -> None:
    catalog = load_catalog(Path("configs/baselines/literature-classification.yaml"))
    registry = build_registry(catalog, baseline_lock_id="1" * 64)
    assert set(registry.method_ids) == REQUIRED_CONTROL_IDS
    assert registry.lock().baseline_lock_id == "1" * 64
    for method_id in registry.method_ids:
        entry = registry[method_id]
        assert entry.factory_importable
        assert len(entry.factory_sha256) == 64

    bundle = materialize_fixture_bundle(tmp_path / "bundle")
    adapter = registry.create(
        "independent_fifo",
        shared_inputs=SharedInputReader(bundle, "independent_fifo"),
    )
    assert isinstance(adapter, IndependentCodeCacheAdapter)
    assert adapter.method_id == "independent_fifo"


def test_registry_fails_if_a_locked_factory_hash_changes() -> None:
    catalog = load_catalog(Path("configs/baselines/literature-classification.yaml"))
    first = build_registry(catalog)
    hashes = {method_id: first[method_id].factory_sha256 for method_id in first.method_ids}
    hashes["per_concept_lora"] = "f" * 64
    with pytest.raises(RegistryError, match="factory source hash changed"):
        build_registry(catalog, expected_factory_sha256=hashes)
