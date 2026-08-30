from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ratemem.experiment.production_config import ProductionExperimentConfig


def test_committed_production_config_binds_real_inputs_and_locked_seed() -> None:
    config = ProductionExperimentConfig.load(
        Path("configs/experiments/sana-ratemem.yaml")
    )

    assert config.profile == "sana-ratemem"
    assert config.seed in {17, 29, 43}
    assert config.dataset_manifest == Path("configs/data/subjects200k.yaml")
    assert config.publication_eligible is False
    assert len(config.sha256) == 64


def test_all_locked_seed_profiles_are_committed() -> None:
    paths = (
        Path("configs/experiments/sana-ratemem.yaml"),
        Path("configs/experiments/sana-ratemem-seed29.yaml"),
        Path("configs/experiments/sana-ratemem-seed43.yaml"),
    )
    configs = tuple(ProductionExperimentConfig.load(path) for path in paths)

    assert tuple(config.seed for config in configs) == (17, 29, 43)
    canonical = tuple(
        config.model_dump(mode="json", exclude={"seed"}) for config in configs
    )
    assert canonical[0] == canonical[1] == canonical[2]


def test_production_config_rejects_unapproved_seed() -> None:
    config = ProductionExperimentConfig.load(
        Path("configs/experiments/sana-ratemem.yaml")
    )
    payload = config.model_dump()
    payload["seed"] = 18

    with pytest.raises(ValidationError):
        ProductionExperimentConfig.model_validate(payload)
