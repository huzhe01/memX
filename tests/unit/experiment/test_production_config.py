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


def test_production_config_rejects_unapproved_seed() -> None:
    config = ProductionExperimentConfig.load(
        Path("configs/experiments/sana-ratemem.yaml")
    )
    payload = config.model_dump()
    payload["seed"] = 18

    with pytest.raises(ValidationError):
        ProductionExperimentConfig.model_validate(payload)
