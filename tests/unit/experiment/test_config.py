from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ratemem.experiment.config import ExperimentConfig


def config_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "memx-experiment-v1",
        "profile": "smoke",
        "seed": 20260830,
        "max_steps": 6,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "learning_rate": 0.01,
        "checkpoint_every": 2,
        "precision": "fp32",
        "dataset_manifest": "configs/data/smoke.yaml",
    }
    payload.update(overrides)
    return payload


def write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_config_hash_changes_when_training_steps_change(tmp_path: Path) -> None:
    first = ExperimentConfig.load(
        write_config(tmp_path / "first.yaml", config_payload(max_steps=4))
    )
    second = ExperimentConfig.load(
        write_config(tmp_path / "second.yaml", config_payload(max_steps=5))
    )

    assert first.sha256 != second.sha256


def test_config_hash_is_independent_of_yaml_key_order(tmp_path: Path) -> None:
    payload = config_payload()
    first = ExperimentConfig.load(write_config(tmp_path / "first.yaml", payload))
    second = ExperimentConfig.load(
        write_config(tmp_path / "second.yaml", dict(reversed(tuple(payload.items()))))
    )

    assert first.sha256 == second.sha256
    assert first.canonical_bytes() == second.canonical_bytes()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", -1, "greater than or equal to 0"),
        ("max_steps", 0, "greater than 0"),
        ("batch_size", True, "valid integer"),
        ("gradient_accumulation", 0, "greater than 0"),
        ("learning_rate", float("nan"), "finite"),
        ("checkpoint_every", 0, "greater than 0"),
    ],
)
def test_config_rejects_invalid_training_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = write_config(tmp_path / "experiment.yaml", config_payload(**{field: value}))

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.load(path)


def test_smoke_profile_requires_fp32() -> None:
    with pytest.raises(ValidationError, match="smoke profile requires fp32"):
        ExperimentConfig.model_validate(config_payload(precision="bf16"))


def test_production_profile_requires_bf16() -> None:
    with pytest.raises(ValidationError, match="sana-ratemem profile requires bf16"):
        ExperimentConfig.model_validate(
            config_payload(profile="sana-ratemem", precision="fp32")
        )


def test_committed_smoke_config_is_runnable() -> None:
    config = ExperimentConfig.load(Path("configs/experiments/smoke.yaml"))

    assert config.profile == "smoke"
    assert config.max_steps == 6
    assert config.checkpoint_every == 2
    assert config.dataset_manifest == Path("configs/data/smoke.yaml")
