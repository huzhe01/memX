from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ratemem.experiment.checkpoint import CheckpointState, CheckpointStore


def state(*, step: int = 2, config_sha256: str = "a" * 64) -> CheckpointState:
    parameter = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    return CheckpointState(
        step=step,
        config_sha256=config_sha256,
        dataset_sha256="b" * 64,
        model_state={"linear.weight": parameter},
        optimizer_state={
            "state": {0: {"step": torch.tensor(2.0), "exp_avg": parameter / 10}},
            "param_groups": [{"lr": 0.01, "params": [0]}],
        },
        torch_rng_state=torch.arange(16, dtype=torch.uint8),
    )


def test_incomplete_staging_directory_is_not_latest(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    incomplete = tmp_path / ".staging-interrupted"
    incomplete.mkdir()
    (incomplete / "model.safetensors").write_bytes(b"partial")

    assert store.latest() is None


def test_checkpoint_round_trip_preserves_training_state(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    expected = state()

    path = store.save(expected)
    observed = store.latest(
        expected_config_sha256=expected.config_sha256,
        expected_dataset_sha256=expected.dataset_sha256,
    )

    assert path.name == "step-00000002"
    assert observed is not None
    assert observed.step == expected.step
    assert observed.config_sha256 == expected.config_sha256
    assert observed.dataset_sha256 == expected.dataset_sha256
    assert torch.equal(
        observed.model_state["linear.weight"], expected.model_state["linear.weight"]
    )
    assert torch.equal(observed.torch_rng_state, expected.torch_rng_state)
    observed_optimizer = observed.optimizer_state
    assert observed_optimizer["param_groups"] == expected.optimizer_state["param_groups"]
    observed_state = observed_optimizer["state"]
    assert isinstance(observed_state, dict)
    assert torch.equal(observed_state[0]["exp_avg"], torch.tensor([[0.1, 0.2], [0.3, 0.4]]))


def test_checkpoint_refuses_to_overwrite_existing_step(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save(state())

    with pytest.raises(FileExistsError, match="checkpoint step already exists"):
        store.save(state())


def test_changed_config_blocks_resume(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save(state())

    with pytest.raises(ValueError, match="configuration hash"):
        store.latest(expected_config_sha256="c" * 64)


def test_changed_dataset_blocks_resume(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save(state())

    with pytest.raises(ValueError, match="dataset hash"):
        store.latest(expected_dataset_sha256="c" * 64)


def test_corrupted_checkpoint_is_rejected(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    path = store.save(state())
    (path / "model.safetensors").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="checkpoint file hash changed"):
        store.latest()


def test_latest_pointer_rejects_path_traversal(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save(state())
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "schema_version": "memx-checkpoint-pointer-v1",
                "checkpoint": "../outside",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latest checkpoint name"):
        store.latest()


def test_checkpoint_model_hash_is_stable_for_identical_tensors(tmp_path: Path) -> None:
    first = CheckpointStore(tmp_path / "first").save(state())
    second = CheckpointStore(tmp_path / "second").save(state())

    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["model_sha256"] == second_manifest["model_sha256"]
