from __future__ import annotations

from pathlib import Path

import pytest

from ratemem.evaluation.canonical import file_sha256
from ratemem.method.config import (
    LockMismatch,
    MethodLockInputs,
    MethodPolicy,
    freeze_method_lock,
)


def test_policy_dimensions_and_training_limits_match_sana() -> None:
    policy = MethodPolicy.from_yaml(Path("configs/method/ratemem-v1.yaml"))
    assert policy.code.projection_count == 120
    assert policy.code.atom_count == 4
    assert policy.code.dimension == 480
    assert policy.code.dimension % policy.codec.group_size == 0
    assert policy.training.segment_length == 2
    assert policy.training.maximum_transformer_passes_per_segment == 2


def test_method_lock_binds_every_visible_scientific_input(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("dataset", "evaluation", "baseline", "train", "validation")
    }
    for name, path in paths.items():
        path.write_text(f'{{"kind":"{name}","revision":"{"1" * 40}"}}\n')
    inputs = MethodLockInputs(
        policy_path=Path("configs/method/ratemem-v1.yaml"),
        dataset_lock_path=paths["dataset"],
        evaluation_lock_path=paths["evaluation"],
        baseline_lock_path=paths["baseline"],
        visible_trace_manifest_paths=(paths["validation"], paths["train"]),
        expected_dataset_lock_sha256=file_sha256(paths["dataset"]),
        expected_evaluation_lock_sha256=file_sha256(paths["evaluation"]),
        expected_baseline_lock_sha256=file_sha256(paths["baseline"]),
    )
    first = freeze_method_lock(inputs)
    second = freeze_method_lock(inputs)
    assert first == second
    assert first.visible_trace_manifest_sha256 == tuple(
        file_sha256(path) for path in sorted((paths["train"], paths["validation"]))
    )

    changed = inputs.model_copy(update={"expected_evaluation_lock_sha256": "0" * 64})
    with pytest.raises(LockMismatch, match="evaluation lock content hash"):
        freeze_method_lock(changed)


def test_method_lock_rejects_final_test_trace_paths(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n")
    final_trace = tmp_path / "final-test-trace.json"
    final_trace.write_text("{}\n")
    inputs = MethodLockInputs(
        policy_path=Path("configs/method/ratemem-v1.yaml"),
        dataset_lock_path=lock,
        evaluation_lock_path=lock,
        baseline_lock_path=lock,
        visible_trace_manifest_paths=(final_trace,),
        expected_dataset_lock_sha256=file_sha256(lock),
        expected_evaluation_lock_sha256=file_sha256(lock),
        expected_baseline_lock_sha256=file_sha256(lock),
    )
    with pytest.raises(LockMismatch, match="final-test"):
        freeze_method_lock(inputs)
