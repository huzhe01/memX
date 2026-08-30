from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from ratemem.data.prepare import PreparedDataset
from ratemem.experiment.checkpoint import CheckpointState, CheckpointStore
from ratemem.experiment.config import ExperimentConfig
from ratemem.experiment.fixture import (
    FixtureExperiment,
    evaluation_batches,
    training_batch,
)
from ratemem.runtime.distributed import DistributedContext

_PATH_TYPE = type(Path())
_TRAIN_SCHEMA = "memx-train-result-v1"
_EVALUATION_SCHEMA = "memx-evaluation-result-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _barrier(context: DistributedContext) -> None:
    if context.ranks.world_size > 1:
        if not torch.distributed.is_initialized():
            raise RuntimeError("distributed context requires an initialized process group")
        torch.distributed.barrier()


def _validate_run_root(path: Path) -> None:
    if type(path) is not _PATH_TYPE:
        raise TypeError("run root must be an exact Path")
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise ValueError("run root must be a real directory")


class _MetricJournal:
    def __init__(self, path: Path, *, expected_steps: int) -> None:
        self.path = path
        if expected_steps < 0:
            raise ValueError("expected metric steps must be nonnegative")
        if not path.exists():
            if expected_steps:
                raise ValueError("checkpoint exists but training metrics are missing")
            return
        lines = path.read_bytes().splitlines(keepends=True)
        if len(lines) != expected_steps:
            raise ValueError("training metric count differs from checkpoint step")
        for expected_step, line in enumerate(lines, start=1):
            if not line.endswith(b"\n"):
                raise ValueError("training metrics must be newline-terminated JSONL")
            try:
                decoded: Any = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError("training metrics contain invalid JSONL") from error
            if (
                type(decoded) is not dict
                or set(decoded) != {"loss", "step"}
                or decoded["step"] != expected_step
                or type(decoded["loss"]) is not float
                or not math.isfinite(decoded["loss"])
                or decoded["loss"] < 0
                or _canonical_json(decoded) + b"\n" != line
            ):
                raise ValueError("training metric record changed")

    def append(self, *, step: int, loss: float) -> None:
        payload = _canonical_json({"loss": loss, "step": step}) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @property
    def sha256(self) -> str:
        return _sha256_file(self.path)


@dataclass(frozen=True, slots=True)
class TrainResult:
    status: Literal["completed", "stopped"]
    step: int
    model_sha256: str
    metrics_sha256: str
    result_sha256: str
    result_path: Path
    metrics_path: Path
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    status: Literal["completed"]
    model_sha256: str
    validation_mse: float
    validation_count: int
    test_mse: float
    test_count: int
    result_sha256: str
    result_path: Path


def _checkpoint_state(
    experiment: FixtureExperiment,
    *,
    step: int,
    config: ExperimentConfig,
    dataset: PreparedDataset,
) -> CheckpointState:
    return CheckpointState(
        step=step,
        config_sha256=config.sha256,
        dataset_sha256=dataset.content_sha256,
        model_state=experiment.model_state_dict(),
        optimizer_state=experiment.optimizer_state_dict(),
        torch_rng_state=torch.get_rng_state(),
    )


def _read_train_result(path: Path) -> TrainResult:
    payload: Any = json.loads(path.read_bytes())
    if type(payload) is not dict or payload.get("schema_version") != _TRAIN_SCHEMA:
        raise ValueError("training result is invalid")
    status = payload.get("status")
    if status not in {"completed", "stopped"}:
        raise ValueError("training result status is invalid")
    checkpoint_name = payload.get("checkpoint")
    if type(checkpoint_name) is not str or "/" in checkpoint_name or ".." in checkpoint_name:
        raise ValueError("training result checkpoint name is invalid")
    return TrainResult(
        status=status,
        step=payload["step"],
        model_sha256=payload["model_sha256"],
        metrics_sha256=payload["metrics_sha256"],
        result_sha256=_sha256_file(path),
        result_path=path,
        metrics_path=path.parent / "metrics.jsonl",
        checkpoint_path=path.parent / "checkpoints" / checkpoint_name,
    )


def train_fixture(
    config: ExperimentConfig,
    dataset: PreparedDataset,
    run_root: Path,
    context: DistributedContext,
    *,
    resume: Literal["never", "auto"],
    stop_after_step: int | None = None,
) -> TrainResult:
    if type(config) is not ExperimentConfig or config.profile != "smoke":
        raise TypeError("fixture training requires an exact smoke ExperimentConfig")
    if type(dataset) is not PreparedDataset:
        raise TypeError("fixture training requires an exact PreparedDataset")
    if type(context) is not DistributedContext:
        raise TypeError("fixture training requires an exact DistributedContext")
    if resume not in {"never", "auto"}:
        raise ValueError("resume must be never or auto")
    if stop_after_step is not None and (
        type(stop_after_step) is not int
        or stop_after_step < 1
        or stop_after_step > config.max_steps
    ):
        raise ValueError("stop_after_step must be within the configured training range")
    _validate_run_root(run_root)
    if resume == "never" and run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError("run already exists; use RESUME=auto to continue it")
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_store = CheckpointStore(run_root / "checkpoints")
    experiment = FixtureExperiment(config, context)

    loaded = (
        checkpoint_store.latest(
            expected_config_sha256=config.sha256,
            expected_dataset_sha256=dataset.content_sha256,
        )
        if resume == "auto"
        else None
    )
    start_step = 0
    if loaded is not None:
        experiment.load_state_dicts(loaded.model_state, loaded.optimizer_state)
        torch.set_rng_state(loaded.torch_rng_state)
        start_step = loaded.step
    metrics_path = run_root / "metrics.jsonl"
    journal = _MetricJournal(metrics_path, expected_steps=start_step)
    if stop_after_step is not None and stop_after_step <= start_step:
        raise ValueError("stop_after_step must be later than the resumed checkpoint")

    completed_step = start_step
    final_checkpoint: Path | None = None
    target_step = stop_after_step or config.max_steps
    for zero_based_step in range(start_step, target_step):
        batch = training_batch(
            dataset,
            config,
            context,
            zero_based_step=zero_based_step,
        )
        observed = experiment.train_step(batch)
        completed_step = zero_based_step + 1
        if context.is_primary:
            journal.append(step=completed_step, loss=observed.loss)
        must_checkpoint = (
            completed_step % config.checkpoint_every == 0
            or completed_step == target_step
            or completed_step == config.max_steps
        )
        if must_checkpoint and context.is_primary:
            final_checkpoint = checkpoint_store.save(
                _checkpoint_state(
                    experiment,
                    step=completed_step,
                    config=config,
                    dataset=dataset,
                )
            )
        if must_checkpoint:
            _barrier(context)

    if completed_step < 1:
        raise RuntimeError("training completed no optimizer steps")
    if final_checkpoint is None:
        latest = checkpoint_store.latest(
            expected_config_sha256=config.sha256,
            expected_dataset_sha256=dataset.content_sha256,
        )
        if latest is None:
            raise RuntimeError("training produced no valid checkpoint")
        final_checkpoint = run_root / "checkpoints" / f"step-{latest.step:08d}"
    status: Literal["completed", "stopped"] = (
        "completed" if completed_step == config.max_steps else "stopped"
    )
    _barrier(context)
    if context.is_primary:
        payload = {
            "schema_version": _TRAIN_SCHEMA,
            "scope": "orchestration_smoke_only",
            "publication_eligible": False,
            "status": status,
            "step": completed_step,
            "config_sha256": config.sha256,
            "dataset_sha256": dataset.content_sha256,
            "model_sha256": experiment.model_sha256(),
            "metrics_sha256": journal.sha256,
            "checkpoint": final_checkpoint.name,
            "runtime": context.runtime.as_manifest(),
            "distributed": context.ranks.as_manifest(),
        }
        _atomic_write(run_root / "train-result.json", _canonical_json(payload))
    _barrier(context)
    return _read_train_result(run_root / "train-result.json")


def _split_metrics(
    experiment: FixtureExperiment,
    config: ExperimentConfig,
    dataset: PreparedDataset,
    context: DistributedContext,
    *,
    split: str,
) -> tuple[float, int]:
    squared_error_sum = 0.0
    example_count = 0
    for batch in evaluation_batches(dataset, config, context, split=split):
        observed = experiment.evaluate(batch)
        squared_error_sum += observed.squared_error_sum
        example_count += observed.example_count
    aggregate = torch.tensor(
        [squared_error_sum, float(example_count)],
        dtype=torch.float64,
        device=context.device,
    )
    if context.ranks.world_size > 1:
        torch.distributed.all_reduce(aggregate, op=torch.distributed.ReduceOp.SUM)
    total_error = float(aggregate[0].cpu())
    total_count = int(aggregate[1].cpu())
    if total_count < 1:
        raise RuntimeError(f"evaluation split {split} contains no examples")
    return total_error / total_count, total_count


def _read_evaluation_result(path: Path) -> EvaluationResult:
    payload: Any = json.loads(path.read_bytes())
    if type(payload) is not dict or payload.get("schema_version") != _EVALUATION_SCHEMA:
        raise ValueError("evaluation result is invalid")
    return EvaluationResult(
        status="completed",
        model_sha256=payload["model_sha256"],
        validation_mse=payload["metrics"]["validation_mse"],
        validation_count=payload["metrics"]["validation_count"],
        test_mse=payload["metrics"]["test_mse"],
        test_count=payload["metrics"]["test_count"],
        result_sha256=_sha256_file(path),
        result_path=path,
    )


def evaluate_fixture(
    config: ExperimentConfig,
    dataset: PreparedDataset,
    run_root: Path,
    context: DistributedContext,
) -> EvaluationResult:
    if type(config) is not ExperimentConfig or config.profile != "smoke":
        raise TypeError("fixture evaluation requires an exact smoke ExperimentConfig")
    if type(dataset) is not PreparedDataset:
        raise TypeError("fixture evaluation requires an exact PreparedDataset")
    if type(context) is not DistributedContext:
        raise TypeError("fixture evaluation requires an exact DistributedContext")
    _validate_run_root(run_root)
    checkpoint = CheckpointStore(run_root / "checkpoints").latest(
        expected_config_sha256=config.sha256,
        expected_dataset_sha256=dataset.content_sha256,
    )
    if checkpoint is None or checkpoint.step != config.max_steps:
        raise RuntimeError("evaluation requires the completed configured checkpoint")
    experiment = FixtureExperiment(config, context)
    experiment.load_state_dicts(checkpoint.model_state, checkpoint.optimizer_state)
    validation_mse, validation_count = _split_metrics(
        experiment,
        config,
        dataset,
        context,
        split="validation",
    )
    test_mse, test_count = _split_metrics(
        experiment,
        config,
        dataset,
        context,
        split="test",
    )
    if context.is_primary:
        payload = {
            "schema_version": _EVALUATION_SCHEMA,
            "scope": "orchestration_smoke_only",
            "publication_eligible": False,
            "status": "completed",
            "config_sha256": config.sha256,
            "dataset_sha256": dataset.content_sha256,
            "model_sha256": experiment.model_sha256(),
            "checkpoint_step": checkpoint.step,
            "metrics": {
                "test_count": test_count,
                "test_mse": test_mse,
                "validation_count": validation_count,
                "validation_mse": validation_mse,
            },
        }
        _atomic_write(run_root / "evaluation.json", _canonical_json(payload))
    _barrier(context)
    return _read_evaluation_result(run_root / "evaluation.json")
