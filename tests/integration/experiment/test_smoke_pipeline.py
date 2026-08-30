from __future__ import annotations

import json
from pathlib import Path

import torch

from ratemem.data.manifest import DatasetManifest
from ratemem.data.prepare import prepare_dataset
from ratemem.experiment.config import ExperimentConfig
from ratemem.experiment.report import render_report
from ratemem.experiment.runner import evaluate_fixture, train_fixture
from ratemem.runtime.device import DeviceRuntime
from ratemem.runtime.distributed import DistributedContext, RankEnvironment


def cpu_context() -> DistributedContext:
    return DistributedContext(
        runtime=DeviceRuntime(
            kind="cpu",
            device=torch.device("cpu"),
            distributed_backend="gloo",
            device_count=0,
            device_names=(),
            bf16_supported=False,
        ),
        ranks=RankEnvironment(rank=0, local_rank=0, world_size=1, local_world_size=1),
    )


def prepared(tmp_path: Path):
    manifest = DatasetManifest.load(Path("configs/data/smoke.yaml"))
    return prepare_dataset(manifest, tmp_path / "data")


def config() -> ExperimentConfig:
    return ExperimentConfig.load(Path("configs/experiments/smoke.yaml"))


def test_resumed_fixture_matches_uninterrupted_run(tmp_path: Path) -> None:
    dataset = prepared(tmp_path)
    experiment_config = config()
    full = train_fixture(
        experiment_config,
        dataset,
        tmp_path / "full",
        cpu_context(),
        resume="never",
    )
    stopped = train_fixture(
        experiment_config,
        dataset,
        tmp_path / "resumed",
        cpu_context(),
        resume="never",
        stop_after_step=2,
    )
    resumed = train_fixture(
        experiment_config,
        dataset,
        tmp_path / "resumed",
        cpu_context(),
        resume="auto",
    )

    assert stopped.status == "stopped"
    assert stopped.step == 2
    assert full.status == resumed.status == "completed"
    assert resumed.model_sha256 == full.model_sha256
    assert resumed.metrics_sha256 == full.metrics_sha256
    assert resumed.step == full.step == experiment_config.max_steps


def test_evaluate_and_report_use_only_validated_outputs(tmp_path: Path) -> None:
    dataset = prepared(tmp_path)
    experiment_config = config()
    training = train_fixture(
        experiment_config,
        dataset,
        tmp_path / "run",
        cpu_context(),
        resume="never",
    )

    evaluation = evaluate_fixture(
        experiment_config,
        dataset,
        tmp_path / "run",
        cpu_context(),
    )
    report = render_report(tmp_path / "run")

    assert evaluation.status == "completed"
    assert evaluation.model_sha256 == training.model_sha256
    assert evaluation.validation_count == 2
    assert evaluation.test_count == 2
    assert evaluation.validation_mse >= 0
    assert evaluation.test_mse >= 0
    assert report.publication_eligible is False
    assert report.train_result_sha256 == training.result_sha256
    assert report.evaluation_sha256 == evaluation.result_sha256
    report_payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert report_payload["scope"] == "orchestration_smoke_only"
    assert report_payload["publication_eligible"] is False
    assert report.csv_path.read_text(encoding="utf-8").splitlines()[0] == (
        "metric,value"
    )


def test_evaluation_does_not_change_checkpoint_or_training_metrics(tmp_path: Path) -> None:
    dataset = prepared(tmp_path)
    experiment_config = config()
    training = train_fixture(
        experiment_config,
        dataset,
        tmp_path / "run",
        cpu_context(),
        resume="never",
    )
    metrics_before = training.metrics_path.read_bytes()
    checkpoint_before = training.checkpoint_path.joinpath("model.safetensors").read_bytes()

    evaluate_fixture(
        experiment_config,
        dataset,
        tmp_path / "run",
        cpu_context(),
    )

    assert training.metrics_path.read_bytes() == metrics_before
    assert training.checkpoint_path.joinpath("model.safetensors").read_bytes() == checkpoint_before


def test_training_refuses_existing_run_without_resume(tmp_path: Path) -> None:
    dataset = prepared(tmp_path)
    experiment_config = config()
    train_fixture(
        experiment_config,
        dataset,
        tmp_path / "run",
        cpu_context(),
        resume="never",
        stop_after_step=2,
    )

    try:
        train_fixture(
            experiment_config,
            dataset,
            tmp_path / "run",
            cpu_context(),
            resume="never",
        )
    except FileExistsError as error:
        assert "RESUME=auto" in str(error)
    else:
        raise AssertionError("an existing run was overwritten")
