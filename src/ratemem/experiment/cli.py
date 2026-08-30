from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from ratemem.data import load_data_manifest
from ratemem.data.manifest import DatasetManifest
from ratemem.data.prepare import PreparedDataset, prepare_dataset
from ratemem.data.subjects200k import (
    Subjects200KManifest,
    prepare_subjects200k_snapshot,
)
from ratemem.experiment.config import ExperimentConfig
from ratemem.experiment.report import render_report
from ratemem.experiment.runner import evaluate_fixture, train_fixture
from ratemem.runtime.device import DeviceRuntime, observe_runtime_probe, resolve_runtime
from ratemem.runtime.distributed import (
    DistributedContext,
    RankEnvironment,
    distributed_session,
)
from ratemem.runtime.preflight import build_preflight_receipt, validate_preflight


def _print_json(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _backend_override() -> str | None:
    value = os.environ.get("RATEMEM_DIST_BACKEND")
    if value is None:
        return None
    if not value or value != value.strip() or value.lower() != value:
        raise ValueError("RATEMEM_DIST_BACKEND must be canonical lowercase text")
    return value


def _runtime(device: str) -> tuple[DeviceRuntime, RankEnvironment, tuple[str, ...]]:
    override = _backend_override()
    probe = observe_runtime_probe(
        additional_backends=() if override is None else (override,)
    )
    runtime = resolve_runtime(device, probe, backend_override=override)
    ranks = RankEnvironment.from_mapping(
        os.environ,
        visible_devices=runtime.device_count,
    )
    compatible = ("pccl",) if override is None else ("pccl", override)
    validate_preflight(runtime, ranks, ppu_compatible_backends=compatible)
    return runtime, ranks, compatible


@contextmanager
def _execution_context(device: str) -> Iterator[DistributedContext]:
    runtime, ranks, _compatible = _runtime(device)
    with distributed_session(runtime, ranks) as context:
        yield context


def _prepared(config: ExperimentConfig, data_root: Path) -> PreparedDataset:
    manifest = DatasetManifest.load(config.dataset_manifest)
    return prepare_dataset(manifest, data_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memx",
        description="Reproducible memX data, training, evaluation, and report pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="prepare immutable datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser("prepare", help="prepare and verify a dataset")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument(
        "--offline",
        action="store_true",
        help="require every pinned remote shard to exist in the local Hugging Face cache",
    )

    runtime = commands.add_parser("runtime", help="inspect launch compatibility")
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    preflight = runtime_commands.add_parser("preflight", help="validate accelerator runtime")
    preflight.add_argument("--device", choices=("auto", "cpu", "nvidia", "ppu"), required=True)

    for name in ("smoke", "train", "evaluate"):
        command = commands.add_parser(name, help=f"run memX {name}")
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument(
            "--device",
            choices=("auto", "cpu", "nvidia", "ppu"),
            required=True,
        )
        if name == "train":
            command.add_argument("--resume", choices=("never", "auto"), default="never")

    report = commands.add_parser("report", help="render validated experiment outputs")
    report.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "data" and arguments.data_command == "prepare":
        manifest = load_data_manifest(arguments.config)
        if type(manifest) is DatasetManifest:
            prepared = prepare_dataset(manifest, arguments.root)
            result: dict[str, object] = {
                "status": "prepared",
                "publication_eligible": False,
                "kind": "episode_store",
                "root": str(prepared.root),
                "episode_count": len(prepared.episodes),
                "dataset_manifest_sha256": prepared.manifest_sha256,
                "content_sha256": prepared.content_sha256,
            }
        elif type(manifest) is Subjects200KManifest:
            snapshot = prepare_subjects200k_snapshot(
                manifest,
                arguments.root,
                offline=arguments.offline,
            )
            result = {
                "status": "prepared",
                "publication_eligible": False,
                "kind": "immutable_snapshot",
                "root": str(snapshot.root),
                "repository_id": snapshot.repository_id,
                "revision": snapshot.revision,
                "shard_count": snapshot.shard_count,
                "total_bytes": snapshot.total_bytes,
                "dataset_manifest_sha256": snapshot.manifest_sha256,
            }
        else:
            raise AssertionError("unsupported validated data manifest")
        _print_json(result)
        return 0
    if arguments.command == "runtime" and arguments.runtime_command == "preflight":
        runtime, ranks, compatible = _runtime(arguments.device)
        _print_json(
            build_preflight_receipt(
                runtime,
                ranks,
                torch_version=importlib.metadata.version("torch"),
                python_version=platform.python_version(),
                ppu_compatible_backends=compatible,
            )
        )
        return 0
    if arguments.command == "report":
        report = render_report(arguments.run_root)
        _print_json(
            {
                "status": "completed",
                "publication_eligible": report.publication_eligible,
                "report": str(report.json_path),
                "report_sha256": report.report_sha256,
            }
        )
        return 0

    config = ExperimentConfig.load(arguments.config)
    prepared = _prepared(config, arguments.data_root)
    with _execution_context(arguments.device) as context:
        if arguments.command == "train":
            training_result = train_fixture(
                config,
                prepared,
                arguments.run_root,
                context,
                resume=arguments.resume,
            )
            _print_json(
                {
                    "status": training_result.status,
                    "publication_eligible": False,
                    "step": training_result.step,
                    "model_sha256": training_result.model_sha256,
                    "result": str(training_result.result_path),
                }
            )
            return 0
        if arguments.command == "evaluate":
            evaluation_result = evaluate_fixture(
                config,
                prepared,
                arguments.run_root,
                context,
            )
            _print_json(
                {
                    "status": evaluation_result.status,
                    "publication_eligible": False,
                    "result": str(evaluation_result.result_path),
                    "model_sha256": evaluation_result.model_sha256,
                }
            )
            return 0
        if arguments.command == "smoke":
            training = train_fixture(
                config,
                prepared,
                arguments.run_root,
                context,
                resume="never",
            )
            evaluate_fixture(
                config,
                prepared,
                arguments.run_root,
                context,
            )
            report = render_report(arguments.run_root)
            _print_json(
                {
                    "status": "completed",
                    "publication_eligible": False,
                    "model_sha256": training.model_sha256,
                    "report": str(report.json_path),
                }
            )
            return 0
    raise RuntimeError("unreachable command dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
