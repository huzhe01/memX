from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ratemem.evaluation.canonical import write_json_atomic
from ratemem.evaluation.dataset_lock import (
    load_inventory,
    seal_dataset_lock,
    write_dataset_lock_and_card,
)
from ratemem.evaluation.statistics import (
    CalibrationRecord,
    PairedPilotEffect,
    RequiredUnits,
    plan_required_units,
)
from ratemem.evaluation.traces import (
    AllPools,
    TraceHashMismatch,
    TraceManifest,
    TracePolicy,
    build_trace_set,
    verify_trace_manifest,
    write_trace_set,
)

POOLS = Path("tests/fixtures/scientific/concept-pools.json")
POLICY = Path("configs/scientific/trace-policy.yaml")


def test_committed_trace_and_calibration_schemas_match_models(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.schema.json"
    trace = tmp_path / "trace.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "stats",
            "schema-calibration",
            "--output",
            str(calibration),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "schema",
            "--output",
            str(trace),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert calibration.read_bytes() == Path(
        "schemas/scientific-calibration-record.schema.json"
    ).read_bytes()
    assert trace.read_bytes() == Path(
        "schemas/scientific-trace-manifest.schema.json"
    ).read_bytes()


def test_trace_manifest_detects_payload_tampering(tmp_path: Path) -> None:
    trace_set = build_trace_set(
        AllPools.load(POOLS),
        TracePolicy.load(POLICY),
        counts={"train": 1, "validation": 1, "final_test": 1},
        event_count=30,
    )["validation"]
    manifests = write_trace_set(trace_set, tmp_path / "traces")
    manifest_path = manifests[0]
    manifest = TraceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    payload_path = manifest_path.parent / manifest.payload_path
    rows = payload_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[-1])
    if row["kind"] in {"read", "probe"}:
        row["generation_seed"] += 1
    else:
        row["event_index"] += 1
    rows[-1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    payload_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(TraceHashMismatch, match="payload SHA-256"):
        verify_trace_manifest(manifest_path)


def test_power_plan_cli_writes_a_hash_bound_required_unit_record(tmp_path: Path) -> None:
    rows = tuple(
        PairedPilotEffect(
            inference_unit_id=f"unit_{index:03d}",
            metric_id="identity_delta",
            paired_effect=(-0.02 + (index % 8) * 0.01),
            source_artifact_sha256=f"{index + 1:064x}",
        )
        for index in range(24)
    )
    calibration = CalibrationRecord.create(
        dataset_lock_id="1" * 64,
        evaluator_revision="2" * 40,
        pool_sha256="3" * 64,
        split="calibration",
        rows=rows,
    )
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "required-units.json"
    write_json_atomic(calibration_path, calibration.model_dump(mode="json"))
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "stats",
            "plan-units",
            "--calibration-record",
            str(calibration_path),
            "--maximum-half-width",
            "0.02",
            "--minimum-effect",
            "0.03",
            "--alpha",
            "0.05",
            "--power",
            "0.80",
            "--minimum-units",
            "12",
            "--simulation-seed",
            "314159",
            "--monte-carlo-draws",
            "256",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    required = RequiredUnits.model_validate_json(output.read_text(encoding="utf-8"))
    assert completed.stdout.startswith("PASS power-plan: final deployment episodes=")
    assert required.required_units == max(
        required.ci_required_units,
        required.power_required_units,
    )


def test_visible_trace_cli_writes_only_train_and_validation(tmp_path: Path) -> None:
    dataset_lock = seal_dataset_lock(
        load_inventory(Path("tests/fixtures/scientific/source-inventory.json")),
        policy_path=Path("configs/scientific/dataset-policy.yaml"),
        mode="synthetic",
    )
    dataset_lock_path = tmp_path / "dataset-lock.yaml"
    write_dataset_lock_and_card(
        dataset_lock,
        dataset_lock_path,
        tmp_path / "data-card.md",
    )
    pools = AllPools.load(POOLS).model_copy(
        update={"dataset_lock_id": dataset_lock.lock_id}
    )
    pools_path = tmp_path / "concept-pools.json"
    write_json_atomic(pools_path, pools.model_dump(mode="json"))

    rows = tuple(
        PairedPilotEffect(
            inference_unit_id=f"visible_{index:03d}",
            metric_id="identity_delta",
            paired_effect=(-0.02 + (index % 8) * 0.01),
            source_artifact_sha256=f"{index + 101:064x}",
        )
        for index in range(24)
    )
    calibration = CalibrationRecord.create(
        dataset_lock_id=dataset_lock.lock_id,
        evaluator_revision="2" * 40,
        pool_sha256="3" * 64,
        split="calibration",
        rows=rows,
    )
    required = plan_required_units(
        calibration,
        0.02,
        0.03,
        0.05,
        0.80,
        12,
        314159,
        monte_carlo_draws=256,
    )
    power_path = tmp_path / "required-units.json"
    write_json_atomic(power_path, required.model_dump(mode="json"))
    output = tmp_path / "visible-traces"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "build-visible",
            "--dataset-lock",
            str(dataset_lock_path),
            "--policy",
            str(POLICY),
            "--power-record",
            str(power_path),
            "--concept-pools",
            str(pools_path),
            "--splits",
            "train,validation",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.endswith(
        "PASS traces: train and validation manifests have disjoint concepts, ids, and seeds\n"
    )
    assert (output / "train").is_dir()
    assert (output / "validation").is_dir()
    assert not (output / "final_test").exists()

    forbidden_output = tmp_path / "forbidden-final"
    forbidden = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "build-visible",
            "--dataset-lock",
            str(dataset_lock_path),
            "--policy",
            str(POLICY),
            "--power-record",
            str(power_path),
            "--concept-pools",
            str(pools_path),
            "--splits",
            "final_test",
            "--output",
            str(forbidden_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert forbidden.returncode == 2
    assert "visible builds accept unique train and validation splits only" in forbidden.stderr
    assert not forbidden_output.exists()
