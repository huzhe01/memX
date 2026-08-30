from __future__ import annotations

import subprocess
from pathlib import Path


def test_committed_evaluation_lock_schema_matches_model(tmp_path: Path) -> None:
    generated = tmp_path / "evaluation-lock.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "lock",
            "schema",
            "--kind",
            "evaluation",
            "--output",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.read_bytes() == Path("schemas/evaluation-lock.schema.json").read_bytes()


def test_evaluation_lock_cli_fails_closed_before_baseline_lock(tmp_path: Path) -> None:
    output = tmp_path / "evaluation-lock.yaml"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "lock",
            "evaluation",
            "--policy",
            "configs/scientific/evaluation-policy.yaml",
            "--dataset-lock",
            str(tmp_path / "dataset-lock.yaml"),
            "--baseline-lock",
            str(tmp_path / "missing-baseline-lock.yaml"),
            "--baseline-audit-receipt",
            str(tmp_path / "baseline-audit.json"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--evaluator-inventory",
            str(tmp_path / "evaluators.json"),
            "--byte-ledger",
            str(tmp_path / "byte-ledger.json"),
            "--margin-record",
            str(tmp_path / "margins.json"),
            "--power-record",
            str(tmp_path / "power.json"),
            "--approvals",
            str(tmp_path / "approvals.json"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr.endswith(
        "BLOCKED evaluation-lock: baseline lock is missing\n"
    )
    assert not output.exists()
