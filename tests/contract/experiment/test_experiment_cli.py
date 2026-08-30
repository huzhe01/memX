"""Contract tests for the public memX experiment CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratemem.experiment.cli import main


def test_help_lists_the_complete_execution_surface(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    assert stopped.value.code == 0
    output = capsys.readouterr().out
    for command in ("data", "runtime", "smoke", "train", "evaluate", "report"):
        assert command in output


def test_data_prepare_cli_emits_canonical_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "data",
            "prepare",
            "--config",
            "configs/data/smoke.yaml",
            "--root",
            str(tmp_path),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "prepared"
    assert payload["episode_count"] == 8
    assert payload["publication_eligible"] is False


def test_smoke_cli_runs_the_complete_offline_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "smoke",
            "--config",
            "configs/experiments/smoke.yaml",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "run"),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["publication_eligible"] is False
    assert Path(payload["report"]).is_file()
