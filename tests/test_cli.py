from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ratemem.cli as cli
from ratemem.codec.progressive import EncodedCode

_EXPECTED = {
    "budget_bytes": 8192,
    "serialized_bytes": 403,
    "status": "passed",
}


def _run(
    command: list[str], *, cwd: Path | None = None, hash_seed: str = "1"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": hash_seed},
    )


def _assert_single_json_line(result: subprocess.CompletedProcess[str]) -> None:
    payload = json.loads(result.stdout)
    assert payload == _EXPECTED
    assert payload["serialized_bytes"] <= payload["budget_bytes"]
    assert result.stdout == json.dumps(payload, sort_keys=True) + "\n"
    assert result.stderr == ""


def test_core_smoke_command() -> None:
    command = [sys.executable, "-m", "ratemem.cli", "smoke-core"]
    result = _run(command)
    repeated = _run(command)

    _assert_single_json_line(result)
    assert repeated.stdout == result.stdout
    assert repeated.stderr == ""


@pytest.mark.parametrize("hash_seed", ["7", "777"])
@pytest.mark.parametrize("entrypoint", ["module", "installed"])
def test_core_smoke_is_deterministic_from_an_external_directory(
    tmp_path: Path, hash_seed: str, entrypoint: str
) -> None:
    if entrypoint == "module":
        command = [sys.executable, "-m", "ratemem.cli", "smoke-core"]
    else:
        command = [str(Path(sys.executable).with_name("ratemem")), "smoke-core"]

    result = _run(command, cwd=tmp_path, hash_seed=hash_seed)
    _assert_single_json_line(result)


def test_smoke_rejects_a_nonfinite_or_misshaped_selected_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        EncodedCode,
        "decode",
        lambda self, packet_count: np.array([np.nan], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="decoded selected prefix"):
        cli.smoke_core()


def test_smoke_requires_strict_selected_prefix_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        EncodedCode,
        "decode",
        lambda self, packet_count: np.zeros(self.shape, dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="strictly improve reconstruction"):
        cli.smoke_core()


def test_smoke_runs_the_lifecycle_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_replay(*args: object, **kwargs: object) -> Any:
        raise RuntimeError("lifecycle-probe-marker")

    monkeypatch.setattr(cli, "replay", fail_replay)
    with pytest.raises(RuntimeError, match="lifecycle-probe-marker"):
        cli.smoke_core()


def test_smoke_round_trips_the_attempt_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ManifestProbe:
        def __init__(self, **values: object) -> None:
            self.values = values

        def model_dump_json(self) -> str:
            return "{}"

        @classmethod
        def model_validate_json(cls, payload: str) -> ManifestProbe:
            raise RuntimeError("manifest-roundtrip-marker")

    monkeypatch.setattr(cli, "AttemptManifest", ManifestProbe)
    with pytest.raises(RuntimeError, match="manifest-roundtrip-marker"):
        cli.smoke_core()
