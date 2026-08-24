import json
import subprocess
import sys


def test_core_smoke_command() -> None:
    command = [sys.executable, "-m", "ratemem.cli", "smoke-core"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    repeated = subprocess.run(command, check=True, capture_output=True, text=True)

    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["serialized_bytes"] <= payload["budget_bytes"]
    assert result.stdout == json.dumps(payload, sort_keys=True) + "\n"
    assert repeated.stdout == result.stdout
    assert result.stderr == ""
    assert repeated.stderr == ""
