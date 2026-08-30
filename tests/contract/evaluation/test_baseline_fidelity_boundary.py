from __future__ import annotations

import subprocess
from pathlib import Path


def test_committed_baseline_fidelity_authorization_schema_matches_model(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "baseline-fidelity-authorization.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "compute",
            "schema-baseline-fidelity",
            "--output",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.read_bytes() == Path(
        "schemas/scientific-baseline-fidelity-authorization.schema.json"
    ).read_bytes()
