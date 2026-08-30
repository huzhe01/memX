from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_duplicate_report_schema_matches_model(tmp_path: Path) -> None:
    generated = tmp_path / "duplicate.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "duplicate-schema",
            "--output",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.read_bytes() == Path(
        "schemas/scientific-duplicate-report.schema.json"
    ).read_bytes()


def test_feature_encoder_lock_cli_rejects_mutable_revision(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"weights")
    inventory = tmp_path / "models.json"
    inventory.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "dinov2_large_duplicate_audit",
                        "repository_uri": "https://github.com/facebookresearch/dinov2",
                        "immutable_revision": "main",
                        "weights_path": str(weights),
                        "weights_sha256": (
                            "9a129038d9a00aed0cf6a7ea059ca50a"
                            "813449061ab87848cf1a13eafdf33b2c"
                        ),
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "encoder-lock.json"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "data",
            "lock-feature-encoder",
            "--model-id",
            "dinov2_large_duplicate_audit",
            "--model-inventory",
            str(inventory),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr.endswith("feature encoder revision must be immutable lowercase hex\n")
    assert not output.exists()
