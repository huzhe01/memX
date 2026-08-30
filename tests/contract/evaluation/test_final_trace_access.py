from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def test_training_sources_cannot_import_or_name_final_trace_access() -> None:
    training_root = Path("src/ratemem/training")
    if not training_root.exists():
        return
    for path in training_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ratemem.evaluation.final_trace"
            if isinstance(node, ast.Attribute):
                assert node.attr != "FINAL_EVALUATION"
        assert "final-test-envelope" not in source


def test_committed_final_trace_schemas_match_strict_models(tmp_path: Path) -> None:
    envelope = tmp_path / "envelope.schema.json"
    freeze = tmp_path / "freeze.schema.json"
    subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "envelope-schema",
            "--output",
            str(envelope),
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
            "freeze-schema",
            "--output",
            str(freeze),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert envelope.read_bytes() == Path(
        "schemas/scientific-final-trace-envelope.schema.json"
    ).read_bytes()
    assert freeze.read_bytes() == Path(
        "schemas/scientific-final-freeze.schema.json"
    ).read_bytes()


def test_ignore_policy_keeps_private_keys_out_but_allows_public_recipient() -> None:
    private = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "artifacts/scientific/final/final-trace.key",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    public = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "configs/scientific/traces/final-trace-recipient.pem",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert private.returncode == 0
    assert public.returncode == 1
