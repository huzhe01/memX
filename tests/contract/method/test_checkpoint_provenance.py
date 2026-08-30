from __future__ import annotations

from pathlib import Path

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.method.checkpoint import MethodCheckpointManifest


def test_method_checkpoint_schema_matches_committed_contract() -> None:
    expected = canonical_json_bytes(MethodCheckpointManifest.model_json_schema()) + b"\n"
    actual = Path("schemas/ratemem-method-checkpoint-v1.schema.json").read_bytes()

    assert actual == expected
