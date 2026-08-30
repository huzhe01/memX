from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.method.config import MethodTrainingLock


def test_committed_method_lock_schema_is_current() -> None:
    path = Path("schemas/ratemem-method-lock-v1.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert path.read_bytes() == canonical_json_bytes(MethodTrainingLock.model_json_schema()) + b"\n"
