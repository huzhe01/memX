from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_PATH_TYPE = type(Path())
_REPORT_SCHEMA = "memx-smoke-report-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_result(path: Path, schema: str) -> dict[str, object]:
    try:
        payload: Any = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"result is unreadable: {path.name}") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != schema
        or payload.get("scope") != "orchestration_smoke_only"
        or payload.get("publication_eligible") is not False
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"result is not a completed orchestration smoke artifact: {path.name}")
    return cast(dict[str, object], payload)


def _metric(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative exact float")
    return value


@dataclass(frozen=True, slots=True)
class ReportResult:
    publication_eligible: bool
    train_result_sha256: str
    evaluation_sha256: str
    report_sha256: str
    json_path: Path
    csv_path: Path
    markdown_path: Path


def render_report(run_root: Path) -> ReportResult:
    if type(run_root) is not _PATH_TYPE:
        raise TypeError("report run root must be an exact Path")
    train_path = run_root / "train-result.json"
    evaluation_path = run_root / "evaluation.json"
    train = _load_result(train_path, "memx-train-result-v1")
    evaluation = _load_result(evaluation_path, "memx-evaluation-result-v1")
    if train.get("model_sha256") != evaluation.get("model_sha256"):
        raise ValueError("training and evaluation model hashes differ")
    metrics_value = evaluation.get("metrics")
    if type(metrics_value) is not dict:
        raise TypeError("evaluation metrics must be an exact mapping")
    metrics = cast(dict[str, object], metrics_value)
    validation_mse = _metric(metrics.get("validation_mse"), "validation_mse")
    test_mse = _metric(metrics.get("test_mse"), "test_mse")
    train_hash = _sha256_file(train_path)
    evaluation_hash = _sha256_file(evaluation_path)
    report_payload = {
        "schema_version": _REPORT_SCHEMA,
        "scope": "orchestration_smoke_only",
        "publication_eligible": False,
        "train_result_sha256": train_hash,
        "evaluation_sha256": evaluation_hash,
        "model_sha256": train["model_sha256"],
        "metrics": {
            "test_mse": test_mse,
            "validation_mse": validation_mse,
        },
    }
    report_root = run_root / "report"
    json_path = report_root / "report.json"
    csv_path = report_root / "metrics.csv"
    markdown_path = report_root / "REPORT.md"
    _atomic_write(json_path, _canonical_json(report_payload))
    _atomic_write(
        csv_path,
        (
            "metric,value\n"
            f"validation_mse,{validation_mse:.12g}\n"
            f"test_mse,{test_mse:.12g}\n"
        ).encode(),
    )
    _atomic_write(
        markdown_path,
        (
            "# memX orchestration smoke report\n\n"
            "This fixture verifies the execution pipeline and is not publication eligible.\n\n"
            f"- Validation MSE: `{validation_mse:.12g}`\n"
            f"- Test MSE: `{test_mse:.12g}`\n"
            f"- Model SHA-256: `{train['model_sha256']}`\n"
        ).encode(),
    )
    return ReportResult(
        publication_eligible=False,
        train_result_sha256=train_hash,
        evaluation_sha256=evaluation_hash,
        report_sha256=_sha256_file(json_path),
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
