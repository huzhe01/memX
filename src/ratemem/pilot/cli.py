"""Fail-closed local operating surface for the single Modal engineering pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Never, cast

import typer

from ratemem.pilot.artifacts import validate_attempt
from ratemem.pilot.config import ModalBudgetConfig
from ratemem.pilot.costs import CostLedger, CostRates, ResourceContract, conservative_bound
from ratemem.pilot.one_shot import (
    GLOBAL_SLOT_PATH,
    GLOBAL_SUBMISSION_RECEIPT_PATH,
    PERMIT_PATH,
    PilotIdentity,
    claim_global_pilot_slot,
    create_launch_permit,
    new_attempt_id,
    validate_consumed_launch_evidence,
    validate_unsubmitted_launch_permit,
)
from ratemem.pilot.private_io import (
    canonical_json_bytes,
    ensure_private_directory,
    file_sha256,
    private_lock,
    read_private_bytes,
    read_private_json,
    write_atomic_private_json,
    write_exclusive_private_bytes,
    write_exclusive_private_json,
)
from ratemem.pilot.runner import pilot_config_sha256
from ratemem.pilot.workspace import (
    capture_workspace_snapshot,
    verify_fresh_attestation_file,
    verify_workspace_snapshot,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
ATTESTATION_PATH = Path("artifacts/pilot/workspace-attestation.json")
LEDGER_PATH = Path("artifacts/pilot/cost-ledger.jsonl")
BUDGET_CONFIG_PATH = Path("configs/pilot/modal-budget.json")
MODAL_CONFIG_PATH = Path("/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml")
SETTLEMENT_DIRECTORY = Path("artifacts/pilot/reconciliation")
INCIDENT_DIRECTORY = Path("artifacts/pilot/incidents")
PROVISION_INTENT_PATH = Path("artifacts/pilot/volume-provision-intent.json")
PROVISION_RECEIPT_PATH = Path("artifacts/pilot/volume-provision-receipt.json")
BUDGET_CONFIRMATION_STATEMENT = (
    "I confirm the Modal dashboard Workspace usage budget is USD 28.00 before credits "
    "and the Workspace spend limit is USD 0.00 after credits."
)
_PROFILE = "ratemem-pilot"
_ENVIRONMENT = "main"
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_FIXED_USD = re.compile(r"(0|[1-9][0-9]*)\.[0-9]{2,6}")
_RECEIPT_SEMANTICS = "lower_bound_may_miss_precommit_reschedule"
_RATE_KEYS = {
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
}
_REQUEST_KEYS = {
    "attempt_id",
    "workspace",
    "source_sha256",
    "git_commit",
    "git_diff_sha256",
    "config_sha256",
    "slot_sha256",
    "permit_sha256",
    "submission_receipt_sha256",
    "known_usage_before_usd",
    "pending_worst_case_usd",
    "phase_bound_usd",
    "rates",
    "rates_sha256",
}
_RECEIPT_KEYS = _REQUEST_KEYS | {
    "function_call_id",
    "input_id",
    "task_id",
    "receipt_id",
    "observed_at",
    "semantics",
}
_PAYLOAD_FILES = {
    "config.json",
    "dataset-manifest.json",
    "execution-receipts.jsonl",
    "metrics.jsonl",
    "rates.json",
}
_RESERVED_ARTIFACT_FILES = {"attempt.pending.json", "checksums.sha256", "attempt.json"}
_REQUIRED_VOLUMES = frozenset({"ratemem-sana-cache", "ratemem-pilot-artifacts"})
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SETTLEMENT_MIN_AGE_SECONDS = 4 * 24 * 60 * 60

app = typer.Typer(
    no_args_is_help=True,
    help="Guarded local controls for the single RateMem Modal engineering pilot.",
)


@dataclass(frozen=True, slots=True)
class SourceTreeIdentity:
    git_commit: str
    source_sha256: str
    git_diff_sha256: str


@dataclass(frozen=True, slots=True)
class VolumeAbsenceEvidence:
    confirmed_at: datetime
    known_usage: Decimal
    sha256: str


def _launch_state(workspace: str) -> tuple[dict[str, Any], dict[str, Any]]:
    permit_path = _operating_path(PERMIT_PATH)
    current_source = source_tree_sha256()
    try:
        validate_unsubmitted_launch_permit(
            permit_path,
            slot=GLOBAL_SLOT_PATH,
            receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
            expected_workspace=workspace,
            current_source_sha256=current_source,
        )
        state: dict[str, Any] = {
            "state": "unsubmitted",
            "permit_sha256": file_sha256(permit_path),
            "slot_sha256": read_private_json(permit_path)["slot_sha256"],
            "submission_receipt_sha256": None,
        }
    except FileExistsError:
        consumed = validate_consumed_launch_evidence(
            permit_path,
            slot=GLOBAL_SLOT_PATH,
            receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
            expected_workspace=workspace,
            current_source_sha256=current_source,
        )
        state = {
            "state": "consumed",
            "permit_sha256": consumed["permit_sha256"],
            "slot_sha256": consumed["slot_sha256"],
            "submission_receipt_sha256": consumed["submission_receipt_sha256"],
        }
    permit = read_private_json(permit_path)
    if (
        permit.get("attempt_id") is None
        or permit.get("workspace") != workspace
        or permit.get("slot_sha256") != state["slot_sha256"]
    ):
        raise ValueError("incident launch state differs from the validated permit")
    return state, permit


def _git(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    )
    if completed.stderr:
        raise RuntimeError("git identity command emitted unexpected stderr")
    return completed.stdout


def source_tree_identity() -> SourceTreeIdentity:
    """Bind a completely clean repository to its exact lowercase HEAD commit."""

    status = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    if status:
        raise ValueError("pilot source tree has tracked, staged, or untracked changes")
    raw_commit = _git(["rev-parse", "--verify", "HEAD"]).strip()
    try:
        commit = raw_commit.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("git HEAD is not a canonical lowercase SHA-1") from error
    if _HEX40.fullmatch(commit) is None:
        raise ValueError("git HEAD is not a canonical lowercase SHA-1")
    tracked_diff = _git(["diff", "--binary", "HEAD", "--"])
    staged_diff = _git(["diff", "--cached", "--binary", "HEAD", "--"])
    if tracked_diff or staged_diff:
        raise ValueError("pilot source tree has tracked, staged, or untracked changes")
    return SourceTreeIdentity(
        git_commit=commit,
        source_sha256=hashlib.sha256(raw_commit).hexdigest(),
        git_diff_sha256=_EMPTY_SHA256,
    )


def source_tree_sha256() -> str:
    """Compatibility helper used by the Modal local entry point."""

    return source_tree_identity().source_sha256


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("secure paths must not contain symbolic links")


def _secure_regular_metadata(path: Path, *, private: bool) -> os.stat_result:
    _assert_no_symlink_ancestors(path)
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or (private and stat.S_IMODE(metadata.st_mode) != 0o600)
    ):
        qualifier = "owner-only mode-0600 " if private else "owner "
        raise PermissionError(f"path must be a {qualifier}regular single-link file")
    return metadata


def _scan_file(path: Path) -> bool:
    before = _secure_regular_metadata(path, private=False)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    assignment = re.compile(
        rb"(?i)(?:MODAL_TOKEN_ID|MODAL_TOKEN_SECRET|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|"
        rb"WANDB_API_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)[\"']?\s*[:=]\s*[\"']?[^\s\"']+"
    )
    opaque = re.compile(rb"(?:ak-|as-|hf_)[A-Za-z0-9_-]{20,}")
    overlap = b""
    found = False
    try:
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            sample = overlap + chunk
            found = found or bool(assignment.search(sample) or opaque.search(sample))
            overlap = sample[-512:]
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("credential-scan input changed while being read")
    return found


def credential_findings(paths: list[Path]) -> list[Path]:
    if type(paths) is not list or any(type(path) is not type(Path()) for path in paths):
        raise TypeError("credential scan paths must be an exact list of Path values")
    findings: list[Path] = []
    for path in paths:
        if _scan_file(path):
            findings.append(path)
    return findings


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    if type(content) is not bytes:
        raise TypeError(f"{label} must be exact bytes")
    try:
        decoded = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if type(decoded) is not dict:
        raise ValueError(f"{label} root must be an exact object")
    payload = cast(dict[str, Any], decoded)
    if canonical_json_bytes(payload) != content:
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _strict_json_lines(content: bytes, *, label: str) -> list[dict[str, Any]]:
    if not content or not content.endswith(b"\n"):
        raise ValueError(f"{label} must be nonempty canonical JSONL with a final newline")
    return [
        _strict_json_bytes(line, label=f"{label} line {index}")
        for index, line in enumerate(content.splitlines(), start=1)
    ]


def create_operator_budget_evidence(
    *,
    screenshot: Path,
    output: Path,
    workspace: str,
    confirmation_statement: str,
) -> Path:
    if type(screenshot) is not type(Path()) or type(output) is not type(Path()):
        raise TypeError("dashboard screenshot and evidence output must be exact Path values")
    if type(workspace) is not str or _WORKSPACE.fullmatch(workspace) is None:
        raise ValueError("workspace must be one canonical lowercase Modal workspace slug")
    if confirmation_statement != BUDGET_CONFIRMATION_STATEMENT:
        raise ValueError("the exact USD 28.00 dashboard confirmation statement is required")
    absolute = Path(os.path.abspath(screenshot))
    before = _secure_regular_metadata(absolute, private=True)
    content = read_private_bytes(absolute)
    after = absolute.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or len(content) != before.st_size:
        raise RuntimeError("dashboard screenshot changed while evidence was created")
    captured_at = datetime.fromtimestamp(before.st_mtime, UTC)
    payload = {
        "kind": "operator-dashboard-budget-v1",
        "profile": _PROFILE,
        "workspace": workspace,
        "environment": _ENVIRONMENT,
        "workspace_budget_usd": "28.00",
        "workspace_spend_limit_usd": "0.00",
        "captured_at": captured_at.isoformat(),
        "dashboard_evidence_path": str(absolute),
        "dashboard_evidence_sha256": hashlib.sha256(content).hexdigest(),
        "confirmation_statement": BUDGET_CONFIRMATION_STATEMENT,
    }
    try:
        write_exclusive_private_json(output, payload)
    except FileExistsError:
        if read_private_bytes(output) != canonical_json_bytes(payload):
            raise ValueError("existing operator evidence differs from this screenshot") from None
    return output


def _fixed_usd_text(value: Decimal, name: str) -> str:
    if type(value) is not Decimal or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be an exact finite nonnegative Decimal")
    exponent = -cast(int, value.as_tuple().exponent)
    if exponent > 6:
        raise ValueError(f"{name} exceeds the artifact's exact six-decimal precision")
    text = format(value, "f")
    if "." not in text:
        text += ".00"
    else:
        whole, fraction = text.split(".", maxsplit=1)
        text = f"{whole}.{fraction.ljust(2, '0')}"
    if _FIXED_USD.fullmatch(text) is None:
        raise ValueError(f"{name} is not a canonical fixed-point USD amount")
    return text


def _require_repository_cwd() -> None:
    if Path.cwd().resolve() != REPOSITORY_ROOT.resolve():
        raise RuntimeError("ratemem-pilot commands must run from the repository root")


def _operating_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _modal_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        )
        if key in os.environ
    }
    environment["MODAL_CONFIG_PATH"] = str(MODAL_CONFIG_PATH)
    environment["MODAL_PROFILE"] = _PROFILE
    environment["MODAL_ENVIRONMENT"] = _ENVIRONMENT
    return environment


def _validate_modal_config_state(state: str) -> None:
    if state not in {"empty", "configured"}:
        raise ValueError("Modal config state must be empty or configured")
    content = read_private_bytes(MODAL_CONFIG_PATH)
    if state == "empty":
        if content:
            raise ValueError("dedicated Modal config must be exactly empty before authentication")
        return
    try:
        decoded = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("dedicated Modal config is not valid UTF-8 TOML") from error
    if type(decoded) is not dict or set(decoded) != {_PROFILE}:
        raise ValueError("dedicated Modal config must contain exactly the pilot profile")
    profile = decoded[_PROFILE]
    if type(profile) is not dict or set(profile) != {"token_id", "token_secret", "active"}:
        raise ValueError("pilot profile must contain only token_id, token_secret, and active")
    if (
        type(profile["token_id"]) is not str
        or not profile["token_id"]
        or type(profile["token_secret"]) is not str
        or not profile["token_secret"]
        or profile["active"] is not True
    ):
        raise ValueError("pilot profile token fields or dedicated activation are invalid")


def _modal_cli_json(arguments: list[str]) -> object:
    if type(arguments) is not list or tuple(arguments) != ("volume", "list", "--env", "main"):
        raise ValueError("Modal JSON command is not allowlisted")
    completed = subprocess.run(
        ["modal", *arguments, "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=_modal_environment(),
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("allowlisted Modal volume query failed")
    try:
        return json.loads(
            completed.stdout,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError("Modal volume query did not return strict JSON") from error


def _modal_cli(arguments: list[str]) -> None:
    allowed = {
        ("volume", "create", "--env", "main", "ratemem-sana-cache"),
        ("volume", "create", "--env", "main", "ratemem-pilot-artifacts"),
    }
    if type(arguments) is not list or tuple(arguments) not in allowed:
        raise ValueError("Modal mutation command is not allowlisted")
    completed = subprocess.run(
        ["modal", *arguments],
        check=False,
        env=_modal_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("allowlisted Modal volume creation failed")


def _volume_names(value: object) -> set[str]:
    if type(value) is not list:
        raise ValueError("Modal volume list must be an exact array")
    names: set[str] = set()
    for entry in cast(list[object], value):
        if type(entry) is not dict:
            raise ValueError("Modal volume entries must be exact objects")
        payload = cast(dict[object, object], entry)
        candidates = [payload[key] for key in ("name", "Name") if key in payload]
        if len(candidates) != 1 or type(candidates[0]) is not str or not candidates[0]:
            raise ValueError("Modal volume entry has no unambiguous exact name")
        if candidates[0] in names:
            raise ValueError("Modal volume list contains duplicate names")
        names.add(candidates[0])
    return names


def _provision_identity(permit: dict[str, Any]) -> dict[str, object]:
    attempt_id = permit.get("attempt_id")
    workspace = permit.get("workspace")
    slot_sha = permit.get("slot_sha256")
    if type(attempt_id) is not str or _UUID7.fullmatch(attempt_id) is None:
        raise ValueError("provision permit attempt must be a canonical UUIDv7")
    if type(workspace) is not str or _WORKSPACE.fullmatch(workspace) is None:
        raise ValueError("provision permit workspace must be a canonical slug")
    if type(slot_sha) is not str or _HEX64.fullmatch(slot_sha) is None:
        raise ValueError("provision permit slot hash must be lowercase SHA-256")
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "workspace": workspace,
        "profile": _PROFILE,
        "environment": _ENVIRONMENT,
        "volume_names": sorted(_REQUIRED_VOLUMES),
        "permit_sha256": file_sha256(_operating_path(PERMIT_PATH)),
        "slot_sha256": slot_sha,
    }


def _read_provision_record(
    path: Path,
    *,
    identity: dict[str, object],
    kind: str,
    timestamp_key: str,
    extra: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload = read_private_json(path)
    if read_private_bytes(path) != canonical_json_bytes(payload):
        raise ValueError("volume provision record must be canonical JSON")
    static = identity | {"kind": kind} | ({} if extra is None else extra)
    if set(payload) != set(static) | {timestamp_key} or any(
        payload[key] != value for key, value in static.items()
    ):
        raise ValueError("volume provision record identity is invalid")
    observed = _canonical_utc(payload[timestamp_key], f"volume provision {timestamp_key}")
    if observed > _now_utc():
        raise ValueError("volume provision record is future-dated")
    return payload


def _verified_unsubmitted_permit(attestation: Path = ATTESTATION_PATH) -> dict[str, Any]:
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    return validate_unsubmitted_launch_permit(
        _operating_path(PERMIT_PATH),
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        expected_workspace=snapshot.workspace,
        current_source_sha256=source_tree_sha256(),
    )


def _artifact_files(path: Path, *, skip_generated: bool = False) -> list[Path]:
    absolute = Path(os.path.abspath(path))
    metadata = absolute.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise PermissionError("security-scan roots must not be symbolic links")
    if stat.S_ISREG(metadata.st_mode):
        if skip_generated and absolute.suffix in {".pyc", ".pyo"}:
            return []
        return [absolute]
    if not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("security-scan roots must be regular files or directories")
    result: list[Path] = []
    generated_directories = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for root, directories, files in os.walk(absolute, followlinks=False):
        current = Path(root)
        for name in directories:
            if (current / name).is_symlink():
                raise PermissionError("security-scan directories must not contain symlinks")
        if skip_generated:
            directories[:] = [name for name in directories if name not in generated_directories]
        for name in files:
            candidate = current / name
            if candidate.is_symlink():
                raise PermissionError("security-scan directories must not contain symlinks")
            if skip_generated and candidate.suffix in {".pyc", ".pyo"}:
                continue
            result.append(candidate)
    return sorted(result)


def _read_artifact_member(
    descriptor: int,
    name: str,
    *,
    capture: bool,
) -> tuple[bytes | None, str, int]:
    member = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=descriptor,
    )
    digest = hashlib.sha256()
    content = bytearray() if capture else None
    count = 0
    try:
        before = os.fstat(member)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise PermissionError(f"artifact member {name} must be owner-only mode 0600")
        while True:
            block = os.read(member, 1024 * 1024)
            if not block:
                break
            count += len(block)
            if capture and count > _MAX_METADATA_BYTES:
                raise ValueError(f"artifact metadata member {name} is unexpectedly large")
            digest.update(block)
            if content is not None:
                content.extend(block)
        after = os.fstat(member)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or count != before.st_size:
            raise RuntimeError(f"artifact member {name} changed while being read")
    finally:
        os.close(member)
    return (None if content is None else bytes(content), digest.hexdigest(), count)


def _canonical_utc(value: object, name: str) -> datetime:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UTC timestamp") from error
    if parsed.tzinfo != UTC or parsed.isoformat(timespec="microseconds") != value:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    return parsed


def _validate_execution_receipts(
    content: bytes,
    *,
    expected_request: dict[str, object],
    attempt: dict[str, Any],
) -> str:
    rows = _strict_json_lines(content, label="execution receipts")
    modal = cast(dict[str, object], attempt["modal"])
    marker_keys = {
        "attempt_id",
        "evidence",
        "forensic_path",
        "raw_snapshot_bytes",
        "raw_snapshot_sha256",
        "scope",
        "status",
    }
    if len(rows) == 1 and set(rows[0]) == marker_keys:
        marker = rows[0]
        if (
            attempt["status"] != "exception"
            or attempt["checkpoint"] is not None
            or marker["attempt_id"] != attempt["attempt_id"]
            or marker["evidence"] != "external_forensic_directory"
            or marker["forensic_path"] != f"execution-receipts/{attempt['attempt_id']}"
            or type(marker["raw_snapshot_bytes"]) is not int
            or marker["raw_snapshot_bytes"] <= 0
            or type(marker["raw_snapshot_sha256"]) is not str
            or _HEX64.fullmatch(marker["raw_snapshot_sha256"]) is None
            or marker["scope"] != "engineering_pilot_only"
            or marker["status"] != "semantic_invalid"
        ):
            raise ValueError("semantic-invalid execution receipt marker is not exact")
        return "external_forensic_directory"
    count = modal["execution_receipt_count"]
    if type(count) is not int or len(rows) != count:
        raise ValueError("execution receipt aggregation count differs from the artifact")
    receipt_ids: list[str] = []
    current_matches = 0
    for receipt in rows:
        if set(receipt) != _RECEIPT_KEYS:
            raise ValueError("execution receipt has missing or unexpected fields")
        if {key: receipt[key] for key in _REQUEST_KEYS} != expected_request:
            raise ValueError("execution receipt request differs from local launch evidence")
        receipt_id = receipt["receipt_id"]
        if type(receipt_id) is not str or _HEX64.fullmatch(receipt_id) is None:
            raise ValueError("execution receipt ID is invalid")
        receipt_ids.append(receipt_id)
        if receipt["semantics"] != _RECEIPT_SEMANTICS:
            raise ValueError("execution receipt semantics changed")
        _canonical_utc(receipt["observed_at"], "execution receipt observed_at")
        for key in ("function_call_id", "input_id", "task_id"):
            if type(receipt[key]) is not str or not receipt[key]:
                raise ValueError(f"execution receipt {key} must be a nonempty exact string")
        if all(receipt[key] == modal[key] for key in ("function_call_id", "input_id", "task_id")):
            current_matches += 1
    if receipt_ids != sorted(receipt_ids) or len(set(receipt_ids)) != len(receipt_ids):
        raise ValueError("execution receipts must be uniquely sorted by receipt ID")
    if current_matches < 1:
        raise ValueError("no execution receipt binds the artifact's Modal execution identity")
    return "validated_canonical_snapshot"


def _validate_metrics(
    content: bytes,
    *,
    permit_sha256: str,
    attempt: dict[str, Any],
    receipt_evidence: str,
) -> None:
    rows = _strict_json_lines(content, label="metrics")
    if len(rows) != 2 or set(rows[0]) != {"scope", "request_permit_sha256", "result"}:
        raise ValueError("metrics.jsonl must contain the exact pilot result and diagnostics rows")
    if (
        rows[0]["scope"] != "engineering_pilot_only"
        or rows[0]["request_permit_sha256"] != permit_sha256
        or type(rows[0]["result"]) is not dict
        or cast(dict[str, object], rows[0]["result"]).get("status") != attempt["status"]
        or type(rows[1]) is not dict
    ):
        raise ValueError("metrics.jsonl is not bound to the pending artifact and permit")
    diagnostics = rows[1]
    result = cast(dict[str, object], rows[0]["result"])
    probes = cast(dict[str, object], attempt["probes"])
    allowed = cast(list[str], probes["allowed_probe_names"])
    probe_results = cast(dict[str, object], probes["results"])
    checkpoint = attempt["checkpoint"]
    backward_probe = probe_results.get("one_timestep_backward")
    backward_loss = result.get("one_timestep_backward_loss")
    if type(backward_probe) is not dict or set(backward_probe) != {"status"}:
        raise ValueError("one-timestep backward probe state is not exact")
    backward_status = backward_probe["status"]
    if backward_status == "pass":
        if type(backward_loss) is not int and type(backward_loss) is not float:
            raise ValueError(
                "one-timestep backward loss must be finite and nonnegative when the probe passes"
            )
        numeric_backward_loss = backward_loss
        if not math.isfinite(numeric_backward_loss) or numeric_backward_loss < 0:
            raise ValueError(
                "one-timestep backward loss must be finite and nonnegative when the probe passes"
            )
    elif backward_loss is not None:
        raise ValueError(
            "one-timestep backward loss must be null unless the probe independently passes"
        )
    if (
        "standalone_backward_loss" not in diagnostics
        or diagnostics["standalone_backward_loss"] != backward_loss
        or type(diagnostics["standalone_backward_loss"]) is not type(backward_loss)
    ):
        raise ValueError(
            "one-timestep backward loss differs from the independent trainer observation"
        )
    expected_result: dict[str, object] = {
        "status": attempt["status"],
        "allowed_probe_names": allowed,
        "results": probe_results,
        "warmup_steps": probes["warmup_steps"],
        "measured_steps": probes["measured_steps"],
        "p50_step_seconds": probes["p50_step_seconds"],
        "p95_step_seconds": probes["p95_step_seconds"],
        "held_in_step_cap": probes["held_in_step_cap"],
        "one_timestep_backward_loss": backward_loss,
        "initial_flow_loss": probes["initial_flow_loss"],
        "final_flow_loss": probes["final_flow_loss"],
        "transformer_passes_per_step": probes["transformer_passes_per_step"],
        "checkpoint_sha256": (
            None if checkpoint is None else cast(dict[str, object], checkpoint)["sha256"]
        ),
        "checkpoint_bytes": (
            None if checkpoint is None else cast(dict[str, object], checkpoint)["bytes"]
        ),
    }
    expected_result.update({name: probe_results[name] for name in allowed})
    if set(result) != set(expected_result) or result != expected_result:
        raise ValueError("metrics result differs from attempt probes or checkpoint identity")
    if receipt_evidence == "external_forensic_directory" and (
        diagnostics.get("execution_receipt_semantic_invalid") is not True
        or diagnostics.get("execution_receipt_evidence") != "external_forensic_directory"
    ):
        raise ValueError(
            "semantic-invalid receipt marker lacks matching external forensic diagnostics"
        )
    if receipt_evidence == "validated_canonical_snapshot" and (
        diagnostics.get("execution_receipt_semantic_invalid") is not False
        or diagnostics.get("execution_receipt_evidence") != "validated_canonical_snapshot"
    ):
        raise ValueError("normal receipts require exact canonical receipt diagnostics")


def _validate_forensic_receipts(
    pending_path: Path,
    directory: Path,
) -> dict[str, Any]:
    """Bind securely downloaded raw receipts to the runner's exact snapshot bytes."""

    if type(pending_path) is not type(Path()) or type(directory) is not type(Path()):
        raise TypeError("forensic validation paths must be exact Path values")
    _require_repository_cwd()

    # This deliberately precedes all artifact and receipt parsing. Even malformed
    # evidence must not bypass the credential boundary.
    scan_files = _artifact_files(directory, skip_generated=False)
    findings = credential_findings(scan_files)
    if findings:
        raise ValueError(f"credential material found in {len(findings)} forensic receipt files")

    pending = _validate_artifact(pending_path)
    attempt_id = cast(str, pending["attempt_id"])
    root = Path(os.path.abspath(directory))
    _assert_no_symlink_ancestors(root)
    before_path = root.lstat()
    if (
        root.name != attempt_id
        or not stat.S_ISDIR(before_path.st_mode)
        or before_path.st_uid != os.getuid()
        or stat.S_IMODE(before_path.st_mode) != 0o700
    ):
        raise PermissionError(
            "forensic receipt directory must match the attempt UUID and be owner-only mode 0700"
        )
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise RuntimeError("forensic receipt directory changed during secure open")
        names = sorted(os.listdir(descriptor))
        if not names or any(re.fullmatch(r"[0-9a-f]{64}\.json", name) is None for name in names):
            raise ValueError("forensic receipt directory has a missing or unexpected file")
        snapshot_parts: list[bytes] = []
        file_rows: list[dict[str, object]] = []
        for name in names:
            content, digest, byte_count = _read_artifact_member(
                descriptor,
                name,
                capture=True,
            )
            assert content is not None
            snapshot_parts.append(content + b"\n")
            file_rows.append({"name": name, "bytes": byte_count, "sha256": digest})
        snapshot_bytes = b"".join(snapshot_parts)
        local_receipts = read_private_bytes(pending_path.parent / "execution-receipts.jsonl")
        rows = _strict_json_lines(local_receipts, label="execution receipts")
        marker_keys = {
            "attempt_id",
            "evidence",
            "forensic_path",
            "raw_snapshot_bytes",
            "raw_snapshot_sha256",
            "scope",
            "status",
        }
        if len(rows) == 1 and set(rows[0]) == marker_keys:
            marker = rows[0]
            expected_bytes = cast(int, marker["raw_snapshot_bytes"])
            expected_sha256 = cast(str, marker["raw_snapshot_sha256"])
            evidence = "external_forensic_directory"
        else:
            expected_bytes = len(local_receipts)
            expected_sha256 = hashlib.sha256(local_receipts).hexdigest()
            evidence = "validated_canonical_snapshot"
        actual_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
        if len(snapshot_bytes) != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError("forensic snapshot bytes or SHA-256 differ from the artifact marker")
        after_path = root.lstat()
        after_open = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(after_path, field) != getattr(before_path, field)
            or getattr(after_open, field) != getattr(opened, field)
            for field in stable
        ):
            raise RuntimeError("forensic receipt directory changed during validation")
        return {
            "schema_version": 1,
            "kind": "ratemem-forensic-receipt-manifest-v1",
            "attempt_id": attempt_id,
            "evidence": evidence,
            "forensic_path": f"execution-receipts/{attempt_id}",
            "files": file_rows,
            "raw_snapshot_bytes": len(snapshot_bytes),
            "raw_snapshot_sha256": actual_sha256,
        }
    finally:
        os.close(descriptor)


def _validate_artifact(path: Path, *, allow_final: bool = False) -> dict[str, Any]:
    if type(path) is not type(Path()) or path.name != "attempt.pending.json":
        raise ValueError("artifact validation requires the canonical attempt.pending.json path")
    _require_repository_cwd()
    root = Path(os.path.abspath(path.parent))
    _assert_no_symlink_ancestors(root)
    before_path = root.lstat()
    if (
        not stat.S_ISDIR(before_path.st_mode)
        or before_path.st_uid != os.getuid()
        or stat.S_IMODE(before_path.st_mode) != 0o700
    ):
        raise PermissionError("artifact attempt root must be owner-only mode 0700")
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise RuntimeError("artifact attempt root changed during secure open")
        pending_bytes, _pending_sha, _pending_size = _read_artifact_member(
            descriptor, "attempt.pending.json", capture=True
        )
        assert pending_bytes is not None
        payload = _strict_json_bytes(pending_bytes, label="attempt.pending.json")
        validate_attempt(payload)
        if payload["cost"]["reconciliation_status"] != "pending":
            raise ValueError("attempt.pending.json must retain pending reconciliation")
        attempt_id = cast(str, payload["attempt_id"])
        if root.name != attempt_id:
            raise ValueError("artifact directory name must equal its canonical attempt UUID")
        checkpoint = payload["checkpoint"]
        expected_payload = set(_PAYLOAD_FILES)
        if checkpoint is not None:
            expected_payload.add("trainable.safetensors")
        expected_names = expected_payload | {"attempt.pending.json", "checksums.sha256"}
        if allow_final and "attempt.json" in os.listdir(descriptor):
            expected_names.add("attempt.json")
        actual_names = set(os.listdir(descriptor))
        if actual_names != expected_names:
            raise ValueError("artifact directory has a missing or unexpected file")

        contents: dict[str, bytes] = {"attempt.pending.json": pending_bytes}
        identities: dict[str, tuple[str, int]] = {}
        for name in sorted(expected_names - {"attempt.pending.json"}):
            captured, digest, byte_count = _read_artifact_member(
                descriptor,
                name,
                capture=name != "trainable.safetensors",
            )
            identities[name] = (digest, byte_count)
            if captured is not None:
                contents[name] = captured

        checksums = contents["checksums.sha256"]
        if hashlib.sha256(checksums).hexdigest() != payload["files"]["checksums_sha256"]:
            raise ValueError("checksums.sha256 digest differs from the pending artifact")
        try:
            checksum_lines = checksums.decode("ascii").splitlines(keepends=True)
        except UnicodeDecodeError as error:
            raise ValueError("checksums.sha256 must be ASCII") from error
        if not checksum_lines or any(not line.endswith("\n") for line in checksum_lines):
            raise ValueError("checksums.sha256 must have canonical newline termination")
        observed_checksums: dict[str, str] = {}
        for line in checksum_lines:
            match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)\n", line)
            if match is None or match.group(2) in observed_checksums:
                raise ValueError("checksums.sha256 contains a malformed or duplicate entry")
            observed_checksums[match.group(2)] = match.group(1)
        if (
            list(observed_checksums) != sorted(expected_payload)
            or set(observed_checksums) != expected_payload
        ):
            raise ValueError("checksums.sha256 does not name the exact canonical payload files")
        for name, claimed in observed_checksums.items():
            if identities[name][0] != claimed:
                raise ValueError(f"artifact checksum mismatch: {name}")

        if checkpoint is not None and (
            cast(dict[str, object], checkpoint)["path"] != "trainable.safetensors"
            or identities["trainable.safetensors"]
            != (
                cast(str, cast(dict[str, object], checkpoint)["sha256"]),
                cast(int, cast(dict[str, object], checkpoint)["bytes"]),
            )
        ):
            raise ValueError("trainable checkpoint identity differs from the artifact")
        config = _strict_json_bytes(contents["config.json"], label="config.json")
        rates = _strict_json_bytes(contents["rates.json"], label="rates.json")
        dataset = _strict_json_bytes(
            contents["dataset-manifest.json"], label="dataset-manifest.json"
        )
        if (
            hashlib.sha256(contents["config.json"]).hexdigest()
            != payload["source"]["config_sha256"]
        ):
            raise ValueError("artifact config bundle hash differs from source identity")
        if (
            hashlib.sha256(contents["dataset-manifest.json"]).hexdigest()
            != payload["dataset"]["manifest_sha256"]
        ):
            raise ValueError("dataset manifest hash differs from the artifact")
        del config, dataset

        source = source_tree_identity()
        permit = read_private_json(_operating_path(PERMIT_PATH))
        launch = validate_consumed_launch_evidence(
            _operating_path(PERMIT_PATH),
            slot=GLOBAL_SLOT_PATH,
            receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
            expected_workspace=cast(str, permit["workspace"]),
            current_source_sha256=source.source_sha256,
        )
        expected_identity = {
            "attempt_id": attempt_id,
            "workspace": payload["modal"]["workspace"],
            "source_sha256": payload["modal"]["launch_source_sha256"],
            "git_commit": payload["source"]["git_commit"],
            "slot_sha256": payload["modal"]["pilot_slot_sha256"],
            "submission_receipt_sha256": payload["modal"]["submission_receipt_sha256"],
        }
        if any(launch[key] != value for key, value in expected_identity.items()):
            raise ValueError("artifact launch identity differs from local consumed evidence")
        permit_sha = cast(str, launch["permit_sha256"])
        if (
            permit["git_diff_sha256"] != payload["source"]["git_diff_sha256"]
            or permit["config_sha256"] != payload["source"]["config_sha256"]
            or permit["known_usage_before_usd"] != payload["cost"]["known_usage_before_usd"]
            or permit["pending_worst_case_usd"] != payload["cost"]["pending_worst_case_usd"]
            or permit["phase_bound_usd"] != payload["cost"]["phase_bound_usd"]
            or permit["rates_sha256"] != payload["cost"]["rates_sha256"]
            or rates != permit["rates"]
            or hashlib.sha256(canonical_json_bytes(rates)).hexdigest() != permit["rates_sha256"]
            or pilot_config_sha256() != permit["config_sha256"]
        ):
            raise ValueError(
                "artifact cost, rate, diff, or config evidence differs from its permit"
            )
        expected_request = {
            key: (
                launch[key]
                if key in {"slot_sha256", "permit_sha256", "submission_receipt_sha256"}
                else permit[key]
            )
            for key in _REQUEST_KEYS
        }
        receipt_evidence = _validate_execution_receipts(
            contents["execution-receipts.jsonl"],
            expected_request=expected_request,
            attempt=payload,
        )
        _validate_metrics(
            contents["metrics.jsonl"],
            permit_sha256=permit_sha,
            attempt=payload,
            receipt_evidence=receipt_evidence,
        )
        findings = credential_findings([root / name for name in sorted(expected_names)])
        if findings:
            raise ValueError(f"credential material found in {len(findings)} artifact files")
        after_path = root.lstat()
        after_open = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(after_path, field) != getattr(before_path, field)
            or getattr(after_open, field) != getattr(opened, field)
            for field in stable
        ):
            raise RuntimeError("artifact directory changed during validation")
        return payload
    finally:
        os.close(descriptor)


def _reconciled_payload(pending: dict[str, Any], reconciled_cost: Decimal) -> dict[str, Any]:
    final = copy.deepcopy(pending)
    final["cost"]["reconciliation_status"] = "reconciled"
    final["cost"]["reconciled_cost_usd"] = _fixed_usd_text(reconciled_cost, "reconciled cost")
    validate_attempt(final)
    return final


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _require_calendar_settlement_window(now: datetime) -> None:
    if type(now) is not datetime or now.tzinfo != UTC:
        raise ValueError("calendar settlement guard requires an exact UTC datetime")
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1, tzinfo=UTC)
    else:
        next_month = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
    if next_month - now < timedelta(days=10):
        _pending_settlement("fewer than ten full days remain before the UTC billing month resets")


def _pending_settlement(message: str) -> Never:
    typer.echo(f"PENDING: {message}; another launch is forbidden", err=True)
    raise typer.Exit(code=3)


def _settlement_candidate(
    *,
    attempt_id: str,
    workspace: str,
    known_usage_before: Decimal,
    observed_usage: Decimal,
    volume_absence_sha256: str,
    allow_equal: bool = False,
) -> None:
    """Require equal metered usage for four days after volume absence."""

    if type(attempt_id) is not str or _UUID7.fullmatch(attempt_id) is None:
        raise ValueError("settlement attempt must be one canonical UUIDv7")
    if type(workspace) is not str or _WORKSPACE.fullmatch(workspace) is None:
        raise ValueError("settlement workspace must be one canonical slug")
    if type(volume_absence_sha256) is not str or _HEX64.fullmatch(volume_absence_sha256) is None:
        raise ValueError("settlement volume absence hash must be lowercase SHA-256")
    if type(allow_equal) is not bool:
        raise TypeError("settlement equality policy must be an exact bool")
    if (
        type(known_usage_before) is not Decimal
        or type(observed_usage) is not Decimal
        or not known_usage_before.is_finite()
        or not observed_usage.is_finite()
        or known_usage_before < 0
        or observed_usage < known_usage_before
        or (observed_usage == known_usage_before and not allow_equal)
    ):
        raise ValueError("settlement usage values violate the exact monotonicity policy")
    directory = _operating_path(SETTLEMENT_DIRECTORY)
    ensure_private_directory(directory)
    candidate_path = directory / f"{attempt_id}.json"
    lock_path = directory / f"{attempt_id}.lock"
    now = _now_utc()
    now_text = now.isoformat(timespec="microseconds")
    body = {
        "schema_version": 1,
        "kind": "ratemem-billing-settlement-candidate",
        "attempt_id": attempt_id,
        "workspace": workspace,
        "known_usage_before_usd": str(known_usage_before),
        "observed_usage_usd": str(observed_usage),
        "first_observed_at_utc": now_text,
        "volume_absence_sha256": volume_absence_sha256,
    }
    with private_lock(lock_path):
        try:
            candidate_path.lstat()
        except FileNotFoundError:
            write_exclusive_private_json(candidate_path, body)
            _pending_settlement("first post-launch billing observation recorded")
        candidate = read_private_json(candidate_path)
        if read_private_bytes(candidate_path) != canonical_json_bytes(candidate):
            raise ValueError("billing settlement candidate must be canonical JSON")
        if set(candidate) != set(body) or any(
            candidate[key] != body[key]
            for key in (
                "schema_version",
                "kind",
                "attempt_id",
                "workspace",
                "known_usage_before_usd",
                "volume_absence_sha256",
            )
        ):
            raise ValueError("billing settlement candidate identity is invalid")
        if type(candidate["observed_usage_usd"]) is not str:
            raise TypeError("billing settlement observed usage must be an exact string")
        try:
            candidate_usage = Decimal(candidate["observed_usage_usd"])
        except InvalidOperation as error:
            raise ValueError("billing settlement observed usage is invalid") from error
        if not candidate_usage.is_finite() or candidate_usage < known_usage_before:
            raise ValueError("billing settlement observed usage is invalid")
        first_observed = _canonical_utc(
            candidate["first_observed_at_utc"], "settlement first observation"
        )
        if candidate_usage != observed_usage:
            write_atomic_private_json(candidate_path, body)
            _pending_settlement("metered usage changed; settlement stability window restarted")
        age = (now - first_observed).total_seconds()
        if age < 0:
            raise ValueError("billing settlement observation is future-dated")
        if age < _SETTLEMENT_MIN_AGE_SECONDS:
            _pending_settlement(
                "metered usage has not remained stable for four days after volume deletion"
            )


def _consumed_launch_evidence(workspace: str) -> dict[str, Any]:
    return validate_consumed_launch_evidence(
        _operating_path(PERMIT_PATH),
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        expected_workspace=workspace,
        current_source_sha256=source_tree_sha256(),
    )


def _absence_path(attempt_id: str) -> Path:
    if type(attempt_id) is not str or _UUID7.fullmatch(attempt_id) is None:
        raise ValueError("volume absence attempt must be one canonical UUIDv7")
    return _operating_path(SETTLEMENT_DIRECTORY) / f"{attempt_id}.volume-absence.json"


def _initial_settlement_body(
    *,
    attempt_id: str,
    workspace: str,
    known_usage_before: Decimal,
    observed_usage: Decimal,
    observed_at: datetime,
    volume_absence_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "ratemem-billing-settlement-candidate",
        "attempt_id": attempt_id,
        "workspace": workspace,
        "known_usage_before_usd": str(known_usage_before),
        "observed_usage_usd": str(observed_usage),
        "first_observed_at_utc": observed_at.isoformat(timespec="microseconds"),
        "volume_absence_sha256": volume_absence_sha256,
    }


def _ensure_initial_settlement_candidate(
    *,
    attempt_id: str,
    workspace: str,
    known_usage_before: Decimal,
    observed_usage: Decimal,
    observed_at: datetime,
    volume_absence_sha256: str,
    allow_absence_reset: bool = False,
) -> None:
    if type(allow_absence_reset) is not bool:
        raise TypeError("settlement candidate reset policy must be an exact bool")
    directory = _operating_path(SETTLEMENT_DIRECTORY)
    ensure_private_directory(directory)
    path = directory / f"{attempt_id}.json"
    body = _initial_settlement_body(
        attempt_id=attempt_id,
        workspace=workspace,
        known_usage_before=known_usage_before,
        observed_usage=observed_usage,
        observed_at=observed_at,
        volume_absence_sha256=volume_absence_sha256,
    )
    try:
        write_exclusive_private_json(path, body)
    except FileExistsError:
        payload = read_private_json(path)
        if read_private_bytes(path) != canonical_json_bytes(payload):
            raise ValueError("billing settlement candidate must be canonical JSON") from None
        required_identity = {
            "schema_version": 1,
            "kind": "ratemem-billing-settlement-candidate",
            "attempt_id": attempt_id,
            "workspace": workspace,
            "known_usage_before_usd": str(known_usage_before),
            "volume_absence_sha256": volume_absence_sha256,
        }
        common_keys = set(required_identity) - {"volume_absence_sha256"}
        if set(payload) != set(body) or any(
            payload[key] != required_identity[key] for key in common_keys
        ):
            raise ValueError("existing billing settlement candidate identity is invalid") from None
        if payload["volume_absence_sha256"] != volume_absence_sha256:
            if not allow_absence_reset:
                raise ValueError(
                    "existing billing settlement candidate absence identity is invalid"
                ) from None
            write_atomic_private_json(path, body)


def _volume_absence_identity(
    *,
    pending: dict[str, Any],
    workspace: str,
    launch: dict[str, Any],
) -> dict[str, object]:
    modal = cast(dict[str, object], pending["modal"])
    expected = {
        "schema_version": 1,
        "kind": "ratemem-pilot-volume-absence-v1",
        "attempt_id": pending["attempt_id"],
        "workspace": workspace,
        "profile": _PROFILE,
        "environment": _ENVIRONMENT,
        "volume_names": sorted(_REQUIRED_VOLUMES),
        "permit_sha256": launch["permit_sha256"],
        "slot_sha256": launch["slot_sha256"],
        "submission_receipt_sha256": launch["submission_receipt_sha256"],
    }
    if (
        launch["attempt_id"] != pending["attempt_id"]
        or launch["workspace"] != workspace
        or modal.get("workspace") != workspace
        or modal.get("pilot_slot_sha256") != launch["slot_sha256"]
        or modal.get("submission_receipt_sha256") != launch["submission_receipt_sha256"]
    ):
        raise ValueError("volume absence launch identity differs from the artifact")
    return expected


def _read_volume_absence(
    *,
    pending: dict[str, Any],
    workspace: str,
    launch: dict[str, Any],
) -> VolumeAbsenceEvidence:
    path = _absence_path(cast(str, pending["attempt_id"]))
    try:
        payload = read_private_json(path)
    except FileNotFoundError:
        _pending_settlement("volume absence has not been attested")
    if read_private_bytes(path) != canonical_json_bytes(payload):
        raise ValueError("volume absence evidence must be canonical JSON")
    identity = _volume_absence_identity(
        pending=pending,
        workspace=workspace,
        launch=launch,
    )
    expected_keys = set(identity) | {
        "known_metered_usage_usd",
        "confirmed_absent_at_utc",
    }
    if set(payload) != expected_keys or any(
        payload[key] != value for key, value in identity.items()
    ):
        raise ValueError("volume absence evidence identity is invalid")
    known_text = payload["known_metered_usage_usd"]
    if type(known_text) is not str or _FIXED_USD.fullmatch(known_text) is None:
        raise ValueError("volume absence usage must be canonical fixed-point USD")
    known = Decimal(known_text)
    if not known.is_finite() or known < 0:
        raise ValueError("volume absence usage must be finite and nonnegative")
    confirmed_at = _canonical_utc(payload["confirmed_absent_at_utc"], "volume absence confirmation")
    if confirmed_at > _now_utc():
        raise ValueError("volume absence confirmation is future-dated")
    evidence = VolumeAbsenceEvidence(
        confirmed_at=confirmed_at,
        known_usage=known,
        sha256=file_sha256(path),
    )
    before = Decimal(cast(str, pending["cost"]["known_usage_before_usd"]))
    _ensure_initial_settlement_candidate(
        attempt_id=cast(str, pending["attempt_id"]),
        workspace=workspace,
        known_usage_before=before,
        observed_usage=evidence.known_usage,
        observed_at=evidence.confirmed_at,
        volume_absence_sha256=evidence.sha256,
    )
    return evidence


def _reject_volume_reappearance(attempt_id: str) -> None:
    directory = _operating_path(SETTLEMENT_DIRECTORY)
    try:
        paths = sorted(directory.glob(f"{attempt_id}.volume-reappearance.*.json"))
    except FileNotFoundError:
        return
    if paths:
        for path in paths:
            read_private_json(path)
        raise RuntimeError("durable volume reappearance permanently invalidates normal settlement")


def _record_volume_reappearance(
    *,
    attempt_id: str,
    workspace: str,
    launch: dict[str, Any],
    volume_names: set[str],
    preceding_absence_sha256: str,
) -> None:
    names = sorted(volume_names)
    if (
        type(preceding_absence_sha256) is not str
        or _HEX64.fullmatch(preceding_absence_sha256) is None
    ):
        raise ValueError("volume reappearance must bind a preceding absence SHA-256")
    fingerprint = hashlib.sha256(
        canonical_json_bytes(
            {
                "preceding_absence_sha256": preceding_absence_sha256,
                "volume_names": names,
            }
        )
    ).hexdigest()
    directory = _operating_path(SETTLEMENT_DIRECTORY)
    ensure_private_directory(directory)
    path = directory / f"{attempt_id}.volume-reappearance.{fingerprint}.json"
    identity: dict[str, object] = {
        "schema_version": 1,
        "kind": "ratemem-pilot-volume-reappearance-v1",
        "attempt_id": attempt_id,
        "workspace": workspace,
        "profile": _PROFILE,
        "environment": _ENVIRONMENT,
        "permit_sha256": launch["permit_sha256"],
        "slot_sha256": launch["slot_sha256"],
        "submission_receipt_sha256": launch["submission_receipt_sha256"],
        "preceding_absence_sha256": preceding_absence_sha256,
        "volume_names": names,
    }
    body = identity | {"observed_at_utc": _now_utc().isoformat(timespec="microseconds")}
    try:
        write_exclusive_private_json(path, body)
    except FileExistsError:
        existing = read_private_json(path)
        if set(existing) != set(body) or any(
            existing[key] != value for key, value in identity.items()
        ):
            raise ValueError("volume reappearance record identity is invalid") from None
        _canonical_utc(existing["observed_at_utc"], "volume reappearance observation")


def _verified_volume_absence(
    *,
    pending: dict[str, Any],
    workspace: str,
) -> VolumeAbsenceEvidence:
    attempt_id = cast(str, pending["attempt_id"])
    present = _volume_names(_modal_cli_json(["volume", "list", "--env", "main"]))
    if present:
        try:
            _absence_path(attempt_id).lstat()
        except FileNotFoundError:
            _pending_settlement("Modal storage volumes are still present in the pilot workspace")
        launch = _consumed_launch_evidence(workspace)
        _record_volume_reappearance(
            attempt_id=attempt_id,
            workspace=workspace,
            launch=launch,
            volume_names=present,
            preceding_absence_sha256=file_sha256(_absence_path(attempt_id)),
        )
        raise RuntimeError("Modal volume reappeared; normal settlement is permanently invalid")
    _reject_volume_reappearance(attempt_id)
    launch = _consumed_launch_evidence(workspace)
    return _read_volume_absence(pending=pending, workspace=workspace, launch=launch)


def _record_hard_budget_observation(
    *,
    attempt_id: str,
    workspace: str,
    known_usage_before: Decimal,
    known_usage: Decimal,
) -> Path:
    if type(attempt_id) is not str or _UUID7.fullmatch(attempt_id) is None:
        raise ValueError("hard-budget observation attempt must be a canonical UUIDv7")
    if type(workspace) is not str or _WORKSPACE.fullmatch(workspace) is None:
        raise ValueError("hard-budget observation workspace must be a canonical slug")
    before_text = _fixed_usd_text(known_usage_before, "hard-budget pre-launch usage")
    usage_text = _fixed_usd_text(known_usage, "hard-budget observed usage")
    if known_usage <= Decimal("28.00") or known_usage < known_usage_before:
        raise ValueError("hard-budget observation must exceed USD 28.00 monotonically")
    fingerprint = hashlib.sha256(f"{attempt_id}\0{usage_text}".encode("ascii")).hexdigest()
    directory = _operating_path(SETTLEMENT_DIRECTORY)
    ensure_private_directory(directory)
    path = directory / f"{attempt_id}.hard-budget.{fingerprint}.json"
    identity: dict[str, object] = {
        "schema_version": 1,
        "kind": "ratemem-hard-budget-observation-v1",
        "attempt_id": attempt_id,
        "workspace": workspace,
        "profile": _PROFILE,
        "environment": _ENVIRONMENT,
        "known_usage_before_usd": before_text,
        "known_metered_usage_usd": usage_text,
        "hard_budget_usd": "28.00",
        "status": "hard_budget_violation",
    }
    body = identity | {"observed_at_utc": _now_utc().isoformat(timespec="microseconds")}
    try:
        write_exclusive_private_json(path, body)
    except FileExistsError:
        existing = read_private_json(path)
        if (
            read_private_bytes(path) != canonical_json_bytes(existing)
            or set(existing) != set(body)
            or any(existing[key] != value for key, value in identity.items())
        ):
            raise ValueError("existing hard-budget observation identity is invalid") from None
        _canonical_utc(existing["observed_at_utc"], "hard-budget observation")
    return path


def _validate_incident(path: Path, *, allow_final: bool = False) -> dict[str, Any]:
    expected_name = "incident.json" if allow_final else "incident.pending.json"
    if type(path) is not type(Path()) or path.name != expected_name:
        raise ValueError(f"incident path must end in {expected_name}")
    payload = _strict_json_bytes(read_private_bytes(path), label=expected_name)
    expected_top = {
        "schema_version",
        "kind",
        "scope",
        "publication_eligible",
        "attempt_id",
        "workspace",
        "profile",
        "environment",
        "status",
        "reason",
        "created_at_utc",
        "launch",
        "cost",
        "volumes_observed",
    }
    if (
        set(payload) != expected_top
        or payload["schema_version"] != 1
        or payload["kind"] != "ratemem-pilot-incident-v1"
        or payload["scope"] != "engineering_pilot_only"
        or payload["publication_eligible"] is not False
        or payload["profile"] != _PROFILE
        or payload["environment"] != _ENVIRONMENT
        or (payload["status"], payload["reason"])
        not in {
            ("artifact_unavailable", "artifact_unavailable_after_launch_command"),
            (
                "attempt_invalidated",
                "volume_reappearance_permanently_invalidated_normal_attempt",
            ),
        }
    ):
        raise ValueError("incident evidence schema or scope is invalid")
    attempt_id = payload["attempt_id"]
    workspace = payload["workspace"]
    if type(attempt_id) is not str or _UUID7.fullmatch(attempt_id) is None:
        raise ValueError("incident attempt must be a canonical UUIDv7")
    if path.parent.name != attempt_id:
        raise ValueError("incident directory must equal its attempt UUID")
    if type(workspace) is not str or _WORKSPACE.fullmatch(workspace) is None:
        raise ValueError("incident workspace must be a canonical slug")
    _canonical_utc(payload["created_at_utc"], "incident created_at")
    launch = payload["launch"]
    if type(launch) is not dict or set(launch) != {
        "state",
        "permit_sha256",
        "slot_sha256",
        "submission_receipt_sha256",
    }:
        raise ValueError("incident launch evidence schema is invalid")
    launch = cast(dict[str, object], launch)
    if launch["state"] not in {"unsubmitted", "consumed"}:
        raise ValueError("incident launch state is invalid")
    for key in ("permit_sha256", "slot_sha256"):
        if type(launch[key]) is not str or _HEX64.fullmatch(cast(str, launch[key])) is None:
            raise ValueError("incident launch hashes must be lowercase SHA-256")
    submission = launch["submission_receipt_sha256"]
    if (launch["state"] == "unsubmitted" and submission is not None) or (
        launch["state"] == "consumed"
        and (type(submission) is not str or _HEX64.fullmatch(submission) is None)
    ):
        raise ValueError("incident submission receipt state is inconsistent")
    current_launch, permit = _launch_state(workspace)
    if current_launch != launch or permit["attempt_id"] != attempt_id:
        raise ValueError("incident launch evidence differs from local one-shot state")
    cost = payload["cost"]
    if type(cost) is not dict or set(cost) != {
        "known_usage_before_usd",
        "known_usage_at_incident_usd",
        "phase_bound_usd",
        "rates_sha256",
        "reconciliation_status",
        "reconciled_cost_usd",
        "hard_budget_violation",
    }:
        raise ValueError("incident cost evidence schema is invalid")
    cost = cast(dict[str, object], cost)
    before = Decimal(cast(str, cost["known_usage_before_usd"]))
    incident_usage = Decimal(cast(str, cost["known_usage_at_incident_usd"]))
    phase = Decimal(cast(str, cost["phase_bound_usd"]))
    for value, label in (
        (before, "incident known usage before"),
        (incident_usage, "incident observed usage"),
        (phase, "incident phase bound"),
    ):
        _fixed_usd_text(value, label)
    if incident_usage < before or phase <= 0:
        raise ValueError("incident cost evidence is nonmonotonic")
    if type(cost["rates_sha256"]) is not str or _HEX64.fullmatch(cost["rates_sha256"]) is None:
        raise ValueError("incident rate hash is invalid")
    expected_status = "reconciled" if allow_final else "pending"
    if cost["reconciliation_status"] != expected_status:
        raise ValueError("incident reconciliation state is invalid")
    if allow_final:
        reconciled = Decimal(cast(str, cost["reconciled_cost_usd"]))
        _fixed_usd_text(reconciled, "incident reconciled cost")
        if type(cost["hard_budget_violation"]) is not bool:
            raise TypeError("incident hard-budget flag must be an exact bool")
    elif cost["reconciled_cost_usd"] is not None or cost["hard_budget_violation"] is not None:
        raise ValueError("pending incident must not claim reconciliation")
    volumes = payload["volumes_observed"]
    if (
        type(volumes) is not list
        or any(type(name) is not str or not name for name in volumes)
        or volumes != sorted(set(volumes))
    ):
        raise ValueError("incident volume observation must be uniquely sorted")
    if (
        permit["known_usage_before_usd"] != cost["known_usage_before_usd"]
        or permit["phase_bound_usd"] != cost["phase_bound_usd"]
        or permit["rates_sha256"] != cost["rates_sha256"]
    ):
        raise ValueError("incident cost evidence differs from the launch permit")
    return payload


def _latest_reappearance_sha256(attempt_id: str) -> str | None:
    directory = _operating_path(SETTLEMENT_DIRECTORY)
    paths = sorted(directory.glob(f"{attempt_id}.volume-reappearance.*.json"))
    observed: list[tuple[datetime, str, Path]] = []
    for path in paths:
        payload = read_private_json(path)
        if (
            read_private_bytes(path) != canonical_json_bytes(payload)
            or payload.get("kind") != "ratemem-pilot-volume-reappearance-v1"
            or payload.get("attempt_id") != attempt_id
        ):
            raise ValueError("volume reappearance record is not canonical or attempt-bound")
        observed.append(
            (
                _canonical_utc(payload.get("observed_at_utc"), "volume reappearance observation"),
                path.name,
                path,
            )
        )
    if not observed:
        return None
    return file_sha256(max(observed)[2])


def _incident_absence_path(attempt_id: str, reappearance_sha256: str | None) -> Path:
    if type(attempt_id) is not str or _UUID7.fullmatch(attempt_id) is None:
        raise ValueError("incident absence attempt must be one canonical UUIDv7")
    if reappearance_sha256 is not None and (
        type(reappearance_sha256) is not str or _HEX64.fullmatch(reappearance_sha256) is None
    ):
        raise ValueError("incident absence reappearance hash must be lowercase SHA-256")
    epoch = "initial" if reappearance_sha256 is None else reappearance_sha256
    return (
        _operating_path(SETTLEMENT_DIRECTORY) / f"{attempt_id}.incident-volume-absence.{epoch}.json"
    )


def _incident_absence_identity(
    incident: dict[str, Any],
    *,
    reappearance_sha256: str | None,
) -> dict[str, object]:
    launch = cast(dict[str, object], incident["launch"])
    return {
        "schema_version": 1,
        "kind": "ratemem-pilot-volume-absence-v1",
        "attempt_id": incident["attempt_id"],
        "workspace": incident["workspace"],
        "profile": _PROFILE,
        "environment": _ENVIRONMENT,
        "volume_names": sorted(_REQUIRED_VOLUMES),
        "permit_sha256": launch["permit_sha256"],
        "slot_sha256": launch["slot_sha256"],
        "submission_receipt_sha256": launch["submission_receipt_sha256"],
        "reappearance_sha256": reappearance_sha256,
    }


def _read_incident_absence(incident: dict[str, Any]) -> VolumeAbsenceEvidence:
    attempt_id = cast(str, incident["attempt_id"])
    reappearance_sha256 = _latest_reappearance_sha256(attempt_id)
    path = _incident_absence_path(attempt_id, reappearance_sha256)
    try:
        path.lstat()
    except FileNotFoundError:
        _pending_settlement("incident volume absence has not been attested")
    payload = read_private_json(path)
    if read_private_bytes(path) != canonical_json_bytes(payload):
        raise ValueError("incident volume absence evidence must be canonical JSON")
    identity = _incident_absence_identity(
        incident,
        reappearance_sha256=reappearance_sha256,
    )
    if set(payload) != set(identity) | {
        "known_metered_usage_usd",
        "confirmed_absent_at_utc",
    } or any(payload[key] != value for key, value in identity.items()):
        raise ValueError("incident volume absence identity is invalid")
    usage_text = payload["known_metered_usage_usd"]
    if type(usage_text) is not str or _FIXED_USD.fullmatch(usage_text) is None:
        raise ValueError("incident volume absence usage is invalid")
    usage = Decimal(usage_text)
    confirmed = _canonical_utc(
        payload["confirmed_absent_at_utc"], "incident volume absence confirmation"
    )
    if confirmed > _now_utc():
        raise ValueError("incident volume absence is future-dated")
    evidence = VolumeAbsenceEvidence(confirmed, usage, file_sha256(path))
    before = Decimal(cast(str, incident["cost"]["known_usage_before_usd"]))
    _ensure_initial_settlement_candidate(
        attempt_id=attempt_id,
        workspace=cast(str, incident["workspace"]),
        known_usage_before=before,
        observed_usage=usage,
        observed_at=confirmed,
        volume_absence_sha256=evidence.sha256,
        allow_absence_reset=True,
    )
    return evidence


def _verified_incident_absence(incident: dict[str, Any]) -> VolumeAbsenceEvidence:
    attempt_id = cast(str, incident["attempt_id"])
    present = _volume_names(_modal_cli_json(["volume", "list", "--env", "main"]))
    if present:
        absence_paths = sorted(
            _operating_path(SETTLEMENT_DIRECTORY).glob(
                f"{attempt_id}.incident-volume-absence.*.json"
            )
        )
        if not absence_paths:
            _pending_settlement("Modal volumes are still present during incident cleanup")
        preceding = max(
            (
                _canonical_utc(
                    read_private_json(path)["confirmed_absent_at_utc"],
                    "incident volume absence confirmation",
                ),
                path.name,
                path,
            )
            for path in absence_paths
        )[2]
        _record_volume_reappearance(
            attempt_id=attempt_id,
            workspace=cast(str, incident["workspace"]),
            launch=cast(dict[str, Any], incident["launch"]),
            volume_names=present,
            preceding_absence_sha256=file_sha256(preceding),
        )
        raise RuntimeError("Modal volume reappeared during incident settlement")
    return _read_incident_absence(incident)


@app.command("attest-workspace")
def attest_workspace(
    evidence: Annotated[Path, typer.Option("--evidence", help="Private dashboard screenshot.")],
    output: Annotated[Path, typer.Option("--output")] = ATTESTATION_PATH,
    operator_evidence: Annotated[Path | None, typer.Option("--operator-evidence")] = None,
) -> None:
    _require_repository_cwd()
    workspace = typer.prompt("Type the exact authorized Modal workspace slug")
    budget = typer.prompt("Type the exact dashboard Workspace usage budget")
    spend_limit = typer.prompt("Type the exact dashboard Workspace spend limit")
    statement = typer.prompt(f"Type exactly: {BUDGET_CONFIRMATION_STATEMENT}")
    if budget != "28.00":
        raise typer.BadParameter("workspace usage budget must be exactly 28.00")
    if spend_limit != "0.00":
        raise typer.BadParameter("workspace spend limit must be exactly 0.00")
    selected_evidence = (
        operator_evidence
        if operator_evidence is not None
        else evidence.with_name(
            f"{evidence.stem}.{file_sha256(evidence)[:16]}.operator-attestation.json"
        )
    )
    create_operator_budget_evidence(
        screenshot=evidence,
        output=selected_evidence,
        workspace=workspace,
        confirmation_statement=statement,
    )
    snapshot = capture_workspace_snapshot(
        evidence_path=selected_evidence,
        confirmed_budget=budget,
        config_path=MODAL_CONFIG_PATH,
    )
    verify_workspace_snapshot(snapshot, expected_workspace=workspace, max_age_seconds=900)
    write_atomic_private_json(_operating_path(output), snapshot.to_json())
    typer.echo(f"PASS workspace={workspace} usage_budget_usd=28.00 spend_limit_usd=0.00")


@app.command("preflight")
def preflight(attestation: Path = ATTESTATION_PATH) -> None:
    _require_repository_cwd()
    _require_calendar_settlement_window(_now_utc())
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    budget = ModalBudgetConfig.load(_operating_path(BUDGET_CONFIG_PATH))
    source = source_tree_identity()
    config_sha = pilot_config_sha256()
    rates = CostRates.normalize(snapshot.rates)
    resources = ResourceContract(
        gpu_count=budget.gpu_count,
        cpu_cores=budget.cpu_cores,
        memory_gib=budget.memory_gib,
        timeout_seconds=budget.timeout_seconds,
        startup_timeout_seconds=budget.startup_timeout_seconds,
        storage_gib_bound=budget.storage_gib_bound,
        non_gpu_setup_allowance_usd=budget.non_gpu_setup_allowance_usd,
    )
    bound = conservative_bound(rates, resources)
    if bound > budget.first_pilot_allocation_usd or bound > Decimal("21.00"):
        raise ValueError("current rates exceed the USD 21.00 first-pilot allocation")
    known = Decimal(snapshot.known_metered_usage_usd)
    known_text = _fixed_usd_text(known, "known metered usage")
    bound_text = _fixed_usd_text(bound, "phase bound")
    if known + bound > budget.internal_limit_usd:
        raise ValueError("known usage plus pilot worst case exceeds USD 27.00")
    if _volume_names(_modal_cli_json(["volume", "list", "--env", "main"])):
        raise ValueError("dedicated pilot workspace must have no pre-existing Modal volumes")
    attempt_id = new_attempt_id()
    identity = PilotIdentity(
        attempt_id=attempt_id,
        workspace=snapshot.workspace,
        source_sha256=source.source_sha256,
        git_commit=source.git_commit,
    )
    rates_sha = hashlib.sha256(canonical_json_bytes(snapshot.rates)).hexdigest()
    if _HEX64.fullmatch(config_sha) is None or source.git_diff_sha256 != _EMPTY_SHA256:
        raise RuntimeError("pilot source or config identity is not canonical")

    ensure_private_directory(GLOBAL_SLOT_PATH.parent)
    ensure_private_directory(_operating_path(PERMIT_PATH).parent)
    ledger = CostLedger(_operating_path(LEDGER_PATH), internal_limit_usd=budget.internal_limit_usd)
    ledger.verify_hash_chain()
    ledger.require_pristine()
    existing_pending = ledger.preview_reservation(
        attempt_id,
        known_usage=known,
        phase_bound=bound,
        rates_sha256=rates_sha,
    )
    if existing_pending != 0:
        raise ValueError("the only pilot requires zero prior open ledger reservations")
    for irreversible in (
        GLOBAL_SLOT_PATH,
        GLOBAL_SUBMISSION_RECEIPT_PATH,
        _operating_path(PERMIT_PATH),
    ):
        try:
            irreversible.lstat()
        except FileNotFoundError:
            continue
        raise FileExistsError(f"one-shot pilot evidence already exists: {irreversible.name}")

    if source_tree_identity() != source or pilot_config_sha256() != config_sha:
        raise RuntimeError("source or pilot config changed during reversible preflight checks")

    claim_global_pilot_slot(GLOBAL_SLOT_PATH, identity=identity)
    ledger.reserve(
        attempt_id,
        known_usage=known,
        phase_bound=bound,
        rates_sha256=rates_sha,
    )
    create_launch_permit(
        _operating_path(PERMIT_PATH),
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        identity=identity,
        known_usage_before_usd=known_text,
        pending_worst_case_usd=bound_text,
        phase_bound_usd=bound_text,
        rates=snapshot.rates,
        rates_sha256=rates_sha,
        git_diff_sha256=source.git_diff_sha256,
        config_sha256=config_sha,
    )
    typer.echo(f"PASS attempt={attempt_id} bound_usd={bound_text} internal_limit_usd=27.00")


@app.command("provision-volumes")
def provision_volumes(attestation: Path = ATTESTATION_PATH) -> None:
    permit = _verified_unsubmitted_permit(attestation)
    if permit["profile"] != _PROFILE or permit["environment"] != _ENVIRONMENT:
        raise ValueError("launch permit profile or environment differs from provisioning")
    identity = _provision_identity(permit)
    intent_path = _operating_path(PROVISION_INTENT_PATH)
    receipt_path = _operating_path(PROVISION_RECEIPT_PATH)
    ensure_private_directory(intent_path.parent)
    existing = _volume_names(_modal_cli_json(["volume", "list", "--env", "main"]))
    try:
        intent_path.lstat()
    except FileNotFoundError:
        if existing:
            raise ValueError(
                "pre-existing Modal volume has no attempt-bound provision intent"
            ) from None
        intent = identity | {
            "kind": "ratemem-volume-provision-intent-v1",
            "confirmed_initially_absent_at_utc": _now_utc().isoformat(timespec="microseconds"),
        }
        try:
            write_exclusive_private_json(intent_path, intent)
        except FileExistsError:
            pass
    _read_provision_record(
        intent_path,
        identity=identity,
        kind="ratemem-volume-provision-intent-v1",
        timestamp_key="confirmed_initially_absent_at_utc",
    )
    if existing - _REQUIRED_VOLUMES:
        raise ValueError("unexpected Modal volume contaminates the dedicated pilot workspace")
    intent_sha = file_sha256(intent_path)
    try:
        receipt_path.lstat()
    except FileNotFoundError:
        pass
    else:
        _read_provision_record(
            receipt_path,
            identity=identity,
            kind="ratemem-volume-provision-receipt-v1",
            timestamp_key="confirmed_present_at_utc",
            extra={"intent_sha256": intent_sha},
        )
        if not _REQUIRED_VOLUMES <= existing:
            raise RuntimeError("attempt-bound Modal volume disappeared after provisioning")
        typer.echo("PASS volumes=ratemem-sana-cache,ratemem-pilot-artifacts")
        return
    for name in sorted(_REQUIRED_VOLUMES - existing):
        _modal_cli(["volume", "create", "--env", "main", name])
    verified = _volume_names(_modal_cli_json(["volume", "list", "--env", "main"]))
    if verified != _REQUIRED_VOLUMES:
        raise RuntimeError("the exact Modal volume set is not durably visible after provisioning")
    receipt = identity | {
        "kind": "ratemem-volume-provision-receipt-v1",
        "intent_sha256": intent_sha,
        "confirmed_present_at_utc": _now_utc().isoformat(timespec="microseconds"),
    }
    try:
        write_exclusive_private_json(receipt_path, receipt)
    except FileExistsError:
        pass
    _read_provision_record(
        receipt_path,
        identity=identity,
        kind="ratemem-volume-provision-receipt-v1",
        timestamp_key="confirmed_present_at_utc",
        extra={"intent_sha256": intent_sha},
    )
    typer.echo("PASS volumes=ratemem-sana-cache,ratemem-pilot-artifacts")


@app.command("permit-field")
def permit_field(field: str) -> None:
    if field not in {"attempt_id", "workspace"}:
        raise typer.BadParameter("field must be attempt_id or workspace")
    snapshot = verify_fresh_attestation_file(
        _operating_path(ATTESTATION_PATH), config_path=MODAL_CONFIG_PATH
    )
    current_source = source_tree_sha256()
    try:
        evidence = validate_unsubmitted_launch_permit(
            _operating_path(PERMIT_PATH),
            slot=GLOBAL_SLOT_PATH,
            receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
            expected_workspace=snapshot.workspace,
            current_source_sha256=current_source,
        )
    except FileExistsError:
        evidence = validate_consumed_launch_evidence(
            _operating_path(PERMIT_PATH),
            slot=GLOBAL_SLOT_PATH,
            receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
            expected_workspace=snapshot.workspace,
            current_source_sha256=current_source,
        )
    typer.echo(cast(str, evidence[field]))


@app.command("validate-modal-config")
def validate_modal_config(state: str) -> None:
    _validate_modal_config_state(state)
    typer.echo(f"PASS modal_config_state={state}")


@app.command("validate-artifact")
def validate_artifact(path: Path) -> None:
    _validate_artifact(path)
    typer.echo(f"PASS artifact={path}")


@app.command("validate-forensic-receipts")
def validate_forensic_receipts(path: Path, directory: Path) -> None:
    manifest = _validate_forensic_receipts(path, directory)
    manifest_path = directory.with_name(f"{directory.name}.manifest.json")
    expected = canonical_json_bytes(manifest)
    try:
        write_exclusive_private_bytes(manifest_path, expected)
    except FileExistsError:
        if read_private_bytes(manifest_path) != expected:
            raise ValueError(
                "existing forensic receipt manifest differs from validated evidence"
            ) from None
    typer.echo(
        f"PASS forensic_receipts={manifest['attempt_id']} "
        f"raw_snapshot_sha256={manifest['raw_snapshot_sha256']}"
    )


@app.command("record-incident")
def record_incident(attestation: Path = ATTESTATION_PATH) -> None:
    _require_repository_cwd()
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    launch, permit = _launch_state(snapshot.workspace)
    attempt_id = cast(str, permit["attempt_id"])
    before = Decimal(cast(str, permit["known_usage_before_usd"]))
    phase = Decimal(cast(str, permit["phase_bound_usd"]))
    usage = Decimal(snapshot.known_metered_usage_usd)
    if usage < before:
        raise ValueError("incident billing data decreased below the launch permit")
    ledger = CostLedger(_operating_path(LEDGER_PATH), internal_limit_usd=Decimal("27.00"))
    record = ledger.attempt_cost(attempt_id)
    if (
        record is None
        or record.known_usage_before != before
        or record.phase_bound != phase
        or record.reconciled_cost is not None
    ):
        raise ValueError("incident requires the exact open launch cost reservation")
    volumes = sorted(_volume_names(_modal_cli_json(["volume", "list", "--env", "main"])))
    if _latest_reappearance_sha256(attempt_id) is None:
        status = "artifact_unavailable"
        reason = "artifact_unavailable_after_launch_command"
    else:
        status = "attempt_invalidated"
        reason = "volume_reappearance_permanently_invalidated_normal_attempt"
    directory = _operating_path(INCIDENT_DIRECTORY) / attempt_id
    ensure_private_directory(directory)
    path = directory / "incident.pending.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "ratemem-pilot-incident-v1",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "attempt_id": attempt_id,
        "workspace": snapshot.workspace,
        "profile": _PROFILE,
        "environment": _ENVIRONMENT,
        "status": status,
        "reason": reason,
        "created_at_utc": _now_utc().isoformat(timespec="microseconds"),
        "launch": launch,
        "cost": {
            "known_usage_before_usd": _fixed_usd_text(before, "incident known usage before"),
            "known_usage_at_incident_usd": _fixed_usd_text(usage, "incident observed usage"),
            "phase_bound_usd": _fixed_usd_text(phase, "incident phase bound"),
            "rates_sha256": permit["rates_sha256"],
            "reconciliation_status": "pending",
            "reconciled_cost_usd": None,
            "hard_budget_violation": None,
        },
        "volumes_observed": volumes,
    }
    try:
        write_exclusive_private_json(path, payload)
    except FileExistsError:
        existing = _validate_incident(path)
        if any(
            existing[key] != payload[key]
            for key in (
                "attempt_id",
                "workspace",
                "profile",
                "environment",
                "status",
                "reason",
                "launch",
            )
        ):
            raise ValueError("existing incident differs from this one-shot launch") from None
    else:
        _validate_incident(path)
    if usage > Decimal("28.00"):
        _record_hard_budget_observation(
            attempt_id=attempt_id,
            workspace=snapshot.workspace,
            known_usage_before=before,
            known_usage=usage,
        )
    typer.echo(f"PASS incident={path}")


@app.command("attest-incident-volume-absence")
def attest_incident_volume_absence(
    path: Path,
    attestation: Path = ATTESTATION_PATH,
) -> None:
    incident = _validate_incident(path)
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    if incident["workspace"] != snapshot.workspace:
        raise ValueError("incident workspace differs from the current attestation")
    attempt_id = cast(str, incident["attempt_id"])
    reappearance_sha256 = _latest_reappearance_sha256(attempt_id)
    evidence_path = _incident_absence_path(attempt_id, reappearance_sha256)
    try:
        evidence_path.lstat()
    except FileNotFoundError:
        pass
    else:
        evidence = _verified_incident_absence(incident)
        typer.echo(
            f"PASS incident={incident['attempt_id']} volumes_absent_since="
            f"{evidence.confirmed_at.isoformat(timespec='microseconds')}"
        )
        return
    present = _volume_names(_modal_cli_json(["volume", "list", "--env", "main"]))
    if present:
        _pending_settlement("Modal volumes are still present during incident cleanup")
    ensure_private_directory(evidence_path.parent)
    identity = _incident_absence_identity(
        incident,
        reappearance_sha256=reappearance_sha256,
    )
    usage = Decimal(snapshot.known_metered_usage_usd)
    body = identity | {
        "known_metered_usage_usd": _fixed_usd_text(usage, "incident volume absence usage"),
        "confirmed_absent_at_utc": _now_utc().isoformat(timespec="microseconds"),
    }
    try:
        write_exclusive_private_json(evidence_path, body)
    except FileExistsError:
        pass
    evidence = _read_incident_absence(incident)
    typer.echo(
        f"PASS incident={incident['attempt_id']} volumes_absent_since="
        f"{evidence.confirmed_at.isoformat(timespec='microseconds')}"
    )


@app.command("reconcile-incident")
def reconcile_incident(
    path: Path,
    attestation: Path = ATTESTATION_PATH,
) -> None:
    incident = _validate_incident(path)
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    if incident["workspace"] != snapshot.workspace:
        raise ValueError("incident workspace differs from the current attestation")
    attempt_id = cast(str, incident["attempt_id"])
    cost_payload = cast(dict[str, object], incident["cost"])
    before = Decimal(cast(str, cost_payload["known_usage_before_usd"]))
    phase = Decimal(cast(str, cost_payload["phase_bound_usd"]))
    after = Decimal(snapshot.known_metered_usage_usd)
    if not after.is_finite() or after < before:
        raise ValueError("incident metered usage is non-finite or decreased")
    ledger = CostLedger(_operating_path(LEDGER_PATH), internal_limit_usd=Decimal("27.00"))
    record = ledger.attempt_cost(attempt_id)
    if record is None or record.known_usage_before != before or record.phase_bound != phase:
        raise ValueError("incident cost identity differs from its ledger reservation")
    hard_violation = after > Decimal("28.00")
    if hard_violation:
        _record_hard_budget_observation(
            attempt_id=attempt_id,
            workspace=snapshot.workspace,
            known_usage_before=before,
            known_usage=after,
        )
    try:
        absence = _verified_incident_absence(incident)
    except typer.Exit:
        if hard_violation:
            raise RuntimeError("HARD BUDGET VIOLATION: incident settlement remains open") from None
        raise
    if after < absence.known_usage:
        raise ValueError("incident usage decreased below its volume absence evidence")
    if record.reconciled_cost is None:
        reconciled_cost = after - before
        try:
            _settlement_candidate(
                attempt_id=attempt_id,
                workspace=snapshot.workspace,
                known_usage_before=before,
                observed_usage=after,
                volume_absence_sha256=absence.sha256,
                allow_equal=True,
            )
        except typer.Exit:
            if hard_violation:
                raise RuntimeError(
                    "HARD BUDGET VIOLATION: incident settlement remains open"
                ) from None
            raise
        ledger.reconcile(
            attempt_id,
            reconciled_cost=reconciled_cost,
            known_usage_after=after,
        )
    else:
        if record.known_usage_after != after:
            raise ValueError("incident usage differs from the durable reconciliation")
        reconciled_cost = record.reconciled_cost
    final = copy.deepcopy(incident)
    final_cost = cast(dict[str, object], final["cost"])
    final_cost["reconciliation_status"] = "reconciled"
    final_cost["reconciled_cost_usd"] = _fixed_usd_text(reconciled_cost, "incident reconciled cost")
    final_cost["hard_budget_violation"] = hard_violation
    final_path = path.with_name("incident.json")
    expected = canonical_json_bytes(final)
    try:
        write_exclusive_private_bytes(final_path, expected)
    except FileExistsError:
        if read_private_bytes(final_path) != expected:
            raise ValueError("existing incident.json differs from durable reconciliation") from None
    _validate_incident(final_path, allow_final=True)
    if hard_violation:
        raise RuntimeError(
            "HARD BUDGET VIOLATION: incident cost was reconciled without attempt.json"
        )
    typer.echo(
        f"PASS incident_reconciled_cost_usd="
        f"{_fixed_usd_text(reconciled_cost, 'incident reconciled cost')}"
    )


@app.command("attest-volume-absence")
def attest_volume_absence(
    path: Path,
    attestation: Path = ATTESTATION_PATH,
) -> None:
    pending = _validate_artifact(path)
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    workspace = cast(str, pending["modal"]["workspace"])
    if workspace != snapshot.workspace:
        raise ValueError("artifact workspace differs from the current attested workspace")
    evidence_path = _absence_path(cast(str, pending["attempt_id"]))
    try:
        evidence_path.lstat()
    except FileNotFoundError:
        pass
    else:
        evidence = _verified_volume_absence(pending=pending, workspace=workspace)
        typer.echo(
            f"PASS attempt={pending['attempt_id']} volumes_absent_since="
            f"{evidence.confirmed_at.isoformat(timespec='microseconds')}"
        )
        return
    present = _volume_names(_modal_cli_json(["volume", "list", "--env", "main"]))
    if present:
        _pending_settlement("Modal storage volumes are still present in the pilot workspace")
    launch = _consumed_launch_evidence(workspace)
    identity = _volume_absence_identity(
        pending=pending,
        workspace=workspace,
        launch=launch,
    )
    now = _now_utc()
    usage = Decimal(snapshot.known_metered_usage_usd)
    usage_text = _fixed_usd_text(usage, "volume absence known usage")
    ensure_private_directory(evidence_path.parent)
    body = identity | {
        "known_metered_usage_usd": usage_text,
        "confirmed_absent_at_utc": now.isoformat(timespec="microseconds"),
    }
    try:
        write_exclusive_private_json(evidence_path, body)
    except FileExistsError:
        existing = _read_volume_absence(
            pending=pending,
            workspace=workspace,
            launch=launch,
        )
        typer.echo(
            f"PASS attempt={pending['attempt_id']} volumes_absent_since="
            f"{existing.confirmed_at.isoformat(timespec='microseconds')}"
        )
        return
    evidence = _read_volume_absence(
        pending=pending,
        workspace=workspace,
        launch=launch,
    )
    typer.echo(
        f"PASS attempt={pending['attempt_id']} volumes_absent_since="
        f"{evidence.confirmed_at.isoformat(timespec='microseconds')}"
    )


@app.command("reconcile")
def reconcile(path: Path, attestation: Path = ATTESTATION_PATH) -> None:
    pending = _validate_artifact(path, allow_final=True)
    snapshot = verify_fresh_attestation_file(
        _operating_path(attestation), config_path=MODAL_CONFIG_PATH
    )
    if pending["modal"]["workspace"] != snapshot.workspace:
        raise ValueError("artifact workspace differs from the current attested workspace")
    attempt_id = cast(str, pending["attempt_id"])
    before = Decimal(cast(str, pending["cost"]["known_usage_before_usd"]))
    phase_bound = Decimal(cast(str, pending["cost"]["phase_bound_usd"]))
    after = Decimal(snapshot.known_metered_usage_usd)
    ledger = CostLedger(_operating_path(LEDGER_PATH), internal_limit_usd=Decimal("27.00"))
    record = ledger.attempt_cost(attempt_id)
    if record is None:
        raise ValueError("artifact attempt has no durable cost reservation")
    if record.known_usage_before != before or record.phase_bound != phase_bound:
        raise ValueError("artifact cost identity differs from its ledger reservation")
    final_path = path.with_name("attempt.json")
    final_exists = final_path.exists() or final_path.is_symlink()
    if not after.is_finite() or after < before:
        raise ValueError("fresh metered usage is non-finite or decreased")
    hard_violation = after > Decimal("28.00")
    if hard_violation:
        _record_hard_budget_observation(
            attempt_id=attempt_id,
            workspace=snapshot.workspace,
            known_usage_before=before,
            known_usage=after,
        )

    try:
        absence = _verified_volume_absence(
            pending=pending,
            workspace=snapshot.workspace,
        )
    except typer.Exit:
        if hard_violation:
            raise RuntimeError(
                "HARD BUDGET VIOLATION: a durable observation exists, settlement "
                "remains open, and attempt.json cannot claim compliance"
            ) from None
        raise
    if after < absence.known_usage:
        raise ValueError("fresh metered usage decreased below volume absence evidence")
    if record.reconciled_cost is None:
        if final_exists:
            raise ValueError("attempt.json exists before a durable ledger reconciliation")
        if after <= before:
            typer.echo(
                "PENDING: billing data has not caught up; another launch is forbidden",
                err=True,
            )
            raise typer.Exit(code=3)
        cost = after - before
        try:
            _settlement_candidate(
                attempt_id=attempt_id,
                workspace=snapshot.workspace,
                known_usage_before=before,
                observed_usage=after,
                volume_absence_sha256=absence.sha256,
            )
        except typer.Exit:
            if hard_violation:
                raise RuntimeError(
                    "HARD BUDGET VIOLATION: a durable observation exists, settlement "
                    "remains open, and attempt.json cannot claim compliance"
                ) from None
            raise
        ledger.reconcile(attempt_id, reconciled_cost=cost, known_usage_after=after)
    else:
        if record.known_usage_after is None or after != record.known_usage_after:
            if hard_violation:
                raise RuntimeError(
                    "HARD BUDGET VIOLATION: fresh pre-credit usage differs from the "
                    "earlier durable reconciliation"
                )
            raise ValueError(
                "fresh metered usage differs from the durable reconciliation; "
                "do not silently finalize stale cost"
            )
        cost = record.reconciled_cost

    if hard_violation:
        raise RuntimeError(
            "HARD BUDGET VIOLATION: actual pre-credit metered usage was durably "
            "reconciled, but attempt.json cannot claim compliance"
        )

    # Revalidate after the ledger transition. A failure here never reopens or releases cost.
    pending = _validate_artifact(path, allow_final=True)
    expected = _reconciled_payload(pending, cost)
    expected_bytes = canonical_json_bytes(expected)
    if final_path.exists() or final_path.is_symlink():
        if read_private_bytes(final_path) != expected_bytes:
            raise ValueError("existing attempt.json differs from the durable reconciliation")
    else:
        write_exclusive_private_bytes(final_path, expected_bytes)
    _validate_artifact(path, allow_final=True)
    observed_final = _strict_json_bytes(read_private_bytes(final_path), label="attempt.json")
    if observed_final != expected:
        raise RuntimeError("published attempt.json failed its idempotent identity check")
    typer.echo(f"PASS reconciled_cost_usd={_fixed_usd_text(cost, 'reconciled cost')}")


@app.command("security-scan")
def security_scan(paths: list[Path]) -> None:
    _require_repository_cwd()
    changed = _git(["diff", "--name-only", "-z", "--diff-filter=ACMR", "HEAD", "--"])
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"])
    candidates: set[Path] = set()
    for name in changed.split(b"\0") + untracked.split(b"\0"):
        if name:
            candidates.add(REPOSITORY_ROOT / name.decode("utf-8"))
    artifact_attempt_parent = Path(os.path.abspath(_operating_path(Path("artifacts/pilot"))))
    for path in paths:
        absolute = Path(os.path.abspath(path))
        is_attempt = (
            absolute.parent == artifact_attempt_parent
            and _UUID7.fullmatch(absolute.name) is not None
        )
        candidates.update(_artifact_files(path, skip_generated=not is_attempt))
    findings = credential_findings(sorted(candidates))
    if findings:
        for finding in findings:
            typer.echo(str(finding), err=True)
        raise typer.Exit(code=4)
    typer.echo(f"PASS security_scan_files={len(candidates)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
