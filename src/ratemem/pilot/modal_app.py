"""Definition-only boundary for the one authorized paid Modal pilot.

This module is run only by the guarded Task 13 launch command.  Importing it locally
defines the application; it does not submit, deploy, or execute paid work.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import modal

APP_NAME = "ratemem-sana-pilot"
CACHE_VOLUME_NAME = "ratemem-sana-cache"
ARTIFACT_VOLUME_NAME = "ratemem-pilot-artifacts"

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
_SHA_KEYS = (
    "source_sha256",
    "git_diff_sha256",
    "config_sha256",
    "slot_sha256",
    "permit_sha256",
    "submission_receipt_sha256",
    "rates_sha256",
)
_RATE_KEYS = {
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
}
_RECEIPT_SEMANTICS = "lower_bound_may_miss_precommit_reschedule"
_RECEIPT_KEYS = _REQUEST_KEYS | {
    "function_call_id",
    "input_id",
    "task_id",
    "receipt_id",
    "observed_at",
    "semantics",
}
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(".", extras=["modal"], groups=[], frozen=True, uv_version="0.8.14")
    .add_local_python_source("ratemem")
    .add_local_dir("configs/pilot", "/opt/ratemem/configs/pilot")
    # add_local_python_source installs ratemem below /root, so artifacts.py's
    # repository-relative schema lookup resolves to /schemas in the container.
    .add_local_dir("schemas", "/schemas")
    .workdir("/opt/ratemem")
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "WANDB_MODE": "disabled",
        }
    )
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=False)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def _lower_sha256(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    checked = value
    if len(checked) != 64 or any(character not in "0123456789abcdef" for character in checked):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return checked


def _decimal_string(value: object, name: str, *, positive: bool = False) -> str:
    if type(value) is not str or re.fullmatch(r"(0|[1-9][0-9]*)\.[0-9]{2,6}", value) is None:
        raise TypeError(f"{name} must be an exact fixed-point decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be decimal") from error
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _positive_rate(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be decimal") from error
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_request(request: object) -> dict[str, object]:
    if type(request) is not dict:
        raise TypeError("pilot request must be an exact object")
    payload = cast(dict[object, object], request)
    if set(payload) != _REQUEST_KEYS:
        raise ValueError("pilot request has missing or unexpected fields")
    if any(type(key) is not str for key in payload):
        raise TypeError("pilot request keys must be exact strings")
    attempt_id = payload["attempt_id"]
    workspace = payload["workspace"]
    if type(attempt_id) is not str or type(workspace) is not str:
        raise TypeError("pilot attempt and workspace identities must be exact strings")
    if _WORKSPACE.fullmatch(workspace) is None:
        raise ValueError("pilot workspace must be one canonical workspace slug")
    parsed_attempt = uuid.UUID(attempt_id)
    if str(parsed_attempt) != attempt_id or parsed_attempt.version != 7:
        raise ValueError("pilot attempt identity must be one canonical UUID version 7")
    git_commit = payload["git_commit"]
    if (
        type(git_commit) is not str
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("git_commit must be 40 lowercase hexadecimal characters")
    source_sha256 = _lower_sha256(payload["source_sha256"], "source_sha256")
    if hashlib.sha256(git_commit.encode("ascii")).hexdigest() != source_sha256:
        raise ValueError("source identity must bind the exact git commit")

    validated: dict[str, object] = {
        "attempt_id": attempt_id,
        "workspace": workspace,
        "git_commit": git_commit,
        "source_sha256": source_sha256,
    }
    for name in _SHA_KEYS:
        if name != "source_sha256":
            validated[name] = _lower_sha256(payload[name], name)

    known = _decimal_string(payload["known_usage_before_usd"], "known usage")
    pending = _decimal_string(payload["pending_worst_case_usd"], "pending worst case")
    phase = _decimal_string(payload["phase_bound_usd"], "phase bound", positive=True)
    if Decimal(known) + Decimal(pending) > Decimal("27.00") or Decimal(phase) > Decimal(pending):
        raise ValueError("launch request exceeds its exact internal cost bounds")
    validated["known_usage_before_usd"] = known
    validated["pending_worst_case_usd"] = pending
    validated["phase_bound_usd"] = phase

    rates = payload["rates"]
    if type(rates) is not dict or set(cast(dict[object, object], rates)) != _RATE_KEYS:
        raise ValueError("launch request rates have missing or unexpected fields")
    normalized_rates: dict[str, str] = {}
    for name in sorted(_RATE_KEYS):
        normalized_rates[name] = _positive_rate(
            cast(dict[object, object], rates)[name], f"rate {name}"
        )
    rates_sha256 = cast(str, validated["rates_sha256"])
    rates_bytes = json.dumps(
        normalized_rates, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if hashlib.sha256(rates_bytes).hexdigest() != rates_sha256:
        raise ValueError("launch request rates hash mismatch")
    validated["rates"] = normalized_rates
    return validated


def _forbidden_credentials() -> list[str]:
    exact = {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    }
    return sorted(name for name in exact if os.environ.get(name))


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("execution receipt write made no progress")
        remaining = remaining[written:]


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _validate_directory(metadata: os.stat_result, *, exact_private: bool) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or (exact_private and stat.S_IMODE(metadata.st_mode) != 0o700)
        or (not exact_private and stat.S_IMODE(metadata.st_mode) & 0o022)
    ):
        raise PermissionError("receipt directory is not a secure owner-controlled directory")


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _validate_open_directory(
    descriptor: int,
    expected: os.stat_result,
    *,
    exact_private: bool,
) -> None:
    current = os.fstat(descriptor)
    _validate_directory(current, exact_private=exact_private)
    if _directory_identity(current) != _directory_identity(expected):
        raise RuntimeError("open receipt directory identity changed")


def _validate_named_directory(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    _validate_open_directory(descriptor, expected, exact_private=True)
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("receipt directory binding disappeared") from error
    _validate_directory(named, exact_private=True)
    if _directory_identity(named) != _directory_identity(expected):
        raise RuntimeError("receipt directory binding changed")


def _validate_directory_chain(
    artifact_root: Path,
    root_descriptor: int,
    root_identity: os.stat_result,
    receipts_descriptor: int,
    receipts_identity: os.stat_result,
    attempt_descriptor: int,
    attempt_identity: os.stat_result,
    attempt_id: str,
) -> None:
    _validate_open_directory(root_descriptor, root_identity, exact_private=False)
    try:
        named_root = artifact_root.lstat()
    except OSError as error:
        raise RuntimeError("artifact root binding disappeared") from error
    _validate_directory(named_root, exact_private=False)
    if _directory_identity(named_root) != _directory_identity(root_identity):
        raise RuntimeError("artifact root directory binding changed")
    _validate_named_directory(
        root_descriptor,
        "execution-receipts",
        receipts_descriptor,
        receipts_identity,
    )
    _validate_named_directory(
        receipts_descriptor,
        attempt_id,
        attempt_descriptor,
        attempt_identity,
    )


def _open_artifact_root(path: Path) -> tuple[int, os.stat_result]:
    if type(path) is not type(Path()) or not path.is_absolute():
        raise TypeError("artifact root must be an exact absolute Path")
    try:
        before = path.lstat()
    except OSError as error:
        raise PermissionError("artifact root must be a real secure directory") from error
    _validate_directory(before, exact_private=False)
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _validate_directory(opened, exact_private=False)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("artifact root changed during secure open")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _open_private_child(
    parent_descriptor: int,
    name: str,
) -> tuple[int, os.stat_result]:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise PermissionError("receipt directory must be a real owner-only directory") from error
    _validate_directory(before, exact_private=True)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        _validate_directory(opened, exact_private=True)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("receipt directory changed during secure open")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _validate_receipt_file(descriptor: int, expected: bytes | None = None) -> bytes:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_nlink != 1
    ):
        raise PermissionError("execution receipt must be an owner-only single-link file")
    content = _read_all(descriptor)
    after = os.fstat(descriptor)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        or len(content) != before.st_size
        or (expected is not None and content != expected)
    ):
        raise RuntimeError("execution receipt changed during verification")
    return content


def _strict_receipt_json(content: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"execution receipt contains non-finite constant {value}")

    try:
        decoded = json.loads(content, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError("execution receipt is not valid strict JSON") from error
    if type(decoded) is not dict or set(decoded) != _RECEIPT_KEYS:
        raise ValueError("execution receipt has missing or unexpected fields")
    payload = cast(dict[str, object], decoded)
    request = {name: payload[name] for name in _REQUEST_KEYS}
    _validate_request(request)
    for name in ("function_call_id", "input_id"):
        if type(payload[name]) is not str or not payload[name]:
            raise ValueError(f"execution receipt {name} must be a nonempty exact string")
    task_id = payload["task_id"]
    if type(task_id) is not str or not task_id:
        raise ValueError("execution receipt task_id must be a nonempty exact string")
    receipt_id = payload["receipt_id"]
    _lower_sha256(receipt_id, "receipt_id")
    observed_at = payload["observed_at"]
    if type(observed_at) is not str:
        raise TypeError("execution receipt observed_at must be an exact string")
    try:
        observed = datetime.fromisoformat(observed_at)
    except ValueError as error:
        raise ValueError("execution receipt observed_at is invalid") from error
    if observed.tzinfo != UTC or observed.isoformat(timespec="microseconds") != observed_at:
        raise ValueError("execution receipt observed_at must be canonical UTC microseconds")
    if payload["semantics"] != _RECEIPT_SEMANTICS:
        raise ValueError("execution receipt semantics changed")
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("execution receipt is not finite canonical JSON") from error
    if canonical != content:
        raise ValueError("execution receipt is not canonical JSON")
    return payload


def _receipt_snapshot(
    directory_descriptor: int,
    expected_request: dict[str, object],
    *,
    current_name: str,
    current_identity: tuple[int, int],
) -> int:
    names = sorted(os.listdir(directory_descriptor))
    for name in names:
        if (
            len(name) != 69
            or not name.endswith(".json")
            or any(character not in "0123456789abcdef" for character in name[:-5])
        ):
            raise ValueError("execution receipt directory contains an unexpected member")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        try:
            named = os.fstat(descriptor)
            if name == current_name and (named.st_dev, named.st_ino) != current_identity:
                raise RuntimeError("published receipt inode identity changed")
            content = _validate_receipt_file(descriptor)
            decoded = _strict_receipt_json(content)
            observed_request = {name: decoded[name] for name in _REQUEST_KEYS}
            if observed_request != expected_request:
                raise ValueError("execution receipt request differs within one attempt")
            if decoded.get("receipt_id") != name[:-5]:
                raise ValueError("execution receipt identity or canonical content is invalid")
        finally:
            os.close(descriptor)
    return len(names)


def _commit_execution_receipt(
    request: dict[str, object],
    *,
    artifact_root: Path = Path("/artifacts"),
) -> tuple[Path, Path, int, str, str]:
    checked_request = _validate_request(request)
    function_call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    if type(function_call_id) is not str or not function_call_id:
        raise RuntimeError("Modal function call identity is unavailable")
    if type(input_id) is not str or not input_id:
        raise RuntimeError("Modal input identity is unavailable")
    task_id = os.environ.get("MODAL_TASK_ID")
    if type(task_id) is not str or not task_id:
        raise RuntimeError("Modal task identity is unavailable")
    receipt_id = hashlib.sha256(
        f"{function_call_id}\0{input_id}\0{task_id}\0{uuid.uuid4().hex}".encode()
    ).hexdigest()
    attempt_id = cast(str, checked_request["attempt_id"])
    receipt_directory = artifact_root / "execution-receipts" / attempt_id
    receipt_path = receipt_directory / f"{receipt_id}.json"
    receipt = dict(checked_request) | {
        "function_call_id": function_call_id,
        "input_id": input_id,
        "task_id": task_id,
        "receipt_id": receipt_id,
        "observed_at": datetime.now(UTC).isoformat(timespec="microseconds"),
        "semantics": _RECEIPT_SEMANTICS,
    }
    content = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    root_descriptor, root_identity = _open_artifact_root(artifact_root)
    receipts_descriptor = -1
    attempt_descriptor = -1
    receipt_descriptor = -1
    named_descriptor = -1
    try:
        receipts_descriptor, receipts_identity = _open_private_child(
            root_descriptor, "execution-receipts"
        )
        attempt_descriptor, attempt_identity = _open_private_child(receipts_descriptor, attempt_id)
        receipt_descriptor = os.open(
            receipt_path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=attempt_descriptor,
        )
        os.fchmod(receipt_descriptor, 0o600)
        created = os.fstat(receipt_descriptor)
        _write_all(receipt_descriptor, content)
        os.fsync(receipt_descriptor)
        verified = os.fstat(receipt_descriptor)
        if (verified.st_dev, verified.st_ino) != (created.st_dev, created.st_ino):
            raise RuntimeError("execution receipt inode changed during publication")
        _validate_receipt_file(receipt_descriptor, content)
        named_descriptor = os.open(
            receipt_path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=attempt_descriptor,
        )
        named = os.fstat(named_descriptor)
        if (named.st_dev, named.st_ino) != (created.st_dev, created.st_ino):
            raise RuntimeError("published receipt name has a different inode identity")
        _validate_receipt_file(named_descriptor, content)
        os.close(named_descriptor)
        named_descriptor = -1
        os.close(receipt_descriptor)
        receipt_descriptor = -1
        os.fsync(attempt_descriptor)
        os.fsync(receipts_descriptor)
        os.fsync(root_descriptor)
        _validate_directory_chain(
            artifact_root,
            root_descriptor,
            root_identity,
            receipts_descriptor,
            receipts_identity,
            attempt_descriptor,
            attempt_identity,
            attempt_id,
        )
        artifact_volume.commit()
        _validate_directory_chain(
            artifact_root,
            root_descriptor,
            root_identity,
            receipts_descriptor,
            receipts_identity,
            attempt_descriptor,
            attempt_identity,
            attempt_id,
        )
        receipt_count = _receipt_snapshot(
            attempt_descriptor,
            checked_request,
            current_name=receipt_path.name,
            current_identity=(created.st_dev, created.st_ino),
        )
        _validate_directory_chain(
            artifact_root,
            root_descriptor,
            root_identity,
            receipts_descriptor,
            receipts_identity,
            attempt_descriptor,
            attempt_identity,
            attempt_id,
        )
        return receipt_path, receipt_directory, receipt_count, function_call_id, input_id
    finally:
        if named_descriptor >= 0:
            os.close(named_descriptor)
        if receipt_descriptor >= 0:
            os.close(receipt_descriptor)
        if attempt_descriptor >= 0:
            os.close(attempt_descriptor)
        if receipts_descriptor >= 0:
            os.close(receipts_descriptor)
        os.close(root_descriptor)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    timeout=7200,
    startup_timeout=1800,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    volumes={"/cache": cache_volume, "/artifacts": artifact_volume},
)
# restrict_modal_access is intentionally omitted: Volume.commit requires Modal resource access.
def run_first_pilot(request: dict[str, object]) -> dict[str, object]:
    checked_request = _validate_request(request)
    receipt_path, receipt_directory, receipt_count, function_call_id, input_id = (
        _commit_execution_receipt(checked_request)
    )

    present = _forbidden_credentials()
    if present:
        raise RuntimeError(f"forbidden credential variables are present: {present}")

    import torch

    if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("pilot requires exactly one observed NVIDIA L40S")

    # This import and all model work are deliberately after durable receipt publication.
    runner_module = importlib.import_module("ratemem.pilot.runner")
    run_real_pilot = runner_module.run_real_pilot
    result = run_real_pilot(
        request=checked_request,
        cache_root=Path("/cache"),
        artifact_root=Path("/artifacts"),
        modal_ids={
            "profile": "ratemem-pilot",
            "workspace": checked_request["workspace"],
            "environment": "main",
            "launch_attempt_id": checked_request["attempt_id"],
            "launch_source_sha256": checked_request["source_sha256"],
            "pilot_slot_sha256": checked_request["slot_sha256"],
            "submission_receipt_sha256": checked_request["submission_receipt_sha256"],
            "function_call_id": function_call_id,
            "input_id": input_id,
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "container_image_id": os.environ["MODAL_IMAGE_ID"],
            "execution_receipt_path": str(receipt_path),
            "execution_receipt_directory": str(receipt_directory),
            "execution_receipt_count": receipt_count,
            "execution_receipt_semantics": _RECEIPT_SEMANTICS,
        },
    )
    if type(result) is not dict:
        raise TypeError("pilot runner must return an exact object")
    artifact_volume.commit()
    cache_volume.commit()
    return cast(dict[str, object], result)


@app.local_entrypoint()
def main() -> None:
    if os.environ.get("MODAL_PROFILE") != "ratemem-pilot":
        raise RuntimeError("MODAL_PROFILE must be ratemem-pilot")
    if os.environ.get("MODAL_ENVIRONMENT") != "main":
        raise RuntimeError("MODAL_ENVIRONMENT must be main")

    cli_module = importlib.import_module("ratemem.pilot.cli")
    one_shot_module = importlib.import_module("ratemem.pilot.one_shot")
    workspace_module = importlib.import_module("ratemem.pilot.workspace")
    GLOBAL_SLOT_PATH = one_shot_module.GLOBAL_SLOT_PATH
    GLOBAL_SUBMISSION_RECEIPT_PATH = one_shot_module.GLOBAL_SUBMISSION_RECEIPT_PATH
    PERMIT_PATH = one_shot_module.PERMIT_PATH
    snapshot = workspace_module.verify_fresh_attestation_file(
        Path("artifacts/pilot/workspace-attestation.json")
    )
    request = one_shot_module.consume_launch_request(
        PERMIT_PATH,
        slot=GLOBAL_SLOT_PATH,
        receipt=GLOBAL_SUBMISSION_RECEIPT_PATH,
        expected_workspace=snapshot.workspace,
        current_source_sha256=cli_module.source_tree_sha256(),
    )
    result = run_first_pilot.remote(request)
    print(json.dumps(result, sort_keys=True))
