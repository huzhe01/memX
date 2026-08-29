"""Immutable one-shot authorization for the single paid Modal engineering pilot.

The Linux abstract ``AF_UNIX`` guard prevents cooperating live processes from
creating independent locks by replacing the state directory. Persistent records
still provide the crash boundary. A same-UID actor that waits for every process to
exit and then deletes *all* local records cannot be detected by any local-only
protocol; that limitation is intentional and does not authorize another launch.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import socket
import stat
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Never, cast

from ratemem.pilot.private_io import canonical_json_bytes, ensure_private_directory

GLOBAL_LOCK_NAME = "modal-pilot-one-shot.lock"
GLOBAL_STATE_DIRECTORY = Path("/home/ubuntu/.local/state/ratemem")
GLOBAL_SLOT_PATH = GLOBAL_STATE_DIRECTORY / "modal-pilot-slot.json"
GLOBAL_SUBMISSION_RECEIPT_PATH = GLOBAL_STATE_DIRECTORY / "modal-pilot-submitted.json"
PERMIT_PATH = Path("artifacts/pilot/launch-permit.json")
_SLOT_NAME = "modal-pilot-slot.json"
_PERMIT_NAME = "launch-permit.json"
_RECEIPT_NAME = "modal-pilot-submitted.json"
_SLOT_KIND = "ratemem-pilot-slot"
_PERMIT_KIND = "ratemem-pilot-launch-permit"
_RECEIPT_KIND = "ratemem-pilot-submission-receipt"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_WORKSPACE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_FIXED_DECIMAL = re.compile(r"(0|[1-9][0-9]*)\.[0-9]{2,6}")
_RATE_KEYS = {
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
}


def new_attempt_id() -> str:
    milliseconds = int(time.time_ns() // 1_000_000) & ((1 << 48) - 1)
    random_bits = secrets.randbits(74)
    value = milliseconds << 80
    value |= 0x7 << 76
    value |= ((random_bits >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= random_bits & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def _exact_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    return value


def _sha(value: object, label: str, pattern: re.Pattern[str] = _HEX64) -> str:
    text = _exact_string(value, label)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{label} must be canonical lowercase hex")
    return text


@dataclass(frozen=True, slots=True)
class PilotIdentity:
    attempt_id: str
    workspace: str
    source_sha256: str
    git_commit: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _exact_string(getattr(self, name), name)
        parsed = uuid.UUID(self.attempt_id)
        if str(parsed) != self.attempt_id or parsed.version != 7 or parsed.variant != uuid.RFC_4122:
            raise ValueError("attempt_id must be a canonical UUIDv7")
        if _WORKSPACE.fullmatch(self.workspace) is None:
            raise ValueError("workspace must be a canonical slug")
        _sha(self.git_commit, "git_commit", _HEX40)
        _sha(self.source_sha256, "source_sha256")
        if hashlib.sha256(self.git_commit.encode("ascii")).hexdigest() != self.source_sha256:
            raise ValueError("source_sha256 must bind the exact git_commit")


def _identity(payload: Mapping[str, Any]) -> PilotIdentity:
    return PilotIdentity(
        payload["attempt_id"], payload["workspace"], payload["source_sha256"], payload["git_commit"]
    )


def _decimal(value: object, label: str) -> Decimal:
    text = _exact_string(value, label)
    if _FIXED_DECIMAL.fullmatch(text) is None:
        raise TypeError(f"{label} must be an exact fixed-point decimal string")
    try:
        result = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{label} is invalid") from error
    if not result.is_finite() or result < 0 or str(result) != text:
        raise ValueError(f"{label} must be canonical, finite, and nonnegative")
    return result


def _rate(value: object) -> Decimal:
    text = _exact_string(value, "rate")
    try:
        result = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("rate must be decimal") from error
    if not result.is_finite() or result <= 0:
        raise ValueError("rate must be finite and positive")
    return result


def _utc_now() -> tuple[datetime, str]:
    value = datetime.now(UTC)
    return value, value.isoformat(timespec="microseconds")


def _utc(value: object, label: str, *, fresh: bool) -> datetime:
    text = _exact_string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from error
    if parsed.tzinfo != UTC or parsed.isoformat(timespec="microseconds") != text:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    if fresh:
        age = datetime.now(UTC) - parsed
        if age > timedelta(minutes=15) or age < -timedelta(minutes=1):
            raise ValueError(f"{label} is stale or implausibly future-dated")
    return parsed


def _schema(payload: Mapping[str, Any], *, kind: str, label: str) -> None:
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise TypeError(f"{label} schema_version must be the exact integer 1")
    if type(payload["kind"]) is not str or payload["kind"] != kind:
        raise ValueError(f"{label} kind is invalid")


def _absolute(path: Path) -> Path:
    if type(path) is not type(Path()):
        raise TypeError("one-shot paths must be exact Path values")
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("one-shot path ancestors must not be symbolic links")


def _validate_directory(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PermissionError("one-shot directories must be owner-only mode 0700")


def _validate_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise PermissionError("one-shot files must be owner-only mode 0600 single-link files")


@dataclass(frozen=True, slots=True)
class _Directory:
    path: Path
    descriptor: int
    metadata: os.stat_result

    def verify(self) -> None:
        current = os.fstat(self.descriptor)
        _validate_directory(current)
        if (
            (current.st_dev, current.st_ino) != (self.metadata.st_dev, self.metadata.st_ino)
            or current.st_mode != self.metadata.st_mode
            or current.st_uid != self.metadata.st_uid
        ):
            raise OSError("one-shot directory descriptor identity changed")
        _assert_no_symlink_ancestors(self.path)
        by_path = self.path.lstat()
        if (
            (by_path.st_dev, by_path.st_ino) != (self.metadata.st_dev, self.metadata.st_ino)
            or by_path.st_mode != self.metadata.st_mode
            or by_path.st_uid != self.metadata.st_uid
        ):
            raise OSError("one-shot directory path identity changed")


def _open_directory(path: Path) -> _Directory:
    absolute = _absolute(path)
    ensure_private_directory(absolute)
    _assert_no_symlink_ancestors(absolute)
    before = absolute.lstat()
    _validate_directory(before)
    descriptor = os.open(
        absolute,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        after = os.fstat(descriptor)
        _validate_directory(after)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise OSError("one-shot directory changed during secure open")
        return _Directory(absolute, descriptor, after)
    except BaseException:
        os.close(descriptor)
        raise


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("one-shot write made no progress")
        remaining = remaining[written:]


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"one-shot JSON contains non-finite constant: {value}")


@dataclass(frozen=True, slots=True)
class _PinnedRecord:
    directory: _Directory
    name: str
    descriptor: int
    metadata: os.stat_result
    content: bytes

    def verify(self) -> None:
        before = os.fstat(self.descriptor)
        _validate_file(before)
        current_content = _read_all(self.descriptor)
        after = os.fstat(self.descriptor)
        _validate_file(after)
        expected_identity = (
            self.metadata.st_dev,
            self.metadata.st_ino,
            self.metadata.st_size,
            self.metadata.st_mtime_ns,
            self.metadata.st_ctime_ns,
        )
        observed_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or observed_identity != expected_identity
            or current_content != self.content
        ):
            raise OSError("one-shot record changed while transaction was active")
        by_name = os.stat(
            self.name,
            dir_fd=self.directory.descriptor,
            follow_symlinks=False,
        )
        _validate_file(by_name)
        if (by_name.st_dev, by_name.st_ino) != (self.metadata.st_dev, self.metadata.st_ino):
            raise OSError("one-shot record path identity changed")


def _open_record(
    directory: _Directory, name: str, keys: set[str]
) -> tuple[dict[str, Any], str, _PinnedRecord]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory.descriptor)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise PermissionError("one-shot record must be a secure regular file") from error
    try:
        before = os.fstat(descriptor)
        _validate_file(before)
        raw = _read_all(descriptor)
        after = os.fstat(descriptor)
        _validate_file(after)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or len(raw) != before.st_size
        ):
            raise OSError("one-shot record changed while being read")
        try:
            value = json.loads(raw, parse_constant=_reject_nonfinite)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError("one-shot JSON is invalid") from error
        if type(value) is not dict or set(value) != keys or canonical_json_bytes(value) != raw:
            raise ValueError("one-shot JSON keys and canonical bytes must match exactly")
        pinned = _PinnedRecord(directory, name, descriptor, after, raw)
        return cast(dict[str, Any], value), hashlib.sha256(raw).hexdigest(), pinned
    except BaseException:
        os.close(descriptor)
        raise


def _entry_exists(directory: _Directory, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _safe_unlink_created(directory: _Directory, name: str, metadata: os.stat_result) -> None:
    try:
        by_name = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        if (by_name.st_dev, by_name.st_ino) != (metadata.st_dev, metadata.st_ino):
            return
        os.unlink(name, dir_fd=directory.descriptor)
        os.fsync(directory.descriptor)
    except OSError:
        pass


def _publish_exclusive(
    directory: _Directory,
    name: str,
    payload: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> str:
    content = canonical_json_bytes(payload)
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("one-shot publication hash differs from the prepared hash")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory.descriptor)
    metadata: os.stat_result | None = None
    committed = False
    try:
        metadata = os.fstat(descriptor)
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        _validate_file(metadata)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        _validate_file(after)
        if (
            (after.st_dev, after.st_ino) != (metadata.st_dev, metadata.st_ino)
            or after.st_size != len(content)
            or _read_all(descriptor) != content
        ):
            raise OSError("one-shot publication changed during durable write")
        by_name = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        _validate_file(by_name)
        if (by_name.st_dev, by_name.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise OSError("one-shot publication path identity changed")
        directory.verify()
        os.fsync(directory.descriptor)
        directory.verify()
        committed = True
        return digest
    except BaseException:
        if metadata is not None and not committed:
            _safe_unlink_created(directory, name, metadata)
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


@dataclass(slots=True)
class _Transaction:
    guard: socket.socket
    state: _Directory
    permit: _Directory | None
    lock_descriptor: int
    lock_metadata: os.stat_result
    pinned: list[_PinnedRecord] = field(default_factory=list)

    def read_state(self, name: str, keys: set[str]) -> tuple[dict[str, Any], str]:
        payload, digest, pinned = _open_record(self.state, name, keys)
        self.pinned.append(pinned)
        return payload, digest

    def read_permit(self, name: str, keys: set[str]) -> tuple[dict[str, Any], str]:
        if self.permit is None:
            raise RuntimeError("permit directory is unavailable")
        payload, digest, pinned = _open_record(self.permit, name, keys)
        self.pinned.append(pinned)
        return payload, digest

    def receipt_exists(self) -> bool:
        return _entry_exists(self.state, _RECEIPT_NAME)

    def precommit(self) -> None:
        self.state.verify()
        if self.permit is not None and self.permit is not self.state:
            self.permit.verify()
        lock_now = os.fstat(self.lock_descriptor)
        _validate_file(lock_now)
        lock_by_name = os.stat(
            GLOBAL_LOCK_NAME,
            dir_fd=self.state.descriptor,
            follow_symlinks=False,
        )
        _validate_file(lock_by_name)
        if (
            (lock_now.st_dev, lock_now.st_ino)
            != (self.lock_metadata.st_dev, self.lock_metadata.st_ino)
            or (lock_by_name.st_dev, lock_by_name.st_ino)
            != (self.lock_metadata.st_dev, self.lock_metadata.st_ino)
        ):
            raise OSError("one-shot lock identity changed")
        for record in self.pinned:
            record.verify()

    def close_best_effort(self) -> None:
        for record in reversed(self.pinned):
            try:
                os.close(record.descriptor)
            except OSError:
                pass
        try:
            fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self.lock_descriptor)
        except OSError:
            pass
        if self.permit is not None and self.permit is not self.state:
            try:
                os.close(self.permit.descriptor)
            except OSError:
                pass
        try:
            os.close(self.state.descriptor)
        except OSError:
            pass
        try:
            self.guard.close()
        except OSError:
            pass


def _kernel_guard(slot: Path) -> socket.socket:
    nominal = os.fsencode(str(_absolute(slot)))
    name = b"\0ratemem-pilot-" + hashlib.sha256(nominal).hexdigest().encode("ascii")
    guard = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        guard.bind(name)
    except OSError as error:
        guard.close()
        if error.errno == errno.EADDRINUSE:
            raise FileExistsError("one-shot pilot transaction is already active") from error
        raise
    return guard


def _open_lock(directory: _Directory) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(GLOBAL_LOCK_NAME, flags, 0o600, dir_fd=directory.descriptor)
    except OSError as error:
        raise PermissionError("one-shot lock must be a secure regular file") from error
    try:
        metadata = os.fstat(descriptor)
        _validate_file(metadata)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = os.fstat(descriptor)
        _validate_file(locked)
        if (locked.st_dev, locked.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise OSError("one-shot lock changed while being acquired")
        return descriptor, locked
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


@contextmanager
def _locked_transaction(slot: Path, permit: Path | None = None) -> Iterator[_Transaction]:
    guard = _kernel_guard(slot)
    state: _Directory | None = None
    permit_directory: _Directory | None = None
    lock_descriptor = -1
    transaction: _Transaction | None = None
    try:
        state = _open_directory(_absolute(slot).parent)
        if permit is not None:
            permit_parent = _absolute(permit).parent
            permit_directory = (
                state if permit_parent == state.path else _open_directory(permit_parent)
            )
        lock_descriptor, lock_metadata = _open_lock(state)
        transaction = _Transaction(
            guard,
            state,
            permit_directory,
            lock_descriptor,
            lock_metadata,
        )
        yield transaction
    finally:
        if transaction is not None:
            transaction.close_best_effort()
        else:
            if lock_descriptor >= 0:
                try:
                    os.close(lock_descriptor)
                except OSError:
                    pass
            if permit_directory is not None and permit_directory is not state:
                try:
                    os.close(permit_directory.descriptor)
                except OSError:
                    pass
            if state is not None:
                try:
                    os.close(state.descriptor)
                except OSError:
                    pass
            try:
                guard.close()
            except OSError:
                pass


def _paths(slot: Path, receipt: Path | None = None) -> Path:
    if type(slot) is not type(Path()) or slot.name != _SLOT_NAME:
        raise ValueError("slot path must use the canonical leaf")
    if receipt is not None and (
        type(receipt) is not type(Path())
        or receipt.name != _RECEIPT_NAME
        or _absolute(receipt).parent != _absolute(slot).parent
    ):
        raise ValueError("receipt must be the canonical slot sibling")
    return slot.with_name(GLOBAL_LOCK_NAME)


_IDENTITY_KEYS = {"attempt_id", "workspace", "source_sha256", "git_commit"}
_SLOT_KEYS = _IDENTITY_KEYS | {"claimed_at_utc", "kind", "schema_version"}
_PERMIT_KEYS = _IDENTITY_KEYS | {
    "authorized_at_utc",
    "kind",
    "schema_version",
    "slot_sha256",
    "git_diff_sha256",
    "config_sha256",
    "known_usage_before_usd",
    "pending_worst_case_usd",
    "phase_bound_usd",
    "rates",
    "rates_sha256",
    "profile",
    "environment",
    "workspace_budget_usd",
    "internal_budget_usd",
}
_RECEIPT_KEYS = _IDENTITY_KEYS | {
    "submitted_at_utc",
    "kind",
    "schema_version",
    "slot_sha256",
    "permit_sha256",
}


def _validate_slot(payload: Mapping[str, Any], *, fresh: bool) -> tuple[PilotIdentity, datetime]:
    _schema(payload, kind=_SLOT_KIND, label="slot")
    identity = _identity(payload)
    claimed = _utc(payload["claimed_at_utc"], "claimed_at_utc", fresh=fresh)
    return identity, claimed


def _validate_permit(
    payload: Mapping[str, Any], *, fresh: bool
) -> tuple[PilotIdentity, datetime]:
    _schema(payload, kind=_PERMIT_KIND, label="permit")
    identity = _identity(payload)
    authorized = _utc(payload["authorized_at_utc"], "authorized_at_utc", fresh=fresh)
    for key in ("slot_sha256", "git_diff_sha256", "config_sha256", "rates_sha256"):
        _sha(payload[key], key)
    known = _decimal(payload["known_usage_before_usd"], "known")
    pending = _decimal(payload["pending_worst_case_usd"], "pending")
    phase = _decimal(payload["phase_bound_usd"], "phase")
    if (
        known + pending > Decimal("27")
        or phase <= 0
        or phase > pending
        or type(payload["rates"]) is not dict
        or set(payload["rates"]) != _RATE_KEYS
        or payload["profile"] != "ratemem-pilot"
        or payload["environment"] != "main"
        or payload["workspace_budget_usd"] != "28.00"
        or payload["internal_budget_usd"] != "27.00"
    ):
        raise ValueError("permit budget or rates are invalid")
    for value in payload["rates"].values():
        _rate(value)
    if (
        hashlib.sha256(canonical_json_bytes(payload["rates"])).hexdigest()
        != payload["rates_sha256"]
    ):
        raise ValueError("permit rates hash binding is invalid")
    return identity, authorized


def claim_global_pilot_slot(slot: Path, *, identity: PilotIdentity) -> dict[str, Any]:
    if type(identity) is not PilotIdentity:
        raise TypeError("identity must be exact PilotIdentity")
    _paths(slot)
    _, claimed_at = _utc_now()
    payload = {
        "attempt_id": identity.attempt_id,
        "claimed_at_utc": claimed_at,
        "git_commit": identity.git_commit,
        "kind": _SLOT_KIND,
        "schema_version": 1,
        "source_sha256": identity.source_sha256,
        "workspace": identity.workspace,
    }
    with _locked_transaction(slot) as transaction:
        if transaction.receipt_exists():
            raise FileExistsError(slot.with_name(_RECEIPT_NAME))
        transaction.precommit()
        _publish_exclusive(transaction.state, _SLOT_NAME, payload)
        return payload


def create_launch_permit(
    permit: Path,
    *,
    slot: Path,
    receipt: Path,
    identity: PilotIdentity,
    known_usage_before_usd: str,
    pending_worst_case_usd: str,
    phase_bound_usd: str,
    rates: Mapping[str, str],
    rates_sha256: str,
    git_diff_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    if (
        type(identity) is not PilotIdentity
        or type(permit) is not type(Path())
        or permit.name != _PERMIT_NAME
    ):
        raise TypeError("exact identity and canonical permit path are required")
    _paths(slot, receipt)
    known, pending, phase = (
        _decimal(known_usage_before_usd, "known_usage_before_usd"),
        _decimal(pending_worst_case_usd, "pending_worst_case_usd"),
        _decimal(phase_bound_usd, "phase_bound_usd"),
    )
    if known + pending > Decimal("27") or phase <= 0 or phase > pending:
        raise ValueError("known+pending must be <=27 and 0<phase<=pending")
    if type(rates) is not dict or set(rates) != _RATE_KEYS:
        raise TypeError("rates must have exactly the four canonical keys")
    checked_rates = {key: _exact_string(value, "rate") for key, value in rates.items()}
    for value in checked_rates.values():
        _rate(value)
    _sha(rates_sha256, "rates_sha256")
    _sha(git_diff_sha256, "git_diff_sha256")
    _sha(config_sha256, "config_sha256")
    if hashlib.sha256(canonical_json_bytes(checked_rates)).hexdigest() != rates_sha256:
        raise ValueError("rates_sha256 must bind the exact canonical rates")

    with _locked_transaction(slot, permit) as transaction:
        if transaction.receipt_exists():
            raise FileExistsError(receipt)
        slot_payload, slot_hash = transaction.read_state(_SLOT_NAME, _SLOT_KEYS)
        slot_identity, claimed = _validate_slot(slot_payload, fresh=True)
        if slot_identity != identity:
            raise ValueError("slot and permit identities must match exactly")
        authorized, authorized_at = _utc_now()
        if authorized < claimed:
            raise ValueError("permit authorization must not precede the slot claim")
        payload = {
            "attempt_id": identity.attempt_id,
            "authorized_at_utc": authorized_at,
            "config_sha256": config_sha256,
            "environment": "main",
            "git_commit": identity.git_commit,
            "git_diff_sha256": git_diff_sha256,
            "internal_budget_usd": "27.00",
            "kind": _PERMIT_KIND,
            "known_usage_before_usd": known_usage_before_usd,
            "pending_worst_case_usd": pending_worst_case_usd,
            "phase_bound_usd": phase_bound_usd,
            "profile": "ratemem-pilot",
            "rates": checked_rates,
            "rates_sha256": rates_sha256,
            "schema_version": 1,
            "slot_sha256": slot_hash,
            "source_sha256": identity.source_sha256,
            "workspace": identity.workspace,
            "workspace_budget_usd": "28.00",
        }
        transaction.precommit()
        if transaction.permit is None:
            raise RuntimeError("permit directory is unavailable")
        _publish_exclusive(transaction.permit, _PERMIT_NAME, payload)
        return payload


def _validate(
    transaction: _Transaction,
    *,
    expected_workspace: str,
    current_source_sha256: str,
    fresh: bool,
) -> tuple[dict[str, Any], str, datetime, dict[str, Any], str, datetime]:
    _exact_string(expected_workspace, "expected_workspace")
    _sha(current_source_sha256, "current_source_sha256")
    slot_payload, slot_hash = transaction.read_state(_SLOT_NAME, _SLOT_KEYS)
    permit_payload, permit_hash = transaction.read_permit(_PERMIT_NAME, _PERMIT_KEYS)
    slot_identity, claimed = _validate_slot(slot_payload, fresh=fresh)
    permit_identity, authorized = _validate_permit(permit_payload, fresh=fresh)
    if (
        permit_identity != slot_identity
        or slot_identity.workspace != expected_workspace
        or slot_identity.source_sha256 != current_source_sha256
        or permit_payload["slot_sha256"] != slot_hash
    ):
        raise ValueError("slot, permit, workspace, and source identities must match exactly")
    if authorized < claimed:
        raise ValueError("permit authorization must not precede the slot claim")
    return slot_payload, slot_hash, claimed, permit_payload, permit_hash, authorized


def validate_unsubmitted_launch_permit(
    permit: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    if type(permit) is not type(Path()) or permit.name != _PERMIT_NAME:
        raise ValueError("permit path must use canonical leaf")
    _paths(slot, receipt)
    with _locked_transaction(slot, permit) as transaction:
        if transaction.receipt_exists():
            raise FileExistsError(receipt)
        _, _, _, permit_payload, _, _ = _validate(
            transaction,
            expected_workspace=expected_workspace,
            current_source_sha256=current_source_sha256,
            fresh=True,
        )
        transaction.precommit()
        return permit_payload


def consume_launch_request(
    permit: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    if type(permit) is not type(Path()) or permit.name != _PERMIT_NAME:
        raise ValueError("permit path must use canonical leaf")
    _paths(slot, receipt)
    with _locked_transaction(slot, permit) as transaction:
        if transaction.receipt_exists():
            raise FileExistsError(receipt)
        slot_payload, slot_hash, _, permit_payload, permit_hash, authorized = _validate(
            transaction,
            expected_workspace=expected_workspace,
            current_source_sha256=current_source_sha256,
            fresh=True,
        )
        submitted, submitted_at = _utc_now()
        if submitted < authorized:
            raise ValueError("submission must not precede permit authorization")
        submitted_payload = {
            **{key: slot_payload[key] for key in _IDENTITY_KEYS},
            "kind": _RECEIPT_KIND,
            "schema_version": 1,
            "submitted_at_utc": submitted_at,
            "permit_sha256": permit_hash,
            "slot_sha256": slot_hash,
        }
        submission_hash = hashlib.sha256(canonical_json_bytes(submitted_payload)).hexdigest()
        result = {
            **{key: permit_payload[key] for key in _IDENTITY_KEYS},
            **{
                key: permit_payload[key]
                for key in (
                    "git_diff_sha256",
                    "config_sha256",
                    "known_usage_before_usd",
                    "pending_worst_case_usd",
                    "phase_bound_usd",
                    "rates",
                    "rates_sha256",
                )
            },
            "slot_sha256": slot_hash,
            "permit_sha256": permit_hash,
            "submission_receipt_sha256": submission_hash,
        }
        transaction.precommit()
        _publish_exclusive(
            transaction.state,
            _RECEIPT_NAME,
            submitted_payload,
            expected_sha256=submission_hash,
        )
        return result


def validate_consumed_launch_evidence(
    permit: Path,
    *,
    slot: Path,
    receipt: Path,
    expected_workspace: str,
    current_source_sha256: str,
) -> dict[str, Any]:
    if type(permit) is not type(Path()) or permit.name != _PERMIT_NAME:
        raise ValueError("permit path must use canonical leaf")
    _paths(slot, receipt)
    with _locked_transaction(slot, permit) as transaction:
        slot_payload, slot_hash, claimed, _, permit_hash, authorized = _validate(
            transaction,
            expected_workspace=expected_workspace,
            current_source_sha256=current_source_sha256,
            fresh=False,
        )
        submitted_payload, submission_hash = transaction.read_state(
            _RECEIPT_NAME, _RECEIPT_KEYS
        )
        _schema(submitted_payload, kind=_RECEIPT_KIND, label="submission receipt")
        submitted = _utc(
            submitted_payload["submitted_at_utc"],
            "submitted_at_utc",
            fresh=False,
        )
        if (
            {key: submitted_payload[key] for key in _IDENTITY_KEYS}
            != {key: slot_payload[key] for key in _IDENTITY_KEYS}
            or submitted_payload["permit_sha256"] != permit_hash
            or submitted_payload["slot_sha256"] != slot_hash
        ):
            raise ValueError("submission receipt binding is invalid")
        if not claimed <= authorized <= submitted:
            raise ValueError("one-shot timestamps violate claim-authorize-submit order")
        transaction.precommit()
        return {**submitted_payload, "submission_receipt_sha256": submission_hash}
