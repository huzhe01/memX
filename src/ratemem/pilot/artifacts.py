"""Fail-closed directory transaction for one engineering pilot attempt."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, cast
from weakref import finalize as weakref_finalize

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from ratemem.adapters.checkpoint import CheckpointFileIdentity

SCHEMA_PATH: Final = Path(__file__).parents[3] / "schemas/ratemem-pilot-attempt-v1.schema.json"
_REQUIRED_FILES_WITHOUT_CHECKPOINT: Final = (
    "config.json",
    "dataset-manifest.json",
    "execution-receipts.jsonl",
    "metrics.jsonl",
    "rates.json",
)
_PROBE_NAMES: Final = (
    "checkpoint_compatibility",
    "dynamic_numerics",
    "gradient_flow",
    "frozen_backbone",
    "peak_memory",
    "one_step_inference",
    "one_timestep_backward",
    "step_timing",
    "held_in_loss",
)
_RESERVED_FILES: Final = frozenset(
    {"attempt.json", "attempt.pending.json", "checksums.sha256"}
)
_CHUNK_BYTES: Final = 1024 * 1024


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("artifact content must be finite canonical JSON") from error


def _strict_schema() -> dict[str, object]:
    try:
        decoded = json.loads(
            SCHEMA_PATH.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite schema constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("pilot artifact schema is unavailable or invalid") from error
    if type(decoded) is not dict:
        raise RuntimeError("pilot artifact schema root must be an exact object")
    schema = cast(dict[str, object], decoded)
    Draft202012Validator.check_schema(schema)
    return schema


def _finite_tree(value: object, path: str = "root") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if type(value) is dict:
        for key, child in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be exact strings")
            _finite_tree(child, f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(cast(list[object], value)):
            _finite_tree(child, f"{path}[{index}]")
        return
    else:
        raise TypeError(f"{path} contains a non-exact JSON value type")


def _usd(value: object, name: str) -> Decimal:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact decimal string")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a valid decimal amount") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return amount


def _timestamp(value: object, name: str) -> datetime:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact date-time string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a valid RFC 3339 date-time") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def validate_attempt(payload: dict[str, Any]) -> None:
    """Validate the exact schema plus semantic and cross-field identities."""

    if type(payload) is not dict:
        raise TypeError("attempt payload must be an exact dict")
    _finite_tree(payload)
    validator = Draft202012Validator(_strict_schema(), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: tuple(error.absolute_path))
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(item) for item in error.absolute_path) or 'root'}: {error.message}"
            for error in errors
        )
        raise ValueError(detail)

    attempt_id = cast(str, payload["attempt_id"])
    parsed_uuid = uuid.UUID(attempt_id)
    if str(parsed_uuid) != attempt_id or parsed_uuid.version != 7:
        raise ValueError("attempt_id must be one canonical RFC UUID version 7")
    modal = cast(dict[str, object], payload["modal"])
    if modal["launch_attempt_id"] != attempt_id:
        raise ValueError("launch attempt identity differs from artifact attempt_id")
    source = cast(dict[str, object], payload["source"])
    expected_source = hashlib.sha256(cast(str, source["git_commit"]).encode("ascii")).hexdigest()
    if modal["launch_source_sha256"] != expected_source:
        raise ValueError("launch source identity differs from the exact HEAD commit hash")

    started = _timestamp(payload["started_at"], "started_at")
    ended = _timestamp(payload["ended_at"], "ended_at")
    if ended < started:
        raise ValueError("ended_at must not precede started_at")
    cost = cast(dict[str, object], payload["cost"])
    known = _usd(cost["known_usage_before_usd"], "known_usage_before_usd")
    pending = _usd(cost["pending_worst_case_usd"], "pending_worst_case_usd")
    phase = _usd(cost["phase_bound_usd"], "phase_bound_usd")
    estimated = _usd(cost["estimated_cost_usd"], "estimated_cost_usd")
    if known + pending > Decimal("27.00"):
        raise ValueError("known usage plus pending worst case exceeds internal USD 27 bound")
    if estimated > phase or phase > pending:
        raise ValueError("estimated cost must not exceed phase or pending worst-case bounds")
    reconciliation = cost["reconciliation_status"]
    reconciled_value = cost["reconciled_cost_usd"]
    if reconciliation == "pending" and reconciled_value is not None:
        raise ValueError("pending reconciliation must have null reconciled cost")
    if reconciliation == "reconciled":
        reconciled = _usd(reconciled_value, "reconciled_cost_usd")
        if reconciled > phase or known + reconciled > Decimal("28.00"):
            raise ValueError("reconciled cost exceeds the phase or workspace bound")

    probes = cast(dict[str, object], payload["probes"])
    results = cast(dict[str, object], probes["results"])
    allowed = tuple(cast(list[str], probes["allowed_probe_names"]))
    if allowed != _PROBE_NAMES or set(results) != set(_PROBE_NAMES):
        raise ValueError("allowed and reported probes must be exactly canonical")
    statuses = tuple(
        cast(dict[str, object], results[name])["status"] for name in _PROBE_NAMES
    )
    p50 = probes["p50_step_seconds"]
    p95 = probes["p95_step_seconds"]
    if (p50 is None) != (p95 is None):
        raise ValueError("p50 and p95 step times must both be measured or both be null")
    if p50 is not None and cast(float, p50) > cast(float, p95):
        raise ValueError("p50 step time must not exceed p95")
    initial_loss = probes["initial_flow_loss"]
    final_loss = probes["final_flow_loss"]
    if (initial_loss is None) != (final_loss is None):
        raise ValueError("initial and final flow losses must both be measured or both be null")
    timing_status = cast(dict[str, object], results["step_timing"])["status"]
    if timing_status == "pass" and p50 is None:
        raise ValueError("a passing timing probe requires measured p50 and p95")
    if timing_status == "not_run" and p50 is not None:
        raise ValueError("a not-run timing probe must not report timing measurements")
    loss_status = cast(dict[str, object], results["held_in_loss"])["status"]
    if loss_status == "pass" and initial_loss is None:
        raise ValueError("a passing held-in loss probe requires paired losses")
    if (
        loss_status == "pass"
        and initial_loss is not None
        and cast(float, final_loss) >= cast(float, initial_loss)
    ):
        raise ValueError("a passing held-in loss probe requires a strict loss decrease")
    if loss_status == "not_run" and initial_loss is not None:
        raise ValueError("a not-run held-in loss probe must not report losses")
    if (
        loss_status == "fail"
        and initial_loss is not None
        and cast(float, final_loss) < cast(float, initial_loss)
    ):
        raise ValueError("a failed held-in loss probe cannot report a loss decrease")
    status = payload["status"]
    if (status == "succeeded") != (payload["error"] is None):
        raise ValueError("only a succeeded attempt may have a null error")
    if status == "succeeded" and any(result != "pass" for result in statuses):
        raise ValueError("a successful attempt requires every canonical probe to pass")
    if status == "succeeded" and (
        p50 is None or initial_loss is None or payload["checkpoint"] is None
    ):
        raise ValueError(
            "a successful attempt requires timing, loss, and checkpoint evidence"
        )
    if status == "probe_failed" and not any(result == "fail" for result in statuses):
        raise ValueError("a probe_failed attempt requires at least one failed probe")
    if status == "probe_failed" and loss_status == "fail" and initial_loss is None:
        raise ValueError("a held-in probe failure requires paired loss evidence")


def _absolute(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("artifact path must be a Path")
    return path if path.is_absolute() else Path.cwd() / path


def _assert_safe_ancestors(path: Path) -> None:
    current = path.parent
    while True:
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("artifact ancestors must be real directories")
        if current == current.parent:
            return
        current = current.parent


def _secure_regular(path: Path, context: str) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise PermissionError(f"{context} must be an owner-only 0600 single-link file")
    return metadata


def _sha_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, _CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("artifact write made no forward progress")
        view = view[written:]


class ArtifactWriter:
    """Create one immutable pending record and one create-only reconciled record.

    All payload validation and copying happens before root creation.  If the current
    uid moves or replaces the new root during its minimal open/fstat sequence, the
    constructor fails closed and may leave the moved empty directory behind; it never
    performs path-based cleanup that could delete a replacement inode.
    """

    def __init__(
        self,
        root: Path,
        attempt: dict[str, Any],
        *,
        checkpoint_identity: CheckpointFileIdentity | None,
    ) -> None:
        validate_attempt(attempt)
        checkpoint = attempt["checkpoint"]
        if checkpoint is None:
            if checkpoint_identity is not None:
                raise ValueError(
                    "checkpoint identity must be absent when attempt checkpoint is null"
                )
        else:
            if type(checkpoint_identity) is not CheckpointFileIdentity:
                raise TypeError(
                    "checkpoint_identity must be an exact CheckpointFileIdentity"
                )
            checkpoint_identity.validate()
            checkpoint_fields = cast(dict[str, object], checkpoint)
            if (
                checkpoint_fields["sha256"] != checkpoint_identity.sha256
                or checkpoint_fields["bytes"] != checkpoint_identity.byte_count
            ):
                raise ValueError(
                    "attempt checkpoint fields differ from checkpoint identity"
                )
        copied_attempt = copy.deepcopy(attempt)
        checked_root = _absolute(root)
        _assert_safe_ancestors(checked_root)
        os.mkdir(checked_root, 0o700)
        root_descriptor = -1
        try:
            metadata = checked_root.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise PermissionError("artifact root must be an owner-only 0700 directory")
            root_descriptor = os.open(
                checked_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            opened_root = os.fstat(root_descriptor)
            if (opened_root.st_dev, opened_root.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise RuntimeError("artifact root identity changed during secure open")
        except BaseException:
            if root_descriptor >= 0:
                os.close(root_descriptor)
            raise
        self.root = checked_root
        self._root_descriptor = root_descriptor
        self._root_identity = (opened_root.st_dev, opened_root.st_ino)
        self._attempt = copied_attempt
        self._checkpoint_identity = checkpoint_identity
        self._required_files = _REQUIRED_FILES_WITHOUT_CHECKPOINT + (
            ("trainable.safetensors",) if checkpoint_identity is not None else ()
        )
        self._sealed = False
        self._finalized = False
        self._closed = False
        self._descriptor_finalizer = weakref_finalize(
            self, os.close, root_descriptor
        )

    @property
    def attempt(self) -> dict[str, Any]:
        self._require_resource()
        return copy.deepcopy(self._attempt)

    def _require_resource(self) -> None:
        if self._closed:
            raise RuntimeError("artifact writer is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._descriptor_finalizer()
        self._root_descriptor = -1
        self._closed = True

    def __enter__(self) -> ArtifactWriter:
        self._require_resource()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.close()

    def _leaf(self, relative_path: str, *, reserved: bool = False) -> Path:
        if type(relative_path) is not str or not relative_path:
            raise TypeError("artifact relative path must be a nonempty exact str")
        if Path(relative_path).name != relative_path or relative_path in {".", ".."}:
            raise ValueError("artifact path must be one plain leaf name")
        if not reserved and relative_path in _RESERVED_FILES:
            raise ValueError("artifact path is reserved")
        return self.root / relative_path

    def _require_open(self) -> None:
        self._require_resource()
        if self._sealed:
            raise RuntimeError("artifact writer is sealed after pending publication")

    def _validate_root_identity(self) -> None:
        try:
            current = self.root.lstat()
        except FileNotFoundError as error:
            raise RuntimeError("artifact root disappeared") from error
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != self._root_identity
            or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) != 0o700
        ):
            raise PermissionError("artifact root identity or permissions changed")

    def _atomic(self, path: Path, blocks: Iterator[bytes]) -> tuple[str, int]:
        self._validate_root_identity()
        temporary_name = f".artifact-{uuid.uuid4().hex}"
        descriptor = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._root_descriptor,
        )
        digest = hashlib.sha256()
        count = 0
        published = False
        try:
            os.fchmod(descriptor, 0o600)
            for block in blocks:
                if type(block) is not bytes:
                    raise TypeError("artifact blocks must be exact bytes")
                _write_all(descriptor, block)
                digest.update(block)
                count += len(block)
            os.fsync(descriptor)
            staged = os.fstat(descriptor)
            if (
                not stat.S_ISREG(staged.st_mode)
                or staged.st_uid != os.getuid()
                or staged.st_nlink != 1
                or stat.S_IMODE(staged.st_mode) != 0o600
            ):
                raise PermissionError("artifact staging file lost its secure identity")
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=self._root_descriptor,
                dst_dir_fd=self._root_descriptor,
                follow_symlinks=False,
            )
            published = True
            linked = os.fstat(descriptor)
            if (
                (linked.st_dev, linked.st_ino) != (staged.st_dev, staged.st_ino)
                or linked.st_nlink != 2
            ):
                raise RuntimeError("published link differs from the staging inode")
            os.unlink(temporary_name, dir_fd=self._root_descriptor)
            unlinked = os.fstat(descriptor)
            if unlinked.st_nlink != 1:
                raise RuntimeError("published inode retained an unexpected link")
            published_descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=self._root_descriptor,
            )
            try:
                metadata = os.fstat(published_descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    raise PermissionError("published artifact file is not secure")
                if (metadata.st_dev, metadata.st_ino) != (
                    staged.st_dev,
                    staged.st_ino,
                ):
                    raise RuntimeError("published inode differs from staging")
                before_read = os.fstat(published_descriptor)
                published_sha256, published_bytes = _sha_descriptor(
                    published_descriptor
                )
                after_read = os.fstat(published_descriptor)
                if (
                    published_sha256 != digest.hexdigest()
                    or published_bytes != count
                    or (
                        after_read.st_size,
                        after_read.st_mtime_ns,
                        after_read.st_ctime_ns,
                    )
                    != (
                        before_read.st_size,
                        before_read.st_mtime_ns,
                        before_read.st_ctime_ns,
                    )
                ):
                    raise RuntimeError("published artifact digest or bytes changed")
            finally:
                os.close(published_descriptor)
            os.fsync(self._root_descriptor)
            self._validate_root_identity()
            return digest.hexdigest(), count
        except BaseException:
            if published:
                try:
                    os.unlink(path.name, dir_fd=self._root_descriptor)
                except FileNotFoundError:
                    pass
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=self._root_descriptor)
            except FileNotFoundError:
                pass

    def write_json(self, relative_path: str, value: object) -> None:
        self.write_bytes(relative_path, _canonical(value))

    def write_bytes(self, relative_path: str, value: bytes) -> None:
        self._require_open()
        if type(value) is not bytes:
            raise TypeError("artifact value must be exact bytes")
        path = self._leaf(relative_path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"artifact file already exists: {relative_path}")
        self._atomic(path, iter((value,)))

    def write_checkpoint(self, source: Path) -> None:
        self._require_open()
        if self._checkpoint_identity is None:
            raise RuntimeError("attempt does not authorize a checkpoint artifact")
        checked_source = _absolute(source)
        _assert_safe_ancestors(checked_source)
        before = _secure_regular(checked_source, "checkpoint source")
        descriptor = os.open(checked_source, os.O_RDONLY | os.O_NOFOLLOW)
        published = False
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError("checkpoint source changed during secure open")
            sha256, byte_count = _sha_descriptor(descriptor)
            if (
                sha256 != self._checkpoint_identity.sha256
                or byte_count != self._checkpoint_identity.byte_count
            ):
                raise ValueError("checkpoint source differs from its Task8 identity")
            os.lseek(descriptor, 0, os.SEEK_SET)

            def blocks() -> Iterator[bytes]:
                while True:
                    block = os.read(descriptor, _CHUNK_BYTES)
                    if not block:
                        return
                    yield block

            published_sha, published_bytes = self._atomic(
                self._leaf("trainable.safetensors"), blocks()
            )
            published = True
            after = os.fstat(descriptor)
            if (
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                or (published_sha, published_bytes) != (sha256, byte_count)
            ):
                raise RuntimeError("checkpoint source changed during artifact copy")
        except BaseException:
            if published:
                try:
                    os.unlink("trainable.safetensors", dir_fd=self._root_descriptor)
                    os.fsync(self._root_descriptor)
                except FileNotFoundError:
                    pass
            raise
        finally:
            os.close(descriptor)

    def _verify_payload_files(self) -> list[str]:
        self._validate_root_identity()
        actual = sorted(
            name
            for name in os.listdir(self._root_descriptor)
            if name not in _RESERVED_FILES and not name.startswith(".artifact-")
        )
        if tuple(actual) != self._required_files:
            raise ValueError("artifact payload file set is not exactly canonical")
        entries: list[str] = []
        for name in actual:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=self._root_descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    raise PermissionError(
                        f"artifact payload {name} is not an owner-only file"
                    )
                sha256, byte_count = _sha_descriptor(descriptor)
            finally:
                os.close(descriptor)
            if name == "trainable.safetensors" and (
                self._checkpoint_identity is None
                or sha256 != self._checkpoint_identity.sha256
                or byte_count != self._checkpoint_identity.byte_count
            ):
                raise ValueError("checkpoint file differs from its Task8 identity")
            entries.append(f"{sha256}  {name}")
        self._validate_root_identity()
        return entries

    def _read_member(self, name: str, context: str) -> bytes:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=self._root_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise PermissionError(f"{context} is not an owner-only file")
            os.lseek(descriptor, 0, os.SEEK_SET)
            blocks: list[bytes] = []
            while True:
                block = os.read(descriptor, _CHUNK_BYTES)
                if not block:
                    return b"".join(blocks)
                blocks.append(block)
        finally:
            os.close(descriptor)

    def write_pending(self) -> Path:
        self._require_open()
        entries = self._verify_payload_files()
        checksum_content = ("\n".join(entries) + "\n").encode("ascii")
        checksum_path = self._leaf("checksums.sha256", reserved=True)
        pending_path = self._leaf("attempt.pending.json", reserved=True)
        previous_checksum = self._attempt["files"]["checksums_sha256"]
        checksum_published = False
        pending_published = False
        try:
            checksum_sha, _ = self._atomic(checksum_path, iter((checksum_content,)))
            checksum_published = True
            if self._verify_payload_files() != entries:
                raise RuntimeError("artifact payload changed before pending publication")
            self._attempt["files"]["checksums_sha256"] = checksum_sha
            validate_attempt(self._attempt)
            self._atomic(pending_path, iter((_canonical(self._attempt),)))
            pending_published = True
            if self._verify_payload_files() != entries:
                raise RuntimeError("artifact payload changed during pending publication")
        except BaseException:
            self._attempt["files"]["checksums_sha256"] = previous_checksum
            created_markers = (
                (("attempt.pending.json",) if pending_published else ())
                + (("checksums.sha256",) if checksum_published else ())
            )
            for name in created_markers:
                try:
                    os.unlink(name, dir_fd=self._root_descriptor)
                except FileNotFoundError:
                    pass
            os.fsync(self._root_descriptor)
            raise
        self._sealed = True
        return pending_path

    def _verify_sealed(self) -> dict[str, Any]:
        if not self._sealed:
            raise RuntimeError("pending artifact must be published before finalization")
        entries = self._verify_payload_files()
        expected_checksums = ("\n".join(entries) + "\n").encode("ascii")
        if self._read_member("checksums.sha256", "checksums file") != expected_checksums:
            raise ValueError("artifact checksum index differs from real payload files")
        checksum_sha = hashlib.sha256(expected_checksums).hexdigest()
        pending_bytes = self._read_member("attempt.pending.json", "pending attempt")
        try:
            decoded = json.loads(pending_bytes)
        except json.JSONDecodeError as error:
            raise ValueError("pending attempt is not valid JSON") from error
        if type(decoded) is not dict or _canonical(decoded) != pending_bytes:
            raise ValueError("pending attempt must remain canonical JSON")
        payload = cast(dict[str, Any], decoded)
        validate_attempt(payload)
        if payload != self._attempt or payload["files"]["checksums_sha256"] != checksum_sha:
            raise ValueError("pending attempt identity or checksum binding changed")
        return payload

    def finalize(self, *, reconciled_cost_usd: str) -> Path:
        self._require_resource()
        if self._finalized:
            raise RuntimeError("artifact attempt is already finalized")
        pending = self._verify_sealed()
        final = copy.deepcopy(pending)
        final["cost"]["reconciliation_status"] = "reconciled"
        final["cost"]["reconciled_cost_usd"] = reconciled_cost_usd
        validate_attempt(final)
        final_path = self._leaf("attempt.json", reserved=True)
        published = False
        try:
            self._atomic(final_path, iter((_canonical(final),)))
            published = True
            if self._verify_sealed() != pending:
                raise RuntimeError("sealed artifact changed during final publication")
        except BaseException:
            if published:
                try:
                    os.unlink("attempt.json", dir_fd=self._root_descriptor)
                    os.fsync(self._root_descriptor)
                except FileNotFoundError:
                    pass
            raise
        self._finalized = True
        return final_path
