from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate private JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"non-finite private JSON constant: {value}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("private JSON value must be a mapping")
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("private JSON must contain finite JSON values") from error


def _absolute(path: Path) -> Path:
    if type(path) is not type(Path()):
        raise TypeError("private path must be an exact Path")
    return path if path.is_absolute() else Path.cwd() / path


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = _absolute(path)
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("directory must be a real non-symlink owned by the current uid")


def _validate_private_directory_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PermissionError("directory must be owned by the current uid with mode 0700")


def ensure_private_directory(path: Path) -> None:
    absolute = _absolute(path)
    _assert_no_symlink_ancestors(absolute.parent)
    try:
        absolute.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = absolute.lstat()
    except OSError as error:
        raise PermissionError(
            "directory must be owned by the current uid with mode 0700"
        ) from error
    _validate_private_directory_metadata(metadata)


@dataclass(frozen=True, slots=True)
class _Directory:
    path: Path
    descriptor: int
    metadata: os.stat_result


def _open_private_directory(path: Path) -> _Directory:
    absolute = _absolute(path)
    ensure_private_directory(absolute)
    before = absolute.lstat()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute, flags)
    try:
        after = os.fstat(descriptor)
        _validate_private_directory_metadata(after)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise OSError("private directory changed during secure open")
        return _Directory(absolute, descriptor, after)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory(directory: _Directory) -> None:
    current = os.fstat(directory.descriptor)
    _validate_private_directory_metadata(current)
    if (
        (current.st_dev, current.st_ino)
        != (directory.metadata.st_dev, directory.metadata.st_ino)
        or current.st_mode != directory.metadata.st_mode
        or current.st_uid != directory.metadata.st_uid
    ):
        raise OSError("private directory changed through its descriptor")
    try:
        _assert_no_symlink_ancestors(directory.path)
        by_path = directory.path.lstat()
    except OSError as error:
        raise OSError("private directory changed while operation was active") from error
    if (
        (by_path.st_dev, by_path.st_ino)
        != (directory.metadata.st_dev, directory.metadata.st_ino)
        or by_path.st_mode != directory.metadata.st_mode
        or by_path.st_uid != directory.metadata.st_uid
    ):
        raise OSError("private directory changed while operation was active")


def _validate_private_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise PermissionError(
            "file must be a regular non-symlink owned by the current uid with mode 0600"
        )


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
            raise OSError("private write made no progress")
        remaining = remaining[written:]


def read_private_bytes(path: Path) -> bytes:
    absolute = _absolute(path)
    directory = _open_private_directory(absolute.parent)
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(absolute.name, flags, dir_fd=directory.descriptor)
        except OSError as error:
            raise PermissionError(
                "file must be a regular non-symlink owned by the current uid with mode 0600"
            ) from error
        before = os.fstat(descriptor)
        _validate_private_file(before)
        content = _read_all(descriptor)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or len(content) != before.st_size
        ):
            raise OSError("private file changed while being read")
        _verify_directory(directory)
        return content
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory.descriptor)


def read_private_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(
            read_private_bytes(path),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError("private JSON is invalid") from error
    if type(decoded) is not dict:
        raise ValueError("private JSON root must be an object")
    return cast(dict[str, Any], decoded)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(read_private_bytes(path)).hexdigest()


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _write_exclusive_at(directory: _Directory, name: str, content: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=directory.descriptor)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _validate_private_file(metadata)
        if metadata.st_size != len(content):
            raise OSError("private file size changed during write")
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        try:
            os.unlink(name, dir_fd=directory.descriptor)
            _fsync_directory(directory.descriptor)
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_exclusive_private_bytes(path: Path, content: bytes) -> None:
    if type(content) is not bytes:
        raise TypeError("private content must be exact bytes")
    absolute = _absolute(path)
    directory = _open_private_directory(absolute.parent)
    created = False
    try:
        _write_exclusive_at(directory, absolute.name, content)
        created = True
        _fsync_directory(directory.descriptor)
        _verify_directory(directory)
    except BaseException:
        if created:
            try:
                os.unlink(absolute.name, dir_fd=directory.descriptor)
                _fsync_directory(directory.descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(directory.descriptor)


def write_exclusive_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_exclusive_private_bytes(path, canonical_json_bytes(payload))


def _validate_existing_at(directory: _Directory, name: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory.descriptor)
    except FileNotFoundError:
        return
    except OSError as error:
        raise PermissionError(
            "file must be a regular non-symlink owned by the current uid with mode 0600"
        ) from error
    try:
        _validate_private_file(os.fstat(descriptor))
    finally:
        os.close(descriptor)


def write_atomic_private_bytes(path: Path, content: bytes) -> None:
    if type(content) is not bytes:
        raise TypeError("private content must be exact bytes")
    absolute = _absolute(path)
    directory = _open_private_directory(absolute.parent)
    temporary = f".{absolute.name}.{uuid.uuid4().hex}.tmp"
    staged = False
    try:
        _validate_existing_at(directory, absolute.name)
        _write_exclusive_at(directory, temporary, content)
        staged = True
        os.replace(
            temporary,
            absolute.name,
            src_dir_fd=directory.descriptor,
            dst_dir_fd=directory.descriptor,
        )
        staged = False
        _fsync_directory(directory.descriptor)
        _validate_existing_at(directory, absolute.name)
        _verify_directory(directory)
    except BaseException:
        if staged:
            try:
                os.unlink(temporary, dir_fd=directory.descriptor)
            except OSError:
                pass
        raise
    finally:
        os.close(directory.descriptor)


def write_atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_atomic_private_bytes(path, canonical_json_bytes(payload))


@contextmanager
def private_lock(path: Path) -> Iterator[None]:
    absolute = _absolute(path)
    directory = _open_private_directory(absolute.parent)
    descriptor = -1
    locked = False
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(absolute.name, flags, 0o600, dir_fd=directory.descriptor)
        metadata = os.fstat(descriptor)
        _validate_private_file(metadata)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        _validate_private_file(os.fstat(descriptor))
        _verify_directory(directory)
        yield
        _validate_private_file(os.fstat(descriptor))
        _verify_directory(directory)
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory.descriptor)
