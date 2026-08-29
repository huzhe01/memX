from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import ratemem.pilot.private_io as private_io
from ratemem.pilot.private_io import (
    canonical_json_bytes,
    ensure_private_directory,
    private_lock,
    read_private_json,
    write_atomic_private_json,
    write_exclusive_private_json,
)


def test_private_directory_and_canonical_files_use_exact_modes(tmp_path: Path) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"
    write_exclusive_private_json(target, {"z": 1, "a": "value"})

    assert private.stat().st_mode & 0o777 == 0o700
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.read_bytes() == b'{"a":"value","z":1}'
    assert read_private_json(target) == {"a": "value", "z": 1}
    assert canonical_json_bytes({"z": 1, "a": "value"}) == target.read_bytes()


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o600])
def test_existing_nonprivate_directory_is_rejected_without_repair(
    tmp_path: Path,
    mode: int,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=mode)
    private.chmod(mode)

    with pytest.raises(PermissionError, match="mode 0700"):
        ensure_private_directory(private)

    assert private.stat().st_mode & 0o777 == mode


def test_symlinked_directory_and_file_and_hardlink_are_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    ensure_private_directory(real)
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="directory"):
        ensure_private_directory(linked_directory)

    target = real / "record.json"
    write_exclusive_private_json(target, {"value": 1})
    symlink = real / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(PermissionError, match="regular non-symlink"):
        read_private_json(symlink)

    hardlink = real / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(PermissionError, match="regular non-symlink"):
        read_private_json(target)


def test_private_json_rejects_non_object_duplicate_and_nonfinite_roots(tmp_path: Path) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    cases = (
        b"[]",
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
    )
    for index, content in enumerate(cases):
        path = private / f"invalid-{index}.json"
        path.write_bytes(content)
        path.chmod(0o600)
        with pytest.raises(ValueError, match="object|duplicate|finite"):
            read_private_json(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_private_json_writer_rejects_nonfinite_values(
    tmp_path: Path,
    value: float,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"
    with pytest.raises(ValueError, match="finite|JSON"):
        write_exclusive_private_json(target, {"value": value})
    assert not target.exists()


def test_exclusive_write_never_replaces_and_atomic_write_rejects_unsafe_target(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"
    write_exclusive_private_json(target, {"value": 1})
    with pytest.raises(FileExistsError):
        write_exclusive_private_json(target, {"value": 2})
    assert read_private_json(target) == {"value": 1}

    target.chmod(0o644)
    with pytest.raises(PermissionError, match="mode 0600"):
        write_atomic_private_json(target, {"value": 2})
    assert target.stat().st_mode & 0o777 == 0o644


def test_atomic_write_cleans_staging_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"
    write_exclusive_private_json(target, {"value": 1})

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(private_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        write_atomic_private_json(target, {"value": 2})

    assert read_private_json(target) == {"value": 1}
    assert tuple(path.name for path in private.iterdir()) == ("record.json",)


def test_exclusive_partial_write_failure_removes_polluted_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"

    def fail_after_partial_write(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[:1])
        raise OSError("injected partial write")

    monkeypatch.setattr(private_io, "_write_all", fail_after_partial_write)
    with pytest.raises(OSError, match="partial"):
        write_exclusive_private_json(target, {"value": 1})
    assert not target.exists()


def test_directory_replacement_during_read_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    target = private / "record.json"
    write_exclusive_private_json(target, {"value": 1})
    displaced = tmp_path / "displaced"
    real_read = private_io._read_all

    def swap_after_read(descriptor: int) -> bytes:
        content = real_read(descriptor)
        private.rename(displaced)
        private.mkdir(mode=0o700)
        private.chmod(0o700)
        return content

    monkeypatch.setattr(private_io, "_read_all", swap_after_read)
    with pytest.raises(OSError, match="directory changed"):
        read_private_json(target)


def test_private_lock_serializes_threads_and_rejects_permissive_existing_lock(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    lock_path = private / "state.lock"
    order: list[int] = []

    def append(value: int) -> None:
        with private_lock(lock_path):
            order.append(value)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(8)))
    assert sorted(order) == list(range(8))
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.stat().st_uid == os.getuid()

    lock_path.chmod(0o644)
    with pytest.raises(PermissionError, match="mode 0600"):
        with private_lock(lock_path):
            raise AssertionError("unreachable")
    assert lock_path.stat().st_mode & 0o777 == 0o644


def test_atomic_write_fsyncs_file_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    ensure_private_directory(private)
    calls: list[int] = []
    real_fsync = private_io.os.fsync

    def observe(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(private_io.os, "fsync", observe)
    write_atomic_private_json(private / "record.json", {"value": 1})
    assert len(calls) >= 2
