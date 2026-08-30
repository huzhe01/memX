from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

_PATH_TYPE = type(Path())
_HEX64 = frozenset("0123456789abcdef")
_STEP_NAME = re.compile(r"step-[0-9]{8}")
_MANIFEST_SCHEMA = "memx-checkpoint-v1"
_POINTER_SCHEMA = "memx-checkpoint-pointer-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX64 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_step(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("checkpoint step must be a positive exact int")
    return value


def _sync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_model_state(value: object) -> dict[str, Tensor]:
    if type(value) is not dict or not value:
        raise TypeError("model_state must be a non-empty exact dict")
    checked = cast(dict[object, object], value)
    if any(type(key) is not str or not key for key in checked):
        raise TypeError("model_state keys must be non-empty exact strings")
    if any(type(tensor) is not Tensor for tensor in checked.values()):
        raise TypeError("model_state values must be exact Tensors")
    return {
        cast(str, name): cast(Tensor, tensor).detach().to(device="cpu").contiguous().clone()
        for name, tensor in sorted(checked.items(), key=lambda item: cast(str, item[0]))
    }


def _safe_tree(value: object, name: str) -> object:
    if type(value) is Tensor:
        return value.detach().to(device="cpu").contiguous().clone()
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is list:
        return [_safe_tree(item, name) for item in value]
    if type(value) is tuple:
        return tuple(_safe_tree(item, name) for item in value)
    if type(value) is dict:
        checked = cast(dict[object, object], value)
        if any(type(key) not in {str, int} for key in checked):
            raise TypeError(f"{name} mapping keys must be exact str or int")
        return {key: _safe_tree(item, name) for key, item in checked.items()}
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class CheckpointState:
    step: int
    config_sha256: str
    dataset_sha256: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, Any]
    torch_rng_state: Tensor

    def __post_init__(self) -> None:
        _positive_step(self.step)
        _require_sha256(self.config_sha256, "configuration hash")
        _require_sha256(self.dataset_sha256, "dataset hash")
        _safe_model_state(self.model_state)
        if type(self.optimizer_state) is not dict:
            raise TypeError("optimizer_state must be an exact dict")
        _safe_tree(self.optimizer_state, "optimizer_state")
        if (
            type(self.torch_rng_state) is not Tensor
            or self.torch_rng_state.dtype is not torch.uint8
            or self.torch_rng_state.ndim != 1
        ):
            raise TypeError("torch_rng_state must be a rank-one uint8 Tensor")


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        if type(root) is not _PATH_TYPE:
            raise TypeError("checkpoint root must be an exact Path")
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
            raise ValueError("checkpoint root must be a real directory")

    def _write_pointer(self, checkpoint_name: str) -> None:
        payload = _canonical_json(
            {"schema_version": _POINTER_SCHEMA, "checkpoint": checkpoint_name}
        )
        temporary = self.root / f".latest-{uuid.uuid4().hex}"
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.root / "latest.json")
        _sync_directory(self.root)

    def save(self, state: CheckpointState) -> Path:
        if type(state) is not CheckpointState:
            raise TypeError("checkpoint state must be an exact CheckpointState")
        name = f"step-{state.step:08d}"
        destination = self.root / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"checkpoint step already exists: {name}")
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)

        model_path = staging / "model.safetensors"
        training_path = staging / "training-state.pt"
        save_file(_safe_model_state(state.model_state), model_path)
        torch.save(
            {
                "optimizer_state": _safe_tree(state.optimizer_state, "optimizer_state"),
                "torch_rng_state": state.torch_rng_state.detach().cpu().contiguous().clone(),
            },
            training_path,
        )
        _sync_file(model_path)
        _sync_file(training_path)
        manifest = {
            "schema_version": _MANIFEST_SCHEMA,
            "step": state.step,
            "config_sha256": state.config_sha256,
            "dataset_sha256": state.dataset_sha256,
            "model_file": model_path.name,
            "model_sha256": _sha256_file(model_path),
            "training_state_file": training_path.name,
            "training_state_sha256": _sha256_file(training_path),
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        _sync_directory(staging)
        staging.rename(destination)
        _sync_directory(self.root)
        self._write_pointer(name)
        return destination

    def latest(
        self,
        *,
        expected_config_sha256: str | None = None,
        expected_dataset_sha256: str | None = None,
    ) -> CheckpointState | None:
        pointer_path = self.root / "latest.json"
        if not pointer_path.exists() and not pointer_path.is_symlink():
            return None
        try:
            pointer: Any = json.loads(pointer_path.read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("latest checkpoint pointer is unreadable") from error
        if (
            type(pointer) is not dict
            or set(pointer) != {"schema_version", "checkpoint"}
            or pointer["schema_version"] != _POINTER_SCHEMA
            or type(pointer["checkpoint"]) is not str
            or _STEP_NAME.fullmatch(pointer["checkpoint"]) is None
        ):
            raise ValueError("latest checkpoint name or pointer fields are invalid")
        return self.load(
            self.root / pointer["checkpoint"],
            expected_config_sha256=expected_config_sha256,
            expected_dataset_sha256=expected_dataset_sha256,
        )

    def load(
        self,
        path: Path,
        *,
        expected_config_sha256: str | None = None,
        expected_dataset_sha256: str | None = None,
    ) -> CheckpointState:
        if type(path) is not _PATH_TYPE or path.parent != self.root:
            raise ValueError("checkpoint path must be a direct child of the store")
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise ValueError("checkpoint path must be a real directory")
        try:
            payload: Any = json.loads((path / "manifest.json").read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("checkpoint manifest is unreadable") from error
        expected_fields = {
            "schema_version",
            "step",
            "config_sha256",
            "dataset_sha256",
            "model_file",
            "model_sha256",
            "training_state_file",
            "training_state_sha256",
        }
        if type(payload) is not dict or set(payload) != expected_fields:
            raise ValueError("checkpoint manifest fields changed")
        manifest = cast(dict[str, object], payload)
        if manifest["schema_version"] != _MANIFEST_SCHEMA:
            raise ValueError("checkpoint schema version changed")
        step = _positive_step(manifest["step"])
        config_hash = _require_sha256(manifest["config_sha256"], "configuration hash")
        dataset_hash = _require_sha256(manifest["dataset_sha256"], "dataset hash")
        if expected_config_sha256 is not None and config_hash != _require_sha256(
            expected_config_sha256, "expected configuration hash"
        ):
            raise ValueError("checkpoint configuration hash differs from the current run")
        if expected_dataset_sha256 is not None and dataset_hash != _require_sha256(
            expected_dataset_sha256, "expected dataset hash"
        ):
            raise ValueError("checkpoint dataset hash differs from the prepared dataset")
        if manifest["model_file"] != "model.safetensors":
            raise ValueError("checkpoint model filename changed")
        if manifest["training_state_file"] != "training-state.pt":
            raise ValueError("checkpoint training-state filename changed")
        model_path = path / "model.safetensors"
        training_path = path / "training-state.pt"
        if _sha256_file(model_path) != _require_sha256(
            manifest["model_sha256"], "model file hash"
        ):
            raise ValueError("checkpoint file hash changed: model.safetensors")
        if _sha256_file(training_path) != _require_sha256(
            manifest["training_state_sha256"], "training state file hash"
        ):
            raise ValueError("checkpoint file hash changed: training-state.pt")
        model_state = load_file(model_path, device="cpu")
        training: Any = torch.load(training_path, map_location="cpu", weights_only=True)
        if type(training) is not dict or set(training) != {"optimizer_state", "torch_rng_state"}:
            raise ValueError("checkpoint training state fields changed")
        optimizer_state = training["optimizer_state"]
        rng_state = training["torch_rng_state"]
        if type(optimizer_state) is not dict:
            raise TypeError("loaded optimizer state must be an exact dict")
        return CheckpointState(
            step=step,
            config_sha256=config_hash,
            dataset_sha256=dataset_hash,
            model_state=model_state,
            optimizer_state=optimizer_state,
            torch_rng_state=rng_state,
        )
