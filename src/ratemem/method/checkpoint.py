"""Safe, provenance-bound checkpoints for the learned RateMem method."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from ratemem.evaluation.canonical import file_sha256, write_json_atomic
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_ALLOWED_PREFIXES = ("adapter_bank.", "amortizer.", "dictionary.", "utility.")
_SAFETENSORS_FORMAT = "ratemem-method-v1"
_PATH_TYPE = type(Path())


class MethodCheckpointError(ValueError):
    """Raised when a method checkpoint is unsafe, corrupt, or incompatible."""


@runtime_checkable
class DictionaryRevisionProvider(Protocol):
    def frozen_dictionary_revision(self) -> str: ...


class MethodProvenance(BaseModel):
    """Immutable identities required to reproduce a learned-method checkpoint."""

    model_config = _MODEL_CONFIG

    git_commit: GitCommit
    git_diff_sha256: Sha256
    backbone_model_id: str
    backbone_revision: str
    support_encoder_revision: str
    method_lock_sha256: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    visible_trace_set_sha256: Sha256
    training_seed: int
    torch_version: str
    diffusers_version: str

    @field_validator(
        "backbone_model_id",
        "backbone_revision",
        "support_encoder_revision",
        "torch_version",
        "diffusers_version",
    )
    @classmethod
    def validate_identity_text(cls, value: str) -> str:
        if type(value) is not str or not value or value != value.strip():
            raise ValueError("provenance identities must be non-empty canonical strings")
        if value.lower() in {"latest", "main", "master", "unknown", "unresolved"}:
            raise ValueError("provenance identities must be immutable resolved values")
        return value

    @field_validator("training_seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("training seed must be a nonnegative signed 64-bit integer")
        return value


class MethodCheckpointManifest(MethodProvenance):
    """Canonical sidecar for a pickle-free learned-method tensor file."""

    schema_version: Literal["1.0"] = "1.0"
    tensor_file: str
    tensor_sha256: Sha256
    tensor_keys: tuple[str, ...]
    dictionary_revision_sha256: Sha256

    @field_validator("tensor_file")
    @classmethod
    def validate_tensor_file(cls, value: str) -> str:
        if (
            type(value) is not str
            or not value
            or Path(value).name != value
            or not value.endswith(".safetensors")
        ):
            raise ValueError("tensor file must be a local safetensors basename")
        return value

    @field_validator("tensor_keys")
    @classmethod
    def validate_tensor_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or value != tuple(sorted(set(value))):
            raise ValueError("tensor keys must be non-empty, unique, and sorted")
        if any(not key.startswith(_ALLOWED_PREFIXES) for key in value):
            raise ValueError("checkpoint contains a forbidden tensor namespace")
        return value

    @model_validator(mode="after")
    def validate_tensor_identity(self) -> MethodCheckpointManifest:
        if self.tensor_file != "method.safetensors":
            raise ValueError("method checkpoint tensor filename changed")
        return self


def _dictionary_revision(method: nn.Module) -> str:
    if not isinstance(method, DictionaryRevisionProvider):
        raise TypeError("method must implement frozen_dictionary_revision()")
    revision = method.frozen_dictionary_revision()
    if (
        type(revision) is not str
        or len(revision) != 64
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise MethodCheckpointError("method returned an invalid dictionary revision")
    return revision


def _collect_trainable_tensors(method: nn.Module) -> dict[str, Tensor]:
    if not isinstance(method, nn.Module):
        raise TypeError("method must be a torch.nn.Module")
    state = method.state_dict()
    selected = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in sorted(state.items())
        if name.startswith(_ALLOWED_PREFIXES)
    }
    if not selected:
        raise MethodCheckpointError("method has no approved trainable tensor namespaces")
    if any(not torch.isfinite(tensor).all().item() for tensor in selected.values()):
        raise MethodCheckpointError("method checkpoint tensors must be finite")
    return selected


def _atomic_save_safetensors(path: Path, tensors: Mapping[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), temporary, metadata={"format": _SAFETENSORS_FORMAT})
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_manifest(path: Path) -> MethodCheckpointManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = MethodCheckpointManifest.model_validate(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MethodCheckpointError(f"invalid method checkpoint manifest: {error}") from error
    canonical = manifest.model_dump(mode="json")
    if payload != canonical:
        raise MethodCheckpointError("method checkpoint manifest is not canonical")
    return manifest


def _verify_tensor_artifact(
    root: Path,
    manifest: MethodCheckpointManifest,
) -> Path:
    tensor_path = root / manifest.tensor_file
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise MethodCheckpointError("method checkpoint tensor file is missing or unsafe")
    if file_sha256(tensor_path) != manifest.tensor_sha256:
        raise MethodCheckpointError("method checkpoint tensor hash changed")
    try:
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            keys = tuple(sorted(handle.keys()))
    except Exception as error:
        raise MethodCheckpointError(f"invalid safetensors checkpoint: {error}") from error
    if metadata != {"format": _SAFETENSORS_FORMAT}:
        raise MethodCheckpointError("method checkpoint safetensors metadata changed")
    if keys != manifest.tensor_keys:
        raise MethodCheckpointError("method checkpoint tensor key list changed")
    return tensor_path


def save_method_checkpoint(
    directory: Path,
    method: nn.Module,
    provenance: MethodProvenance,
) -> MethodCheckpointManifest:
    """Atomically write the approved learned tensors and canonical manifest."""

    if type(directory) is not _PATH_TYPE:
        raise TypeError("checkpoint directory must be an exact pathlib.Path")
    if type(provenance) is not MethodProvenance:
        raise TypeError("provenance must be an exact MethodProvenance")
    if directory.is_symlink():
        raise MethodCheckpointError("checkpoint directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / "method.safetensors"
    manifest_path = directory / "manifest.json"
    if tensor_path.exists() or tensor_path.is_symlink():
        raise FileExistsError("method checkpoint tensor file already exists")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("method checkpoint manifest already exists")

    tensors = _collect_trainable_tensors(method)
    _atomic_save_safetensors(tensor_path, tensors)
    manifest = MethodCheckpointManifest(
        **provenance.model_dump(),
        tensor_file=tensor_path.name,
        tensor_sha256=file_sha256(tensor_path),
        tensor_keys=tuple(tensors),
        dictionary_revision_sha256=_dictionary_revision(method),
    )
    try:
        write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    except Exception:
        tensor_path.unlink(missing_ok=True)
        raise
    return manifest


def inspect_method_checkpoint(directory: Path) -> MethodCheckpointManifest:
    """Validate the manifest, tensor hash, metadata, and exact allowed key set."""

    if type(directory) is not _PATH_TYPE or directory.is_symlink() or not directory.is_dir():
        raise MethodCheckpointError("method checkpoint must be a real directory")
    manifest = _read_manifest(directory / "manifest.json")
    _verify_tensor_artifact(directory, manifest)
    return manifest


def load_method_checkpoint(
    directory: Path,
    method: nn.Module,
    expected_provenance: MethodProvenance,
) -> MethodCheckpointManifest:
    """Load a checkpoint only after exact provenance and dictionary validation."""

    if not isinstance(method, nn.Module):
        raise TypeError("method must be a torch.nn.Module")
    if type(expected_provenance) is not MethodProvenance:
        raise TypeError("expected provenance must be an exact MethodProvenance")
    manifest = inspect_method_checkpoint(directory)
    observed_provenance = MethodProvenance.model_validate(
        manifest.model_dump(include=set(MethodProvenance.model_fields))
    )
    if observed_provenance != expected_provenance:
        differing = sorted(
            field
            for field in MethodProvenance.model_fields
            if getattr(observed_provenance, field) != getattr(expected_provenance, field)
        )
        raise MethodCheckpointError(
            "method checkpoint provenance mismatch: " + ", ".join(differing)
        )

    current = method.state_dict()
    expected_keys = tuple(
        sorted(name for name in current if name.startswith(_ALLOWED_PREFIXES))
    )
    if expected_keys != manifest.tensor_keys:
        raise MethodCheckpointError("method architecture does not match checkpoint tensor keys")
    try:
        loaded = load_file(directory / manifest.tensor_file, device="cpu")
    except Exception as error:
        raise MethodCheckpointError(f"cannot load method safetensors: {error}") from error
    merged = dict(current)
    for name in manifest.tensor_keys:
        target = current[name]
        source = loaded[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise MethodCheckpointError(f"checkpoint tensor shape or dtype changed: {name}")
        merged[name] = source.to(device=target.device)
    method.load_state_dict(cast(dict[str, Tensor], merged), strict=True)
    if _dictionary_revision(method) != manifest.dictionary_revision_sha256:
        raise MethodCheckpointError("loaded dictionary revision does not match manifest")
    return manifest


__all__ = [
    "DictionaryRevisionProvider",
    "MethodCheckpointError",
    "MethodCheckpointManifest",
    "MethodProvenance",
    "inspect_method_checkpoint",
    "load_method_checkpoint",
    "save_method_checkpoint",
]
