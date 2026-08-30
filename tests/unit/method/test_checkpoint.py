from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from ratemem.method.checkpoint import (
    MethodCheckpointError,
    MethodProvenance,
    inspect_method_checkpoint,
    load_method_checkpoint,
    save_method_checkpoint,
)


class TrainableMethod(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_bank = nn.Linear(3, 3)
        self.amortizer = nn.Linear(3, 2)
        self.dictionary = nn.Linear(2, 2, bias=False)
        self.utility = nn.Linear(2, 1)
        self.backbone = nn.Linear(3, 3)
        self.backbone.requires_grad_(False)

    def frozen_dictionary_revision(self) -> str:
        tensor = self.dictionary.weight.detach().float().cpu().contiguous()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def provenance(**updates: object) -> MethodProvenance:
    payload: dict[str, object] = {
        "git_commit": "1" * 40,
        "git_diff_sha256": "2" * 64,
        "backbone_model_id": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "backbone_revision": "3" * 40,
        "support_encoder_revision": "4" * 40,
        "method_lock_sha256": "5" * 64,
        "dataset_lock_sha256": "6" * 64,
        "evaluation_lock_sha256": "7" * 64,
        "baseline_lock_sha256": "8" * 64,
        "visible_trace_set_sha256": "9" * 64,
        "training_seed": 1729,
        "torch_version": "2.13.0",
        "diffusers_version": "0.40.0",
    }
    payload.update(updates)
    return MethodProvenance.model_validate(payload)


def test_method_checkpoint_round_trip_excludes_forbidden_state(tmp_path: Path) -> None:
    torch.manual_seed(11)
    source = TrainableMethod()
    expected_allowed = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if not name.startswith("backbone.")
    }
    manifest = save_method_checkpoint(tmp_path, source, provenance())
    saved_backbone = source.backbone.weight.detach().clone()

    torch.manual_seed(22)
    target = TrainableMethod()
    target_backbone = target.backbone.weight.detach().clone()
    load_method_checkpoint(tmp_path, target, provenance())

    assert manifest.tensor_keys == tuple(sorted(expected_allowed))
    assert all(not name.startswith("backbone.") for name in manifest.tensor_keys)
    for name, expected in expected_allowed.items():
        assert torch.equal(target.state_dict()[name], expected)
    assert torch.equal(source.backbone.weight, saved_backbone)
    assert torch.equal(target.backbone.weight, target_backbone)


def test_method_checkpoint_rejects_provenance_mismatch(tmp_path: Path) -> None:
    method = TrainableMethod()
    save_method_checkpoint(tmp_path, method, provenance())

    changed = provenance(training_seed=1730)
    with pytest.raises(MethodCheckpointError, match="training_seed"):
        load_method_checkpoint(tmp_path, TrainableMethod(), changed)


def test_method_checkpoint_rejects_tensor_corruption(tmp_path: Path) -> None:
    save_method_checkpoint(tmp_path, TrainableMethod(), provenance())
    tensor_path = tmp_path / "method.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")

    with pytest.raises(MethodCheckpointError, match="tensor hash"):
        inspect_method_checkpoint(tmp_path)


def test_method_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    method = TrainableMethod()
    save_method_checkpoint(tmp_path, method, provenance())

    with pytest.raises(FileExistsError, match="already exists"):
        save_method_checkpoint(tmp_path, method, provenance())


def test_method_checkpoint_manifest_is_canonical(tmp_path: Path) -> None:
    manifest = save_method_checkpoint(tmp_path, TrainableMethod(), provenance())
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert payload == manifest.model_dump(mode="json")
    assert inspect_method_checkpoint(tmp_path) == manifest
