from __future__ import annotations

import builtins
import gc
import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from diffusers import SanaTransformer2DModel
from safetensors import SafetensorError, safe_open
from safetensors.torch import save as save_safetensors
from torch import Tensor, nn

import ratemem.adapters.checkpoint as checkpoint_module
from ratemem.adapters.checkpoint import (
    CheckpointFileIdentity,
    CheckpointProvenance,
    TrainableCheckpointMetadata,
    load_trainable_checkpoint,
    save_trainable_checkpoint,
)
from ratemem.adapters.sana_layout import (
    PRODUCTION_ATOM_COUNT,
    PRODUCTION_BLOCK_COUNT,
    PRODUCTION_RANK,
    PRODUCTION_WIDTH,
    SANA_LAYOUT_VERSION,
    SanaAdapterLayout,
    SanaDynamicAdapterBank,
    install_sana_dynamic_atoms,
)
from ratemem.support.amortizer import SupportAmortizer


def _tiny_sana() -> SanaTransformer2DModel:
    return SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        mlp_ratio=1.0,
        sample_size=4,
        patch_size=1,
    )


def _components(
    seed: int,
    *,
    heads: int = 4,
    bank_dtype: torch.dtype = torch.float32,
) -> tuple[SanaTransformer2DModel, SanaDynamicAdapterBank, SupportAmortizer]:
    torch.manual_seed(seed)
    transformer = _tiny_sana().to(dtype=bank_dtype).requires_grad_(False).eval()
    bank = install_sana_dynamic_atoms(
        transformer,
        rank=2,
        atom_count=4,
        expected_blocks=1,
    )
    amortizer = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=16,
        projection_count=6,
        atom_count=4,
        layers=1,
        heads=heads,
    ).eval()
    return transformer, bank, amortizer


def _private_directory(tmp_path: Path, name: str = "private") -> Path:
    root = tmp_path / name
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        model_id="test/sana",
        model_revision="1" * 40,
        support_model_id="test/dino",
        support_model_revision="2" * 40,
    )


def _snapshot(
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> dict[str, Tensor]:
    values = {
        f"adapter_bank.{name}": parameter.detach().clone()
        for name, parameter in bank.named_parameters()
    }
    values.update(
        {
            f"amortizer.{name}": parameter.detach().clone()
            for name, parameter in amortizer.named_parameters()
        }
    )
    return values


def _assert_snapshot(
    expected: dict[str, Tensor],
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> None:
    actual = _snapshot(bank, amortizer)
    assert tuple(sorted(actual)) == tuple(sorted(expected))
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0.0, atol=0.0)


def _header(path: Path) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    with safe_open(path, framework="pt", device="cpu", backend="pread") as handle:
        metadata = handle.metadata()
        assert metadata is not None
        return metadata, tuple(handle.keys()), tuple(handle.offset_keys())


def _payload(path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu", backend="pread") as handle:
        metadata = handle.metadata()
        assert metadata is not None
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
    return tensors, metadata


def _write_payload(
    path: Path,
    tensors: dict[str, Tensor],
    metadata: dict[str, str],
) -> CheckpointFileIdentity:
    content = save_safetensors(tensors, metadata=metadata)
    path.write_bytes(content)
    path.chmod(0o600)
    return CheckpointFileIdentity(
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
    )


def _write_bytes(path: Path, content: bytes) -> CheckpointFileIdentity:
    path.write_bytes(content)
    path.chmod(0o600)
    return CheckpointFileIdentity(
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
    )


def _replace_raw_header(content: bytes, header_text: str) -> bytes:
    old_length = int.from_bytes(content[:8], "little", signed=False)
    encoded = header_text.encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return len(encoded).to_bytes(8, "little", signed=False) + encoded + content[8 + old_length :]


def _parameters(
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> tuple[Tensor, ...]:
    return (*bank.parameters(), *amortizer.parameters())


def _versions(
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in _parameters(bank, amortizer))


def _parameter_identity(
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter.untyped_storage().nbytes(),
        )
        for parameter in _parameters(bank, amortizer)
    )


def _frozen_base_parameters(bank: SanaDynamicAdapterBank) -> tuple[Tensor, ...]:
    return tuple(parameter for wrapper in bank.wrappers for parameter in wrapper.base.parameters())


def _frozen_base_contract(
    bank: SanaDynamicAdapterBank,
) -> tuple[tuple[int, int, int, int, Tensor], ...]:
    return tuple(
        (
            id(parameter),
            parameter.untyped_storage().data_ptr(),
            parameter.untyped_storage().nbytes(),
            parameter._version,
            parameter.detach().clone(),
        )
        for parameter in _frozen_base_parameters(bank)
    )


def _assert_frozen_base_contract(
    expected: tuple[tuple[int, int, int, int, Tensor], ...],
    bank: SanaDynamicAdapterBank,
) -> None:
    for contract, parameter in zip(expected, _frozen_base_parameters(bank), strict=True):
        identity, data_ptr, nbytes, version, value = contract
        assert id(parameter) == identity
        assert parameter.untyped_storage().data_ptr() == data_ptr
        assert parameter.untyped_storage().nbytes() == nbytes
        assert parameter._version == version
        assert parameter.grad is None
        torch.testing.assert_close(parameter, value, rtol=0.0, atol=0.0)


def _set_distinct_gradients(
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> tuple[tuple[int, Tensor], ...]:
    gradients: list[tuple[int, Tensor]] = []
    for index, parameter in enumerate(_parameters(bank, amortizer), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
        assert parameter.grad is not None
        gradients.append((id(parameter.grad), parameter.grad.detach().clone()))
    return tuple(gradients)


def _assert_gradients(
    expected: tuple[tuple[int, Tensor], ...],
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> None:
    for (expected_id, expected_value), parameter in zip(
        expected,
        _parameters(bank, amortizer),
        strict=True,
    ):
        assert parameter.grad is not None
        assert id(parameter.grad) == expected_id
        torch.testing.assert_close(parameter.grad, expected_value, rtol=0.0, atol=0.0)


def _forbid_materialization(_handle: object, _key: str) -> Tensor:
    raise AssertionError("tensor materialization occurred before header validation")


def _joint_output_and_gradients(
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    support_value: Tensor,
    description_value: Tensor,
    adapter_input_value: Tensor,
) -> tuple[Tensor | None, ...]:
    for parameter in _parameters(bank, amortizer):
        parameter.grad = None
    support = support_value.detach().clone().requires_grad_(True)
    description = description_value.detach().clone().requires_grad_(True)
    adapter_input = adapter_input_value.detach().clone().requires_grad_(True)
    mask = torch.ones(support.shape[:2], dtype=torch.bool)
    prediction = amortizer(support, mask, description)
    wrapper = bank.wrappers[0]
    with wrapper.use_coefficients(prediction.coefficients[:, 0, :]):
        output = wrapper(adapter_input)
        loss = (
            output.square().sum()
            + prediction.logits.square().sum()
            + prediction.scales.square().sum()
        )
        loss.backward()
    values: list[Tensor | None] = [
        output.detach().clone(),
        prediction.logits.detach().clone(),
        prediction.scales.detach().clone(),
        prediction.coefficients.detach().clone(),
        support.grad.detach().clone(),
        description.grad.detach().clone(),
        adapter_input.grad.detach().clone(),
    ]
    values.extend(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in _parameters(bank, amortizer)
    )
    assert all(parameter.grad is None for parameter in wrapper.base.parameters())
    return tuple(values)


def _assert_optional_tensors_equal(
    expected: tuple[Tensor | None, ...],
    actual: tuple[Tensor | None, ...],
) -> None:
    assert len(expected) == len(actual)
    for expected_value, actual_value in zip(expected, actual, strict=True):
        if expected_value is None:
            assert actual_value is None
        else:
            assert actual_value is not None
            torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)


def test_public_identity_types_are_frozen_and_strict() -> None:
    provenance = _provenance()
    identity = CheckpointFileIdentity(sha256="a" * 64, byte_count=1)
    assert provenance.model_revision == "1" * 40
    assert identity.byte_count == 1
    with pytest.raises(FrozenInstanceError):
        identity.byte_count = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match="byte_count"):
        CheckpointFileIdentity(sha256="a" * 64, byte_count=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sha256"):
        CheckpointFileIdentity(sha256="A" * 64, byte_count=1)
    with pytest.raises(TypeError, match="model_id"):
        CheckpointProvenance(
            model_id=object(),  # type: ignore[arg-type]
            model_revision="1" * 40,
            support_model_id="test/dino",
            support_model_revision="2" * 40,
        )


def test_runtime_serializer_version_is_part_of_the_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bank, amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    monkeypatch.setattr(checkpoint_module.safetensors, "__version__", "0.8.1")

    with pytest.raises(RuntimeError, match="safetensors 0.8.0"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()


def test_public_functions_reject_non_path_and_non_exact_contract_types(
    tmp_path: Path,
) -> None:
    _, bank, amortizer = _components(23)
    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    with pytest.raises(TypeError, match="exact Path"):
        save_trainable_checkpoint(  # type: ignore[arg-type]
            str(path),
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )
    with pytest.raises(TypeError, match="provenance"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=object(),  # type: ignore[arg-type]
        )
    assert not path.exists()


def test_save_uses_canonical_paths_single_metadata_key_and_deterministic_bytes(
    tmp_path: Path,
) -> None:
    _, bank, amortizer = _components(23)
    before = _snapshot(bank, amortizer)
    before_versions = _versions(bank, amortizer)
    identities = _parameter_identity(bank, amortizer)
    gradients = _set_distinct_gradients(bank, amortizer)
    frozen = _frozen_base_contract(bank)
    first_root = _private_directory(tmp_path, "first")
    second_root = _private_directory(tmp_path, "second")
    first = first_root / "trainable.safetensors"
    second = second_root / "trainable.safetensors"

    first_identity = save_trainable_checkpoint(
        first,
        adapter_bank=bank,
        amortizer=amortizer,
        provenance=_provenance(),
    )
    second_identity = save_trainable_checkpoint(
        second,
        adapter_bank=bank,
        amortizer=amortizer,
        provenance=_provenance(),
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_identity == second_identity
    assert first_identity.sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_identity.byte_count == first.stat().st_size
    metadata, keys, offset_keys = _header(first)
    assert set(metadata) == {"ratemem"}
    manifest = json.loads(metadata["ratemem"])
    assert (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        == metadata["ratemem"]
    )
    assert manifest["format"] == "safetensors"
    assert manifest["framework"] == "pt"
    assert manifest["schema"] == "ratemem-trainable-checkpoint"
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["serializer_version"] == "0.8.0"
    assert set(manifest) == {
        "amortizer",
        "format",
        "framework",
        "layout",
        "model",
        "schema",
        "schema_version",
        "serializer_version",
        "support_model",
        "tensors",
    }
    assert manifest["layout"] == {
        "atom_count": 4,
        "atom_tensor_count": 12,
        "code_dim": 24,
        "num_blocks": 1,
        "projection_count": 6,
        "rank": 2,
        "version": SANA_LAYOUT_VERSION,
    }
    assert manifest["amortizer"] == {
        "architecture_canonical": amortizer.architecture_canonical,
        "architecture_sha256": amortizer.architecture_signature,
    }
    assert manifest["model"] == {"id": "test/sana", "revision": "1" * 40}
    assert manifest["support_model"] == {
        "id": "test/dino",
        "revision": "2" * 40,
    }
    assert set(manifest["tensors"]) == {
        "amortizer_tensor_count",
        "bank_tensor_count",
        "spec_sha256",
        "total_tensor_count",
    }
    assert manifest["tensors"]["bank_tensor_count"] == 12
    assert manifest["tensors"]["total_tensor_count"] == len(keys)
    assert keys == tuple(sorted(keys))
    assert offset_keys == tuple(sorted(offset_keys))
    assert "adapter_bank.transformer_blocks.0.attn1.to_q.atom_down" in keys
    assert "adapter_bank.transformer_blocks.0.attn1.to_q.atom_up" in keys
    assert not any(key.startswith("adapters.") or ".base." in key for key in keys)
    assert tuple(key for key in keys if key.startswith("adapter_bank.")) == tuple(
        sorted(
            f"adapter_bank.{projection}.{atom_name}"
            for projection in bank.layout.projection_names
            for atom_name in ("atom_down", "atom_up")
        )
    )

    expected = _snapshot(bank, amortizer)
    tensor_spec = [
        {
            "dtype": "F32",
            "key": key,
            "shape": list(expected[key].shape),
        }
        for key in sorted(expected)
    ]
    tensor_spec_canonical = json.dumps(
        tensor_spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert (
        manifest["tensors"]["spec_sha256"]
        == hashlib.sha256(tensor_spec_canonical.encode("ascii")).hexdigest()
    )
    with safe_open(first, framework="pt", device="cpu", backend="pread") as handle:
        loaded = tuple((key, handle.get_tensor(key)) for key in handle.keys())
    assert tuple(key for key, _tensor in loaded) == tuple(sorted(expected))
    storage_identities: list[tuple[int, int]] = []
    for key, tensor in loaded:
        assert type(tensor) is Tensor
        assert tensor.device.type == "cpu"
        assert tensor.dtype is expected[key].dtype
        assert tuple(tensor.shape) == tuple(expected[key].shape)
        assert tensor.layout is torch.strided
        assert tensor.is_contiguous()
        assert not tensor.is_inference()
        assert not tensor.requires_grad
        assert bool(torch.isfinite(tensor).all())
        storage = tensor.untyped_storage()
        storage_identities.append((storage.data_ptr(), storage.nbytes()))
    assert len(storage_identities) == len(set(storage_identities))

    mode = stat.S_IMODE(first.stat().st_mode)
    assert mode == 0o600
    assert stat.S_ISREG(first.stat().st_mode)
    assert first.stat().st_nlink == 1
    _assert_snapshot(before, bank, amortizer)
    assert _versions(bank, amortizer) == before_versions
    assert _parameter_identity(bank, amortizer) == identities
    _assert_gradients(gradients, bank, amortizer)
    _assert_frozen_base_contract(frozen, bank)


def test_mixed_fp32_bf16_checkpoint_uses_safetensors_canonical_offset_order(
    tmp_path: Path,
) -> None:
    source_transformer, bank, amortizer = _components(23, bank_dtype=torch.bfloat16)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=bank,
        amortizer=amortizer,
        provenance=_provenance(),
    )
    _, keys, offset_keys = _header(path)
    expected_offset_keys = (
        *sorted(key for key in keys if key.startswith("amortizer.")),
        *sorted(key for key in keys if key.startswith("adapter_bank.")),
    )
    assert offset_keys == expected_offset_keys

    destination_transformer, destination_bank, destination_amortizer = _components(
        29,
        bank_dtype=torch.bfloat16,
    )
    load_trainable_checkpoint(
        path,
        adapter_bank=destination_bank,
        amortizer=destination_amortizer,
        expected_provenance=_provenance(),
        expected_file=identity,
    )
    _assert_snapshot(_snapshot(bank, amortizer), destination_bank, destination_amortizer)
    assert source_transformer is not destination_transformer


def test_checkpoint_bytes_are_deterministic_across_fresh_processes(tmp_path: Path) -> None:
    first = _private_directory(tmp_path, "process-one") / "trainable.safetensors"
    second = _private_directory(tmp_path, "process-two") / "trainable.safetensors"
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import torch
        from diffusers import SanaTransformer2DModel

        from ratemem.adapters.checkpoint import CheckpointProvenance, save_trainable_checkpoint
        from ratemem.adapters.sana_layout import install_sana_dynamic_atoms
        from ratemem.support.amortizer import SupportAmortizer

        torch.manual_seed(23)
        transformer = SanaTransformer2DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=4,
            cross_attention_dim=8,
            caption_channels=8,
            mlp_ratio=1.0,
            sample_size=4,
            patch_size=1,
        ).requires_grad_(False).eval()
        bank = install_sana_dynamic_atoms(
            transformer,
            rank=2,
            atom_count=4,
            expected_blocks=1,
        )
        amortizer = SupportAmortizer(
            support_dim=6,
            description_dim=8,
            hidden_dim=16,
            projection_count=6,
            atom_count=4,
            layers=1,
            heads=4,
        ).eval()
        identity = save_trainable_checkpoint(
            Path(sys.argv[1]),
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=CheckpointProvenance(
                model_id="test/sana",
                model_revision="1" * 40,
                support_model_id="test/dino",
                support_model_revision="2" * 40,
            ),
        )
        print(identity.sha256, identity.byte_count)
        """
    )
    outputs: list[str] = []
    for hash_seed, path in (("1", first), ("987654", second)):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0] == outputs[1]
    assert first.read_bytes() == second.read_bytes()


def test_load_requires_external_identity_and_strictly_restores_state(
    tmp_path: Path,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    expected = _snapshot(source_bank, source_amortizer)

    _, destination_bank, destination_amortizer = _components(29)
    before_versions = _versions(destination_bank, destination_amortizer)
    identities = _parameter_identity(destination_bank, destination_amortizer)
    gradients = _set_distinct_gradients(destination_bank, destination_amortizer)
    frozen = _frozen_base_contract(destination_bank)
    metadata = load_trainable_checkpoint(
        path,
        adapter_bank=destination_bank,
        amortizer=destination_amortizer,
        expected_provenance=_provenance(),
        expected_file=identity,
    )

    assert type(metadata) is TrainableCheckpointMetadata
    assert metadata.provenance == _provenance()
    assert metadata.layout_version == SANA_LAYOUT_VERSION
    assert metadata.rank == 2
    assert metadata.atom_count == 4
    assert metadata.projection_count == 6
    assert metadata.code_dim == 24
    with pytest.raises(ValueError, match="canonical JSON and SHA-256 disagree"):
        replace(metadata, amortizer_architecture_sha256="f" * 64)
    _assert_snapshot(expected, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == tuple(
        version + 1 for version in before_versions
    )
    assert _parameter_identity(destination_bank, destination_amortizer) == identities
    _assert_gradients(gradients, destination_bank, destination_amortizer)
    _assert_frozen_base_contract(frozen, destination_bank)


def test_roundtrip_preserves_joint_adapter_amortizer_outputs_and_every_gradient(
    tmp_path: Path,
) -> None:
    source_transformer, source_bank, source_amortizer = _components(23)
    destination_transformer, destination_bank, destination_amortizer = _components(23)
    with torch.no_grad():
        for parameter in _parameters(destination_bank, destination_amortizer):
            parameter.add_(0.25)
    generator = torch.Generator().manual_seed(101)
    support = torch.randn(1, 3, 6, generator=generator)
    description = torch.randn(1, 8, generator=generator)
    adapter_input = torch.randn(
        1,
        2,
        source_bank.wrappers[0].base.in_features,
        generator=generator,
    )
    expected = _joint_output_and_gradients(
        source_bank,
        source_amortizer,
        support,
        description,
        adapter_input,
    )
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    load_trainable_checkpoint(
        path,
        adapter_bank=destination_bank,
        amortizer=destination_amortizer,
        expected_provenance=_provenance(),
        expected_file=identity,
    )

    actual = _joint_output_and_gradients(
        destination_bank,
        destination_amortizer,
        support,
        description,
        adapter_input,
    )

    _assert_optional_tensors_equal(expected, actual)
    assert source_transformer is not destination_transformer


def test_load_rejects_wrong_file_identity_without_mutation(tmp_path: Path) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)

    with pytest.raises(ValueError, match="sha256"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=replace(identity, sha256="f" * 64),
        )
    _assert_snapshot(before, destination_bank, destination_amortizer)


@pytest.mark.parametrize(
    ("case", "expected_error", "expected_message"),
    [
        ("noncanonical", ValueError, "canonical JSON"),
        ("extra_outer_key", ValueError, "metadata keys"),
        ("extra_manifest_key", ValueError, "unexpected keys"),
        ("bool_for_rank", TypeError, "wrong exact type"),
        ("wrong_schema_version", ValueError, "expected checkpoint contract"),
        ("wrong_model_revision", ValueError, "expected checkpoint contract"),
    ],
)
def test_manifest_schema_is_exact_and_validation_precedes_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    tensors, metadata = _payload(path)
    manifest = json.loads(metadata["ratemem"])
    assert type(manifest) is dict
    if case == "noncanonical":
        metadata["ratemem"] = json.dumps(manifest)
    elif case == "extra_outer_key":
        metadata["unexpected"] = "value"
    elif case == "extra_manifest_key":
        manifest["unexpected"] = "value"
        metadata["ratemem"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    elif case == "bool_for_rank":
        manifest["layout"]["rank"] = True
        metadata["ratemem"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    elif case == "wrong_schema_version":
        manifest["schema_version"] = "2.0.0"
        metadata["ratemem"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    elif case == "wrong_model_revision":
        manifest["model"]["revision"] = "3" * 40
        metadata["ratemem"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    else:  # pragma: no cover - exhaustive test parameter guard
        raise AssertionError(case)
    identity = _write_payload(path, tensors, metadata)
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    monkeypatch.setattr(checkpoint_module, "_materialize_tensor", _forbid_materialization)

    with pytest.raises(expected_error, match=expected_message):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing", "order"),
        ("extra_base", "order"),
        ("wrong_dtype", "dtype"),
        ("wrong_shape", "shape"),
    ],
)
def test_tensor_header_contract_rejects_wrong_keys_shapes_and_dtypes_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_message: str,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    tensors, metadata = _payload(path)
    first_key = sorted(tensors)[0]
    raw_content: bytes | None = None
    if case == "missing":
        del tensors[first_key]
    elif case == "extra_base":
        tensors["adapter_bank.transformer_blocks.0.attn1.to_q.base.weight"] = torch.zeros(1)
    elif case == "wrong_dtype":
        content = path.read_bytes()
        header_length = int.from_bytes(content[:8], "little", signed=False)
        header = json.loads(content[8 : 8 + header_length].rstrip(b" ").decode("utf-8"))
        header[first_key]["dtype"] = "BF16"
        raw_content = _replace_raw_header(
            content,
            json.dumps(header, separators=(",", ":")),
        )
    elif case == "wrong_shape":
        tensors[first_key] = tensors[first_key].reshape(-1)[:-1]
    else:  # pragma: no cover - exhaustive test parameter guard
        raise AssertionError(case)
    identity = (
        _write_payload(path, tensors, metadata)
        if raw_content is None
        else _write_bytes(path, raw_content)
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    monkeypatch.setattr(checkpoint_module, "_materialize_tensor", _forbid_materialization)

    with pytest.raises(ValueError, match=expected_message):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)


def test_nonfinite_tensor_is_rejected_after_materialization_without_mutation(
    tmp_path: Path,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    tensors, metadata = _payload(path)
    first_key = sorted(tensors)[0]
    tensors[first_key].flatten()[0] = float("nan")
    identity = _write_payload(path, tensors, metadata)
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)

    with pytest.raises(ValueError, match="finite"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == before_versions


@pytest.mark.parametrize("case", ["inference", "alias"])
def test_materialized_tensors_must_be_normal_unaliased_cpu_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    real_materialize = checkpoint_module._materialize_tensor
    first_down: Tensor | None = None

    def pathological_materialization(handle: object, key: str) -> Tensor:
        nonlocal first_down
        tensor = real_materialize(handle, key)
        if case == "inference" and first_down is None:
            first_down = tensor
            with torch.inference_mode():
                return tensor.clone()
        if case == "alias" and key.endswith("atom_down"):
            if first_down is None:
                first_down = tensor
            elif tensor.shape == first_down.shape and tensor.dtype is first_down.dtype:
                return first_down
        return tensor

    monkeypatch.setattr(
        checkpoint_module,
        "_materialize_tensor",
        pathological_materialization,
    )
    expected_message = "inference" if case == "inference" else "aliases"
    with pytest.raises(ValueError, match=expected_message):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == before_versions


@pytest.mark.parametrize(
    "case",
    [
        "short",
        "truncated_data",
        "trailing_data",
        "duplicate_header_key",
        "invalid_data_offsets",
        "noncanonical_header",
    ],
)
def test_corrupt_or_truncated_files_fail_before_tensor_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    content = path.read_bytes()
    if case == "short":
        content = content[:4]
    elif case == "truncated_data":
        content = content[:-1]
    elif case == "trailing_data":
        content += b"unexpected"
    elif case == "duplicate_header_key":
        header_length = int.from_bytes(content[:8], "little", signed=False)
        header = json.loads(content[8 : 8 + header_length].rstrip(b" ").decode("utf-8"))
        duplicate_key = next(key for key in header if key != "__metadata__")
        duplicate_entry = json.dumps(header[duplicate_key], separators=(",", ":"))
        normal = json.dumps(header, separators=(",", ":"))
        content = _replace_raw_header(
            content,
            f'{normal[:-1]},"{duplicate_key}":{duplicate_entry}}}',
        )
    elif case == "invalid_data_offsets":
        header_length = int.from_bytes(content[:8], "little", signed=False)
        header = json.loads(content[8 : 8 + header_length].rstrip(b" ").decode("utf-8"))
        tensor_key = next(key for key in header if key != "__metadata__")
        header[tensor_key]["data_offsets"] = [0, 0]
        content = _replace_raw_header(
            content,
            json.dumps(header, separators=(",", ":")),
        )
    elif case == "noncanonical_header":
        header_length = int.from_bytes(content[:8], "little", signed=False)
        header = json.loads(content[8 : 8 + header_length].rstrip(b" ").decode("utf-8"))
        content = _replace_raw_header(content, json.dumps(header))
    else:  # pragma: no cover - exhaustive test parameter guard
        raise AssertionError(case)
    identity = _write_bytes(path, content)
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    monkeypatch.setattr(checkpoint_module, "_materialize_tensor", _forbid_materialization)

    with pytest.raises((ValueError, RuntimeError, SafetensorError)):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)

    with pytest.raises(ValueError, match="byte count"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=replace(identity, byte_count=identity.byte_count + 1),
        )
    _assert_snapshot(before, destination_bank, destination_amortizer)


def test_same_shape_different_head_architecture_is_rejected_before_load(
    tmp_path: Path,
) -> None:
    _, source_bank, source_amortizer = _components(23, heads=4)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29, heads=8)
    before = _snapshot(destination_bank, destination_amortizer)

    with pytest.raises(ValueError, match="amortizer architecture"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )
    _assert_snapshot(before, destination_bank, destination_amortizer)


@contextmanager
def _active_bank(bank: SanaDynamicAdapterBank) -> Iterator[None]:
    with bank.activate(torch.zeros(bank.layout.code_dim)):
        yield


def test_save_and_load_reject_active_bank(tmp_path: Path) -> None:
    _, bank, amortizer = _components(23)
    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    with _active_bank(bank), pytest.raises(RuntimeError, match="active"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )
    assert not path.exists()

    identity = save_trainable_checkpoint(
        path,
        adapter_bank=bank,
        amortizer=amortizer,
        provenance=_provenance(),
    )
    with _active_bank(bank), pytest.raises(RuntimeError, match="active"):
        load_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )


def test_save_is_create_only_and_requires_a_private_existing_parent(
    tmp_path: Path,
) -> None:
    _, bank, amortizer = _components(23)
    missing = tmp_path / "missing" / "trainable.safetensors"
    with pytest.raises(FileNotFoundError):
        save_trainable_checkpoint(
            missing,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )
    assert not missing.parent.exists()

    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    path.write_bytes(b"original")
    path.chmod(0o600)
    with pytest.raises(FileExistsError):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )
    assert path.read_bytes() == b"original"

    permissive = tmp_path / "permissive"
    permissive.mkdir(mode=0o755)
    permissive.chmod(0o755)
    with pytest.raises(PermissionError, match="0700"):
        save_trainable_checkpoint(
            permissive / "trainable.safetensors",
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )


def test_load_rejects_permissive_symlink_and_hardlinked_files(tmp_path: Path) -> None:
    _, bank, amortizer = _components(23)
    first_root = _private_directory(tmp_path, "first")
    path = first_root / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=bank,
        amortizer=amortizer,
        provenance=_provenance(),
    )

    path.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        load_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )
    path.chmod(0o600)

    second_root = _private_directory(tmp_path, "second")
    symlink = second_root / "symlink.safetensors"
    symlink.symlink_to(path)
    with pytest.raises(OSError, match="symlink"):
        load_trainable_checkpoint(
            symlink,
            adapter_bank=bank,
            amortizer=amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    hardlink = second_root / "hardlink.safetensors"
    os.link(path, hardlink)
    with pytest.raises(OSError, match="hard link"):
        load_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )


def test_save_rejects_symlinked_parent_and_never_follows_existing_target(
    tmp_path: Path,
) -> None:
    _, bank, amortizer = _components(23)
    real_root = _private_directory(tmp_path, "real")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(OSError, match="symlink"):
        save_trainable_checkpoint(
            linked_root / "trainable.safetensors",
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    victim = real_root / "victim"
    victim.write_bytes(b"victim")
    victim.chmod(0o600)
    target = real_root / "trainable.safetensors"
    target.symlink_to(victim)
    with pytest.raises(FileExistsError):
        save_trainable_checkpoint(
            target,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )
    assert victim.read_bytes() == b"victim"


@pytest.mark.parametrize(
    "fault_name",
    ["_write_all", "_publish_no_replace", "_fsync_directory"],
)
def test_atomic_save_failure_leaves_no_final_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_name: str,
) -> None:
    _, bank, amortizer = _components(23)
    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    before = _snapshot(bank, amortizer)
    before_versions = _versions(bank, amortizer)
    identities = _parameter_identity(bank, amortizer)
    gradients = _set_distinct_gradients(bank, amortizer)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {fault_name} failure")

    monkeypatch.setattr(checkpoint_module, fault_name, fail)
    with pytest.raises(OSError, match="injected"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()
    assert tuple(root.iterdir()) == ()
    _assert_snapshot(before, bank, amortizer)
    assert _versions(bank, amortizer) == before_versions
    assert _parameter_identity(bank, amortizer) == identities
    _assert_gradients(gradients, bank, amortizer)


def test_save_detects_value_drift_that_bypasses_tensor_version_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bank, amortizer = _components(23)
    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    parameter = next(bank.parameters())
    before_version = parameter._version
    before = parameter.detach().clone()
    real_save = checkpoint_module.save

    def mutate_after_serialization(
        tensors: dict[str, Tensor],
        metadata: dict[str, str],
    ) -> bytes:
        content = real_save(tensors, metadata=metadata)
        parameter.data.add_(1.0)
        assert parameter._version == before_version
        return content

    monkeypatch.setattr(checkpoint_module, "save", mutate_after_serialization)
    with pytest.raises(RuntimeError, match="checkpoint source value changed"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()
    with torch.no_grad():
        parameter.copy_(before)


def test_load_detects_precommit_value_drift_that_bypasses_version_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    parameter = next(destination_bank.parameters())
    before = _snapshot(destination_bank, destination_amortizer)
    before_version = parameter._version
    real_stage = checkpoint_module._stage_loaded_tensors

    def mutate_after_staging(handle: object, contract: object) -> tuple[Tensor, ...]:
        staged = real_stage(handle, contract)  # type: ignore[arg-type]
        parameter.data.add_(1.0)
        assert parameter._version == before_version
        return staged

    monkeypatch.setattr(checkpoint_module, "_stage_loaded_tensors", mutate_after_staging)
    with pytest.raises(RuntimeError, match="checkpoint destination value changed"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    after = _snapshot(destination_bank, destination_amortizer)
    changed_key = next(iter(before))
    assert torch.equal(after[changed_key], before[changed_key] + 1.0)
    for key in tuple(before)[1:]:
        torch.testing.assert_close(after[key], before[key], rtol=0.0, atol=0.0)
    with torch.no_grad():
        parameter.copy_(before[changed_key])


def test_load_rejects_file_metadata_change_during_materialization_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    real_stage = checkpoint_module._stage_loaded_tensors

    def touch_after_staging(handle: object, contract: object) -> tuple[Tensor, ...]:
        staged = real_stage(handle, contract)  # type: ignore[arg-type]
        metadata = path.stat()
        os.utime(
            path,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
        )
        return staged

    monkeypatch.setattr(checkpoint_module, "_stage_loaded_tensors", touch_after_staging)
    with pytest.raises(OSError, match="changed during validation or materialization"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == before_versions


@pytest.mark.parametrize(
    "mutation",
    [
        "nonfinite",
        "alias",
        "base_trainable",
        "amortizer_not_trainable",
        "amortizer_same_shape_head",
        "inference_parameter",
        "noncontiguous_parameter",
        "wrapper_extra_buffer",
        "base_extra_module",
    ],
)
def test_save_rejects_unhealthy_or_aliased_component_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, bank, amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    if mutation == "nonfinite":
        with torch.no_grad():
            bank.wrappers[0].atom_down.flatten()[0] = float("nan")
    elif mutation == "alias":
        bank.wrappers[1].atom_down = bank.wrappers[0].atom_down
    elif mutation == "base_trainable":
        bank.wrappers[0].base.weight.requires_grad_(True)
    elif mutation == "amortizer_not_trainable":
        next(amortizer.parameters()).requires_grad_(False)
    elif mutation == "amortizer_same_shape_head":
        amortizer.head = nn.Linear(
            amortizer.hidden_dim,
            amortizer.projection_count * amortizer.atom_count,
        )
    elif mutation == "inference_parameter":
        original = bank.wrappers[0].atom_down
        with torch.inference_mode():
            bank.wrappers[0].atom_down = nn.Parameter(torch.empty_like(original))
    elif mutation == "noncontiguous_parameter":
        original = bank.wrappers[0].atom_down
        backing = torch.empty(
            *original.shape[:-1],
            original.shape[-1] * 2,
            dtype=original.dtype,
            device=original.device,
        )
        bank.wrappers[0].atom_down = nn.Parameter(backing[..., ::2])
    elif mutation == "wrapper_extra_buffer":
        bank.wrappers[0].register_buffer("unexpected", torch.zeros(1))
    elif mutation == "base_extra_module":
        bank.wrappers[0].base.add_module("unexpected", nn.Identity())
    else:  # pragma: no cover - exhaustive test parameter guard
        raise AssertionError(mutation)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_save_rejects_nonfinite_frozen_base_without_serializing_it(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    _, bank, amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    with torch.no_grad():
        bank.wrappers[0].base.weight.flatten()[0] = nonfinite

    with pytest.raises(ValueError, match="base.weight must be finite"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()


def test_load_rejects_nonfinite_destination_frozen_base_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    with torch.no_grad():
        destination_bank.wrappers[0].base.weight.flatten()[0] = float("inf")
    before = _snapshot(destination_bank, destination_amortizer)
    monkeypatch.setattr(checkpoint_module, "_materialize_tensor", _forbid_materialization)

    with pytest.raises(ValueError, match="base.weight must be finite"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)


def test_save_rejects_an_unbound_bank_with_canonical_looking_wrappers(
    tmp_path: Path,
) -> None:
    transformer, installed_bank, amortizer = _components(23)
    unbound_bank = SanaDynamicAdapterBank(
        installed_bank.layout,
        installed_bank.wrappers,
    )
    path = _private_directory(tmp_path) / "trainable.safetensors"

    with pytest.raises(RuntimeError, match="canonically installed and bound"):
        save_trainable_checkpoint(
            path,
            adapter_bank=unbound_bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert transformer is not None
    assert not path.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("root_trainable", "trainable parameter inventory"),
        ("root_nonfinite", "transformer.*finite"),
        ("root_parameter_alias", "transformer.*object aliases"),
        ("root_storage_alias", "transformer.*storage aliases"),
        ("root_trainable_buffer", "transformer buffer.*must not require gradients"),
        ("root_training_mode", "transformer modules must remain in eval mode"),
    ],
)
def test_checkpoint_validates_the_complete_bound_transformer_inventory(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    transformer, bank, amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    if mutation == "root_trainable":
        transformer.scale_shift_table.requires_grad_(True)
    elif mutation == "root_nonfinite":
        with torch.no_grad():
            transformer.scale_shift_table.flatten()[0] = float("nan")
    elif mutation == "root_parameter_alias":
        transformer.register_parameter("root_atom_alias", bank.wrappers[0].atom_down)
    elif mutation == "root_storage_alias":
        transformer.register_buffer(
            "root_atom_storage_alias",
            bank.wrappers[0].atom_down.detach(),
        )
    elif mutation == "root_trainable_buffer":
        transformer.register_buffer(
            "unexpected_trainable_buffer",
            torch.ones(1, requires_grad=True),
        )
    elif mutation == "root_training_mode":
        transformer.training = True
    else:  # pragma: no cover - exhaustive test parameter guard
        raise AssertionError(mutation)

    with pytest.raises((ValueError, RuntimeError), match=expected_message):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()


def test_save_holds_the_validated_parent_directory_fd_across_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bank, amortizer = _components(23)
    root = _private_directory(tmp_path)
    displaced = tmp_path / "displaced"
    path = root / "trainable.safetensors"
    real_contract = checkpoint_module._component_contract
    swapped = False

    def replace_parent_once(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        if not swapped:
            swapped = True
            root.rename(displaced)
            root.mkdir(mode=0o700)
            root.chmod(0o700)
        return real_contract(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(checkpoint_module, "_component_contract", replace_parent_once)
    with pytest.raises(OSError, match="checkpoint parent changed"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert tuple(root.iterdir()) == ()
    assert tuple(displaced.iterdir()) == ()


def test_load_holds_the_validated_parent_directory_fd_across_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    root = _private_directory(tmp_path)
    displaced = tmp_path / "displaced"
    path = root / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    content = path.read_bytes()
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    real_contract = checkpoint_module._component_contract
    swapped = False

    def replace_parent_once(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        if not swapped:
            swapped = True
            root.rename(displaced)
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            path.write_bytes(content)
            path.chmod(0o600)
        return real_contract(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(checkpoint_module, "_component_contract", replace_parent_once)
    with pytest.raises(OSError, match="checkpoint parent changed"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == before_versions


def test_load_rolls_back_if_parent_is_replaced_during_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    root = _private_directory(tmp_path)
    displaced = tmp_path / "displaced"
    path = root / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    real_copy = checkpoint_module._copy_checkpoint_value
    swapped = False

    def replace_parent_after_copy(parameter: Tensor, value: Tensor) -> None:
        nonlocal swapped
        real_copy(parameter, value)  # type: ignore[arg-type]
        if not swapped:
            swapped = True
            root.rename(displaced)
            root.mkdir(mode=0o700)
            root.chmod(0o700)

    monkeypatch.setattr(
        checkpoint_module,
        "_copy_checkpoint_value",
        replace_parent_after_copy,
    )
    with pytest.raises(OSError, match="checkpoint parent changed"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == tuple(
        version + 2 for version in before_versions
    )
    assert tuple(root.iterdir()) == ()
    assert (displaced / "trainable.safetensors").read_bytes()


def test_save_rejects_same_inode_same_size_corruption_after_hardlink_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bank, amortizer = _components(23)
    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    real_publish = checkpoint_module._publish_no_replace

    def publish_then_corrupt(*args: object, **kwargs: object) -> None:
        real_publish(*args, **kwargs)  # type: ignore[arg-type]
        if len(args) == 2:
            final_path = args[1]
            assert type(final_path) is type(Path())
            descriptor = os.open(final_path, os.O_RDWR)  # type: ignore[arg-type]
        else:
            directory_fd, _temporary_name, final_name = args
            assert type(directory_fd) is int
            assert type(final_name) is str
            descriptor = os.open(final_name, os.O_RDWR, dir_fd=directory_fd)
        try:
            original = os.pread(descriptor, 1, 16)
            assert len(original) == 1
            os.pwrite(descriptor, bytes([original[0] ^ 0xFF]), 16)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(checkpoint_module, "_publish_no_replace", publish_then_corrupt)
    with pytest.raises(OSError, match="published checkpoint bytes changed"):
        save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=_provenance(),
        )

    assert not path.exists()
    assert tuple(root.iterdir()) == ()


def test_rollback_failure_persistently_poisons_both_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    root = _private_directory(tmp_path)
    path = root / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    gradients = _set_distinct_gradients(destination_bank, destination_amortizer)
    identities = _parameter_identity(destination_bank, destination_amortizer)
    real_copy = checkpoint_module._copy_checkpoint_value
    calls = 0

    def fail_during_copy(parameter: Tensor, value: Tensor) -> None:
        nonlocal calls
        calls += 1
        real_copy(parameter, value)  # type: ignore[arg-type]
        if calls == 3:
            raise RuntimeError("injected copy failure")

    def fail_during_rollback(_parameter: Tensor, _value: Tensor) -> None:
        raise RuntimeError("injected rollback failure")

    monkeypatch.setattr(checkpoint_module, "_copy_checkpoint_value", fail_during_copy)
    monkeypatch.setattr(
        checkpoint_module,
        "_rollback_checkpoint_value",
        fail_during_rollback,
    )
    with pytest.raises(RuntimeError, match="destination state is poisoned"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_gradients(gradients, destination_bank, destination_amortizer)
    assert _parameter_identity(destination_bank, destination_amortizer) == identities
    poison_path = root / "poisoned.safetensors"
    with pytest.raises(RuntimeError, match="poisoned"):
        save_trainable_checkpoint(
            poison_path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            provenance=_provenance(),
        )
    with pytest.raises(RuntimeError, match="poisoned"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )
    reason = checkpoint_module.trainable_checkpoint_poison_reason(
        adapter_bank=destination_bank,
        amortizer=destination_amortizer,
    )
    assert reason is not None and "rollback failed" in reason
    other_transformer, other_bank, other_amortizer = _components(31)
    with pytest.raises(RuntimeError, match="poisoned"):
        save_trainable_checkpoint(
            root / "poisoned-bank.safetensors",
            adapter_bank=destination_bank,
            amortizer=other_amortizer,
            provenance=_provenance(),
        )
    with pytest.raises(RuntimeError, match="poisoned"):
        save_trainable_checkpoint(
            root / "poisoned-amortizer.safetensors",
            adapter_bank=other_bank,
            amortizer=destination_amortizer,
            provenance=_provenance(),
        )
    assert other_transformer is not None
    assert not poison_path.exists()


def test_dead_checkpoint_poison_records_are_removed_without_leaking_registry_entries() -> None:
    with checkpoint_module._POISON_LOCK:
        original_registry = dict(checkpoint_module._POISONED_COMPONENTS)
        checkpoint_module._POISONED_COMPONENTS.clear()
    try:
        transformer, bank, amortizer = _components(37)
        checkpoint_module._mark_checkpoint_poisoned(bank, amortizer, "dead record")
        bank_key = ("bank", id(bank))
        amortizer_key = ("amortizer", id(amortizer))
        with checkpoint_module._POISON_LOCK:
            record = checkpoint_module._POISONED_COMPONENTS[bank_key]
            assert checkpoint_module._POISONED_COMPONENTS[amortizer_key] is record

        del transformer, bank, amortizer
        gc.collect()

        with checkpoint_module._POISON_LOCK:
            assert bank_key not in checkpoint_module._POISONED_COMPONENTS
            assert amortizer_key not in checkpoint_module._POISONED_COMPONENTS
    finally:
        with checkpoint_module._POISON_LOCK:
            checkpoint_module._POISONED_COMPONENTS.clear()
            checkpoint_module._POISONED_COMPONENTS.update(original_registry)


def test_stale_reused_id_cleanup_cannot_unpoison_an_unrelated_live_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with checkpoint_module._POISON_LOCK:
        original_registry = dict(checkpoint_module._POISONED_COMPONENTS)
        checkpoint_module._POISONED_COMPONENTS.clear()
    try:
        stale_transformer, stale_bank, stale_amortizer = _components(41)
        checkpoint_module._mark_checkpoint_poisoned(
            stale_bank, stale_amortizer, "stale record"
        )
        stale_bank_key = ("bank", id(stale_bank))
        stale_amortizer_key = ("amortizer", id(stale_amortizer))
        with checkpoint_module._POISON_LOCK:
            stale_record = checkpoint_module._POISONED_COMPONENTS[stale_bank_key]

        del stale_transformer, stale_bank, stale_amortizer
        gc.collect()
        assert stale_record.bank_ref() is None
        assert stale_record.amortizer_ref() is None

        protected_transformer, protected_bank, protected_amortizer = _components(43)
        checkpoint_module._mark_checkpoint_poisoned(
            protected_bank, protected_amortizer, "protected live component"
        )
        protected_amortizer_key = ("amortizer", id(protected_amortizer))
        del protected_bank
        gc.collect()

        # Reinsert one deliberately stale record under its original key to model a
        # delayed weakref cleanup followed by reuse of exactly that numeric object id.
        with checkpoint_module._POISON_LOCK:
            checkpoint_module._POISONED_COMPONENTS[stale_bank_key] = stale_record

        query_transformer, query_bank, query_amortizer = _components(47)
        real_id = builtins.id

        def reused_id(value: object) -> int:
            if value is query_bank:
                return stale_bank_key[1]
            return real_id(value)

        monkeypatch.setattr(checkpoint_module, "id", reused_id, raising=False)
        assert checkpoint_module.trainable_checkpoint_poison_reason(
            adapter_bank=query_bank,
            amortizer=protected_amortizer,
        ) == "protected live component"

        # The stale lookup may clean only the stale record's original keys. It must
        # not derive a deletion key from the unrelated component used for this query.
        with checkpoint_module._POISON_LOCK:
            assert checkpoint_module._POISONED_COMPONENTS[protected_amortizer_key].reason == (
                "protected live component"
            )
        other_transformer, other_bank, _other_amortizer = _components(53)
        assert checkpoint_module.trainable_checkpoint_poison_reason(
            adapter_bank=other_bank,
            amortizer=protected_amortizer,
        ) == "protected live component"
        assert protected_transformer is not None
        assert query_transformer is not None
        assert query_amortizer is not None
        assert other_transformer is not None
        assert stale_amortizer_key != protected_amortizer_key
    finally:
        with checkpoint_module._POISON_LOCK:
            checkpoint_module._POISONED_COMPONENTS.clear()
            checkpoint_module._POISONED_COMPONENTS.update(original_registry)


@pytest.mark.parametrize("failure_index", [3, 14])
def test_joint_load_rolls_back_bank_and_amortizer_when_a_copy_mutates_then_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    identities = _parameter_identity(destination_bank, destination_amortizer)
    gradients = _set_distinct_gradients(destination_bank, destination_amortizer)
    real_copy = checkpoint_module._copy_checkpoint_value
    calls = 0

    def mutate_then_fail(parameter: Tensor, value: Tensor) -> None:
        nonlocal calls
        calls += 1
        real_copy(parameter, value)  # type: ignore[arg-type]
        if calls == failure_index:
            raise RuntimeError("injected copy failure")

    monkeypatch.setattr(checkpoint_module, "_copy_checkpoint_value", mutate_then_fail)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _parameter_identity(destination_bank, destination_amortizer) == identities
    _assert_gradients(gradients, destination_bank, destination_amortizer)
    after_versions = _versions(destination_bank, destination_amortizer)
    assert after_versions[:failure_index] == tuple(
        version + 2 for version in before_versions[:failure_index]
    )
    assert after_versions[failure_index:] == before_versions[failure_index:]


def test_materialization_failure_is_precommit_and_does_not_touch_versions_or_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    identities = _parameter_identity(destination_bank, destination_amortizer)
    gradients = _set_distinct_gradients(destination_bank, destination_amortizer)
    real_materialize = checkpoint_module._materialize_tensor
    calls = 0

    def fail_during_materialization(handle: object, key: str) -> Tensor:
        nonlocal calls
        calls += 1
        tensor = real_materialize(handle, key)
        if calls == 3:
            raise RuntimeError("injected materialization failure")
        return tensor

    monkeypatch.setattr(
        checkpoint_module,
        "_materialize_tensor",
        fail_during_materialization,
    )
    with pytest.raises(RuntimeError, match="injected materialization failure"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == before_versions
    assert _parameter_identity(destination_bank, destination_amortizer) == identities
    _assert_gradients(gradients, destination_bank, destination_amortizer)


def test_postcommit_validation_failure_rolls_back_every_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source_bank, source_amortizer = _components(23)
    path = _private_directory(tmp_path) / "trainable.safetensors"
    identity = save_trainable_checkpoint(
        path,
        adapter_bank=source_bank,
        amortizer=source_amortizer,
        provenance=_provenance(),
    )
    _, destination_bank, destination_amortizer = _components(29)
    before = _snapshot(destination_bank, destination_amortizer)
    before_versions = _versions(destination_bank, destination_amortizer)
    real_validate = checkpoint_module._assert_same_topology
    calls = 0

    def fail_once(
        before_contract: object,
        after_contract: object,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected postcommit failure")
        real_validate(before_contract, after_contract)  # type: ignore[arg-type]

    monkeypatch.setattr(checkpoint_module, "_assert_same_topology", fail_once)
    with pytest.raises(RuntimeError, match="injected postcommit failure"):
        load_trainable_checkpoint(
            path,
            adapter_bank=destination_bank,
            amortizer=destination_amortizer,
            expected_provenance=_provenance(),
            expected_file=identity,
        )

    _assert_snapshot(before, destination_bank, destination_amortizer)
    assert _versions(destination_bank, destination_amortizer) == tuple(
        version + 2 for version in before_versions
    )


def test_production_metadata_constants_are_not_weakened() -> None:
    architecture = SupportAmortizer(
        support_dim=384,
        description_dim=2304,
        hidden_dim=256,
        projection_count=120,
        atom_count=4,
        layers=2,
        heads=8,
    )
    assert architecture.architecture_canonical == (
        '{"atom_count":4,"description_dim":2304,"heads":8,"hidden_dim":256,'
        '"layers":2,"projection_count":120,'
        '"schema_version":"ratemem-support-amortizer-v1","support_dim":384}'
    )
    assert architecture.architecture_signature == (
        "b48d5f323a80803196ebacec91aea6c381399396639ea26b20c2f0b044bd9c9c"
    )
    assert len(tuple(architecture.named_parameters())) == 38
    assert sum(parameter.numel() for parameter in architecture.parameters()) == 2_130_264
    layout = SanaAdapterLayout(
        num_blocks=PRODUCTION_BLOCK_COUNT,
        atom_count=PRODUCTION_ATOM_COUNT,
    )
    assert layout.projection_count == 120
    assert layout.code_dim == 480
    assert layout.atom_tensor_count == 240
    assert (
        layout.trainable_parameter_count(
            width=PRODUCTION_WIDTH,
            rank=PRODUCTION_RANK,
        )
        == 8_601_600
    )
    assert layout.atom_tensor_count + len(tuple(architecture.named_parameters())) == 278
    total_trainable_values = 8_601_600 + sum(
        parameter.numel() for parameter in architecture.parameters()
    )
    assert total_trainable_values == 10_731_864
