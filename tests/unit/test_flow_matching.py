from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel

from ratemem.adapters.sana_layout import install_sana_dynamic_atoms
from ratemem.pilot.data import PilotCacheReceipt, PrecomputedPilotData
from ratemem.sana.flow import (
    FlowBatch,
    FlowDraw,
    OneTimestepFlowTrainer,
    flow_interpolate,
    flow_target,
    sigma_for_timesteps,
)
from ratemem.support.amortizer import SupportAmortizer


def _schedule() -> tuple[tuple[float, ...], tuple[float, ...]]:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    return (
        tuple(float(value) for value in scheduler.timesteps),
        tuple(float(value) for value in scheduler.sigmas),
    )


def _tiny_sana() -> SanaTransformer2DModel:
    model = SanaTransformer2DModel(
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
        qk_norm=None,
    )
    model.requires_grad_(False)
    model.eval()
    return model


def _trainer() -> OneTimestepFlowTrainer:
    transformer = _tiny_sana()
    bank = install_sana_dynamic_atoms(
        transformer,
        rank=2,
        atom_count=4,
        expected_blocks=1,
    )
    transformer.enable_gradient_checkpointing()
    amortizer = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=16,
        projection_count=6,
        atom_count=4,
        layers=1,
        heads=4,
    ).train()
    optimizer = torch.optim.AdamW(
        [*bank.parameters(), *amortizer.parameters()],
        lr=1e-3,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    timesteps, sigmas = _schedule()
    return OneTimestepFlowTrainer(
        transformer,
        bank,
        amortizer,
        timesteps,
        sigmas,
        optimizer,
        expected_amortizer_signature=amortizer.architecture_signature,
    )


def _batch(batch_size: int = 2) -> FlowBatch:
    return FlowBatch(
        clean_latents=torch.randn(batch_size, 4, 4, 4),
        prompt_embeddings=torch.randn(batch_size, 3, 8),
        prompt_attention_mask=torch.tensor(
            [[1, 1, 0], [1, 1, 1]], dtype=torch.int64
        )[:batch_size],
        support_features=torch.randn(batch_size, 2, 6),
        support_mask=torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool
        )[:batch_size],
        description_features=torch.randn(batch_size, 8),
    )


def _production_cache_tensors() -> dict[str, torch.Tensor]:
    prompt_mask = torch.zeros((8, 300), dtype=torch.int64)
    prompt_mask[:, :17] = 1
    return {
        "clean_latents": torch.zeros((8, 32, 32, 32), dtype=torch.float32),
        "prompt_embeddings": torch.zeros((8, 300, 2304), dtype=torch.float32),
        "prompt_attention_mask": prompt_mask,
        "support_features": torch.zeros((8, 1, 384), dtype=torch.float32),
        "support_mask": torch.ones((8, 1), dtype=torch.bool),
        "description_features": torch.zeros((8, 2304), dtype=torch.float32),
    }


def _cache(tensors: dict[str, torch.Tensor]) -> PrecomputedPilotData:
    receipt = PilotCacheReceipt(
        identity_sha256="1" * 64,
        manifest_sha256="2" * 64,
        manifest_byte_count=1,
        features_sha256="3" * 64,
        features_byte_count=1,
    )
    return PrecomputedPilotData(
        root=Path("/unused"),
        tensors=MappingProxyType(tensors),
        manifest={},
        receipt=receipt,
    )


def test_flow_endpoints_target_and_per_example_shapes_are_exact() -> None:
    clean = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noise = torch.tensor([[[[5.0]]], [[[7.0]]]])
    sigma = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1)
    actual = flow_interpolate(clean, noise, sigma)
    torch.testing.assert_close(actual[0], clean[0])
    torch.testing.assert_close(actual[1], noise[1])
    torch.testing.assert_close(flow_target(clean, noise), noise - clean)
    assert actual.dtype == torch.float32 and actual.shape == clean.shape


def test_sigma_lookup_preserves_batch_order_and_requires_unique_exact_matches() -> None:
    schedule_timesteps = torch.tensor([1000.0, 500.0, 1.0])
    schedule_sigmas = torch.tensor([1.0, 0.5, 0.001])
    selected = torch.tensor([1.0, 1000.0])
    sigma = sigma_for_timesteps(
        selected,
        schedule_timesteps,
        schedule_sigmas,
        n_dim=4,
    )
    torch.testing.assert_close(
        sigma[:, 0, 0, 0], torch.tensor([0.001, 1.0])
    )
    with pytest.raises(ValueError, match="exactly once"):
        sigma_for_timesteps(
            torch.tensor([2.0]),
            schedule_timesteps,
            schedule_sigmas,
            n_dim=4,
        )
    with pytest.raises(ValueError, match="exactly once"):
        sigma_for_timesteps(
            torch.tensor([1.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.1, 0.2]),
            n_dim=4,
        )


def test_flow_matches_official_flowmatch_scheduler_scale_noise() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    clean = torch.randn(3, 4, 2, 2)
    noise = torch.randn_like(clean)
    indices = torch.tensor([0, 499, 999])
    timesteps = scheduler.timesteps[indices]
    sigmas = scheduler.sigmas[indices].reshape(3, 1, 1, 1)
    expected = scheduler.scale_noise(clean, timesteps, noise)
    torch.testing.assert_close(
        flow_interpolate(clean, noise, sigmas),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(flow_target(clean, noise), noise - clean)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("clean_latents", torch.zeros(2, 4, 4, 4, dtype=torch.float64), "float32"),
        ("prompt_attention_mask", torch.ones(2, 3, dtype=torch.bool), "int64"),
        ("support_mask", torch.ones(2, 2, dtype=torch.int64), "bool"),
        ("description_features", torch.zeros(1, 8), "batch"),
        (
            "prompt_attention_mask",
            torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.int64),
            "right-padded",
        ),
        (
            "support_mask",
            torch.tensor([[False, False], [True, True]]),
            "valid support",
        ),
    ],
)
def test_flow_batch_fails_closed_on_dtype_shape_and_mask_contracts(
    field: str,
    replacement: torch.Tensor,
    error: str,
) -> None:
    trainer = _trainer()
    invalid = replace(_batch(), **{field: replacement})
    with pytest.raises((TypeError, ValueError), match=error):
        invalid.validate(trainer.transformer, trainer.amortizer)


def test_flow_batch_rejects_nonfinite_values_except_masked_support_padding() -> None:
    trainer = _trainer()
    batch = _batch()
    support = batch.support_features.clone()
    support[0, 1] = torch.nan
    replace(batch, support_features=support).validate(
        trainer.transformer, trainer.amortizer
    )

    valid_support = support.clone()
    valid_support[1, 1] = torch.inf
    with pytest.raises(ValueError, match="valid support"):
        replace(batch, support_features=valid_support).validate(
            trainer.transformer, trainer.amortizer
        )
    with pytest.raises(ValueError, match="prompt embeddings"):
        replace(
            batch,
            prompt_embeddings=torch.full_like(batch.prompt_embeddings, torch.nan),
        ).validate(trainer.transformer, trainer.amortizer)


def test_from_cache_requires_exact_six_key_order_and_makes_normal_device_tensors() -> None:
    tensors = _production_cache_tensors()
    cache = _cache(tensors)
    with torch.inference_mode():
        batch = FlowBatch.from_cache(cache, device=torch.device("cpu"))
    assert tuple(batch.as_mapping()) == (
        "clean_latents",
        "prompt_embeddings",
        "prompt_attention_mask",
        "support_features",
        "support_mask",
        "description_features",
    )
    assert all(
        tensor.device.type == "cpu"
        and not tensor.requires_grad
        and not tensor.is_inference()
        for tensor in batch.as_mapping().values()
    )
    for name, tensor in tensors.items():
        assert batch.as_mapping()[name] is not tensor

    reordered = {name: tensors[name] for name in reversed(tuple(tensors))}
    bad_cache = replace(cache, tensors=MappingProxyType(reordered))
    with pytest.raises(ValueError, match="keys and order"):
        FlowBatch.from_cache(bad_cache, device=torch.device("cpu"))


def test_from_cache_can_select_exact_rows_without_aliasing_source() -> None:
    tensors = _production_cache_tensors()
    tensors["clean_latents"][:, 0, 0, 0] = torch.arange(8)
    cache = _cache(tensors)
    batch = FlowBatch.from_cache(
        cache,
        device=torch.device("cpu"),
        row_indices=(7, 2, 7),
    )
    assert batch.clean_latents.shape[0] == 3
    torch.testing.assert_close(
        batch.clean_latents[:, 0, 0, 0], torch.tensor([7.0, 2.0, 7.0])
    )
    batch.clean_latents[0, 0, 0, 0] = -1
    assert tensors["clean_latents"][7, 0, 0, 0] == 7


def test_from_cache_rejects_noncontiguous_source_before_normalizing_it() -> None:
    tensors = _production_cache_tensors()
    tensors["description_features"] = (
        torch.zeros((8, 2304, 2), dtype=torch.float32)[:, :, 0]
    )
    assert not tensors["description_features"].is_contiguous()
    cache = _cache(tensors)
    with pytest.raises(ValueError, match="contiguous"):
        FlowBatch.from_cache(cache, device=torch.device("cpu"))


def test_flow_draw_is_frozen_and_validates_exact_batch_shape_dtype_and_indices() -> None:
    batch = _batch()
    draw = FlowDraw(
        noise=torch.randn_like(batch.clean_latents),
        timestep_indices=torch.tensor([0, 999], dtype=torch.int64),
    )
    draw.validate(batch, schedule_length=1000)
    with pytest.raises(FrozenInstanceError):
        draw.noise = torch.zeros_like(draw.noise)  # type: ignore[misc]
    with pytest.raises(ValueError, match="range"):
        replace(
            draw,
            timestep_indices=torch.tensor([0, 1000], dtype=torch.int64),
        ).validate(batch, schedule_length=1000)
    with pytest.raises(TypeError, match="float32"):
        replace(draw, noise=draw.noise.double()).validate(
            batch, schedule_length=1000
        )


def test_constructor_caches_the_exact_schedule_once_and_rejects_nearby_values() -> None:
    trainer = _trainer()
    first = trainer.schedule_tensors(torch.device("cpu"))
    second = trainer.schedule_tensors(torch.device("cpu"))
    assert first[0] is second[0] and first[1] is second[1]
    assert first[0].dtype == first[1].dtype == torch.float32
    assert first[0].shape == first[1].shape == (1000,)
    assert tuple(float(value) for value in first[0]) == trainer.training_timesteps
    assert tuple(float(value) for value in first[1]) == trainer.training_sigmas

    timesteps = list(trainer.training_timesteps)
    timesteps[-1] = 1.000001
    with pytest.raises(ValueError, match="canonical index 999"):
        OneTimestepFlowTrainer(
            trainer.transformer,
            trainer.adapter_bank,
            trainer.amortizer,
            tuple(timesteps),
            trainer.training_sigmas,
            trainer.optimizer,
            expected_amortizer_signature=trainer.amortizer.architecture_signature,
        )


def test_cached_schedule_data_mutation_is_detected_before_a_flow_side_effect() -> None:
    trainer = _trainer()
    timesteps, _sigmas = trainer.schedule_tensors(torch.device("cpu"))
    timesteps.data[-1] = 2.0
    with pytest.raises(RuntimeError, match="cached training timesteps changed"):
        trainer.train_step(
            _batch(),
            generator=torch.Generator().manual_seed(73),
        )


def test_flow_source_has_no_schedule_regeneration_or_scheduler_dependency() -> None:
    source_path = Path("src/ratemem/sana/flow.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_calls = {"linspace", "arange", "set_timesteps"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            imported = [alias.name for alias in node.names]
            assert not any("scheduler" in name.lower() for name in imported)
            if isinstance(node, ast.ImportFrom):
                assert "scheduler" not in (node.module or "").lower()
        if isinstance(node, ast.Call):
            name: str | None = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            assert name not in forbidden_calls


@pytest.mark.parametrize(
    "bad_value",
    [None, [], (1.0,), tuple(float(index) for index in range(1000))],
)
def test_schedule_constructor_requires_exact_canonical_1000_float_tuples(
    bad_value: Any,
) -> None:
    trainer = _trainer()
    with pytest.raises((TypeError, ValueError)):
        OneTimestepFlowTrainer(
            trainer.transformer,
            trainer.adapter_bank,
            trainer.amortizer,
            bad_value,
            trainer.training_sigmas,
            trainer.optimizer,
            expected_amortizer_signature=trainer.amortizer.architecture_signature,
        )
