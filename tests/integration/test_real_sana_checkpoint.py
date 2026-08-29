from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from diffusers import (
    AutoencoderDC,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from torch import nn
from transformers import BitImageProcessor, Dinov2Model, Gemma2Model, GemmaTokenizer

from ratemem.adapters.sana_layout import (
    install_sana_dynamic_atoms,
    validate_production_sana_layout,
)
from ratemem.pilot.config import SanaPilotConfig
from ratemem.sana.components import (
    hydrate_pinned_snapshots,
    load_pinned_components,
)

CONFIG_PATH = Path("configs/pilot/sana-1.5-1.6b.json")


def _assert_exact_attributes(value: object, expected: dict[str, object]) -> None:
    for name, expected_value in expected.items():
        actual = getattr(value, name)
        assert type(actual) is type(expected_value)
        assert actual == expected_value


def _assert_frozen_eval_placement(
    module: nn.Module, *, dtype: torch.dtype
) -> None:
    assert all(not child.training for child in module.modules())
    assert all(not parameter.requires_grad for parameter in module.parameters())
    tensors = (*module.parameters(), *module.buffers())
    assert tensors
    assert all(tensor.device.type == "cuda" for tensor in tensors)
    assert all(
        not tensor.is_floating_point() or tensor.dtype is dtype for tensor in tensors
    )


@pytest.mark.cuda
@pytest.mark.real_sana
@pytest.mark.paid_modal
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("RATEMEM_RUN_REAL_SANA") != "1",
    reason="explicit paid CUDA real-SANA opt-in is required",
)
def test_fixed_real_sana_checkpoint_loads_offline_then_accepts_exact_atoms() -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    cache_dir = Path(os.environ.get("RATEMEM_REAL_SANA_CACHE", "/cache/huggingface"))
    snapshots = hydrate_pinned_snapshots(config, cache_dir=cache_dir)
    bundle = load_pinned_components(
        config,
        snapshots=snapshots,
        device=torch.device("cuda"),
    )

    assert type(bundle.transformer) is SanaTransformer2DModel
    assert type(bundle.vae) is AutoencoderDC
    assert type(bundle.tokenizer) is GemmaTokenizer
    assert type(bundle.text_encoder) is Gemma2Model
    assert type(bundle.training_scheduler) is FlowMatchEulerDiscreteScheduler
    assert type(bundle.inference_scheduler) is DPMSolverMultistepScheduler
    assert type(bundle.support_processor) is BitImageProcessor
    assert type(bundle.support_encoder) is Dinov2Model

    _assert_exact_attributes(
        bundle.transformer.config,
        {
            "_class_name": "SanaTransformer2DModel",
            "sample_size": 32,
            "patch_size": 1,
            "num_layers": 20,
            "num_attention_heads": 70,
            "attention_head_dim": 32,
            "num_cross_attention_heads": 20,
            "cross_attention_head_dim": 112,
            "cross_attention_dim": 2240,
            "caption_channels": 2304,
            "in_channels": 32,
            "out_channels": 32,
            "attention_bias": False,
            "qk_norm": "rms_norm_across_heads",
            "mlp_ratio": 2.5,
            "dropout": 0.0,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "interpolation_scale": None,
            "guidance_embeds": False,
            "guidance_embeds_scale": 0.1,
            "timestep_scale": 1.0,
        },
    )
    _assert_exact_attributes(
        bundle.vae.config,
        {
            "_class_name": "AutoencoderDC",
            "in_channels": 3,
            "latent_channels": 32,
            "attention_head_dim": 32,
            "encoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "decoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            "encoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "encoder_layers_per_block": [2, 2, 2, 3, 3, 3],
            "decoder_layers_per_block": [3, 3, 3, 3, 3, 3],
            "encoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
            "decoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
            "upsample_block_type": "interpolate",
            "downsample_block_type": "Conv",
            "decoder_norm_types": "rms_norm",
            "decoder_act_fns": "silu",
            "encoder_out_shortcut": True,
            "decoder_in_shortcut": True,
            "decoder_conv_act_fn": "relu",
            "scaling_factor": 0.41407,
        },
    )
    _assert_exact_attributes(
        bundle.tokenizer,
        {
            "vocab_size": 256000,
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
            "unk_token_id": 3,
            "padding_side": "left",
        },
    )
    _assert_exact_attributes(
        bundle.text_encoder.config,
        {"architectures": ["Gemma2Model"], "hidden_size": 2304},
    )
    _assert_exact_attributes(
        bundle.support_encoder.config,
        {"architectures": ["Dinov2Model"], "hidden_size": 384},
    )
    _assert_exact_attributes(
        bundle.inference_scheduler.config,
        {
            "_class_name": "DPMSolverMultistepScheduler",
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "trained_betas": None,
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
            "solver_type": "midpoint",
            "prediction_type": "flow_prediction",
            "thresholding": False,
            "dynamic_thresholding_ratio": 0.995,
            "sample_max_value": 1.0,
            "lower_order_final": True,
            "euler_at_final": False,
            "flow_shift": 3.0,
            "use_flow_sigmas": True,
            "final_sigmas_type": "zero",
            "timestep_spacing": "linspace",
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
            "use_lu_lambdas": False,
            "use_dynamic_shifting": False,
            "time_shift_type": "exponential",
            "variance_type": None,
            "steps_offset": 0,
            "rescale_betas_zero_snr": False,
            "lambda_min_clipped": float("-inf"),
        },
    )
    _assert_exact_attributes(
        bundle.training_scheduler.config,
        {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "use_dynamic_shifting": False,
        },
    )
    processor = bundle.support_processor
    assert processor.crop_size.height == processor.crop_size.width == 224
    assert processor.size.shortest_edge == 256
    _assert_exact_attributes(
        processor,
        {
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": (0.485, 0.456, 0.406),
            "image_std": (0.229, 0.224, 0.225),
            "image_processor_type": "BitImageProcessor",
            "resample": 3,
            "rescale_factor": 1 / 255,
        },
    )

    for module, dtype in (
        (bundle.transformer, torch.bfloat16),
        (bundle.text_encoder, torch.bfloat16),
        (bundle.vae, torch.float32),
        (bundle.support_encoder, torch.float32),
    ):
        _assert_frozen_eval_placement(module, dtype=dtype)

    canonical_sigmas = torch.linspace(1, 1000, 1000, dtype=torch.float32).flip(0) / 1000
    canonical_timesteps = canonical_sigmas * 1000
    assert bundle.training_timesteps == tuple(
        float(value) for value in canonical_timesteps
    )
    assert bundle.training_sigmas == tuple(float(value) for value in canonical_sigmas)

    pipeline = bundle.inference_pipeline()
    assert type(pipeline) is SanaPipeline
    assert pipeline.tokenizer is bundle.tokenizer
    assert pipeline.text_encoder is bundle.text_encoder
    assert pipeline.vae is bundle.vae
    assert pipeline.transformer is bundle.transformer
    assert pipeline.scheduler is bundle.inference_scheduler

    layout = validate_production_sana_layout(
        bundle.transformer,
        rank=config.rank,
        atom_count=config.atom_count,
    )
    bank = install_sana_dynamic_atoms(
        bundle.transformer,
        rank=config.rank,
        atom_count=config.atom_count,
        expected_blocks=config.num_blocks,
    )
    atom_parameters = tuple(bank.parameters())
    assert layout.projection_count == len(bank.wrappers) == 120
    assert layout.atom_tensor_count == len(atom_parameters) == 240
    assert len({id(parameter) for parameter in atom_parameters}) == 240
    assert sum(parameter.numel() for parameter in atom_parameters) == 8_601_600
    assert all(
        parameter.device.type == "cuda" and parameter.dtype is torch.bfloat16
        for parameter in atom_parameters
    )
    trainable_ids = {
        id(parameter)
        for parameter in bundle.transformer.parameters()
        if parameter.requires_grad
    }
    assert trainable_ids == {id(parameter) for parameter in atom_parameters}
