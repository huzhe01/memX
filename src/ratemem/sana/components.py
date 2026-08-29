from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import torch
from diffusers import (
    AutoencoderDC,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from huggingface_hub import snapshot_download
from torch import Tensor, nn
from transformers import BitImageProcessor, Dinov2Model, Gemma2Model, GemmaTokenizer
from transformers.image_utils import SizeDict

from ratemem.pilot.config import SanaPilotConfig

SANA_FILES: Final[tuple[str, ...]] = (
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
DINO_FILES: Final[tuple[str, ...]] = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
)
SANA_CONTROL_FILE_SHA256: Final[tuple[tuple[str, str], ...]] = (
    (
        "scheduler/scheduler_config.json",
        "f9256042828841b26561487c7e0c33fff8717e98ac0fef5c1f6d05bfdd66e908",
    ),
    (
        "transformer/config.json",
        "70863bf60b87cbeab5780c9827ffc5b880cd1ec9ce22bf033409b7e257e8fc68",
    ),
    (
        "vae/config.json",
        "ba6f3d3e44d75d44fdd3760097c069173b5b925e6d14604d5d3582628d09cca6",
    ),
    (
        "text_encoder/config.json",
        "733f241a6692770dfba10383e2c5a56a4f88b320732d9ee8fa16118737eca84d",
    ),
    (
        "text_encoder/model.safetensors.index.json",
        "92764588f700e36874c52f9f05bba143857e5069fc69b14450f907a1cdf879ed",
    ),
    (
        "tokenizer/tokenizer_config.json",
        "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89",
    ),
    (
        "tokenizer/tokenizer.json",
        "5f7eee611703c5ce5d1eee32d9cdcfe465647b8aff0c1dfb3bed7ad7dbb05060",
    ),
)
DINO_CONTROL_FILE_SHA256: Final[tuple[tuple[str, str], ...]] = (
    (
        "config.json",
        "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    ),
    (
        "preprocessor_config.json",
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    ),
)

_TRANSFORMER_CONFIG: Final[dict[str, object]] = {
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
}
_VAE_CONFIG: Final[dict[str, object]] = {
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
}
_TEXT_CONFIG: Final[dict[str, object]] = {
    "architectures": ["Gemma2Model"],
    "hidden_size": 2304,
}
_TOKENIZER_CONFIG: Final[dict[str, object]] = {
    "vocab_size": 256000,
    "add_bos_token": True,
    "add_eos_token": False,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "pad_token_id": 0,
    "unk_token_id": 3,
    "padding_side": "left",
}
_DINO_CONFIG: Final[dict[str, object]] = {
    "architectures": ["Dinov2Model"],
    "hidden_size": 384,
}
_PROCESSOR_CONFIG: Final[dict[str, object]] = {
    "crop_size": SizeDict(height=224, width=224),
    "size": SizeDict(shortest_edge=256),
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
}
_INFERENCE_SCHEDULER_CONFIG: Final[dict[str, object]] = {
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
}
_TRAINING_SCHEDULER_CONFIG: Final[dict[str, object]] = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": False,
}

class _Configured(Protocol):
    config: object


class _TrainingSchedulerState(Protocol):
    config: object
    timesteps: Tensor
    sigmas: Tensor


@dataclass(frozen=True, slots=True)
class PinnedSnapshotPaths:
    sana: Path
    dino: Path

    def __post_init__(self) -> None:
        if not isinstance(self.sana, Path) or not isinstance(self.dino, Path):
            raise TypeError("snapshot paths must be pathlib.Path values")


def _require_exact_config(config: SanaPilotConfig) -> None:
    if type(config) is not SanaPilotConfig:
        raise TypeError("config must be an exact SanaPilotConfig")
    config.validate()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_snapshot(
    snapshot: Path,
    *,
    revision: str,
    files: tuple[str, ...],
    control_hashes: tuple[tuple[str, str], ...],
    label: str,
) -> Path:
    try:
        resolved = snapshot.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{label} snapshot directory is missing: {snapshot}"
        ) from error
    if resolved.name != revision:
        raise ValueError(
            f"{label} snapshot revision basename must be exact: {revision}"
        )
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} snapshot directory is missing: {resolved}")
    missing = [relative for relative in files if not (resolved / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} snapshot files are missing: {missing}")
    for relative, expected_hash in control_hashes:
        actual_hash = _sha256_file(resolved / relative)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{label} snapshot control-file SHA-256 changed: {relative}"
            )
    return resolved


def _validate_snapshot_paths(
    config: SanaPilotConfig, snapshots: PinnedSnapshotPaths
) -> PinnedSnapshotPaths:
    if type(snapshots) is not PinnedSnapshotPaths:
        raise TypeError("snapshots must be an exact PinnedSnapshotPaths")
    sana = _validate_snapshot(
        snapshots.sana,
        revision=config.revision,
        files=SANA_FILES,
        control_hashes=SANA_CONTROL_FILE_SHA256,
        label="SANA",
    )
    dino = _validate_snapshot(
        snapshots.dino,
        revision=config.support_revision,
        files=DINO_FILES,
        control_hashes=DINO_CONTROL_FILE_SHA256,
        label="DINO",
    )
    return PinnedSnapshotPaths(sana=sana, dino=dino)


def hydrate_pinned_snapshots(
    config: SanaPilotConfig, *, cache_dir: Path
) -> PinnedSnapshotPaths:
    """The sole network-enabled boundary for the fixed Hub snapshots."""

    _require_exact_config(config)
    sana = _validate_snapshot(
        Path(
            snapshot_download(
                repo_id=config.model_id,
                repo_type="model",
                revision=config.revision,
                cache_dir=cache_dir,
                allow_patterns=cast(list[str], SANA_FILES),
                token=False,
                local_files_only=False,
                force_download=False,
            )
        ),
        revision=config.revision,
        files=SANA_FILES,
        control_hashes=SANA_CONTROL_FILE_SHA256,
        label="SANA",
    )
    dino = _validate_snapshot(
        Path(
            snapshot_download(
                repo_id=config.support_model_id,
                repo_type="model",
                revision=config.support_revision,
                cache_dir=cache_dir,
                allow_patterns=cast(list[str], DINO_FILES),
                token=False,
                local_files_only=False,
                force_download=False,
            )
        ),
        revision=config.support_revision,
        files=DINO_FILES,
        control_hashes=DINO_CONTROL_FILE_SHA256,
        label="DINO",
    )
    return PinnedSnapshotPaths(sana=sana, dino=dino)


def _require_exact_value(actual: object, expected: object, context: str) -> None:
    if type(actual) is not type(expected):
        raise RuntimeError(f"{context} config has the wrong exact type")
    if type(expected) is dict:
        actual_dict = cast(dict[str, object], actual)
        expected_dict = cast(dict[str, object], expected)
        if tuple(actual_dict) != tuple(expected_dict):
            raise RuntimeError(f"{context} config keys or order changed")
        for key, expected_child in expected_dict.items():
            _require_exact_value(actual_dict[key], expected_child, f"{context}.{key}")
        return
    if type(expected) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        if len(actual_list) != len(expected_list):
            raise RuntimeError(f"{context} config list length changed")
        for index, (actual_child, expected_child) in enumerate(
            zip(actual_list, expected_list, strict=True)
        ):
            _require_exact_value(actual_child, expected_child, f"{context}[{index}]")
        return
    if type(expected) is tuple:
        actual_tuple = cast(tuple[object, ...], actual)
        expected_tuple = cast(tuple[object, ...], expected)
        if len(actual_tuple) != len(expected_tuple):
            raise RuntimeError(f"{context} config tuple length changed")
        for index, (actual_child, expected_child) in enumerate(
            zip(actual_tuple, expected_tuple, strict=True)
        ):
            _require_exact_value(actual_child, expected_child, f"{context}[{index}]")
        return
    if type(expected) is SizeDict:
        actual_size = actual
        expected_size = expected
        for field in (
            "height",
            "width",
            "longest_edge",
            "shortest_edge",
            "max_height",
            "max_width",
        ):
            _require_exact_value(
                getattr(actual_size, field),
                getattr(expected_size, field),
                f"{context}.{field}",
            )
        return
    if actual != expected:
        raise RuntimeError(f"{context} config value changed")


def _require_attributes(value: object, expected: dict[str, object], context: str) -> None:
    actual: dict[str, object] = {}
    for name in expected:
        try:
            actual[name] = getattr(value, name)
        except AttributeError as error:
            raise RuntimeError(f"{context} component config is missing {name}") from error
    _require_exact_value(actual, expected, context)


def _freeze_and_move(
    module: nn.Module, *, device: torch.device, dtype: torch.dtype
) -> None:
    module.requires_grad_(False)
    module.eval()
    module.to(device=device, dtype=dtype)


def _module_device(module: nn.Module, context: str) -> torch.device:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise RuntimeError(f"{context} component exposes no placement tensors")
    return tensors[0].device


def _assert_module_contract(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    context: str,
) -> None:
    if any(child.training for child in module.modules()):
        raise RuntimeError(f"{context} component is not recursively in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError(f"{context} component exposes trainable parameters")
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device != device:
            raise RuntimeError(f"{context} component is on the wrong device")
        if tensor.is_floating_point() and tensor.dtype != dtype:
            raise RuntimeError(f"{context} component has the wrong dtype")


def _tensor_values(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, Tensor):
        raise RuntimeError(f"{context} scheduler values must be a Tensor")
    return tuple(float(item) for item in value.detach().cpu().flatten())


def _canonical_training_schedule() -> tuple[tuple[float, ...], tuple[float, ...]]:
    sigmas = torch.linspace(1, 1000, 1000, dtype=torch.float32).flip(0) / 1000
    timesteps = sigmas * 1000
    return (
        tuple(float(value) for value in timesteps),
        tuple(float(value) for value in sigmas),
    )


@dataclass(frozen=True, slots=True)
class PinnedComponents:
    transformer: SanaTransformer2DModel
    vae: AutoencoderDC
    tokenizer: GemmaTokenizer
    text_encoder: Gemma2Model
    training_scheduler: FlowMatchEulerDiscreteScheduler
    inference_scheduler: DPMSolverMultistepScheduler
    support_processor: BitImageProcessor
    support_encoder: Dinov2Model
    training_timesteps: tuple[float, ...]
    training_sigmas: tuple[float, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self) is not PinnedComponents:
            raise TypeError("component bundle must be an exact PinnedComponents")
        expected_types = (
            (self.transformer, SanaTransformer2DModel, "transformer"),
            (self.vae, AutoencoderDC, "VAE"),
            (self.tokenizer, GemmaTokenizer, "tokenizer"),
            (self.text_encoder, Gemma2Model, "text encoder"),
            (
                self.training_scheduler,
                FlowMatchEulerDiscreteScheduler,
                "training scheduler",
            ),
            (
                self.inference_scheduler,
                DPMSolverMultistepScheduler,
                "inference scheduler",
            ),
            (self.support_processor, BitImageProcessor, "support processor"),
            (self.support_encoder, Dinov2Model, "support encoder"),
        )
        for component, expected_type, component_context in expected_types:
            if type(component) is not expected_type:
                raise TypeError(
                    f"{component_context} component must have an exact explicit class"
                )
        if type(self.training_timesteps) is not tuple:
            raise TypeError("training timesteps must be an exact immutable tuple")
        if type(self.training_sigmas) is not tuple:
            raise TypeError("training sigmas must be an exact immutable tuple")
        if any(type(value) is not float for value in self.training_timesteps):
            raise TypeError("every training timestep must be an exact built-in float")
        if any(type(value) is not float for value in self.training_sigmas):
            raise TypeError("every training sigma must be an exact built-in float")

        _require_attributes(
            cast(_Configured, self.transformer).config,
            _TRANSFORMER_CONFIG,
            "transformer",
        )
        _require_attributes(
            cast(_Configured, self.vae).config,
            _VAE_CONFIG,
            "VAE",
        )
        _require_attributes(self.tokenizer, _TOKENIZER_CONFIG, "tokenizer")
        _require_attributes(self.text_encoder.config, _TEXT_CONFIG, "text encoder")
        _require_attributes(self.support_encoder.config, _DINO_CONFIG, "support encoder")
        _require_attributes(self.support_processor, _PROCESSOR_CONFIG, "support processor")
        _require_attributes(
            cast(_Configured, self.inference_scheduler).config,
            _INFERENCE_SCHEDULER_CONFIG,
            "inference scheduler",
        )
        training_scheduler = cast(
            _TrainingSchedulerState, self.training_scheduler
        )
        _require_attributes(
            training_scheduler.config,
            _TRAINING_SCHEDULER_CONFIG,
            "training scheduler",
        )
        scheduler_timesteps = _tensor_values(
            training_scheduler.timesteps, "training timestep"
        )
        scheduler_sigmas = _tensor_values(
            training_scheduler.sigmas, "training sigma"
        )
        for schedule_value, schedule_context in (
            (training_scheduler.timesteps, "training timesteps"),
            (training_scheduler.sigmas, "training sigmas"),
        ):
            if (
                type(schedule_value) is not Tensor
                or schedule_value.dtype is not torch.float32
                or schedule_value.device.type != "cpu"
                or schedule_value.shape != (1000,)
            ):
                raise RuntimeError(
                    f"{schedule_context} must preserve the pinned float32 CPU construction"
                )
        canonical_timesteps, canonical_sigmas = _canonical_training_schedule()
        if (
            len(self.training_timesteps) != 1000
            or len(self.training_sigmas) != 1000
            or len(scheduler_timesteps) != 1000
            or len(scheduler_sigmas) != 1000
        ):
            raise RuntimeError("training scheduler immutable arrays changed")
        for index, (
            timestep,
            sigma,
            scheduler_timestep,
            scheduler_sigma,
            canonical_timestep,
            canonical_sigma,
        ) in enumerate(
            zip(
                self.training_timesteps,
                self.training_sigmas,
                scheduler_timesteps,
                scheduler_sigmas,
                canonical_timesteps,
                canonical_sigmas,
                strict=True,
            )
        ):
            if (
                timestep != canonical_timestep
                or sigma != canonical_sigma
                or scheduler_timestep != canonical_timestep
                or scheduler_sigma != canonical_sigma
            ):
                raise RuntimeError(
                    f"training scheduler changed at canonical index {index}"
                )

        device = _module_device(cast(nn.Module, self.transformer), "transformer")
        for module, dtype, context in (
            (cast(nn.Module, self.transformer), torch.bfloat16, "transformer"),
            (cast(nn.Module, self.text_encoder), torch.bfloat16, "text encoder"),
            (cast(nn.Module, self.vae), torch.float32, "VAE"),
            (cast(nn.Module, self.support_encoder), torch.float32, "support encoder"),
        ):
            _assert_module_contract(
                module,
                device=device,
                dtype=dtype,
                context=context,
            )

    def inference_pipeline(self) -> SanaPipeline:
        self.validate()
        pipeline = SanaPipeline(  # type: ignore[no-untyped-call]
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.inference_scheduler,
        )
        if type(pipeline) is not SanaPipeline:
            raise TypeError("inference pipeline must be an exact SanaPipeline")
        for name, expected in (
            ("tokenizer", self.tokenizer),
            ("text_encoder", self.text_encoder),
            ("vae", self.vae),
            ("transformer", self.transformer),
            ("scheduler", self.inference_scheduler),
        ):
            if getattr(pipeline, name) is not expected:
                raise RuntimeError(f"inference pipeline replaced the {name} component")
        return pipeline


def load_pinned_components(
    config: SanaPilotConfig,
    *,
    snapshots: PinnedSnapshotPaths,
    device: torch.device,
) -> PinnedComponents:
    """Load only previously hydrated, fully verified local snapshots."""

    _require_exact_config(config)
    snapshots = _validate_snapshot_paths(config, snapshots)

    inference_scheduler = DPMSolverMultistepScheduler.from_pretrained(  # type: ignore[no-untyped-call]
        snapshots.sana,
        subfolder="scheduler",
        local_files_only=True,
        token=False,
        force_download=False,
    )
    transformer = SanaTransformer2DModel.from_pretrained(  # type: ignore[no-untyped-call]
        snapshots.sana,
        subfolder="transformer",
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.bfloat16,
        use_safetensors=True,
    )
    vae = AutoencoderDC.from_pretrained(  # type: ignore[no-untyped-call]
        snapshots.sana,
        subfolder="vae",
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.float32,
        use_safetensors=True,
    )
    tokenizer = GemmaTokenizer.from_pretrained(
        snapshots.sana,
        subfolder="tokenizer",
        local_files_only=True,
        token=False,
        force_download=False,
        trust_remote_code=False,
    )
    text_encoder = Gemma2Model.from_pretrained(
        snapshots.sana,
        subfolder="text_encoder",
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.bfloat16,
        use_safetensors=True,
        weights_only=True,
        trust_remote_code=False,
    )
    support_processor = BitImageProcessor.from_pretrained(
        snapshots.dino,
        local_files_only=True,
        token=False,
        force_download=False,
    )
    support_encoder = Dinov2Model.from_pretrained(
        snapshots.dino,
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.float32,
        use_safetensors=True,
        weights_only=True,
        trust_remote_code=False,
    )
    training_scheduler = FlowMatchEulerDiscreteScheduler(  # type: ignore[no-untyped-call]
        num_train_timesteps=config.num_train_timesteps,
        shift=config.flow_shift,
        use_dynamic_shifting=config.use_dynamic_shifting,
    )

    if type(inference_scheduler) is not DPMSolverMultistepScheduler:
        raise TypeError("inference scheduler component must have an exact explicit class")
    if type(transformer) is not SanaTransformer2DModel:
        raise TypeError("transformer component must have an exact explicit class")
    if type(vae) is not AutoencoderDC:
        raise TypeError("VAE component must have an exact explicit class")
    if type(tokenizer) is not GemmaTokenizer:
        raise TypeError("tokenizer component must have an exact explicit class")
    if type(text_encoder) is not Gemma2Model:
        raise TypeError("text encoder component must have an exact explicit class")
    if type(support_processor) is not BitImageProcessor:
        raise TypeError("support processor component must have an exact explicit class")
    if type(support_encoder) is not Dinov2Model:
        raise TypeError("support encoder component must have an exact explicit class")

    _freeze_and_move(
        cast(nn.Module, transformer), device=device, dtype=torch.bfloat16
    )
    _freeze_and_move(cast(nn.Module, vae), device=device, dtype=torch.float32)
    _freeze_and_move(
        cast(nn.Module, text_encoder), device=device, dtype=torch.bfloat16
    )
    _freeze_and_move(
        cast(nn.Module, support_encoder), device=device, dtype=torch.float32
    )
    timesteps, sigmas = _canonical_training_schedule()
    components = PinnedComponents(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        training_scheduler=training_scheduler,
        inference_scheduler=inference_scheduler,
        support_processor=support_processor,
        support_encoder=support_encoder,
        training_timesteps=timesteps,
        training_sigmas=sigmas,
    )
    _validate_snapshot_paths(config, snapshots)
    return components


def assert_frozen(modules: dict[str, nn.Module]) -> None:
    changed = [
        f"{module_name}.{name}"
        for module_name, module in modules.items()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    training = [
        f"{module_name}.{name}"
        for module_name, module in modules.items()
        for name, child in module.named_modules()
        if child.training
    ]
    if changed or training:
        raise RuntimeError(
            f"frozen modules violate the contract: trainable={changed[:5]}, "
            f"training={training[:5]}"
        )
