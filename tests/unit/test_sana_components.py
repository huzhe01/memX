from __future__ import annotations

import ast
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn
from transformers import BitImageProcessor
from transformers.image_utils import SizeDict

import ratemem.sana.components as components_module
from ratemem.pilot.config import SanaPilotConfig
from ratemem.sana.components import (
    DINO_CONTROL_FILE_SHA256,
    DINO_FILES,
    SANA_CONTROL_FILE_SHA256,
    SANA_FILES,
    PinnedComponents,
    PinnedSnapshotPaths,
    hydrate_pinned_snapshots,
    load_pinned_components,
)

CONFIG_PATH = Path("configs/pilot/sana-1.5-1.6b.json")
COMPONENTS_PATH = Path("src/ratemem/sana/components.py")
SANA_REVISION = "b77948f2b4eed5c728e9b828ccff07f7427b43cc"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"

EXPECTED_SANA_FILES = (
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
EXPECTED_DINO_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
)
EXPECTED_SANA_CONTROL_FILE_SHA256 = (
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
EXPECTED_DINO_CONTROL_FILE_SHA256 = (
    (
        "config.json",
        "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    ),
    (
        "preprocessor_config.json",
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    ),
)


@pytest.fixture(autouse=True)
def _mock_fixture_control_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path], str]:
    real_sha256_file = components_module._sha256_file
    expected = {
        (SANA_REVISION, relative): digest
        for relative, digest in EXPECTED_SANA_CONTROL_FILE_SHA256
    } | {
        (DINO_REVISION, relative): digest
        for relative, digest in EXPECTED_DINO_CONTROL_FILE_SHA256
    }

    def fixture_hash(path: Path) -> str:
        for revision in (SANA_REVISION, DINO_REVISION):
            if revision in path.parts:
                index = path.parts.index(revision)
                relative = Path(*path.parts[index + 1 :]).as_posix()
                return expected[(revision, relative)]
        return real_sha256_file(path)

    monkeypatch.setattr(components_module, "_sha256_file", fixture_hash)
    return real_sha256_file


def _write_snapshot(root: Path, revision: str, files: tuple[str, ...]) -> Path:
    snapshot = root / revision
    for relative in files:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return snapshot


def _snapshots(tmp_path: Path) -> PinnedSnapshotPaths:
    return PinnedSnapshotPaths(
        sana=_write_snapshot(tmp_path / "sana", SANA_REVISION, EXPECTED_SANA_FILES),
        dino=_write_snapshot(tmp_path / "dino", DINO_REVISION, EXPECTED_DINO_FILES),
    )


class _FakeModule(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.child = nn.Linear(1, 1)
        self.config = config
        self.train(True)


class _FakeSanaTransformer(_FakeModule):
    from_pretrained = Mock()


class _FakeVae(_FakeModule):
    from_pretrained = Mock()


class _FakeTextEncoder(_FakeModule):
    from_pretrained = Mock()


class _FakeSupportEncoder(_FakeModule):
    from_pretrained = Mock()


class _FakeTokenizer:
    from_pretrained = Mock()


class _FakeProcessor:
    from_pretrained = Mock()


class _FakeInferenceScheduler:
    from_pretrained = Mock()

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            _class_name="DPMSolverMultistepScheduler",
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            trained_betas=None,
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            prediction_type="flow_prediction",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1.0,
            lower_order_final=True,
            euler_at_final=False,
            flow_shift=3.0,
            use_flow_sigmas=True,
            final_sigmas_type="zero",
            timestep_spacing="linspace",
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
            use_lu_lambdas=False,
            use_dynamic_shifting=False,
            time_shift_type="exponential",
            variance_type=None,
            steps_offset=0,
            rescale_betas_zero_snr=False,
            lambda_min_clipped=float("-inf"),
        )


class _FakePipeline:
    from_pretrained = Mock(side_effect=AssertionError("pipeline loader called"))

    def __init__(
        self,
        *,
        tokenizer: object,
        text_encoder: nn.Module,
        vae: nn.Module,
        transformer: nn.Module,
        scheduler: object,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler


@dataclass
class _LoaderDoubles:
    transformer: _FakeSanaTransformer
    vae: _FakeVae
    tokenizer: _FakeTokenizer
    text_encoder: _FakeTextEncoder
    support_processor: _FakeProcessor
    support_encoder: _FakeSupportEncoder
    inference_scheduler: _FakeInferenceScheduler

    @property
    def loader_mocks(self) -> tuple[Mock, ...]:
        return (
            _FakeInferenceScheduler.from_pretrained,
            _FakeSanaTransformer.from_pretrained,
            _FakeVae.from_pretrained,
            _FakeTokenizer.from_pretrained,
            _FakeTextEncoder.from_pretrained,
            _FakeProcessor.from_pretrained,
            _FakeSupportEncoder.from_pretrained,
        )


def _patch_loaders(monkeypatch: pytest.MonkeyPatch) -> _LoaderDoubles:
    transformer = _FakeSanaTransformer(
        SimpleNamespace(
            _class_name="SanaTransformer2DModel",
            sample_size=32,
            patch_size=1,
            num_layers=20,
            num_attention_heads=70,
            attention_head_dim=32,
            num_cross_attention_heads=20,
            cross_attention_head_dim=112,
            cross_attention_dim=2240,
            caption_channels=2304,
            in_channels=32,
            out_channels=32,
            attention_bias=False,
            qk_norm="rms_norm_across_heads",
            mlp_ratio=2.5,
            dropout=0.0,
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            interpolation_scale=None,
            guidance_embeds=False,
            guidance_embeds_scale=0.1,
            timestep_scale=1.0,
        )
    )
    vae = _FakeVae(
        SimpleNamespace(
            _class_name="AutoencoderDC",
            in_channels=3,
            latent_channels=32,
            attention_head_dim=32,
            encoder_block_types=[
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            decoder_block_types=[
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
            ],
            encoder_block_out_channels=[128, 256, 512, 512, 1024, 1024],
            decoder_block_out_channels=[128, 256, 512, 512, 1024, 1024],
            encoder_layers_per_block=[2, 2, 2, 3, 3, 3],
            decoder_layers_per_block=[3, 3, 3, 3, 3, 3],
            encoder_qkv_multiscales=[[], [], [], [5], [5], [5]],
            decoder_qkv_multiscales=[[], [], [], [5], [5], [5]],
            upsample_block_type="interpolate",
            downsample_block_type="Conv",
            decoder_norm_types="rms_norm",
            decoder_act_fns="silu",
            encoder_out_shortcut=True,
            decoder_in_shortcut=True,
            decoder_conv_act_fn="relu",
            scaling_factor=0.41407,
        )
    )
    tokenizer = _FakeTokenizer()
    tokenizer.vocab_size = 256000
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
    tokenizer.bos_token_id = 2
    tokenizer.eos_token_id = 1
    tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 3
    tokenizer.padding_side = "left"
    text_encoder = _FakeTextEncoder(
        SimpleNamespace(architectures=["Gemma2Model"], hidden_size=2304)
    )
    support_processor = _FakeProcessor()
    support_processor.crop_size = SizeDict(height=224, width=224)
    support_processor.size = SizeDict(shortest_edge=256)
    support_processor.do_center_crop = True
    support_processor.do_convert_rgb = True
    support_processor.do_normalize = True
    support_processor.do_rescale = True
    support_processor.do_resize = True
    support_processor.image_mean = (0.485, 0.456, 0.406)
    support_processor.image_std = (0.229, 0.224, 0.225)
    support_processor.image_processor_type = "BitImageProcessor"
    support_processor.resample = 3
    support_processor.rescale_factor = 1 / 255
    support_encoder = _FakeSupportEncoder(
        SimpleNamespace(architectures=["Dinov2Model"], hidden_size=384)
    )
    inference_scheduler = _FakeInferenceScheduler()
    values = _LoaderDoubles(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        support_processor=support_processor,
        support_encoder=support_encoder,
        inference_scheduler=inference_scheduler,
    )
    loader_values = (
        inference_scheduler,
        transformer,
        vae,
        tokenizer,
        text_encoder,
        support_processor,
        support_encoder,
    )
    for loader, value in zip(values.loader_mocks, loader_values, strict=True):
        loader.reset_mock()
        loader.side_effect = None
        loader.return_value = value
    _FakePipeline.from_pretrained.reset_mock()

    monkeypatch.setattr(components_module, "SanaTransformer2DModel", _FakeSanaTransformer)
    monkeypatch.setattr(components_module, "AutoencoderDC", _FakeVae)
    monkeypatch.setattr(components_module, "GemmaTokenizer", _FakeTokenizer)
    monkeypatch.setattr(components_module, "Gemma2Model", _FakeTextEncoder)
    monkeypatch.setattr(components_module, "BitImageProcessor", _FakeProcessor)
    monkeypatch.setattr(components_module, "Dinov2Model", _FakeSupportEncoder)
    monkeypatch.setattr(
        components_module,
        "DPMSolverMultistepScheduler",
        _FakeInferenceScheduler,
    )
    monkeypatch.setattr(components_module, "SanaPipeline", _FakePipeline)
    return values


LOADED_CONFIG_TAMPERS = (
    ("transformer", "_class_name", "WrongTransformer"),
    ("transformer", "sample_size", 31),
    ("transformer", "sample_size", 32.0),
    ("transformer", "patch_size", 2),
    ("transformer", "num_layers", 19),
    ("transformer", "num_layers", 20.0),
    ("transformer", "num_attention_heads", 69),
    ("transformer", "attention_head_dim", 31),
    ("transformer", "num_cross_attention_heads", 19),
    ("transformer", "cross_attention_head_dim", 111),
    ("transformer", "cross_attention_dim", 2239),
    ("transformer", "caption_channels", 2303),
    ("transformer", "in_channels", 31),
    ("transformer", "out_channels", 31),
    ("transformer", "attention_bias", True),
    ("transformer", "attention_bias", 0),
    ("transformer", "qk_norm", "rms_norm"),
    ("transformer", "mlp_ratio", 2.0),
    ("transformer", "dropout", 0),
    ("transformer", "norm_elementwise_affine", 0),
    ("transformer", "norm_eps", 1e-5),
    ("transformer", "interpolation_scale", 1),
    ("transformer", "guidance_embeds", 0),
    ("transformer", "guidance_embeds_scale", 0.2),
    ("transformer", "timestep_scale", 1),
    ("vae", "_class_name", "WrongVae"),
    ("vae", "in_channels", 3.0),
    ("vae", "latent_channels", 31),
    ("vae", "attention_head_dim", 32.0),
    (
        "vae",
        "encoder_block_types",
        [
            "WrongBlock",
            "ResBlock",
            "ResBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
        ],
    ),
    (
        "vae",
        "decoder_block_types",
        (
            "ResBlock",
            "ResBlock",
            "ResBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
        ),
    ),
    ("vae", "encoder_block_out_channels", [127, 256, 512, 512, 1024, 1024]),
    ("vae", "decoder_block_out_channels", [128, 256, 512, 512, 1024, 512]),
    ("vae", "encoder_layers_per_block", [1, 2, 2, 3, 3, 3]),
    ("vae", "decoder_layers_per_block", [3, 3, 3, 3, 3, 2]),
    ("vae", "encoder_qkv_multiscales", [[], [], [], [3], [5], [5]]),
    ("vae", "decoder_qkv_multiscales", [[], [], [], [5], [5], [3]]),
    ("vae", "upsample_block_type", "pixel_shuffle"),
    ("vae", "downsample_block_type", "pixel_unshuffle"),
    ("vae", "decoder_norm_types", "batch_norm"),
    ("vae", "decoder_act_fns", "relu"),
    ("vae", "encoder_out_shortcut", 1),
    ("vae", "decoder_in_shortcut", 1),
    ("vae", "decoder_conv_act_fn", "silu"),
    ("vae", "scaling_factor", 0.5),
    ("text_encoder", "architectures", ["WrongModel"]),
    ("text_encoder", "architectures", ("Gemma2Model",)),
    ("text_encoder", "hidden_size", 2303),
    ("text_encoder", "hidden_size", 2304.0),
    ("tokenizer", "vocab_size", 255999),
    ("tokenizer", "vocab_size", 256000.0),
    ("tokenizer", "add_bos_token", False),
    ("tokenizer", "add_bos_token", 1),
    ("tokenizer", "add_eos_token", True),
    ("tokenizer", "add_eos_token", 0),
    ("tokenizer", "bos_token_id", 3),
    ("tokenizer", "eos_token_id", 2),
    ("tokenizer", "pad_token_id", 1),
    ("tokenizer", "unk_token_id", 4),
    ("tokenizer", "padding_side", "right"),
    ("support_encoder", "architectures", ["WrongModel"]),
    ("support_encoder", "architectures", ("Dinov2Model",)),
    ("support_encoder", "hidden_size", 383),
    ("support_encoder", "hidden_size", 384.0),
    ("support_processor", "crop_size", {"height": 223, "width": 224}),
    ("support_processor", "size", {"shortest_edge": 255}),
    ("support_processor", "size", {"shortest_edge": 256.0}),
    ("support_processor", "do_center_crop", False),
    ("support_processor", "do_center_crop", 1),
    ("support_processor", "do_convert_rgb", False),
    ("support_processor", "do_normalize", False),
    ("support_processor", "do_rescale", False),
    ("support_processor", "do_resize", False),
    ("support_processor", "image_mean", [0.5, 0.456, 0.406]),
    ("support_processor", "image_mean", [0.485, 0.456, 0.406]),
    ("support_processor", "image_std", [0.2, 0.224, 0.225]),
    ("support_processor", "image_std", [0.229, 0.224, 0.225]),
    ("support_processor", "image_processor_type", "WrongProcessor"),
    ("support_processor", "resample", 2),
    ("support_processor", "resample", 3.0),
    ("support_processor", "rescale_factor", 1.0),
    ("inference_scheduler", "_class_name", "WrongScheduler"),
    ("inference_scheduler", "num_train_timesteps", 999),
    ("inference_scheduler", "num_train_timesteps", 1000.0),
    ("inference_scheduler", "beta_start", 0.001),
    ("inference_scheduler", "beta_end", 0.2),
    ("inference_scheduler", "beta_schedule", "scaled_linear"),
    ("inference_scheduler", "trained_betas", []),
    ("inference_scheduler", "algorithm_type", "dpmsolver"),
    ("inference_scheduler", "solver_order", 3),
    ("inference_scheduler", "solver_type", "heun"),
    ("inference_scheduler", "prediction_type", "epsilon"),
    ("inference_scheduler", "thresholding", 0),
    ("inference_scheduler", "dynamic_thresholding_ratio", 0.9),
    ("inference_scheduler", "sample_max_value", 1),
    ("inference_scheduler", "lower_order_final", 1),
    ("inference_scheduler", "euler_at_final", 0),
    ("inference_scheduler", "flow_shift", 2.0),
    ("inference_scheduler", "flow_shift", 3),
    ("inference_scheduler", "use_flow_sigmas", False),
    ("inference_scheduler", "use_flow_sigmas", 1),
    ("inference_scheduler", "final_sigmas_type", "sigma_min"),
    ("inference_scheduler", "timestep_spacing", "leading"),
    ("inference_scheduler", "use_karras_sigmas", True),
    ("inference_scheduler", "use_exponential_sigmas", True),
    ("inference_scheduler", "use_beta_sigmas", True),
    ("inference_scheduler", "use_lu_lambdas", 0),
    ("inference_scheduler", "use_dynamic_shifting", True),
    ("inference_scheduler", "use_dynamic_shifting", 0),
    ("inference_scheduler", "time_shift_type", "linear"),
    ("inference_scheduler", "variance_type", "learned"),
    ("inference_scheduler", "steps_offset", 0.0),
    ("inference_scheduler", "rescale_betas_zero_snr", 0),
    ("inference_scheduler", "lambda_min_clipped", 0.0),
)


def test_file_allowlists_are_exact_and_contain_no_executable_or_legacy_weights() -> None:
    assert SANA_FILES == EXPECTED_SANA_FILES
    assert DINO_FILES == EXPECTED_DINO_FILES
    assert type(SANA_FILES) is tuple and type(DINO_FILES) is tuple
    for relative in (*SANA_FILES, *DINO_FILES):
        assert "*" not in relative
        assert not relative.endswith((".py", ".bin"))


def test_control_file_sha256_manifests_are_exact_and_allowlisted() -> None:
    assert SANA_CONTROL_FILE_SHA256 == EXPECTED_SANA_CONTROL_FILE_SHA256
    assert DINO_CONTROL_FILE_SHA256 == EXPECTED_DINO_CONTROL_FILE_SHA256
    assert {path for path, _digest in SANA_CONTROL_FILE_SHA256} <= set(SANA_FILES)
    assert {path for path, _digest in DINO_CONTROL_FILE_SHA256} <= set(DINO_FILES)
    for _path, digest in (*SANA_CONTROL_FILE_SHA256, *DINO_CONTROL_FILE_SHA256):
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)


def test_sha256_file_hashes_streamed_bytes(
    tmp_path: Path, _mock_fixture_control_hashes: Callable[[Path], str]
) -> None:
    path = tmp_path / "control.json"
    path.write_bytes(b"abc")

    assert _mock_fixture_control_hashes(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize("corrupt_repo", ["sana", "dino"])
def test_hydrator_hashes_each_control_file_before_any_later_download(
    corrupt_repo: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _mock_fixture_control_hashes: Callable[[Path], str],
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    target_revision = SANA_REVISION if corrupt_repo == "sana" else DINO_REVISION
    fixture_hash = components_module._sha256_file

    def hash_with_corruption(path: Path) -> str:
        if target_revision in path.parts:
            return _mock_fixture_control_hashes(path)
        return fixture_hash(path)

    download = Mock(side_effect=[str(snapshots.sana), str(snapshots.dino)])
    monkeypatch.setattr(components_module, "_sha256_file", hash_with_corruption)
    monkeypatch.setattr(components_module, "snapshot_download", download)

    with pytest.raises(ValueError, match="SHA-256"):
        hydrate_pinned_snapshots(config, cache_dir=tmp_path / "cache")

    assert download.call_count == (1 if corrupt_repo == "sana" else 2)


def test_local_control_file_hash_failure_precedes_every_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _mock_fixture_control_hashes: Callable[[Path], str],
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    doubles = _patch_loaders(monkeypatch)
    fixture_hash = components_module._sha256_file

    def hash_with_corruption(path: Path) -> str:
        if path.name == "scheduler_config.json":
            return _mock_fixture_control_hashes(path)
        return fixture_hash(path)

    monkeypatch.setattr(components_module, "_sha256_file", hash_with_corruption)

    with pytest.raises(ValueError, match="SHA-256"):
        load_pinned_components(
            config,
            snapshots=snapshots,
            device=torch.device("cpu"),
        )

    assert all(loader.call_count == 0 for loader in doubles.loader_mocks)


def test_local_loader_rehashes_control_files_after_all_component_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    doubles = _patch_loaders(monkeypatch)
    fixture_hash = components_module._sha256_file
    control_changed = False

    def hash_before_and_after_load(path: Path) -> str:
        if control_changed and path.name == "scheduler_config.json":
            return "0" * 64
        return fixture_hash(path)

    def load_then_change_control(*_args: object, **_kwargs: object) -> object:
        nonlocal control_changed
        control_changed = True
        return doubles.inference_scheduler

    monkeypatch.setattr(
        components_module,
        "_sha256_file",
        hash_before_and_after_load,
    )
    _FakeInferenceScheduler.from_pretrained.side_effect = load_then_change_control

    with pytest.raises(ValueError, match="SHA-256"):
        load_pinned_components(
            config,
            snapshots=snapshots,
            device=torch.device("cpu"),
        )

    assert all(loader.call_count == 1 for loader in doubles.loader_mocks)


def test_pinned_bit_processor_raw_json_normalizes_to_the_exact_runtime_contract() -> None:
    processor = BitImageProcessor.from_dict(
        {
            "crop_size": {"height": 224, "width": 224},
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_processor_type": "BitImageProcessor",
            "image_std": [0.229, 0.224, 0.225],
            "resample": 3,
            "rescale_factor": 1 / 255,
            "size": {"shortest_edge": 256},
        }
    )

    assert type(processor.crop_size) is SizeDict
    assert type(processor.size) is SizeDict
    assert type(processor.image_mean) is tuple
    assert type(processor.image_std) is tuple
    components_module._require_attributes(
        processor,
        components_module._PROCESSOR_CONFIG,
        "support processor",
    )


def test_hydrator_makes_exactly_two_pinned_allowlisted_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    expected = _snapshots(tmp_path)
    download = Mock(side_effect=[str(expected.sana), str(expected.dino)])
    monkeypatch.setattr(components_module, "snapshot_download", download)
    cache_dir = tmp_path / "cache"

    actual = hydrate_pinned_snapshots(config, cache_dir=cache_dir)

    assert actual == expected
    assert download.call_args_list == [
        call(
            repo_id=config.model_id,
            repo_type="model",
            revision=config.revision,
            cache_dir=cache_dir,
            allow_patterns=SANA_FILES,
            token=False,
            local_files_only=False,
            force_download=False,
        ),
        call(
            repo_id=config.support_model_id,
            repo_type="model",
            revision=config.support_revision,
            cache_dir=cache_dir,
            allow_patterns=DINO_FILES,
            token=False,
            local_files_only=False,
            force_download=False,
        ),
    ]


@pytest.mark.parametrize("failure", ["sana-basename", "dino-basename", "missing-file"])
def test_hydrator_rejects_unverified_download_results(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    sana = snapshots.sana
    dino = snapshots.dino
    if failure == "sana-basename":
        sana = _write_snapshot(tmp_path / "bad-sana", "0" * 40, EXPECTED_SANA_FILES)
    elif failure == "dino-basename":
        dino = _write_snapshot(tmp_path / "bad-dino", "1" * 40, EXPECTED_DINO_FILES)
    else:
        (sana / SANA_FILES[-1]).unlink()
    download = Mock(side_effect=[str(sana), str(dino)])
    monkeypatch.setattr(components_module, "snapshot_download", download)

    with pytest.raises((FileNotFoundError, ValueError), match="snapshot|revision|missing"):
        hydrate_pinned_snapshots(config, cache_dir=tmp_path / "cache")

    expected_calls = 2 if failure == "dino-basename" else 1
    assert download.call_count == expected_calls


def test_hydrator_rejects_a_sha_named_symlink_to_the_wrong_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    wrong = _write_snapshot(tmp_path / "actual", "0" * 40, EXPECTED_SANA_FILES)
    disguised = tmp_path / "disguised" / SANA_REVISION
    disguised.parent.mkdir()
    disguised.symlink_to(wrong, target_is_directory=True)
    download = Mock(return_value=str(disguised))
    monkeypatch.setattr(components_module, "snapshot_download", download)

    with pytest.raises(ValueError, match="revision"):
        hydrate_pinned_snapshots(config, cache_dir=tmp_path / "cache")

    download.assert_called_once()


def test_hydrator_returns_strictly_resolved_snapshot_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    actual = _snapshots(tmp_path / "actual")
    sana_alias = tmp_path / "aliases" / "sana" / SANA_REVISION
    dino_alias = tmp_path / "aliases" / "dino" / DINO_REVISION
    sana_alias.parent.mkdir(parents=True)
    dino_alias.parent.mkdir(parents=True)
    sana_alias.symlink_to(actual.sana, target_is_directory=True)
    dino_alias.symlink_to(actual.dino, target_is_directory=True)
    download = Mock(side_effect=[str(sana_alias), str(dino_alias)])
    monkeypatch.setattr(components_module, "snapshot_download", download)

    snapshots = hydrate_pinned_snapshots(config, cache_dir=tmp_path / "cache")

    assert snapshots.sana == actual.sana.resolve(strict=True)
    assert snapshots.dino == actual.dino.resolve(strict=True)


def test_hydrator_hashes_the_resolved_symlink_target_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _mock_fixture_control_hashes: Callable[[Path], str],
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    actual = _snapshots(tmp_path / "actual")
    alias = tmp_path / "alias" / SANA_REVISION
    alias.parent.mkdir()
    alias.symlink_to(actual.sana, target_is_directory=True)
    fixture_hash = components_module._sha256_file

    def hash_resolved_scheduler(path: Path) -> str:
        if path.name == "scheduler_config.json":
            return _mock_fixture_control_hashes(path)
        return fixture_hash(path)

    download = Mock(side_effect=[str(alias), str(actual.dino)])
    monkeypatch.setattr(components_module, "_sha256_file", hash_resolved_scheduler)
    monkeypatch.setattr(components_module, "snapshot_download", download)

    with pytest.raises(ValueError, match="SHA-256"):
        hydrate_pinned_snapshots(config, cache_dir=tmp_path / "cache")

    download.assert_called_once()


def test_hydrator_revalidates_exact_config_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    object.__setattr__(config, "revision", "0" * 40)
    download = Mock()
    monkeypatch.setattr(components_module, "snapshot_download", download)

    with pytest.raises(ValueError, match="canonical|exact|changed"):
        hydrate_pinned_snapshots(config, cache_dir=tmp_path)

    download.assert_not_called()


def test_hydrator_rejects_config_subclass_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _ConfigSubclass(SanaPilotConfig):
        pass

    config = object.__new__(_ConfigSubclass)
    download = Mock()
    monkeypatch.setattr(components_module, "snapshot_download", download)

    with pytest.raises(TypeError, match="exact SanaPilotConfig"):
        hydrate_pinned_snapshots(config, cache_dir=tmp_path)

    download.assert_not_called()


def test_local_loaders_receive_complete_offline_safe_kwargs_and_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    doubles = _patch_loaders(monkeypatch)
    network = Mock(side_effect=AssertionError("network boundary crossed"))
    monkeypatch.setattr(components_module, "snapshot_download", network)
    monkeypatch.setattr(socket, "socket", network)

    bundle = load_pinned_components(
        config,
        snapshots=snapshots,
        device=torch.device("cpu"),
    )

    _FakeInferenceScheduler.from_pretrained.assert_called_once_with(
        snapshots.sana,
        subfolder="scheduler",
        local_files_only=True,
        token=False,
        force_download=False,
    )
    _FakeSanaTransformer.from_pretrained.assert_called_once_with(
        snapshots.sana,
        subfolder="transformer",
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.bfloat16,
        use_safetensors=True,
    )
    _FakeVae.from_pretrained.assert_called_once_with(
        snapshots.sana,
        subfolder="vae",
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.float32,
        use_safetensors=True,
    )
    _FakeTokenizer.from_pretrained.assert_called_once_with(
        snapshots.sana,
        subfolder="tokenizer",
        local_files_only=True,
        token=False,
        force_download=False,
        trust_remote_code=False,
    )
    _FakeTextEncoder.from_pretrained.assert_called_once_with(
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
    _FakeProcessor.from_pretrained.assert_called_once_with(
        snapshots.dino,
        local_files_only=True,
        token=False,
        force_download=False,
    )
    _FakeSupportEncoder.from_pretrained.assert_called_once_with(
        snapshots.dino,
        local_files_only=True,
        token=False,
        force_download=False,
        dtype=torch.float32,
        use_safetensors=True,
        weights_only=True,
        trust_remote_code=False,
    )
    network.assert_not_called()
    assert type(bundle) is PinnedComponents
    assert type(bundle.training_scheduler) is FlowMatchEulerDiscreteScheduler
    assert bundle.training_scheduler.config.num_train_timesteps == 1000
    assert bundle.training_scheduler.config.shift == 1.0
    assert bundle.training_scheduler.config.use_dynamic_shifting is False
    assert len(bundle.training_timesteps) == len(bundle.training_sigmas) == 1000
    assert bundle.training_timesteps[:2] == (1000.0, 999.0)
    assert bundle.training_timesteps[-1] == 1.0
    assert bundle.training_sigmas[0] == 1.0
    assert bundle.training_sigmas[-1] == pytest.approx(0.001)
    canonical_sigmas = torch.linspace(1, 1000, 1000, dtype=torch.float32).flip(0) / 1000
    canonical_timesteps = canonical_sigmas * 1000
    assert bundle.training_timesteps == tuple(
        float(value) for value in canonical_timesteps
    )
    assert bundle.training_sigmas == tuple(float(value) for value in canonical_sigmas)
    torch.testing.assert_close(
        torch.tensor(bundle.training_sigmas),
        torch.tensor(bundle.training_timesteps) / 1000.0,
    )
    for module, dtype in (
        (doubles.transformer, torch.bfloat16),
        (doubles.text_encoder, torch.bfloat16),
        (doubles.vae, torch.float32),
        (doubles.support_encoder, torch.float32),
    ):
        assert all(not child.training for child in module.modules())
        assert all(not parameter.requires_grad for parameter in module.parameters())
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        assert all(parameter.dtype is dtype for parameter in module.parameters())

    pipeline = bundle.inference_pipeline()
    assert type(pipeline) is _FakePipeline
    assert pipeline.tokenizer is bundle.tokenizer
    assert pipeline.text_encoder is bundle.text_encoder
    assert pipeline.vae is bundle.vae
    assert pipeline.transformer is bundle.transformer
    assert pipeline.scheduler is bundle.inference_scheduler
    _FakePipeline.from_pretrained.assert_not_called()


@pytest.mark.parametrize("field", ["training_timesteps", "training_sigmas"])
def test_training_schedule_is_canonical_at_every_index_even_if_both_views_are_tampered(
    field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    _patch_loaders(monkeypatch)
    bundle = load_pinned_components(
        config,
        snapshots=snapshots,
        device=torch.device("cpu"),
    )
    values = list(getattr(bundle, field))
    values[500] += 0.25
    replacement = tuple(values)
    object.__setattr__(bundle, field, replacement)
    scheduler_tensor = (
        bundle.training_scheduler.timesteps
        if field == "training_timesteps"
        else bundle.training_scheduler.sigmas
    )
    scheduler_tensor[500] += 0.25

    with pytest.raises(RuntimeError, match="scheduler|canonical|index"):
        bundle.validate()


def test_training_schedule_tuple_requires_exact_builtin_float_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _EqualFloat(float):
        pass

    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    _patch_loaders(monkeypatch)
    bundle = load_pinned_components(
        config,
        snapshots=snapshots,
        device=torch.device("cpu"),
    )
    values = list(bundle.training_timesteps)
    values[500] = _EqualFloat(values[500])
    object.__setattr__(bundle, "training_timesteps", tuple(values))

    with pytest.raises(TypeError, match="float|timestep"):
        bundle.validate()


@pytest.mark.parametrize("failure", ["missing", "wrong-sana-sha", "wrong-dino-sha"])
def test_local_validation_fails_before_every_loader(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    if failure == "missing":
        (snapshots.dino / DINO_FILES[-1]).unlink()
    elif failure == "wrong-sana-sha":
        snapshots = PinnedSnapshotPaths(
            sana=_write_snapshot(tmp_path / "bad", "0" * 40, EXPECTED_SANA_FILES),
            dino=snapshots.dino,
        )
    else:
        snapshots = PinnedSnapshotPaths(
            sana=snapshots.sana,
            dino=_write_snapshot(tmp_path / "bad", "1" * 40, EXPECTED_DINO_FILES),
        )
    doubles = _patch_loaders(monkeypatch)

    with pytest.raises((FileNotFoundError, ValueError), match="snapshot|revision|missing"):
        load_pinned_components(
            config,
            snapshots=snapshots,
            device=torch.device("cpu"),
        )

    assert all(loader.call_count == 0 for loader in doubles.loader_mocks)


def test_local_loader_rejects_snapshot_subclass_before_every_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SnapshotSubclass(PinnedSnapshotPaths):
        pass

    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    subclass = _SnapshotSubclass(sana=snapshots.sana, dino=snapshots.dino)
    doubles = _patch_loaders(monkeypatch)

    with pytest.raises(TypeError, match="exact PinnedSnapshotPaths"):
        load_pinned_components(
            config,
            snapshots=subclass,
            device=torch.device("cpu"),
        )

    assert all(loader.call_count == 0 for loader in doubles.loader_mocks)


def test_local_loader_revalidates_exact_config_before_every_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    object.__setattr__(config, "support_revision", "0" * 40)
    snapshots = _snapshots(tmp_path)
    doubles = _patch_loaders(monkeypatch)

    with pytest.raises(ValueError, match="canonical|exact|changed"):
        load_pinned_components(
            config,
            snapshots=snapshots,
            device=torch.device("cpu"),
        )

    assert all(loader.call_count == 0 for loader in doubles.loader_mocks)


@pytest.mark.parametrize(
    ("component_name", "field_name", "replacement"), LOADED_CONFIG_TAMPERS
)
def test_every_loaded_component_config_leaf_is_validated_exactly(
    component_name: str,
    field_name: str,
    replacement: object,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    doubles = _patch_loaders(monkeypatch)
    component = getattr(doubles, component_name)
    target = (
        component
        if component_name in {"tokenizer", "support_processor"}
        else component.config
    )
    setattr(target, field_name, replacement)

    with pytest.raises(RuntimeError, match="component|config|scheduler|processor"):
        load_pinned_components(
            config,
            snapshots=snapshots,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "component_name",
    [
        "transformer",
        "vae",
        "tokenizer",
        "text_encoder",
        "support_processor",
        "support_encoder",
        "inference_scheduler",
    ],
)
def test_every_loader_result_requires_an_exact_explicit_class(
    component_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    snapshots = _snapshots(tmp_path)
    doubles = _patch_loaders(monkeypatch)
    original = getattr(doubles, component_name)
    subclass = type(f"Subclass{type(original).__name__}", (type(original),), {})
    if isinstance(original, _FakeModule):
        replacement = subclass(original.config)
    else:
        replacement = subclass()
    loader_by_component = {
        "transformer": _FakeSanaTransformer.from_pretrained,
        "vae": _FakeVae.from_pretrained,
        "tokenizer": _FakeTokenizer.from_pretrained,
        "text_encoder": _FakeTextEncoder.from_pretrained,
        "support_processor": _FakeProcessor.from_pretrained,
        "support_encoder": _FakeSupportEncoder.from_pretrained,
        "inference_scheduler": _FakeInferenceScheduler.from_pretrained,
    }
    loader_by_component[component_name].return_value = replacement

    with pytest.raises(TypeError, match="exact|component"):
        load_pinned_components(
            config,
            snapshots=snapshots,
            device=torch.device("cpu"),
        )


def test_component_source_has_one_network_boundary_and_no_dynamic_loader_paths() -> None:
    source = COMPONENTS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id.startswith("Auto")
        and node.id != "AutoencoderDC"
    }
    assert forbidden_names == set()
    assert "DiffusionPipeline" not in source
    assert "custom_pipeline" not in source
    for forbidden_network_api in (
        "hf_hub_download",
        "requests",
        "httpx",
        "urllib",
        "urlopen",
    ):
        assert forbidden_network_api not in source

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "set_timesteps":
                pytest.fail("training scheduler arrays must never be regenerated")
            if (
                node.func.attr == "from_pretrained"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {
                    "FlowMatchEulerDiscreteScheduler",
                    "SanaPipeline",
                }
            ):
                pytest.fail(f"forbidden loader: {node.func.value.id}.from_pretrained")
        for keyword in node.keywords:
            if keyword.arg == "custom_pipeline":
                pytest.fail("custom pipelines are forbidden")
            if (
                keyword.arg == "trust_remote_code"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                pytest.fail("dynamic remote code is forbidden")

    snapshot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "snapshot_download"
    ]
    assert len(snapshot_calls) == 2
    hydrate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "hydrate_pinned_snapshots"
    )
    assert all(call_node in set(ast.walk(hydrate)) for call_node in snapshot_calls)
