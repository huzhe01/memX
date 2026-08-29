from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import BitImageProcessor
from transformers.image_utils import SizeDict

from ratemem.support.features import (
    encode_support_images,
    masked_mean_description,
    verify_frozen_encoder,
)


class _RecordingProcessor:
    crop_size = {"height": 4, "width": 4}

    def __init__(self, pixel_values: Tensor | None = None) -> None:
        self.pixel_values = pixel_values
        self.calls: list[tuple[list[Image.Image], str]] = []

    def __call__(self, *, images: list[Image.Image], return_tensors: str) -> dict[str, Any]:
        self.calls.append((images, return_tensors))
        pixels = self.pixel_values
        if pixels is None:
            pixels = torch.stack(
                [torch.full((3, 4, 4), float(index + 1)) for index in range(len(images))]
            )
        return {"pixel_values": pixels}


class _TinyFrozenEncoder(nn.Module):
    def __init__(self, feature_dim: int = 5) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_dim, bias=False)
        self.register_buffer("running", torch.zeros(1))
        self.config = SimpleNamespace(hidden_size=feature_dim)
        self.grad_enabled_during_forward: bool | None = None
        self.last_hidden_override: object | None = None
        self.last_pixel_values: Tensor | None = None

    def forward(self, *, pixel_values: Tensor) -> object:
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        self.last_pixel_values = pixel_values
        if self.last_hidden_override is not None:
            if self.last_hidden_override == "missing":
                return SimpleNamespace()
            return SimpleNamespace(last_hidden_state=self.last_hidden_override)
        pooled = pixel_values.mean(dim=(-1, -2))
        class_token = self.projection(pooled)
        return SimpleNamespace(last_hidden_state=torch.stack((class_token, class_token + 1), dim=1))


class _CallbackProcessor(_RecordingProcessor):
    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self.callback = callback

    def __call__(self, *, images: list[Image.Image], return_tensors: str) -> dict[str, Any]:
        self.callback()
        return super().__call__(images=images, return_tensors=return_tensors)


class _MutatingCropProcessor(_RecordingProcessor):
    def __init__(self, callback: Callable[[], None], *, fail: bool = False) -> None:
        super().__init__()
        self.callback = callback
        self.fail = fail

    @property
    def crop_size(self) -> dict[str, int]:  # type: ignore[override]
        self.callback()
        if self.fail:
            raise ValueError("crop getter exploded")
        return {"height": 4, "width": 4}


class _WeightMutatingEncoder(_TinyFrozenEncoder):
    def forward(self, *, pixel_values: Tensor) -> object:
        with torch.no_grad():
            self.projection.weight.add_(1.0)
        return super().forward(pixel_values=pixel_values)


def _encoder(feature_dim: int = 5) -> _TinyFrozenEncoder:
    return _TinyFrozenEncoder(feature_dim).requires_grad_(False).eval()


def _images(count: int = 2) -> list[Image.Image]:
    return [Image.new("RGB", (6, 7), color=(index, index, index)) for index in range(count)]


def test_verify_frozen_encoder_accepts_recursive_eval_frozen_module() -> None:
    verify_frozen_encoder(_encoder())


def test_verify_frozen_encoder_rejects_non_module() -> None:
    with pytest.raises(TypeError, match="encoder must be an nn.Module"):
        verify_frozen_encoder(object())  # type: ignore[arg-type]


def test_verify_frozen_encoder_rejects_trainable_parameter() -> None:
    encoder = _encoder()
    encoder.projection.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="projection.weight.*requires_grad"):
        verify_frozen_encoder(encoder)


def test_verify_frozen_encoder_rejects_training_descendant() -> None:
    encoder = _encoder()
    encoder.projection.train()
    with pytest.raises(RuntimeError, match="projection.*eval mode"):
        verify_frozen_encoder(encoder)


def test_encode_support_images_is_frozen_fp32_cpu_and_autograd_safe() -> None:
    torch.manual_seed(3)
    encoder = _encoder()
    processor = _RecordingProcessor()

    with torch.inference_mode():
        features = encode_support_images(
            _images(), processor=processor, encoder=encoder, device=torch.device("cpu")
        )

    assert features.shape == (2, 5)
    assert features.dtype == torch.float32
    assert features.device == torch.device("cpu")
    assert features.is_contiguous()
    assert not features.requires_grad
    assert not features.is_inference()
    assert encoder.grad_enabled_during_forward is False
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert len(processor.calls) == 1
    called_images, return_tensors = processor.calls[0]
    assert called_images == _images()
    assert return_tensors == "pt"

    head = nn.Linear(5, 2)
    head(features).sum().backward()
    assert head.weight.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_encode_support_images_accepts_pinned_processor_sizedict_contract() -> None:
    processor = _RecordingProcessor()
    processor.crop_size = SizeDict(height=4, width=4)  # type: ignore[assignment]
    features = encode_support_images(
        _images(), processor=processor, encoder=_encoder(), device=torch.device("cpu")
    )
    assert features.shape == (2, 5)


def test_encode_support_images_accepts_real_local_bit_image_processor_batch_feature() -> None:
    processor = BitImageProcessor.from_dict(
        {
            "do_resize": True,
            "size": {"shortest_edge": 4},
            "resample": 3,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "do_center_crop": True,
            "crop_size": {"height": 4, "width": 4},
            "do_convert_rgb": True,
        }
    )
    encoder = _encoder()
    features = encode_support_images(
        [Image.new("RGB", (6, 7), "red")],
        processor=processor,
        encoder=encoder,
        device=torch.device("cpu"),
    )
    assert processor.do_convert_rgb is True
    assert isinstance(processor.crop_size, SizeDict)
    assert encoder.last_pixel_values is not None
    assert encoder.last_pixel_values.shape == (1, 3, 4, 4)
    assert encoder.last_pixel_values.dtype == torch.float32
    assert features.shape == (1, 5)
    assert features.dtype == torch.float32 and features.device.type == "cpu"


def test_encode_support_images_rejects_encoder_weight_mutation_during_forward() -> None:
    encoder = _WeightMutatingEncoder().requires_grad_(False).eval()
    with pytest.raises(RuntimeError, match="encoder state mutated during support encoding"):
        encode_support_images(
            _images(),
            processor=_RecordingProcessor(),
            encoder=encoder,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda encoder: encoder.projection.weight.data.add_(1.0),
        lambda encoder: setattr(
            encoder.projection.weight,
            "data",
            encoder.projection.weight.detach().clone(),
        ),
        lambda encoder: setattr(
            encoder.projection,
            "weight",
            nn.Parameter(encoder.projection.weight.detach().clone(), requires_grad=False),
        ),
        lambda encoder: setattr(encoder, "running", torch.ones_like(encoder.running)),
        lambda encoder: encoder.add_module("injected", nn.Identity().eval()),
        lambda encoder: encoder.projection.train(),
    ],
)
def test_encode_support_images_rejects_replacement_topology_or_mode_mutation(
    mutation: Callable[[_TinyFrozenEncoder], None],
) -> None:
    encoder = _encoder()
    processor = _CallbackProcessor(lambda: mutation(encoder))
    with pytest.raises(RuntimeError, match="encoder state mutated during support encoding"):
        encode_support_images(
            _images(), processor=processor, encoder=encoder, device=torch.device("cpu")
        )


def test_encoder_postcondition_failure_chains_original_body_exception() -> None:
    encoder = _encoder()

    def mutate_then_fail() -> None:
        encoder.running.add_(1.0)
        raise ValueError("processor exploded")

    with pytest.raises(RuntimeError, match="encoder state mutated during support encoding") as exc:
        encode_support_images(
            _images(),
            processor=_CallbackProcessor(mutate_then_fail),
            encoder=encoder,
            device=torch.device("cpu"),
        )
    assert isinstance(exc.value.__cause__, ValueError)
    assert str(exc.value.__cause__) == "processor exploded"


@pytest.mark.parametrize("fail", [False, True])
def test_encoder_snapshot_precedes_mutating_processor_crop_getter(fail: bool) -> None:
    encoder = _encoder()

    def mutate() -> None:
        encoder.running.data.add_(1.0)

    with pytest.raises(RuntimeError, match="encoder state mutated during support encoding") as exc:
        encode_support_images(
            _images(),
            processor=_MutatingCropProcessor(mutate, fail=fail),
            encoder=encoder,
            device=torch.device("cpu"),
        )
    if fail:
        assert isinstance(exc.value.__cause__, ValueError)
        assert str(exc.value.__cause__) == "crop getter exploded"


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ([], "at least one"),
        ([object()], "PIL RGB image"),
        ([Image.new("L", (4, 4))], "PIL RGB image"),
        ([Image.new("RGB", (0, 4))], "positive dimensions"),
    ],
)
def test_encode_support_images_rejects_invalid_image_sequences(
    images: list[object], message: str
) -> None:
    processor = _RecordingProcessor()
    with pytest.raises((TypeError, ValueError), match=message):
        encode_support_images(  # type: ignore[arg-type]
            images, processor=processor, encoder=_encoder(), device=torch.device("cpu")
        )
    assert processor.calls == []


def test_encode_support_images_rejects_non_sequence_and_non_device() -> None:
    processor = _RecordingProcessor()
    with pytest.raises(TypeError, match="images must be a sequence"):
        encode_support_images(  # type: ignore[arg-type]
            iter(_images()), processor=processor, encoder=_encoder(), device=torch.device("cpu")
        )
    with pytest.raises(TypeError, match="device must be a torch.device"):
        encode_support_images(  # type: ignore[arg-type]
            _images(), processor=processor, encoder=_encoder(), device="cpu"
        )
    assert processor.calls == []


def test_encode_support_images_rejects_encoder_placement_before_preprocessing() -> None:
    processor = _RecordingProcessor()
    with pytest.raises(ValueError, match="encoder tensors must be on cpu"):
        encode_support_images(
            _images(),
            processor=processor,
            encoder=_encoder().to(device="meta"),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="floating encoder tensors must be torch.float32"):
        encode_support_images(
            _images(),
            processor=processor,
            encoder=_encoder().double(),
            device=torch.device("cpu"),
        )
    assert processor.calls == []


@pytest.mark.parametrize(
    ("pixel_values", "message"),
    [
        (torch.ones(2, 3, 4), "rank-4"),
        (torch.ones(1, 3, 4, 4), "batch size"),
        (torch.ones(2, 1, 4, 4), "three channels"),
        (torch.ones(2, 3, 5, 4), "4x4"),
        (torch.ones(2, 3, 4, 4, dtype=torch.float64), "torch.float32"),
        (torch.full((2, 3, 4, 4), torch.nan), "finite"),
    ],
)
def test_encode_support_images_validates_processor_pixels(
    pixel_values: Tensor, message: str
) -> None:
    processor = _RecordingProcessor(pixel_values)
    encoder = _encoder()
    with pytest.raises((TypeError, ValueError), match=message):
        encode_support_images(
            _images(), processor=processor, encoder=encoder, device=torch.device("cpu")
        )
    assert encoder.grad_enabled_during_forward is None


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("missing", "last_hidden_state"),
        (object(), "last_hidden_state must be a Tensor"),
        (torch.ones(2, 5), "rank-3"),
        (torch.ones(1, 2, 5), "batch size"),
        (torch.ones(2, 0, 5), "at least one token"),
        (torch.ones(2, 2, 6), "hidden size must be 5"),
        (torch.ones(2, 2, 5, dtype=torch.float64), "torch.float32"),
        (torch.full((2, 2, 5), torch.inf), "finite"),
    ],
)
def test_encode_support_images_validates_encoder_output(override: object, message: str) -> None:
    encoder = _encoder()
    encoder.last_hidden_override = override
    with pytest.raises((TypeError, ValueError), match=message):
        encode_support_images(
            _images(),
            processor=_RecordingProcessor(),
            encoder=encoder,
            device=torch.device("cpu"),
        )


def test_masked_mean_description_uses_only_unmasked_tokens_and_routes_gradients() -> None:
    tokens = torch.tensor(
        [[[1.0, 3.0], [5.0, 7.0], [float("nan"), float("inf")]]],
        requires_grad=True,
    )
    mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
    pooled = masked_mean_description(tokens, mask)
    torch.testing.assert_close(pooled, torch.tensor([[3.0, 5.0]]), rtol=0.0, atol=0.0)
    pooled.sum().backward()
    assert tokens.grad is not None
    torch.testing.assert_close(tokens.grad[0, :2], torch.full((2, 2), 0.5))
    torch.testing.assert_close(tokens.grad[0, 2], torch.zeros(2))


def test_masked_mean_description_returns_autograd_safe_cached_features() -> None:
    with torch.inference_mode():
        pooled = masked_mean_description(
            torch.ones(2, 3, 5),
            torch.ones(2, 3, dtype=torch.bool),
        )
    assert not pooled.is_inference()
    head = nn.Linear(5, 2)
    head(pooled).sum().backward()
    assert head.weight.grad is not None


def test_masked_mean_description_rejects_nonfinite_post_reduction_overflow() -> None:
    maximum = torch.finfo(torch.float32).max
    tokens = torch.full((1, 2, 1), maximum)
    with pytest.raises(ValueError, match="pooled description features must be finite"):
        masked_mean_description(tokens, torch.ones(1, 2, dtype=torch.bool))


@pytest.mark.parametrize(
    "mask_dtype",
    [
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_masked_mean_description_accepts_binary_boolean_or_integer_masks(
    mask_dtype: torch.dtype,
) -> None:
    tokens = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    mask = torch.tensor([[True, False]], dtype=mask_dtype)
    torch.testing.assert_close(
        masked_mean_description(tokens, mask), torch.tensor([[1.0, 3.0]])
    )


def test_masked_mean_description_rejects_empty_or_all_masked_rows() -> None:
    with pytest.raises(ValueError, match="positive batch, token, and feature dimensions"):
        masked_mean_description(torch.empty(1, 0, 2), torch.empty(1, 0, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one unmasked token"):
        masked_mean_description(torch.ones(2, 2, 3), torch.tensor([[1, 0], [0, 0]]))


@pytest.mark.parametrize(
    ("tokens", "mask", "message"),
    [
        (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool), "rank-3"),
        (torch.ones(2, 3, 4), torch.ones(2, 4, dtype=torch.bool), "shape"),
        (
            torch.ones(2, 3, 4, dtype=torch.float64),
            torch.ones(2, 3, dtype=torch.bool),
            "torch.float32",
        ),
        (torch.ones(2, 3, 4), torch.ones(2, 3), "boolean or integer"),
        (torch.ones(2, 3, 4), torch.tensor([[1, 2, 1], [1, 1, 1]]), "binary"),
        (
            torch.tensor([[[1.0], [float("inf")]]]),
            torch.tensor([[1, 1]]),
            "unmasked description features must be finite",
        ),
    ],
)
def test_masked_mean_description_validates_inputs(
    tokens: Tensor, mask: Tensor, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        masked_mean_description(tokens, mask)
