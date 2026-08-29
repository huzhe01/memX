from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

_INTEGER_MASK_DTYPES = {
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class _ModuleState:
    name: str
    identity: int
    training: bool


@dataclass(frozen=True)
class _TensorState:
    name: str
    identity: int
    requires_grad: bool
    device: torch.device
    dtype: torch.dtype
    version: int
    storage_data_ptr: int
    storage_nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    value: Tensor = field(compare=False, repr=False)


@dataclass(frozen=True)
class _EncoderState:
    modules: tuple[_ModuleState, ...]
    parameters: tuple[_TensorState, ...]
    buffers: tuple[_TensorState, ...]


def _tracked_version(tensor: Tensor, *, name: str) -> int:
    try:
        return int(tensor._version)
    except RuntimeError as error:
        raise RuntimeError(f"support encoder tensor {name} must have a tracked version") from error


def _tensor_state(name: str, tensor: Tensor) -> _TensorState:
    return _TensorState(
        name=name,
        identity=id(tensor),
        requires_grad=tensor.requires_grad,
        device=tensor.device,
        dtype=tensor.dtype,
        version=_tracked_version(tensor, name=name),
        storage_data_ptr=tensor.untyped_storage().data_ptr(),
        storage_nbytes=tensor.untyped_storage().nbytes(),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        # DINO-small is snapshotted only during one-time precompute. The extra exact
        # state copy is intentional: version counters do not detect `.data` mutation.
        value=tensor.detach().clone(),
    )


def _snapshot_encoder(encoder: nn.Module) -> _EncoderState:
    return _EncoderState(
        modules=tuple(
            _ModuleState(name, id(module), module.training)
            for name, module in encoder.named_modules(remove_duplicate=False)
        ),
        parameters=tuple(
            _tensor_state(name, parameter)
            for name, parameter in encoder.named_parameters(remove_duplicate=False)
        ),
        buffers=tuple(
            _tensor_state(name, buffer)
            for name, buffer in encoder.named_buffers(remove_duplicate=False)
        ),
    )


def _require_encoder_unchanged(encoder: nn.Module, expected: _EncoderState) -> None:
    try:
        current = _snapshot_encoder(encoder)
        metadata_matches = current == expected
        values_match = metadata_matches and all(
            torch.equal(before.value, after.value)
            for before, after in zip(
                (*expected.parameters, *expected.buffers),
                (*current.parameters, *current.buffers),
                strict=True,
            )
        )
    except BaseException as error:
        raise RuntimeError("encoder state mutated during support encoding") from error
    if not metadata_matches or not values_match:
        raise RuntimeError("encoder state mutated during support encoding")


def verify_frozen_encoder(module: nn.Module) -> None:
    if not isinstance(module, nn.Module):
        raise TypeError("encoder must be an nn.Module")
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            raise RuntimeError(f"support encoder parameter {name} has requires_grad=True")
    for name, descendant in module.named_modules():
        if descendant.training:
            label = name or "<root>"
            raise RuntimeError(f"support encoder module {label} must be in eval mode")


def _validate_encoder_placement(encoder: nn.Module, device: torch.device) -> None:
    tensors = [
        *(encoder.named_parameters()),
        *(encoder.named_buffers()),
    ]
    if not tensors:
        raise ValueError("support encoder must own at least one parameter or buffer")
    for name, tensor in tensors:
        if tensor.device != device:
            raise ValueError(f"encoder tensors must be on {device}; {name} is on {tensor.device}")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(
                f"floating encoder tensors must be torch.float32; {name} has {tensor.dtype}"
            )
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"floating encoder tensor {name} must be finite")


def _validate_images(images: Sequence[Image.Image]) -> list[Image.Image]:
    if not isinstance(images, Sequence) or isinstance(images, str | bytes | bytearray):
        raise TypeError("images must be a sequence of PIL RGB images")
    copied = list(images)
    if not copied:
        raise ValueError("at least one support image is required")
    for image in copied:
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise TypeError("each support image must be a PIL RGB image")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("support images must have positive dimensions")
    return copied


def _processor_crop_size(processor: object) -> tuple[int, int]:
    if not callable(processor):
        raise TypeError("processor must be callable")
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, Mapping):
        if set(crop_size) != {"height", "width"}:
            raise ValueError("processor crop_size must contain exactly height and width")
        height = crop_size["height"]
        width = crop_size["width"]
    else:
        height = getattr(crop_size, "height", None)
        width = getattr(crop_size, "width", None)
        alternative_fields = ("longest_edge", "shortest_edge", "max_height", "max_width")
        if any(getattr(crop_size, field, None) is not None for field in alternative_fields):
            raise ValueError("processor crop_size must contain exactly height and width")
    if type(height) is not int or height <= 0 or type(width) is not int or width <= 0:
        raise ValueError("processor crop dimensions must be exact positive integers")
    return height, width


def _encoder_hidden_size(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("support encoder config.hidden_size must be an exact positive integer")
    return hidden_size


def _processed_pixels(
    processed: object,
    *,
    batch_size: int,
    height: int,
    width: int,
) -> Tensor:
    if not isinstance(processed, Mapping) or "pixel_values" not in processed:
        raise TypeError("processor output must be a mapping containing pixel_values")
    pixel_values = processed["pixel_values"]
    if not isinstance(pixel_values, Tensor):
        raise TypeError("processor pixel_values must be a Tensor")
    if pixel_values.ndim != 4:
        raise ValueError("processor pixel_values must be rank-4")
    if pixel_values.shape[0] != batch_size:
        raise ValueError("processor pixel_values batch size must match images")
    if pixel_values.shape[1] != 3:
        raise ValueError("processor pixel_values must have three channels")
    if pixel_values.shape[2:] != (height, width):
        raise ValueError(f"processor pixel_values must have spatial size {height}x{width}")
    if pixel_values.dtype != torch.float32:
        raise TypeError("processor pixel_values must have dtype torch.float32")
    if pixel_values.device.type != "cpu":
        raise ValueError("processor pixel_values must be on cpu")
    if not bool(torch.isfinite(pixel_values).all()):
        raise ValueError("processor pixel_values must be finite")
    return pixel_values


def encode_support_images(
    images: Sequence[Image.Image],
    *,
    processor: object,
    encoder: nn.Module,
    device: torch.device,
) -> Tensor:
    if type(device) is not torch.device:
        raise TypeError("device must be a torch.device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("support encoding device must be cpu or cuda")
    image_list = _validate_images(images)
    verify_frozen_encoder(encoder)
    _validate_encoder_placement(encoder, device)
    encoder_state = _snapshot_encoder(encoder)
    body_error: BaseException | None = None
    result: Tensor | None = None
    try:
        height, width = _processor_crop_size(processor)
        hidden_size = _encoder_hidden_size(encoder)
        processed = processor(images=image_list, return_tensors="pt")  # type: ignore[operator]
        pixel_values = _processed_pixels(
            processed,
            batch_size=len(image_list),
            height=height,
            width=width,
        ).to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=False):
                output: Any = encoder(pixel_values=pixel_values)
            hidden = getattr(output, "last_hidden_state", None)
            if not isinstance(hidden, Tensor):
                raise TypeError("support encoder last_hidden_state must be a Tensor")
            if hidden.ndim != 3:
                raise ValueError("support encoder last_hidden_state must be rank-3")
            if hidden.shape[0] != len(image_list):
                raise ValueError("support encoder output batch size must match images")
            if hidden.shape[1] <= 0:
                raise ValueError("support encoder output must contain at least one token")
            if hidden.shape[2] != hidden_size:
                raise ValueError(f"support encoder output hidden size must be {hidden_size}")
            if hidden.dtype != torch.float32:
                raise TypeError("support encoder last_hidden_state must have dtype torch.float32")
            if hidden.device != device:
                raise ValueError(
                    "support encoder last_hidden_state must be on the requested device"
                )
            if not bool(torch.isfinite(hidden).all()):
                raise ValueError("support encoder last_hidden_state must be finite")
            class_features = hidden[:, 0].detach()

        # The caller may itself be inside inference_mode (the pilot precompute path is).
        # Clone with inference mode explicitly disabled so the cached features remain
        # valid inputs to a trainable amortizer.
        with torch.inference_mode(False):
            result = class_features.to(device="cpu", dtype=torch.float32).contiguous().clone()
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            _require_encoder_unchanged(encoder, encoder_state)
        except BaseException as invariant_error:
            if body_error is not None:
                raise invariant_error from body_error
            raise
    if result is None:
        raise RuntimeError("support encoder produced no cached features")
    return result


def masked_mean_description(token_features: Tensor, attention_mask: Tensor) -> Tensor:
    if not isinstance(token_features, Tensor):
        raise TypeError("token_features must be a Tensor")
    if not isinstance(attention_mask, Tensor):
        raise TypeError("attention_mask must be a Tensor")
    if token_features.ndim != 3:
        raise ValueError("token_features must be rank-3")
    if attention_mask.ndim != 2 or attention_mask.shape != token_features.shape[:2]:
        raise ValueError("attention_mask shape must match the first two token feature dimensions")
    if any(dimension <= 0 for dimension in token_features.shape):
        raise ValueError("token_features must have positive batch, token, and feature dimensions")
    if token_features.dtype != torch.float32:
        raise TypeError("token_features must have dtype torch.float32")
    if attention_mask.dtype != torch.bool and attention_mask.dtype not in _INTEGER_MASK_DTYPES:
        raise TypeError("attention_mask must have boolean or integer dtype")
    if attention_mask.device != token_features.device:
        raise ValueError("token_features and attention_mask must be on the same device")
    if attention_mask.dtype == torch.bool:
        valid = attention_mask
    else:
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must be binary")
        valid = attention_mask.to(torch.bool)
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("each description requires at least one unmasked token")
    if not bool(torch.isfinite(token_features[valid]).all()):
        raise ValueError("unmasked description features must be finite")

    sanitized = torch.where(valid.unsqueeze(-1), token_features, torch.zeros_like(token_features))
    denominator = valid.sum(dim=1, keepdim=True).to(dtype=torch.float32)
    pooled = sanitized.sum(dim=1) / denominator
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("pooled description features must be finite")
    if pooled.is_inference():
        with torch.inference_mode(False):
            return pooled.clone()
    return pooled
