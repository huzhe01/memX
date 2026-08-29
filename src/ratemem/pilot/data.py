from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import struct
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Never, cast

import torch
from datasets import Features, Value, load_dataset  # type: ignore[import-untyped]
from datasets import Image as DatasetImage
from datasets import config as datasets_config
from diffusers.models.autoencoders.vae import EncoderOutput
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
from torchvision.transforms import functional as vision_functional
from transformers.modeling_outputs import BaseModelOutputWithPast

from ratemem.pilot.config import (
    SUBJECTS_PILOT_CANONICAL_SHA256,
    SanaPilotConfig,
    SubjectsPilotConfig,
)
from ratemem.sana.components import PinnedComponents
from ratemem.support.features import encode_support_images, masked_mean_description

_SCHEMA_VERSION = "1.0.0"
_SCOPE = "engineering_pilot_only"
_SANA_REVISION = "b77948f2b4eed5c728e9b828ccff07f7427b43cc"
_SUPPORT_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
_DATASET_REVISION = "0d1cf6536239888f1a8e218790649344810067bc"
_COMPLETE_MARKER = b"ratemem-cache-complete-v1\n"
_HEX_DIGITS = frozenset("0123456789abcdef")

CACHE_TENSOR_SPECS: Mapping[str, tuple[tuple[int, ...], torch.dtype]] = (
    MappingProxyType(
        {
            "clean_latents": ((8, 32, 32, 32), torch.float32),
            "prompt_embeddings": ((8, 300, 2304), torch.float32),
            "prompt_attention_mask": ((8, 300), torch.int64),
            "support_features": ((8, 1, 384), torch.float32),
            "support_mask": ((8, 1), torch.bool),
            "description_features": ((8, 2304), torch.float32),
        }
    )
)

_SAFETENSORS_DTYPES = {
    torch.float32: "F32",
    torch.int64: "I64",
    torch.bool: "BOOL",
}


def _reject_nonfinite_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be an exact lowercase SHA-256")
    return cast(str, value)


def _require_subjects_config(config: SubjectsPilotConfig) -> SubjectsPilotConfig:
    if type(config) is not SubjectsPilotConfig:
        raise TypeError("dataset config must be an exact SubjectsPilotConfig")
    config.validate()
    return config


def _require_sana_config(config: SanaPilotConfig) -> SanaPilotConfig:
    if type(config) is not SanaPilotConfig:
        raise TypeError("SANA config must be an exact SanaPilotConfig")
    config.validate()
    return config


def rgb_content_sha256(image: Image.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if image.mode != "RGB":
        raise TypeError("image must be in exact RGB mode")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("RGB image dimensions must be positive")
    pixels = image.tobytes()
    expected_bytes = image.width * image.height * 3
    if len(pixels) != expected_bytes:
        raise ValueError("RGB image exposes an unexpected raw byte count")
    digest = hashlib.sha256()
    digest.update(b"ratemem-rgb-v1\0")
    digest.update(struct.pack(">II", image.width, image.height))
    digest.update(pixels)
    return digest.hexdigest()


def split_composite_pair(
    image: Image.Image,
    config: SubjectsPilotConfig,
) -> tuple[Image.Image, Image.Image]:
    locked = _require_subjects_config(config)
    if not isinstance(image, Image.Image) or image.mode != locked.mode:
        raise TypeError("Subjects200K composite must be a PIL image in exact RGB mode")
    if image.size != locked.size:
        raise ValueError(
            "Subjects200K composite must be "
            f"{locked.size[0]}x{locked.size[1]}, got {image.width}x{image.height}"
        )
    support = image.crop(cast(tuple[int, int, int, int], locked.left_crop))
    query = image.crop(cast(tuple[int, int, int, int], locked.right_crop))
    expected = (locked.image_size, locked.image_size)
    if support.size != expected or query.size != expected:
        raise RuntimeError("locked Subjects200K crop boxes produced unexpected geometry")
    if support.mode != "RGB" or query.mode != "RGB":
        raise RuntimeError("locked Subjects200K crops did not preserve RGB mode")
    return support, query


def _require_exact_dict_keys(
    value: object, expected: tuple[str, ...], context: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an exact dict")
    result = cast(dict[str, object], value)
    if tuple(result) != expected:
        raise ValueError(f"{context} keys and order must be exactly canonical")
    return result


def _require_exact_json_value(actual: object, expected: object, context: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(
            f"{context} must have exact type {type(expected).__name__}"
        )
    if type(expected) is dict:
        actual_dict = cast(dict[str, object], actual)
        expected_dict = cast(dict[str, object], expected)
        if tuple(actual_dict) != tuple(expected_dict):
            raise ValueError(f"{context} JSON key order changed")
        for key, expected_child in expected_dict.items():
            _require_exact_json_value(
                actual_dict[key], expected_child, f"{context}.{key}"
            )
        return
    if type(expected) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        if len(actual_list) != len(expected_list):
            raise ValueError(f"{context} JSON list length changed")
        for index, (actual_child, expected_child) in enumerate(
            zip(actual_list, expected_list, strict=True)
        ):
            _require_exact_json_value(
                actual_child, expected_child, f"{context}[{index}]"
            )
        return
    if actual != expected:
        raise ValueError(f"{context} changed from its exact value")


def _validate_quality(
    value: object, config: SubjectsPilotConfig
) -> tuple[int, int, int] | None:
    if value is None:
        return None
    quality = _require_exact_dict_keys(
        value, config.quality_field_order, "quality assessment"
    )
    values: list[int] = []
    for name in config.quality_field_order:
        score = quality[name]
        if type(score) is not int or not 0 <= score <= 5:
            raise ValueError(f"quality field {name} must be an exact integer from 0 to 5")
        values.append(score)
    return cast(tuple[int, int, int], tuple(values))


def _validate_description(
    value: object, config: SubjectsPilotConfig
) -> tuple[str, str, str, str, bool]:
    description = _require_exact_dict_keys(
        value, config.description_field_order, "description"
    )
    strings: list[str] = []
    for name in config.description_field_order[:-1]:
        field = description[name]
        if type(field) is not str or not field.strip():
            raise TypeError(f"description field {name} must be a non-empty exact str")
        strings.append(field)
    valid = description[config.description_field_order[-1]]
    if type(valid) is not bool:
        raise TypeError("description_valid must be an exact bool")
    if not valid:
        raise ValueError("locked pilot rows require description_valid=true")
    return strings[0], strings[1], strings[2], strings[3], valid


def _row_identity_payload(
    *,
    row_index: int,
    collection: str,
    quality_assessment: tuple[int, int, int] | None,
    item: str,
    description_0: str,
    description_1: str,
    category: str,
    description_valid: bool,
    composite_sha256: str,
    support_sha256: str,
    query_sha256: str,
    config: SubjectsPilotConfig,
) -> dict[str, object]:
    return {
        "schema_version": "ratemem-subjects-row-v1",
        "dataset_id": config.dataset_id,
        "revision": config.revision,
        "config_name": config.config_name,
        "split": config.split,
        "source_file": config.source_file,
        "source_file_sha256": config.source_file_sha256,
        "row_index": row_index,
        "collection": collection,
        "quality_assessment": (
            None
            if quality_assessment is None
            else {
                name: quality_assessment[index]
                for index, name in enumerate(config.quality_field_order)
            }
        ),
        "description": {
            "item": item,
            "description_0": description_0,
            "description_1": description_1,
            "category": category,
            "description_valid": description_valid,
        },
        "composite": {
            "mode": config.mode,
            "size": list(config.size),
            "sha256": composite_sha256,
        },
        "support": {
            "side": config.support_side,
            "crop": list(config.left_crop),
            "sha256": support_sha256,
        },
        "query": {
            "side": config.query_side,
            "crop": list(config.right_crop),
            "sha256": query_sha256,
        },
    }


@dataclass(frozen=True, slots=True)
class PilotExample:
    row_index: int
    collection: str
    quality_assessment: tuple[int, int, int] | None
    item: str
    description_0: str
    description_1: str
    category: str
    description_valid: bool
    support_rgb: bytes
    query_rgb: bytes
    composite_sha256: str
    support_sha256: str
    query_sha256: str
    row_sha256: str

    @property
    def concept_description(self) -> str:
        return self.item

    @property
    def query_prompt(self) -> str:
        return self.description_1

    def support_image(self) -> Image.Image:
        return Image.frombytes("RGB", (512, 512), self.support_rgb)

    def query_image(self) -> Image.Image:
        return Image.frombytes("RGB", (512, 512), self.query_rgb)

    def validate(self, config: SubjectsPilotConfig) -> None:
        locked = _require_subjects_config(config)
        if type(self) is not PilotExample:
            raise TypeError("example must be an exact PilotExample")
        if type(self.row_index) is not int or self.row_index not in locked.row_indices:
            raise ValueError("example row index is outside the locked rows")
        if type(self.collection) is not str or not self.collection:
            raise TypeError("example collection must be a non-empty exact str")
        if self.quality_assessment is not None:
            if type(self.quality_assessment) is not tuple or len(self.quality_assessment) != 3:
                raise TypeError("example quality assessment must be an exact immutable triple")
            if any(
                type(value) is not int or not 0 <= value <= 5
                for value in self.quality_assessment
            ):
                raise ValueError("example quality assessment values must be integers from 0 to 5")
        for name in ("item", "description_0", "description_1", "category"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise TypeError(f"example {name} must be a non-empty exact str")
        if type(self.description_valid) is not bool or not self.description_valid:
            raise ValueError("example description must remain exactly valid")
        expected_bytes = locked.image_size * locked.image_size * 3
        for name in ("support_rgb", "query_rgb"):
            value = getattr(self, name)
            if type(value) is not bytes or len(value) != expected_bytes:
                raise ValueError(f"example {name} has the wrong immutable RGB byte count")
        for name in (
            "composite_sha256",
            "support_sha256",
            "query_sha256",
            "row_sha256",
        ):
            _require_sha256(getattr(self, name), f"example {name}")
        if rgb_content_sha256(self.support_image()) != self.support_sha256:
            raise RuntimeError("example support RGB bytes or hash were mutated")
        if rgb_content_sha256(self.query_image()) != self.query_sha256:
            raise RuntimeError("example query RGB bytes or hash were mutated")
        payload = _row_identity_payload(
            row_index=self.row_index,
            collection=self.collection,
            quality_assessment=self.quality_assessment,
            item=self.item,
            description_0=self.description_0,
            description_1=self.description_1,
            category=self.category,
            description_valid=self.description_valid,
            composite_sha256=self.composite_sha256,
            support_sha256=self.support_sha256,
            query_sha256=self.query_sha256,
            config=locked,
        )
        if hashlib.sha256(_canonical_json(payload)).hexdigest() != self.row_sha256:
            raise RuntimeError("example row identity or metadata were mutated")


def build_example(
    row_index: int,
    row: Mapping[str, object],
    config: SubjectsPilotConfig,
) -> PilotExample:
    locked = _require_subjects_config(config)
    if type(row_index) is not int or row_index not in locked.row_indices:
        raise ValueError("row index must be one of the exact locked rows 0 through 7")
    row_dict = _require_exact_dict_keys(row, locked.feature_order, "dataset row")
    image = row_dict["image"]
    if not isinstance(image, Image.Image):
        raise TypeError("dataset row image must be a PIL image")
    collection = row_dict["collection"]
    if type(collection) is not str or not collection:
        raise TypeError("dataset row collection must be a non-empty exact str")
    quality = _validate_quality(row_dict["quality_assessment"], locked)
    item, description_0, description_1, category, description_valid = (
        _validate_description(row_dict["description"], locked)
    )
    support, query = split_composite_pair(image, locked)
    composite_sha256 = rgb_content_sha256(image)
    support_sha256 = rgb_content_sha256(support)
    query_sha256 = rgb_content_sha256(query)
    identity = _row_identity_payload(
        row_index=row_index,
        collection=collection,
        quality_assessment=quality,
        item=item,
        description_0=description_0,
        description_1=description_1,
        category=category,
        description_valid=description_valid,
        composite_sha256=composite_sha256,
        support_sha256=support_sha256,
        query_sha256=query_sha256,
        config=locked,
    )
    example = PilotExample(
        row_index=row_index,
        collection=collection,
        quality_assessment=quality,
        item=item,
        description_0=description_0,
        description_1=description_1,
        category=category,
        description_valid=description_valid,
        support_rgb=support.tobytes(),
        query_rgb=query.tobytes(),
        composite_sha256=composite_sha256,
        support_sha256=support_sha256,
        query_sha256=query_sha256,
        row_sha256=hashlib.sha256(_canonical_json(identity)).hexdigest(),
    )
    example.validate(locked)
    return example


def _validate_features(features: object, config: SubjectsPilotConfig) -> None:
    if type(features) is not Features:
        raise TypeError("dataset features must be an exact datasets.Features")
    locked = _require_subjects_config(config)
    typed = cast(Features, features)
    if tuple(typed) != locked.feature_order:
        raise ValueError("dataset feature schema order changed")
    image = typed["image"]
    if type(image) is not DatasetImage or image.mode is not None or image.decode is not True:
        raise ValueError("dataset image feature changed from the decoded image contract")
    collection = typed["collection"]
    if type(collection) is not Value or collection.dtype != "string":
        raise ValueError("dataset collection feature must be string")
    for section_name, field_order, expected_dtypes in (
        (
            "quality_assessment",
            locked.quality_field_order,
            ("int64", "int64", "int64"),
        ),
        (
            "description",
            locked.description_field_order,
            ("string", "string", "string", "string", "bool"),
        ),
    ):
        section = typed[section_name]
        if type(section) is not dict or tuple(section) != field_order:
            raise ValueError(f"dataset {section_name} nested feature order changed")
        for name, expected_dtype in zip(field_order, expected_dtypes, strict=True):
            feature = section[name]
            if type(feature) is not Value or feature.dtype != expected_dtype:
                raise ValueError(f"dataset {section_name}.{name} feature dtype changed")


def _parquet_thread_shutdown_barrier() -> None:
    """Wait for the pinned datasets/PyArrow early-iteration shutdown barrier."""

    delay = datasets_config.SLEEP_TIME_ON_THREADS_SHUTDOWN
    if type(delay) is not int or delay != 5:
        raise RuntimeError("datasets Parquet shutdown delay changed from pinned value 5")
    time.sleep(delay)


def hydrate_locked_examples(
    config: SubjectsPilotConfig,
    *,
    cache_dir: Path,
) -> tuple[PilotExample, ...]:
    """Hydrate exactly rows 0--7; this is the module's sole network boundary."""

    locked = _require_subjects_config(config)
    if not isinstance(cache_dir, Path):
        raise TypeError("dataset cache_dir must be a Path")
    # source_file_sha256 is the published LFS object identity at the pinned
    # revision. Streaming eight rows does not claim a local full-shard byte hash.
    rows = load_dataset(
        locked.dataset_id,
        name=locked.config_name,
        data_files={locked.split: [locked.source_file]},
        split=locked.split,
        revision=locked.revision,
        streaming=True,
        cache_dir=str(cache_dir),
        token=False,
    )
    _validate_features(getattr(rows, "features", None), locked)
    iterator = iter(rows)
    examples: list[PilotExample] = []
    try:
        for row_index in locked.row_indices:
            try:
                row = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "dataset ended before every locked row was returned"
                ) from error
            examples.append(build_example(row_index, row, locked))
    finally:
        close = getattr(iterator, "close", None)
        if not callable(close):
            raise TypeError("streaming dataset iterator must expose close()")
        try:
            close()
        finally:
            _parquet_thread_shutdown_barrier()
    return tuple(examples)


def preprocess_query_image(image: Image.Image, *, resolution: int) -> Tensor:
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise TypeError("query image must be a PIL RGB image")
    if image.size != (512, 512):
        raise ValueError("query crop must be exactly 512x512")
    if type(resolution) is not int or resolution != 1024:
        raise ValueError("query preprocessing resolution must remain exactly 1024")
    resized = vision_functional.resize(
        image,
        resolution,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    cropped = vision_functional.center_crop(resized, [resolution, resolution])
    pixels = vision_functional.to_tensor(cropped)
    result = vision_functional.normalize(pixels, [0.5], [0.5])
    if not isinstance(result, Tensor):
        raise TypeError("official SANA image preprocessing must return a Tensor")
    if result.shape != (3, resolution, resolution) or result.dtype != torch.float32:
        raise RuntimeError("official SANA image preprocessing produced an invalid tensor")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("query preprocessing produced non-finite pixels")
    return result


def _module_device(module: nn.Module, context: str) -> torch.device:
    if not isinstance(module, nn.Module):
        raise TypeError(f"{context} must be an nn.Module")
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise ValueError(f"{context} must expose at least one placement tensor")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"{context} tensors must share one device")
    return next(iter(devices))


def _validate_token_batch(value: object, batch_size: int) -> tuple[Tensor, Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("tokenizer output must be a mapping")
    if set(value) != {"input_ids", "attention_mask"}:
        raise ValueError("tokenizer output must contain exactly input_ids and attention_mask")
    input_ids = value["input_ids"]
    attention_mask = value["attention_mask"]
    if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
        raise TypeError("tokenizer outputs must be tensors")
    expected_shape = (batch_size, 300)
    if input_ids.shape != expected_shape or attention_mask.shape != expected_shape:
        raise ValueError("tokenizer outputs must have exact batch by 300 shape")
    if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
        raise TypeError("tokenizer IDs and masks must have dtype torch.int64")
    if input_ids.device.type != "cpu" or attention_mask.device.type != "cpu":
        raise ValueError("tokenizer outputs must begin on CPU")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("prompt attention mask must be binary")
    if bool((attention_mask.sum(dim=1) == 0).any()):
        raise ValueError("every prompt must contain at least one unmasked token")
    if bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
        raise ValueError("prompt attention mask must be monotonically right-padded")
    return input_ids, attention_mask


def _text_hidden(
    texts: list[str],
    *,
    tokenizer: Any,
    text_encoder: nn.Module,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tokenized = tokenizer(
        texts,
        padding="max_length",
        max_length=300,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = _validate_token_batch(tokenized, len(texts))
    with torch.inference_mode():
        output = text_encoder(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            return_dict=True,
        )
    if type(output) is not BaseModelOutputWithPast:
        raise TypeError("text encoder must return an exact BaseModelOutputWithPast")
    hidden = output.last_hidden_state
    if not isinstance(hidden, Tensor):
        raise TypeError("text encoder last_hidden_state must be a Tensor")
    if hidden.shape != (len(texts), 300, 2304):
        raise ValueError("text encoder output must have exact batch x 300 x 2304 shape")
    if hidden.dtype != torch.bfloat16:
        raise TypeError("pinned text encoder output must have dtype torch.bfloat16")
    if hidden.device != device:
        raise ValueError("text encoder output is on the wrong device")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("text encoder output must be finite")
    return hidden, attention_mask


def _encode_text_pair(
    examples: tuple[PilotExample, ...],
    *,
    tokenizer: Any,
    text_encoder: nn.Module,
) -> tuple[Tensor, Tensor, Tensor]:
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("pinned tokenizer must begin with padding_side='left'")
    device = _module_device(text_encoder, "text encoder")
    try:
        tokenizer.padding_side = "right"
        prompts = [example.query_prompt.lower().strip() for example in examples]
        descriptions = [
            example.concept_description.lower().strip() for example in examples
        ]
        prompt_hidden, prompt_mask = _text_hidden(
            prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
        )
        description_hidden, description_mask = _text_hidden(
            descriptions,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
        )
    finally:
        tokenizer.padding_side = "left"
    description_features = masked_mean_description(
        description_hidden.float(), description_mask.to(device)
    )
    return prompt_hidden.float(), prompt_mask, description_features


def _encode_query_latents(
    examples: tuple[PilotExample, ...],
    *,
    vae: nn.Module,
    resolution: int,
) -> Tensor:
    device = _module_device(vae, "VAE")
    config = getattr(vae, "config", None)
    scaling_factor = getattr(config, "scaling_factor", None)
    if type(scaling_factor) is not float or scaling_factor != 0.41407:
        raise ValueError("VAE scaling factor must remain the exact float 0.41407")
    latents: list[Tensor] = []
    encode_method = getattr(vae, "encode", None)
    if not callable(encode_method):
        raise TypeError("VAE must expose a callable encode method")
    encode = cast(Callable[..., object], encode_method)
    for example in examples:
        pixels = preprocess_query_image(
            example.query_image(), resolution=resolution
        ).unsqueeze(0)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=False):
                output = encode(
                    pixels.to(device=device, dtype=torch.float32), return_dict=True
                )
        if type(output) is not EncoderOutput:
            raise TypeError("AutoencoderDC.encode must return an exact EncoderOutput")
        latent = output.latent
        if not isinstance(latent, Tensor):
            raise TypeError("EncoderOutput.latent must be a Tensor")
        if latent.shape != (1, 32, 32, 32):
            raise ValueError("VAE latent shape must be exactly 1x32x32x32 per microbatch")
        if latent.dtype != torch.float32 or latent.device != device:
            raise TypeError("VAE latent must be float32 on the VAE device")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("VAE latent must be finite")
        latents.append(latent * scaling_factor)
    return torch.cat(latents, dim=0)


def _normal_cpu_tensor(tensor: Tensor, *, dtype: torch.dtype) -> Tensor:
    with torch.inference_mode(False):
        return tensor.detach().to(device="cpu", dtype=dtype).contiguous().clone()


def _validate_cache_tensors(tensors: Mapping[str, Tensor]) -> None:
    if tuple(tensors) != tuple(CACHE_TENSOR_SPECS):
        raise ValueError("cache tensor keys and order changed from the exact contract")
    for name, (expected_shape, expected_dtype) in CACHE_TENSOR_SPECS.items():
        tensor = tensors[name]
        if type(tensor) is not Tensor:
            raise TypeError(f"cache tensor {name} must be an exact Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"cache tensor {name} has the wrong exact shape")
        if tensor.dtype != expected_dtype:
            raise TypeError(f"cache tensor {name} has the wrong exact dtype")
        if tensor.device.type != "cpu":
            raise ValueError(f"cache tensor {name} must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError(f"cache tensor {name} must be contiguous")
        if tensor.requires_grad or tensor.is_inference():
            raise ValueError(f"cache tensor {name} must be detached and non-inference")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"cache tensor {name} must be finite")
    prompt_mask = tensors["prompt_attention_mask"]
    if not bool(((prompt_mask == 0) | (prompt_mask == 1)).all()):
        raise ValueError("cached prompt attention mask must be binary")
    if bool((prompt_mask.sum(dim=1) == 0).any()):
        raise ValueError("every cached prompt must contain a valid token")
    if bool((prompt_mask[:, 1:] > prompt_mask[:, :-1]).any()):
        raise ValueError("cached prompt mask must be monotonically right-padded")
    support_mask = tensors["support_mask"]
    if not bool(support_mask.all()):
        raise ValueError("cached support mask must be entirely true")


def _validate_examples(
    examples: Sequence[PilotExample], config: SubjectsPilotConfig
) -> tuple[PilotExample, ...]:
    if not isinstance(examples, Sequence) or isinstance(examples, str | bytes | bytearray):
        raise TypeError("examples must be a sequence")
    copied = tuple(examples)
    if len(copied) != 8:
        raise ValueError("precompute requires exactly eight locked examples")
    for example in copied:
        if type(example) is not PilotExample:
            raise TypeError("every example must be an exact PilotExample")
        example.validate(config)
    if tuple(example.row_index for example in copied) != config.row_indices:
        raise ValueError("examples must preserve exact locked row order 0 through 7")
    return copied


def _require_pinned_components(components: object) -> PinnedComponents:
    if type(components) is not PinnedComponents:
        raise TypeError("components must be an exact PinnedComponents")
    pinned = components
    pinned.validate()
    return pinned


def _precompute_tensors_impl(
    examples: Sequence[PilotExample],
    components: Any,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> dict[str, Tensor]:
    sana = _require_sana_config(sana_config)
    subjects = _require_subjects_config(dataset_config)
    locked_examples = _validate_examples(examples, subjects)
    vae = cast(nn.Module, components.vae)
    text_encoder = cast(nn.Module, components.text_encoder)
    support_encoder = cast(nn.Module, components.support_encoder)
    clean_latents = _encode_query_latents(
        locked_examples, vae=vae, resolution=sana.resolution
    )
    prompt_embeddings, prompt_attention_mask, description_features = (
        _encode_text_pair(
            locked_examples,
            tokenizer=components.tokenizer,
            text_encoder=text_encoder,
        )
    )
    support_device = _module_device(support_encoder, "support encoder")
    support_features = encode_support_images(
        [example.support_image() for example in locked_examples],
        processor=components.support_processor,
        encoder=support_encoder,
        device=support_device,
    ).unsqueeze(1)
    raw = {
        "clean_latents": clean_latents,
        "prompt_embeddings": prompt_embeddings,
        "prompt_attention_mask": prompt_attention_mask,
        "support_features": support_features,
        "support_mask": torch.ones((8, 1), dtype=torch.bool),
        "description_features": description_features,
    }
    tensors = {
        name: _normal_cpu_tensor(raw[name], dtype=dtype)
        for name, (_, dtype) in CACHE_TENSOR_SPECS.items()
    }
    _validate_cache_tensors(tensors)
    return tensors


def precompute_tensors(
    examples: Sequence[PilotExample],
    components: PinnedComponents,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> dict[str, Tensor]:
    """Offline deterministic production boundary for exact pinned components."""

    pinned = _require_pinned_components(components)
    return _precompute_tensors_impl(
        examples, pinned, sana_config, dataset_config
    )


def _precompute_tensors_for_test(
    examples: Sequence[PilotExample],
    components: object,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> dict[str, Tensor]:
    """Synthetic-only seam; production callers must use precompute_tensors."""

    return _precompute_tensors_impl(
        examples, components, sana_config, dataset_config
    )


def _identity_payload(
    examples: tuple[PilotExample, ...],
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> dict[str, object]:
    return {
        "sana_revision": sana_config.revision,
        "support_revision": sana_config.support_revision,
        "dataset_revision": dataset_config.revision,
        "dataset_config_sha256": dataset_config.canonical_sha256,
        "row_indices": [example.row_index for example in examples],
        "row_sha256": [example.row_sha256 for example in examples],
    }


def cache_metadata(
    *,
    identity_sha256: str,
    sana_revision: str,
    support_revision: str,
    dataset_revision: str,
) -> dict[str, str]:
    _require_sha256(identity_sha256, "cache identity")
    expected_revisions = (
        (sana_revision, _SANA_REVISION, "SANA"),
        (support_revision, _SUPPORT_REVISION, "support"),
        (dataset_revision, _DATASET_REVISION, "dataset"),
    )
    for actual, expected, context in expected_revisions:
        if type(actual) is not str or actual != expected:
            raise ValueError(f"{context} revision changed from the pinned cache contract")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "scope": _SCOPE,
        "publication_eligible": False,
        "identity_sha256": identity_sha256,
        "sana_revision": sana_revision,
        "support_revision": support_revision,
        "dataset_revision": dataset_revision,
    }
    return {"ratemem": _canonical_json(payload).decode("utf-8")}


def _absolute_without_resolution(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("cache path must be a Path")
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = _absolute_without_resolution(path)
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError(f"cache path is unsafe because {current} is a symlink")


def _validate_private_directory(path: Path, context: str) -> os.stat_result:
    _assert_no_symlink_ancestors(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{context} does not exist: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"{context} must be a real directory")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{context} must be owned by the current uid")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError(f"{context} must have exact mode 0700")
    return metadata


def _validate_private_file(path: Path, context: str) -> os.stat_result:
    _assert_no_symlink_ancestors(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{context} does not exist: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(f"{context} must be a regular non-symlink file")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{context} must be owned by the current uid")
    if metadata.st_nlink != 1:
        raise OSError(f"{context} must have exactly one hard link")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError(f"{context} must have exact mode 0600")
    return metadata


def _read_private_file(path: Path, context: str) -> bytes:
    before = _validate_private_file(path, context)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise OSError(f"{context} changed during secure open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_file(path: Path, content: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("private cache file write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _validate_private_file(path, "new cache file")


@contextmanager
def _cache_lock(path: Path) -> Iterator[None]:
    _assert_no_symlink_ancestors(path)
    created = False
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(path, flags)
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise PermissionError("cache lock must be an owner-only 0600 single-link file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_private_file(path, "cache lock")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _sha256_file(path: Path) -> tuple[str, int]:
    content = _read_private_file(path, "cache content file")
    return hashlib.sha256(content).hexdigest(), len(content)


def _tensor_specs_manifest() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(shape),
            "dtype": _SAFETENSORS_DTYPES[dtype],
        }
        for name, (shape, dtype) in CACHE_TENSOR_SPECS.items()
    ]


def _manifest_payload(
    *,
    identity: dict[str, object],
    identity_sha256: str,
    features_sha256: str,
    features_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": _SCOPE,
        "publication_eligible": False,
        "identity_sha256": identity_sha256,
        "identity": identity,
        "features": {
            "filename": "features.safetensors",
            "sha256": features_sha256,
            "bytes": features_bytes,
            "tensors": _tensor_specs_manifest(),
        },
    }


def _validate_identity_payload(value: object) -> dict[str, object]:
    identity = _require_exact_dict_keys(
        value,
        (
            "sana_revision",
            "support_revision",
            "dataset_revision",
            "dataset_config_sha256",
            "row_indices",
            "row_sha256",
        ),
        "cache identity",
    )
    expected = (
        (identity["sana_revision"], _SANA_REVISION, "SANA revision"),
        (identity["support_revision"], _SUPPORT_REVISION, "support revision"),
        (identity["dataset_revision"], _DATASET_REVISION, "dataset revision"),
    )
    for actual, pinned, context in expected:
        if type(actual) is not str or actual != pinned:
            raise ValueError(f"cache {context} changed")
    if (
        type(identity["dataset_config_sha256"]) is not str
        or identity["dataset_config_sha256"] != SUBJECTS_PILOT_CANONICAL_SHA256
    ):
        raise ValueError("cache dataset config hash changed from the unique canonical config")
    row_indices = identity["row_indices"]
    row_hashes = identity["row_sha256"]
    _require_exact_json_value(
        row_indices, list(range(8)), "cache row indices"
    )
    if type(row_hashes) is not list or len(row_hashes) != 8:
        raise ValueError("cache must bind exactly eight row hashes")
    for row_hash in row_hashes:
        _require_sha256(row_hash, "cache row hash")
    return identity


def _validate_manifest(value: object) -> dict[str, Any]:
    manifest = _require_exact_dict_keys(
        value,
        (
            "schema_version",
            "scope",
            "publication_eligible",
            "identity_sha256",
            "identity",
            "features",
        ),
        "cache manifest",
    )
    if manifest["schema_version"] != _SCHEMA_VERSION or type(
        manifest["schema_version"]
    ) is not str:
        raise ValueError("cache manifest schema version changed")
    if manifest["scope"] != _SCOPE or type(manifest["scope"]) is not str:
        raise ValueError("cache manifest scope changed")
    if type(manifest["publication_eligible"]) is not bool or manifest[
        "publication_eligible"
    ]:
        raise ValueError("cache manifest must remain publication ineligible")
    identity_sha256 = _require_sha256(
        manifest["identity_sha256"], "cache identity"
    )
    identity = _validate_identity_payload(manifest["identity"])
    if hashlib.sha256(_canonical_json(identity)).hexdigest() != identity_sha256:
        raise RuntimeError("cache identity checksum does not match its payload")
    features = _require_exact_dict_keys(
        manifest["features"],
        ("filename", "sha256", "bytes", "tensors"),
        "cache features manifest",
    )
    if type(features["filename"]) is not str or features["filename"] != (
        "features.safetensors"
    ):
        raise ValueError("cache feature filename changed")
    _require_sha256(features["sha256"], "cache feature checksum")
    if type(features["bytes"]) is not int or features["bytes"] <= 0:
        raise ValueError("cache feature byte count must be an exact positive integer")
    _require_exact_json_value(
        features["tensors"], _tensor_specs_manifest(), "cache tensor manifest specs"
    )
    return cast(dict[str, Any], manifest)


def _validate_safetensors_header(
    path: Path,
    *,
    expected_metadata: dict[str, str],
) -> None:
    _validate_private_file(path, "features.safetensors")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if handle.metadata() != expected_metadata:
            raise RuntimeError("safetensors metadata changed from the exact cache contract")
        if tuple(handle.keys()) != tuple(sorted(CACHE_TENSOR_SPECS)):
            raise ValueError("safetensors keys changed from the exact cache contract")
        for name, (shape, dtype) in CACHE_TENSOR_SPECS.items():
            tensor_slice = handle.get_slice(name)
            if tuple(tensor_slice.get_shape()) != shape:
                raise ValueError(f"safetensors {name} header has the wrong shape")
            if tensor_slice.get_dtype() != _SAFETENSORS_DTYPES[dtype]:
                raise TypeError(f"safetensors {name} header has the wrong dtype")
    _validate_private_file(path, "features.safetensors")


@dataclass(frozen=True, slots=True)
class PilotCacheReceipt:
    identity_sha256: str
    manifest_sha256: str
    manifest_byte_count: int
    features_sha256: str
    features_byte_count: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self) is not PilotCacheReceipt:
            raise TypeError("receipt must be an exact PilotCacheReceipt")
        for name in (
            "identity_sha256",
            "manifest_sha256",
            "features_sha256",
        ):
            _require_sha256(getattr(self, name), f"receipt {name}")
        for name in ("manifest_byte_count", "features_byte_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"receipt {name} must be an exact positive byte count")


@dataclass(frozen=True, slots=True)
class PrecomputedPilotData:
    root: Path
    tensors: Mapping[str, Tensor]
    manifest: dict[str, Any]
    receipt: PilotCacheReceipt

    def __post_init__(self) -> None:
        if type(self) is not PrecomputedPilotData:
            raise TypeError("pilot data must be an exact PrecomputedPilotData")
        if type(self.receipt) is not PilotCacheReceipt:
            raise TypeError("pilot data receipt must be an exact PilotCacheReceipt")
        self.receipt.validate()

    @property
    def identity_sha256(self) -> str:
        return self.receipt.identity_sha256


def load_precomputed_cache(
    root: Path,
    *,
    expected_receipt: PilotCacheReceipt,
) -> PrecomputedPilotData:
    """Strictly load an existing cache; missing/corrupt input never triggers hydration."""

    if type(expected_receipt) is not PilotCacheReceipt:
        raise TypeError("expected_receipt must be an exact PilotCacheReceipt")
    expected_receipt.validate()
    cache_root = _absolute_without_resolution(root)
    _validate_private_directory(cache_root, "pilot cache")
    expected_entries = {"manifest.json", "features.safetensors", "complete"}
    if {entry.name for entry in cache_root.iterdir()} != expected_entries:
        raise RuntimeError("pilot cache is partial or contains unexpected files")
    manifest_path = cache_root / "manifest.json"
    features_path = cache_root / "features.safetensors"
    complete_path = cache_root / "complete"
    for path, context in (
        (manifest_path, "cache manifest"),
        (features_path, "features.safetensors"),
        (complete_path, "cache completion marker"),
    ):
        _validate_private_file(path, context)

    complete = _read_private_file(complete_path, "cache completion marker")
    if complete != _COMPLETE_MARKER:
        raise RuntimeError("cache completion marker changed")
    manifest_bytes = _read_private_file(manifest_path, "cache manifest")
    if len(manifest_bytes) != expected_receipt.manifest_byte_count:
        raise RuntimeError("cache manifest byte count does not match external receipt")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_receipt.manifest_sha256:
        raise RuntimeError("cache manifest checksum does not match external receipt")
    decoded = json.loads(
        manifest_bytes.decode("utf-8"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_nonfinite_constant,
    )
    if _canonical_json(decoded) != manifest_bytes:
        raise ValueError("cache manifest bytes are not exact canonical JSON")
    manifest = _validate_manifest(decoded)
    if manifest["identity_sha256"] != expected_receipt.identity_sha256:
        raise ValueError("cache identity does not match the external receipt")

    features_sha256, features_bytes = _sha256_file(features_path)
    if features_bytes != expected_receipt.features_byte_count:
        raise RuntimeError("features byte count does not match external receipt")
    if features_sha256 != expected_receipt.features_sha256:
        raise RuntimeError("features checksum does not match external receipt")
    features_manifest = cast(dict[str, Any], manifest["features"])
    if features_sha256 != features_manifest["sha256"]:
        raise RuntimeError("features.safetensors checksum changed")
    if features_bytes != features_manifest["bytes"]:
        raise RuntimeError("features.safetensors byte count changed")
    identity = cast(dict[str, object], manifest["identity"])
    metadata = cache_metadata(
        identity_sha256=expected_receipt.identity_sha256,
        sana_revision=cast(str, identity["sana_revision"]),
        support_revision=cast(str, identity["support_revision"]),
        dataset_revision=cast(str, identity["dataset_revision"]),
    )
    _validate_safetensors_header(features_path, expected_metadata=metadata)
    loaded = load_file(features_path, device="cpu")
    tensors = {name: loaded[name] for name in CACHE_TENSOR_SPECS}
    _validate_cache_tensors(tensors)
    return PrecomputedPilotData(
        root=cache_root,
        tensors=MappingProxyType(tensors),
        manifest=manifest,
        receipt=expected_receipt,
    )


def _discover_waiter_receipt(root: Path) -> PilotCacheReceipt:
    """Discover only a cache that appeared while this owner waited on its lock."""

    cache_root = _absolute_without_resolution(root)
    _validate_private_directory(cache_root, "concurrent pilot cache")
    expected_entries = {"manifest.json", "features.safetensors", "complete"}
    if {entry.name for entry in cache_root.iterdir()} != expected_entries:
        raise RuntimeError("concurrent pilot cache is partial or unexpected")
    complete = _read_private_file(cache_root / "complete", "cache completion marker")
    if complete != _COMPLETE_MARKER:
        raise RuntimeError("concurrent cache completion marker changed")
    manifest_bytes = _read_private_file(cache_root / "manifest.json", "cache manifest")
    decoded = json.loads(
        manifest_bytes.decode("utf-8"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_nonfinite_constant,
    )
    if _canonical_json(decoded) != manifest_bytes:
        raise ValueError("concurrent cache manifest is not canonical JSON")
    manifest = _validate_manifest(decoded)
    features_sha256, features_bytes = _sha256_file(
        cache_root / "features.safetensors"
    )
    return PilotCacheReceipt(
        identity_sha256=cast(str, manifest["identity_sha256"]),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_byte_count=len(manifest_bytes),
        features_sha256=features_sha256,
        features_byte_count=features_bytes,
    )


def _remove_owned_staging(path: Path) -> None:
    if not path.exists():
        return
    for name in ("complete", "manifest.json", "features.safetensors"):
        candidate = path / name
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid():
            candidate.unlink()
    try:
        path.rmdir()
    except OSError:
        pass


def _build_precomputed_cache_impl(
    examples: Sequence[PilotExample],
    output_dir: Path,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
    tensor_builder: Callable[[], Mapping[str, Tensor]],
) -> PrecomputedPilotData:
    sana = _require_sana_config(sana_config)
    subjects = _require_subjects_config(dataset_config)
    locked_examples = _validate_examples(examples, subjects)
    output = _absolute_without_resolution(output_dir)
    _assert_no_symlink_ancestors(output)
    parent = output.parent
    _validate_private_directory(parent, "pilot cache parent")
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "pilot cache already exists; reuse requires load_precomputed_cache "
            "with its retained external receipt"
        )
    lock_path = output.with_name(f"{output.name}.lock")
    with _cache_lock(lock_path):
        _assert_no_symlink_ancestors(output)
        identity = _identity_payload(locked_examples, sana, subjects)
        identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
        if output.exists() or output.is_symlink():
            waiter_receipt = _discover_waiter_receipt(output)
            if waiter_receipt.identity_sha256 != identity_sha256:
                raise ValueError("concurrent pilot cache has a different identity")
            return load_precomputed_cache(
                output, expected_receipt=waiter_receipt
            )

        tensors = dict(tensor_builder())
        _validate_cache_tensors(tensors)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent)
        )
        published = False
        try:
            os.chmod(staging, 0o700)
            _validate_private_directory(staging, "cache staging directory")
            features_path = staging / "features.safetensors"
            metadata = cache_metadata(
                identity_sha256=identity_sha256,
                sana_revision=sana.revision,
                support_revision=sana.support_revision,
                dataset_revision=subjects.revision,
            )
            save_file(dict(tensors), features_path, metadata=metadata)
            os.chmod(features_path, 0o600)
            _validate_private_file(features_path, "staged features.safetensors")
            _fsync_file(features_path)
            features_sha256, features_bytes = _sha256_file(features_path)
            manifest = _manifest_payload(
                identity=identity,
                identity_sha256=identity_sha256,
                features_sha256=features_sha256,
                features_bytes=features_bytes,
            )
            manifest_bytes = _canonical_json(manifest)
            _write_private_file(staging / "manifest.json", manifest_bytes)
            _write_private_file(staging / "complete", _COMPLETE_MARKER)
            receipt = PilotCacheReceipt(
                identity_sha256=identity_sha256,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                manifest_byte_count=len(manifest_bytes),
                features_sha256=features_sha256,
                features_byte_count=features_bytes,
            )
            _fsync_directory(staging)
            if output.exists() or output.is_symlink():
                raise FileExistsError("pilot cache final path appeared before atomic commit")
            os.rename(staging, output)
            published = True
            _fsync_directory(parent)
        except BaseException:
            if published and output.exists() and not staging.exists():
                try:
                    os.rename(output, staging)
                except OSError:
                    pass
            _remove_owned_staging(staging)
            raise
        return load_precomputed_cache(
            output, expected_receipt=receipt
        )


def build_precomputed_cache(
    examples: Sequence[PilotExample],
    components: PinnedComponents,
    output_dir: Path,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> PrecomputedPilotData:
    """Build a new production cache from exact validated pinned components."""

    pinned = _require_pinned_components(components)
    return _build_precomputed_cache_impl(
        examples,
        output_dir,
        sana_config,
        dataset_config,
        lambda: _precompute_tensors_impl(
            examples, pinned, sana_config, dataset_config
        ),
    )


def _build_precomputed_cache_for_test(
    examples: Sequence[PilotExample],
    components: object,
    output_dir: Path,
    sana_config: SanaPilotConfig,
    dataset_config: SubjectsPilotConfig,
) -> PrecomputedPilotData:
    """Synthetic-only seam; production callers must use build_precomputed_cache."""

    return _build_precomputed_cache_impl(
        examples,
        output_dir,
        sana_config,
        dataset_config,
        lambda: _precompute_tensors_impl(
            examples, components, sana_config, dataset_config
        ),
    )
