from __future__ import annotations

import ast
import hashlib
import inspect
import json
import multiprocessing
import os
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Features, Value
from datasets import Image as DatasetImage
from diffusers.models.autoencoders.vae import EncoderOutput
from PIL import Image, PngImagePlugin
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional
from transformers.modeling_outputs import BaseModelOutputWithPast

import ratemem.pilot.data as pilot_data
from ratemem.pilot.config import SanaPilotConfig, SubjectsPilotConfig
from ratemem.pilot.data import (
    CACHE_TENSOR_SPECS,
    PilotCacheReceipt,
    PilotExample,
    build_example,
    cache_metadata,
    hydrate_locked_examples,
    load_precomputed_cache,
    preprocess_query_image,
    rgb_content_sha256,
    split_composite_pair,
)
from ratemem.sana.components import PinnedComponents

_test_precompute_tensors = pilot_data._precompute_tensors_for_test
_test_build_precomputed_cache = pilot_data._build_precomputed_cache_for_test

SUBJECTS_CONFIG_PATH = Path("configs/pilot/subjects200k-held-in.json")
SANA_CONFIG_PATH = Path("configs/pilot/sana-1.5-1.6b.json")
DATASET_REVISION = "0d1cf6536239888f1a8e218790649344810067bc"
SHARD_SHA256 = "3d696ccbdfc736961e75e5b7ce33adae40cd70ffb69cdc27020a25d643971903"

QUALITY_BY_ROW: tuple[tuple[int, int, int] | None, ...] = (
    (5, 5, 5),
    None,
    (5, 5, 5),
    (0, 1, 5),
    (5, 5, 5),
    (5, 5, 5),
    (0, 4, 5),
    (5, 5, 5),
)


def _subjects_payload() -> dict[str, Any]:
    value = json.loads(SUBJECTS_CONFIG_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _write_json(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "subjects.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if type(value) is dict:
        result: list[tuple[object, ...]] = []
        for key, child in value.items():
            result.extend(_leaf_paths(child, (*prefix, key)))
        return result
    if type(value) is list:
        result = []
        for index, child in enumerate(value):
            result.extend(_leaf_paths(child, (*prefix, index)))
        return result
    return [prefix]


def _at_path(payload: object, path: tuple[object, ...]) -> Any:
    value: Any = payload
    for segment in path:
        value = value[segment]
    return value


def _set_path(payload: object, path: tuple[object, ...], value: object) -> None:
    parent: Any = payload
    for segment in path[:-1]:
        parent = parent[segment]
    parent[path[-1]] = value


def _different_value(value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return value + "-tampered"
    raise AssertionError(f"unhandled canonical value {value!r}")


def _wrong_equal_type_cases() -> list[tuple[tuple[object, ...], object]]:
    if not SUBJECTS_CONFIG_PATH.exists():
        return []
    payload = _subjects_payload()
    result: list[tuple[tuple[object, ...], object]] = []
    for path in _leaf_paths(payload):
        value = _at_path(payload, path)
        if type(value) is bool:
            result.append((path, int(value)))
        elif type(value) is int:
            result.append((path, float(value)))
    return result


def _dataset_features() -> Features:
    return Features(
        {
            "image": DatasetImage(),
            "collection": Value("string"),
            "quality_assessment": {
                "compositeStructure": Value("int64"),
                "objectConsistency": Value("int64"),
                "imageQuality": Value("int64"),
            },
            "description": {
                "item": Value("string"),
                "description_0": Value("string"),
                "description_1": Value("string"),
                "category": Value("string"),
                "description_valid": Value("bool"),
            },
        }
    )


def _composite(seed: int = 0) -> Image.Image:
    image = Image.new("RGB", (1056, 528), (seed, seed, seed))
    image.paste(Image.new("RGB", (512, 512), (255, seed, 0)), (8, 8))
    image.paste(Image.new("RGB", (512, 512), (0, seed, 255)), (528, 8))
    return image


def _quality_payload(row_index: int) -> dict[str, int] | None:
    quality = QUALITY_BY_ROW[row_index]
    if quality is None:
        return None
    return {
        "compositeStructure": quality[0],
        "objectConsistency": quality[1],
        "imageQuality": quality[2],
    }


def _row(row_index: int) -> dict[str, object]:
    return {
        "image": _composite(row_index),
        "collection": "collection_1",
        "quality_assessment": _quality_payload(row_index),
        "description": {
            "item": "Eames Lounge Chair",
            "description_0": f"Studio chair {row_index}",
            "description_1": f"Outdoor chair {row_index}",
            "category": "furniture",
            "description_valid": True,
        },
    }


def _examples(config: SubjectsPilotConfig | None = None) -> tuple[PilotExample, ...]:
    locked = config or SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    return tuple(build_example(index, _row(index), locked) for index in range(8))


def _valid_tensors() -> dict[str, Tensor]:
    return {
        "clean_latents": torch.zeros((8, 32, 32, 32), dtype=torch.float32),
        "prompt_embeddings": torch.zeros((8, 300, 2304), dtype=torch.float32),
        "prompt_attention_mask": torch.ones((8, 300), dtype=torch.int64),
        "support_features": torch.zeros((8, 1, 384), dtype=torch.float32),
        "support_mask": torch.ones((8, 1), dtype=torch.bool),
        "description_features": torch.zeros((8, 2304), dtype=torch.float32),
    }


def _secure_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    return parent


def _rechain_cache_manifest(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = output / "features.safetensors"
    manifest["features"]["sha256"] = hashlib.sha256(features.read_bytes()).hexdigest()
    manifest["features"]["bytes"] = features.stat().st_size
    encoded = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    manifest_path.write_bytes(encoded)
    os.chmod(manifest_path, 0o600)
    (output / "complete").write_text(
        "ratemem-cache-complete-v1\n", encoding="ascii"
    )
    os.chmod(output / "complete", 0o600)


def _write_rechained_manifest(output: Path, manifest: dict[str, Any]) -> None:
    encoded = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    (output / "manifest.json").write_bytes(encoded)
    os.chmod(output / "manifest.json", 0o600)
    (output / "complete").write_text(
        "ratemem-cache-complete-v1\n", encoding="ascii"
    )
    os.chmod(output / "complete", 0o600)


def _fake_compute(
    monkeypatch: pytest.MonkeyPatch, tensors: dict[str, Tensor] | None = None
) -> None:
    payload = tensors or _valid_tensors()
    monkeypatch.setattr(
        pilot_data,
        "_precompute_tensors_impl",
        lambda *args, **kwargs: {key: value.clone() for key, value in payload.items()},
    )


def _dummy_receipt(identity_sha256: str = "0" * 64) -> PilotCacheReceipt:
    return PilotCacheReceipt(
        identity_sha256=identity_sha256,
        manifest_sha256="1" * 64,
        manifest_byte_count=1,
        features_sha256="2" * 64,
        features_byte_count=1,
    )


def _receipt_from_files(
    output: Path, *, identity_sha256: str
) -> PilotCacheReceipt:
    manifest = (output / "manifest.json").read_bytes()
    features = (output / "features.safetensors").read_bytes()
    return PilotCacheReceipt(
        identity_sha256=identity_sha256,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_byte_count=len(manifest),
        features_sha256=hashlib.sha256(features).hexdigest(),
        features_byte_count=len(features),
    )


def test_pilot_cache_receipt_is_exact_frozen_and_self_validating() -> None:
    receipt = _dummy_receipt()
    receipt.validate()
    assert replace(receipt) == receipt
    with pytest.raises(FrozenInstanceError):
        receipt.features_byte_count = 2  # type: ignore[misc]
    for field_name, replacement in (
        ("identity_sha256", "g" * 64),
        ("manifest_sha256", "0" * 63),
        ("manifest_byte_count", True),
        ("features_sha256", "f" * 63),
        ("features_byte_count", 0),
    ):
        with pytest.raises((TypeError, ValueError), match="receipt|SHA|byte|exact"):
            replace(receipt, **{field_name: replacement})
    object.__setattr__(receipt, "features_sha256", "g" * 64)
    with pytest.raises(ValueError, match="receipt|SHA"):
        receipt.validate()

    class _Subclass(PilotCacheReceipt):
        pass

    with pytest.raises(TypeError, match="exact PilotCacheReceipt"):
        _Subclass(
            identity_sha256="0" * 64,
            manifest_sha256="1" * 64,
            manifest_byte_count=1,
            features_sha256="2" * 64,
            features_byte_count=1,
        )


def test_production_precompute_and_build_reject_arbitrary_component_bundles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    examples = _examples()
    sana = SanaPilotConfig.load(SANA_CONFIG_PATH)
    subjects = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    fake = _fake_components()
    with pytest.raises(TypeError, match="exact PinnedComponents"):
        pilot_data.precompute_tensors(examples, fake, sana, subjects)
    absent_parent = tmp_path / "must-not-be-created"
    with pytest.raises(TypeError, match="exact PinnedComponents"):
        pilot_data.build_precomputed_cache(
            examples, fake, absent_parent / "cache", sana, subjects
        )
    assert not absent_parent.exists()

    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        examples, fake, output, sana, subjects
    )
    with pytest.raises(TypeError, match="exact PinnedComponents"):
        pilot_data.build_precomputed_cache(
            examples, fake, output, sana, subjects
        )
    assert load_precomputed_cache(
        output, expected_receipt=built.receipt
    ).receipt == built.receipt


def test_exact_pinned_components_validate_before_any_production_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exact = object.__new__(PinnedComponents)
    calls = 0

    def reject(self: PinnedComponents) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("component validation failed")

    monkeypatch.setattr(PinnedComponents, "validate", reject)
    with pytest.raises(RuntimeError, match="component validation failed"):
        pilot_data.precompute_tensors(
            _examples(),
            exact,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    output = tmp_path / "must-not-exist" / "cache"
    with pytest.raises(RuntimeError, match="component validation failed"):
        pilot_data.build_precomputed_cache(
            _examples(),
            exact,
            output,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert calls == 2
    assert not output.parent.exists()


def test_production_entry_rejects_pinned_components_subclasses(tmp_path: Path) -> None:
    class _Subclass(PinnedComponents):
        pass

    fake_subclass = object.__new__(_Subclass)
    with pytest.raises(TypeError, match="exact PinnedComponents"):
        pilot_data.precompute_tensors(
            _examples(),
            fake_subclass,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    with pytest.raises(TypeError, match="exact PinnedComponents"):
        pilot_data.build_precomputed_cache(
            _examples(),
            fake_subclass,
            tmp_path / "cache",
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )


def test_committed_subjects_config_is_the_exact_engineering_only_contract() -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)

    assert config.schema_version == "1.0.0"
    assert config.scope == "engineering_pilot_only"
    assert config.publication_eligible is False
    assert config.dataset_id == "Yuanshi/Subjects200K"
    assert config.revision == DATASET_REVISION
    assert config.config_name == "default"
    assert config.split == "train"
    assert config.source_file == "data/train-00000-of-00032.parquet"
    assert config.source_file_sha256 == SHARD_SHA256
    assert config.row_indices == tuple(range(8))
    assert config.held_in is True
    assert (
        config.held_in_meaning
        == "public_train_rows_engineering_smoke_not_scientific_holdout"
    )
    assert config.mode == "RGB"
    assert config.size == (1056, 528)
    assert config.left_crop == (8, 8, 520, 520)
    assert config.right_crop == (528, 8, 1040, 520)
    assert config.feature_order == (
        "image",
        "collection",
        "quality_assessment",
        "description",
    )
    with pytest.raises(FrozenInstanceError):
        config.split = "test"  # type: ignore[misc]


@pytest.mark.parametrize(
    "leaf_path", _leaf_paths(_subjects_payload()) if SUBJECTS_CONFIG_PATH.exists() else []
)
def test_every_subjects_config_leaf_is_locked(
    leaf_path: tuple[object, ...], tmp_path: Path
) -> None:
    payload = _subjects_payload()
    _set_path(payload, leaf_path, _different_value(_at_path(payload, leaf_path)))
    with pytest.raises(ValueError, match="canonical|exact|changed|type"):
        SubjectsPilotConfig.load(_write_json(tmp_path, payload))


@pytest.mark.parametrize(("leaf_path", "replacement"), _wrong_equal_type_cases())
def test_subjects_config_rejects_equal_values_with_wrong_exact_types(
    leaf_path: tuple[object, ...], replacement: object, tmp_path: Path
) -> None:
    payload = _subjects_payload()
    _set_path(payload, leaf_path, replacement)
    with pytest.raises(ValueError, match="canonical|exact|type"):
        SubjectsPilotConfig.load(_write_json(tmp_path, payload))


@pytest.mark.parametrize(
    "section",
    [None, "dataset", "semantics", "composite"],
)
def test_subjects_config_object_order_is_canonical(
    section: str | None, tmp_path: Path
) -> None:
    payload = _subjects_payload()
    target = payload if section is None else payload[section]
    assert type(target) is dict
    first = next(iter(target))
    target[first] = target.pop(first)
    with pytest.raises(ValueError, match="order|canonical|exact"):
        SubjectsPilotConfig.load(_write_json(tmp_path, payload))


@pytest.mark.parametrize(
    "list_path",
    [
        ("dataset", "row_indices"),
        ("composite", "size"),
        ("composite", "left_crop"),
        ("composite", "right_crop"),
        ("feature_order",),
        ("quality_field_order",),
        ("description_field_order",),
    ],
)
def test_subjects_config_list_order_and_members_are_canonical(
    list_path: tuple[str, ...], tmp_path: Path
) -> None:
    payload = _subjects_payload()
    values = _at_path(payload, list_path)
    assert type(values) is list and len(values) > 1
    values.reverse()
    with pytest.raises(ValueError, match="canonical|exact|changed"):
        SubjectsPilotConfig.load(_write_json(tmp_path, payload))


def test_subjects_config_rejects_duplicates_nonfinite_and_coordinated_geometry(
    tmp_path: Path,
) -> None:
    text = SUBJECTS_CONFIG_PATH.read_text(encoding="utf-8")
    invalid_texts = (
        text.replace(
            '  "schema_version": "1.0.0",',
            '  "schema_version": "1.0.0",\n  "schema_version": "1.0.0",',
            1,
        ),
        text.replace(
            f'    "revision": "{DATASET_REVISION}",',
            (
                f'    "revision": "{DATASET_REVISION}",\n'
                f'    "revision": "{DATASET_REVISION}",'
            ),
            1,
        ),
        text.replace('    "image_size": 512,', '    "image_size": NaN,', 1),
        text.replace('    "image_size": 512,', '    "image_size": Infinity,', 1),
        text.replace('    "image_size": 512,', '    "image_size": -Infinity,', 1),
    )
    for index, invalid in enumerate(invalid_texts):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate|finite|constant"):
            SubjectsPilotConfig.load(path)

    coordinated = _subjects_payload()
    coordinated["composite"].update(
        {
            "size": [2080, 1040],
            "image_size": 1024,
            "padding_pixels": 8,
            "left_crop": [8, 8, 1032, 1032],
            "right_crop": [1040, 8, 2064, 1032],
        }
    )
    with pytest.raises(ValueError, match="canonical|exact|changed"):
        SubjectsPilotConfig.load(_write_json(tmp_path, coordinated))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate-list"])
def test_subjects_config_rejects_missing_extra_and_duplicate_members(
    mutation: str, tmp_path: Path
) -> None:
    payload = _subjects_payload()
    if mutation == "missing":
        payload["dataset"].pop("license_spdx")
    elif mutation == "extra":
        payload["semantics"]["claim"] = "scientific"
    else:
        payload["dataset"]["row_indices"][-1] = 0
    with pytest.raises(ValueError, match="canonical|exact|order|changed"):
        SubjectsPilotConfig.load(_write_json(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("dataset", "dataset_id"), "Yuanshi/Subjects201K"),
        (("dataset", "revision"), "0" * 40),
        (("dataset", "source_file_sha256"), "f" * 64),
    ],
)
def test_subjects_config_rejects_plausible_alternative_ids_and_full_hashes(
    path: tuple[str, ...], replacement: str, tmp_path: Path
) -> None:
    payload = _subjects_payload()
    _set_path(payload, path, replacement)
    with pytest.raises(ValueError, match="canonical|exact|changed"):
        SubjectsPilotConfig.load(_write_json(tmp_path, payload))


def test_subjects_config_direct_reconstruction_replace_and_forced_mutation() -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    assert replace(config) == config

    for field in fields(config):
        value = getattr(config, field.name)
        if type(value) is tuple:
            replacement: object = tuple(reversed(value))
        else:
            replacement = _different_value(value)
        with pytest.raises((TypeError, ValueError), match="canonical|exact|changed|type"):
            replace(config, **{field.name: replacement})

    object.__setattr__(config, "revision", "0" * 40)
    with pytest.raises(ValueError, match="canonical|exact|changed"):
        config.validate()


def test_subjects_config_rejects_subclasses_without_weakening_sana_config() -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)

    class _Subclass(SubjectsPilotConfig):
        pass

    with pytest.raises(TypeError, match="exact SubjectsPilotConfig"):
        _Subclass(**{field.name: getattr(config, field.name) for field in fields(config)})

    SanaPilotConfig.load(SANA_CONFIG_PATH).validate()


def test_split_uses_the_locked_asymmetric_crop_and_immutable_rgb_bytes() -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    support, query = split_composite_pair(_composite(7), config)
    assert support.size == query.size == (512, 512)
    assert support.mode == query.mode == "RGB"
    assert support.getpixel((10, 10)) == (255, 7, 0)
    assert query.getpixel((10, 10)) == (0, 7, 255)

    example = build_example(7, _row(7), config)
    assert type(example.support_rgb) is bytes
    assert type(example.query_rgb) is bytes
    assert example.support_image().getpixel((10, 10)) == (255, 7, 0)
    assert example.query_image().getpixel((10, 10)) == (0, 7, 255)
    assert example.concept_description == "Eames Lounge Chair"
    assert example.query_prompt == "Outdoor chair 7"


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (1055, 528)),
        Image.new("RGB", (1056, 527)),
        Image.new("L", (1056, 528)),
    ],
)
def test_split_rejects_wrong_composite_geometry_or_mode(image: Image.Image) -> None:
    with pytest.raises((TypeError, ValueError), match="RGB|1056x528"):
        split_composite_pair(image, SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH))


def test_rgb_hash_uses_raw_pixels_not_png_encoding_metadata(tmp_path: Path) -> None:
    image = _composite(4)
    plain = tmp_path / "plain.png"
    annotated = tmp_path / "annotated.png"
    image.save(plain)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("volatile", "different encoder metadata")
    image.save(annotated, pnginfo=metadata, compress_level=1)

    with Image.open(plain) as first, Image.open(annotated) as second:
        assert plain.read_bytes() != annotated.read_bytes()
        assert rgb_content_sha256(first.convert("RGB")) == rgb_content_sha256(
            second.convert("RGB")
        )

    changed = image.copy()
    changed.putpixel((0, 0), (1, 2, 3))
    assert rgb_content_sha256(changed) != rgb_content_sha256(image)


def test_row_identity_binds_every_metadata_group_crop_and_pixel() -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    baseline = build_example(0, _row(0), config).row_sha256
    mutations: list[dict[str, object]] = []

    collection = _row(0)
    collection["collection"] = "other"
    mutations.append(collection)
    quality = _row(0)
    quality["quality_assessment"]["imageQuality"] = 4  # type: ignore[index]
    mutations.append(quality)
    for key in ("item", "description_0", "description_1", "category"):
        row = _row(0)
        row["description"][key] += " changed"  # type: ignore[index,operator]
        mutations.append(row)
    pixels = _row(0)
    pixels["image"].putpixel((9, 9), (3, 2, 1))  # type: ignore[union-attr]
    mutations.append(pixels)

    for row in mutations:
        assert build_example(0, row, config).row_sha256 != baseline
    assert build_example(1, _row(1), config).row_sha256 != baseline


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("collection"), "keys|order"),
        (lambda row: row.update(collection=1), "collection"),
        (lambda row: row.update(quality_assessment="bad"), "quality"),
        (lambda row: row["quality_assessment"].update(imageQuality=True), "quality"),
        (lambda row: row["quality_assessment"].update(imageQuality=6), "quality"),
        (lambda row: row["description"].pop("category"), "description"),
        (lambda row: row["description"].update(description_valid=1), "description"),
        (lambda row: row["description"].update(description_valid=False), "valid"),
    ],
)
def test_build_example_rejects_invalid_row_contract(
    mutation: Any, message: str
) -> None:
    row = _row(0)
    mutation(row)
    with pytest.raises((TypeError, ValueError), match=message):
        build_example(0, row, SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH))


class _EightOnlyIterator:
    def __init__(self) -> None:
        self.next_calls = 0

    def __iter__(self) -> _EightOnlyIterator:
        return self

    def __next__(self) -> dict[str, object]:
        if self.next_calls >= 8:
            raise AssertionError("row 8 must never be requested")
        index = self.next_calls
        self.next_calls += 1
        return _row(index)


class _FakeDataset:
    def __init__(self, features: Features | None = None) -> None:
        self.features = features or _dataset_features()
        self.iterator = _EightOnlyIterator()

    def __iter__(self) -> _EightOnlyIterator:
        return self.iterator


def test_hydrator_is_the_single_exact_network_boundary_and_stops_before_row_8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    dataset = _FakeDataset()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeDataset:
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(pilot_data, "load_dataset", fake_load_dataset)
    examples = hydrate_locked_examples(config, cache_dir=tmp_path)

    assert calls == [
        (
            ("Yuanshi/Subjects200K",),
            {
                "name": "default",
                "data_files": {"train": ["data/train-00000-of-00032.parquet"]},
                "split": "train",
                "revision": DATASET_REVISION,
                "streaming": True,
                "cache_dir": str(tmp_path),
                "token": False,
            },
        )
    ]
    assert tuple(example.row_index for example in examples) == tuple(range(8))
    assert dataset.iterator.next_calls == 8


def test_hydrator_validates_features_before_iterating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    features = _dataset_features()
    image = features.pop("image")
    features["image"] = image
    dataset = _FakeDataset(features)
    monkeypatch.setattr(pilot_data, "load_dataset", lambda *a, **k: dataset)

    with pytest.raises(ValueError, match="feature|schema|order"):
        hydrate_locked_examples(
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH), cache_dir=tmp_path
        )
    assert dataset.iterator.next_calls == 0


def test_hydrator_rejects_nested_feature_reordering_before_iterating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    features = _dataset_features()
    description = features["description"]
    item = description.pop("item")
    description["item"] = item
    dataset = _FakeDataset(features)
    monkeypatch.setattr(pilot_data, "load_dataset", lambda *a, **k: dataset)

    with pytest.raises(ValueError, match="nested feature order"):
        hydrate_locked_examples(
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH), cache_dir=tmp_path
        )
    assert dataset.iterator.next_calls == 0


def test_data_module_ast_has_one_network_call_and_no_alternative_clients() -> None:
    source = inspect.getsource(pilot_data)
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    load_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "load_dataset"
    ]
    assert len(load_calls) == 1
    forbidden = {
        "snapshot_download",
        "hf_hub_download",
        "requests",
        "httpx",
        "urllib",
        "urlopen",
        "socket",
        "load_dataset_builder",
        "load_from_disk",
        "import_module",
        "runpy",
        "exec",
        "eval",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not forbidden.intersection(names | attributes)
    assert "trust_remote_code" not in source


def test_cache_tensor_contract_is_immutable() -> None:
    assert type(CACHE_TENSOR_SPECS).__name__ == "mappingproxy"


def test_safetensors_metadata_is_one_canonical_typed_json_value() -> None:
    metadata = cache_metadata(
        identity_sha256="a" * 64,
        sana_revision="b77948f2b4eed5c728e9b828ccff07f7427b43cc",
        support_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        dataset_revision=DATASET_REVISION,
    )
    assert tuple(metadata) == ("ratemem",)
    payload = json.loads(metadata["ratemem"])
    assert tuple(payload) == (
        "schema_version",
        "scope",
        "publication_eligible",
        "identity_sha256",
        "sana_revision",
        "support_revision",
        "dataset_revision",
    )
    assert payload["publication_eligible"] is False
    assert metadata["ratemem"] == json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def test_single_key_metadata_makes_safetensors_bytes_stable_across_processes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    script = """
import sys
import torch
from safetensors.torch import save_file
from ratemem.pilot.data import cache_metadata

metadata = cache_metadata(
    identity_sha256="a" * 64,
    sana_revision="b77948f2b4eed5c728e9b828ccff07f7427b43cc",
    support_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
    dataset_revision="0d1cf6536239888f1a8e218790649344810067bc",
)
tensors = {
    "clean_latents": torch.tensor([1.0], dtype=torch.float32),
    "prompt_embeddings": torch.tensor([2.0], dtype=torch.float32),
    "prompt_attention_mask": torch.tensor([1], dtype=torch.int64),
    "support_features": torch.tensor([3.0], dtype=torch.float32),
    "support_mask": torch.tensor([True], dtype=torch.bool),
    "description_features": torch.tensor([4.0], dtype=torch.float32),
}
save_file(tensors, sys.argv[1], metadata=metadata)
"""
    for path in (first, second):
        subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_query_preprocessing_matches_official_bilinear_reference_not_bicubic() -> None:
    grid = torch.arange(3 * 512 * 512, dtype=torch.int64).reshape(3, 512, 512)
    image = vision_functional.to_pil_image((grid % 256).to(torch.uint8))

    actual = preprocess_query_image(image, resolution=1024)
    reference = vision_functional.resize(
        image,
        1024,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    reference = vision_functional.center_crop(reference, [1024, 1024])
    reference_tensor = vision_functional.normalize(
        vision_functional.to_tensor(reference), [0.5], [0.5]
    )
    bicubic = vision_functional.resize(
        image,
        1024,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    bicubic_tensor = vision_functional.normalize(
        vision_functional.to_tensor(bicubic), [0.5], [0.5]
    )

    assert torch.equal(actual, reference_tensor)
    assert not torch.equal(actual, bicubic_tensor)
    assert actual.shape == (3, 1024, 1024)
    assert actual.dtype == torch.float32


class _FakeTokenizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.padding_side = "left"
        self.fail = fail
        self.calls: list[tuple[list[str], dict[str, object], str]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, Tensor]:
        self.calls.append((texts, kwargs, self.padding_side))
        if self.fail:
            raise RuntimeError("tokenizer failed")
        count = len(texts)
        input_ids = torch.zeros((count, 300), dtype=torch.int64)
        attention_mask = torch.zeros((count, 300), dtype=torch.int64)
        attention_mask[:, :3] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _NonmonotonicMaskTokenizer(_FakeTokenizer):
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, Tensor]:
        output = super().__call__(texts, **kwargs)
        output["attention_mask"][:, 1] = 0
        output["attention_mask"][:, 2] = 1
        return output


class _FakeTextEncoder(nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
        self.fail = fail

    def forward(self, **kwargs: object) -> BaseModelOutputWithPast:
        if self.fail:
            raise RuntimeError("encoder failed")
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, Tensor)
        hidden = torch.ones(
            (input_ids.shape[0], input_ids.shape[1], 2304),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        return BaseModelOutputWithPast(last_hidden_state=hidden)


class _FakeVAE(nn.Module):
    def __init__(self, output: str = "latent") -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.41407)
        self.output = output
        self.calls: list[tuple[int, ...]] = []

    def encode(self, pixels: Tensor, *, return_dict: bool = True) -> object:
        assert return_dict is True
        self.calls.append(tuple(pixels.shape))
        latent = torch.full(
            (1, 32, 32, 32), 2.0, dtype=torch.float32, device=pixels.device
        )
        if self.output == "latent":
            return EncoderOutput(latent=latent)
        if self.output == "latent_dist":
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))
        if self.output == "nonfinite":
            latent[0, 0, 0, 0] = float("nan")
            return EncoderOutput(latent=latent)
        return EncoderOutput(latent=latent[:, :31])


def _fake_components(
    *, tokenizer: _FakeTokenizer | None = None, text_encoder: _FakeTextEncoder | None = None,
    vae: _FakeVAE | None = None,
) -> SimpleNamespace:
    support_encoder = nn.Linear(1, 1, bias=False).requires_grad_(False).eval()
    return SimpleNamespace(
        tokenizer=tokenizer or _FakeTokenizer(),
        text_encoder=text_encoder or _FakeTextEncoder(),
        vae=vae or _FakeVAE(),
        support_processor=object(),
        support_encoder=support_encoder,
    )


def test_precompute_uses_exact_text_contract_vae_microbatch_and_task5_dino_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)
    sana = SanaPilotConfig.load(SANA_CONFIG_PATH)
    tokenizer = _FakeTokenizer()
    vae = _FakeVAE()
    components = _fake_components(tokenizer=tokenizer, vae=vae)
    dino_calls: list[dict[str, object]] = []

    def fake_dino(images: list[Image.Image], **kwargs: object) -> Tensor:
        dino_calls.append({"images": images, **kwargs})
        return torch.ones((8, 384), dtype=torch.float32)

    monkeypatch.setattr(pilot_data, "encode_support_images", fake_dino)
    tensors = _test_precompute_tensors(_examples(config), components, sana, config)

    assert len(vae.calls) == 8
    assert vae.calls == [(1, 3, 1024, 1024)] * 8
    assert torch.equal(
        tensors["clean_latents"],
        torch.full((8, 32, 32, 32), 2.0 * 0.41407),
    )
    assert len(tokenizer.calls) == 2
    query_call, description_call = tokenizer.calls
    assert query_call[0] == [f"outdoor chair {index}" for index in range(8)]
    assert description_call[0] == ["eames lounge chair"] * 8
    for _, kwargs, padding_side in tokenizer.calls:
        assert kwargs == {
            "padding": "max_length",
            "max_length": 300,
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt",
        }
        assert padding_side == "right"
    assert tokenizer.padding_side == "left"
    assert len(dino_calls) == 1
    assert len(dino_calls[0]["images"]) == 8
    assert tensors["prompt_embeddings"].shape == (8, 300, 2304)
    assert tensors["description_features"].shape == (8, 2304)


@pytest.mark.parametrize("failure", ["tokenizer", "encoder"])
def test_text_encoding_restores_left_padding_on_every_error(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tokenizer = _FakeTokenizer(fail=failure == "tokenizer")
    text_encoder = _FakeTextEncoder(fail=failure == "encoder")
    components = _fake_components(tokenizer=tokenizer, text_encoder=text_encoder)
    monkeypatch.setattr(
        pilot_data,
        "encode_support_images",
        lambda *a, **k: torch.ones((8, 384), dtype=torch.float32),
    )

    with pytest.raises(RuntimeError, match="failed"):
        _test_precompute_tensors(
            _examples(),
            components,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert tokenizer.padding_side == "left"


def test_text_encoding_rejects_nonmonotonic_right_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _NonmonotonicMaskTokenizer()
    monkeypatch.setattr(
        pilot_data,
        "encode_support_images",
        lambda *a, **k: torch.ones((8, 384), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="right-padded|0-to-1"):
        _test_precompute_tensors(
            _examples(),
            _fake_components(tokenizer=tokenizer),
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert tokenizer.padding_side == "left"


@pytest.mark.parametrize("vae_output", ["latent_dist", "nonfinite", "bad_shape"])
def test_precompute_rejects_non_encoderoutput_nonfinite_and_bad_vae_latents(
    monkeypatch: pytest.MonkeyPatch, vae_output: str
) -> None:
    monkeypatch.setattr(
        pilot_data,
        "encode_support_images",
        lambda *a, **k: torch.ones((8, 384), dtype=torch.float32),
    )
    with pytest.raises((TypeError, ValueError), match="EncoderOutput|finite|latent|shape"):
        _test_precompute_tensors(
            _examples(),
            _fake_components(vae=_FakeVAE(vae_output)),
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )


def test_outer_inference_mode_still_returns_six_trainable_safe_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_data,
        "encode_support_images",
        lambda *a, **k: torch.ones((8, 384), dtype=torch.float32),
    )
    with torch.inference_mode():
        tensors = _test_precompute_tensors(
            _examples(),
            _fake_components(),
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )

    assert tuple(tensors) == tuple(CACHE_TENSOR_SPECS)
    for tensor in tensors.values():
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert not tensor.requires_grad
        assert not tensor.is_inference()
    head = nn.Linear(2304, 1)
    head(tensors["description_features"]).sum().backward()
    assert head.weight.grad is not None


def test_build_then_strict_reload_has_exact_manifest_metadata_and_private_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    result = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )

    assert result.root == output.resolve()
    assert tuple(result.tensors) == tuple(CACHE_TENSOR_SPECS)
    assert result.manifest["scope"] == "engineering_pilot_only"
    assert result.manifest["publication_eligible"] is False
    assert result.manifest["features"]["sha256"] == hashlib.sha256(
        (output / "features.safetensors").read_bytes()
    ).hexdigest()
    with safe_open(output / "features.safetensors", framework="pt") as handle:
        assert tuple(handle.metadata() or {}) == ("ratemem",)
    for path, mode in (
        (output, 0o700),
        (output / "manifest.json", 0o600),
        (output / "features.safetensors", 0o600),
        (output / "complete", 0o600),
        (parent / "cache.lock", 0o600),
    ):
        metadata = path.lstat()
        assert stat.S_IMODE(metadata.st_mode) == mode
        assert metadata.st_uid == os.getuid()
        if path.is_file():
            assert metadata.st_nlink == 1
    reloaded = load_precomputed_cache(
        output, expected_receipt=result.receipt
    )
    assert reloaded.manifest == result.manifest
    for name, tensor in reloaded.tensors.items():
        expected_shape, expected_dtype = CACHE_TENSOR_SPECS[name]
        assert tuple(tensor.shape) == expected_shape
        assert tensor.dtype == expected_dtype


def test_loader_never_networks_or_falls_back_for_missing_partial_or_corrupt_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("offline loader attempted network access")

    monkeypatch.setattr(pilot_data, "load_dataset", forbidden_network)
    parent = _secure_parent(tmp_path)
    with pytest.raises(FileNotFoundError, match="cache"):
        load_precomputed_cache(parent / "missing", expected_receipt=_dummy_receipt())

    partial = parent / "partial"
    partial.mkdir(mode=0o700)
    with pytest.raises(RuntimeError, match="partial|complete|cache"):
        load_precomputed_cache(partial, expected_receipt=_dummy_receipt())


def test_cache_rejects_checksum_mode_symlink_and_identity_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    features = output / "features.safetensors"
    original = features.read_bytes()
    features.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    os.chmod(features, 0o600)
    with pytest.raises(RuntimeError, match="checksum|sha256"):
        load_precomputed_cache(output, expected_receipt=built.receipt)

    features.write_bytes(original)
    os.chmod(features, 0o640)
    with pytest.raises(PermissionError, match="0600|mode"):
        load_precomputed_cache(output, expected_receipt=built.receipt)
    os.chmod(features, 0o600)
    with pytest.raises(ValueError, match="identity"):
        load_precomputed_cache(
            output,
            expected_receipt=replace(built.receipt, identity_sha256="0" * 64),
        )


def test_cache_rejects_rechained_bad_safetensors_metadata_keys_shape_and_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    tensors = load_file(output / "features.safetensors")
    metadata = pilot_data.cache_metadata(
        identity_sha256=built.identity_sha256,
        sana_revision=SanaPilotConfig.load(SANA_CONFIG_PATH).revision,
        support_revision=SanaPilotConfig.load(SANA_CONFIG_PATH).support_revision,
        dataset_revision=DATASET_REVISION,
    )

    cases: list[tuple[dict[str, Tensor], dict[str, str], str]] = []
    wrong_metadata = dict(metadata)
    wrong_metadata["scope"] = "scientific"
    cases.append((tensors, wrong_metadata, "metadata"))
    missing_key = dict(tensors)
    missing_key.pop("support_mask")
    cases.append((missing_key, metadata, "keys"))
    wrong_shape = dict(tensors)
    wrong_shape["support_mask"] = torch.ones((8, 2), dtype=torch.bool)
    cases.append((wrong_shape, metadata, "shape"))
    wrong_dtype = dict(tensors)
    wrong_dtype["prompt_attention_mask"] = tensors["prompt_attention_mask"].bool()
    cases.append((wrong_dtype, metadata, "dtype"))

    for case_index, (case_tensors, case_metadata, message) in enumerate(cases):
        case_output = parent / f"case-{case_index}"
        _fake_compute(monkeypatch)
        other = _test_build_precomputed_cache(
            _examples(),
            _fake_components(),
            case_output,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
        save_file(case_tensors, case_output / "features.safetensors", metadata=case_metadata)
        os.chmod(case_output / "features.safetensors", 0o600)
        _rechain_cache_manifest(case_output)
        with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
            load_precomputed_cache(
                case_output,
                expected_receipt=_receipt_from_files(
                    case_output, identity_sha256=other.identity_sha256
                ),
            )


@pytest.mark.parametrize(
    "mutation", ["nonfinite", "prompt_mask", "prompt_hole", "support_mask"]
)
def test_cache_rejects_rechained_unhealthy_tensor_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    tensors = load_file(output / "features.safetensors")
    if mutation == "nonfinite":
        tensors["description_features"][0, 0] = float("nan")
        message = "finite"
    elif mutation == "prompt_mask":
        tensors["prompt_attention_mask"][0, 0] = 2
        message = "binary"
    elif mutation == "prompt_hole":
        tensors["prompt_attention_mask"][0, 1] = 0
        tensors["prompt_attention_mask"][0, 2] = 1
        message = "right-padded|0-to-1"
    else:
        tensors["support_mask"][0, 0] = False
        message = "entirely true"
    identity = built.manifest["identity"]
    save_file(
        tensors,
        output / "features.safetensors",
        metadata=cache_metadata(
            identity_sha256=built.identity_sha256,
            sana_revision=identity["sana_revision"],
            support_revision=identity["support_revision"],
            dataset_revision=identity["dataset_revision"],
        ),
    )
    os.chmod(output / "features.safetensors", 0o600)
    _rechain_cache_manifest(output)
    with pytest.raises(ValueError, match=message):
        load_precomputed_cache(
            output,
            expected_receipt=_receipt_from_files(
                output, identity_sha256=built.identity_sha256
            ),
        )


def test_external_receipt_rejects_rechained_healthy_same_shape_tensor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    tensors = load_file(output / "features.safetensors")
    tensors["description_features"][0, 0] = 17.0
    save_file(
        tensors,
        output / "features.safetensors",
        metadata=cache_metadata(
            identity_sha256=built.identity_sha256,
            sana_revision=built.manifest["identity"]["sana_revision"],
            support_revision=built.manifest["identity"]["support_revision"],
            dataset_revision=built.manifest["identity"]["dataset_revision"],
        ),
    )
    os.chmod(output / "features.safetensors", 0o600)
    _rechain_cache_manifest(output)
    with pytest.raises(RuntimeError, match="external receipt"):
        load_precomputed_cache(output, expected_receipt=built.receipt)


def test_manifest_requires_the_unique_canonical_subjects_config_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    manifest = deepcopy(built.manifest)
    manifest["identity"]["dataset_config_sha256"] = "f" * 64
    changed_identity = hashlib.sha256(
        json.dumps(
            manifest["identity"], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest["identity_sha256"] = changed_identity
    _write_rechained_manifest(output, manifest)
    with pytest.raises(ValueError, match="unique canonical config"):
        load_precomputed_cache(
            output,
            expected_receipt=_receipt_from_files(
                output, identity_sha256=changed_identity
            ),
        )


@pytest.mark.parametrize("mutation", ["row_index_type", "tensor_shape_type"])
def test_cache_manifest_rejects_python_equal_values_with_wrong_json_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    built = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    manifest = deepcopy(built.manifest)
    expected_identity = built.identity_sha256
    if mutation == "row_index_type":
        manifest["identity"]["row_indices"][0] = False
        expected_identity = hashlib.sha256(
            json.dumps(
                manifest["identity"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        manifest["identity_sha256"] = expected_identity
    else:
        specs = manifest["features"]["tensors"]
        support_mask = next(spec for spec in specs if spec["name"] == "support_mask")
        support_mask["shape"][1] = True
    _write_rechained_manifest(output, manifest)

    with pytest.raises(ValueError, match="exact type|row indices|tensor manifest"):
        load_precomputed_cache(
            output,
            expected_receipt=_receipt_from_files(
                output, identity_sha256=expected_identity
            ),
        )


def test_cache_build_failure_never_publishes_partial_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = _secure_parent(tmp_path)
    output = parent / "cache"

    def fail(*args: object, **kwargs: object) -> dict[str, Tensor]:
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(pilot_data, "_precompute_tensors_impl", fail)
    with pytest.raises(RuntimeError, match="encoder exploded"):
        _test_build_precomputed_cache(
            _examples(),
            _fake_components(),
            output,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert not output.exists()


def test_cache_fsync_failure_never_publishes_partial_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    monkeypatch.setattr(
        pilot_data,
        "_fsync_file",
        lambda path: (_ for _ in ()).throw(OSError("fsync exploded")),
    )
    with pytest.raises(OSError, match="fsync exploded"):
        _test_build_precomputed_cache(
            _examples(),
            _fake_components(),
            output,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert not output.exists()
    assert list(parent.glob(".cache.staging-*")) == []


@pytest.mark.parametrize("failure", ["chmod", "validation"])
def test_every_post_mkdtemp_failure_cleans_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    if failure == "chmod":
        real_chmod = os.chmod

        def fail_staging_chmod(path: os.PathLike[str] | str, mode: int) -> None:
            if Path(path).name.startswith(".cache.staging-"):
                raise OSError("staging chmod failed")
            real_chmod(path, mode)

        monkeypatch.setattr(pilot_data.os, "chmod", fail_staging_chmod)
        message = "staging chmod failed"
    else:
        real_validate = pilot_data._validate_private_directory

        def fail_staging_validation(path: Path, context: str) -> os.stat_result:
            if context == "cache staging directory":
                raise PermissionError("staging validation failed")
            return real_validate(path, context)

        monkeypatch.setattr(
            pilot_data, "_validate_private_directory", fail_staging_validation
        )
        message = "staging validation failed"

    with pytest.raises((OSError, PermissionError), match=message):
        _test_build_precomputed_cache(
            _examples(),
            _fake_components(),
            output,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert not output.exists()
    assert list(parent.glob(".cache.staging-*")) == []


def test_same_identity_concurrent_build_has_one_encoder_winner_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    call_count = 0
    counter_lock = threading.Lock()
    compute_started = threading.Event()
    release_compute = threading.Event()

    def compute(*args: object, **kwargs: object) -> dict[str, Tensor]:
        nonlocal call_count
        with counter_lock:
            call_count += 1
        compute_started.set()
        assert release_compute.wait(timeout=10)
        return _valid_tensors()

    monkeypatch.setattr(pilot_data, "_precompute_tensors_impl", compute)
    args = (
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_test_build_precomputed_cache, *args)
        assert compute_started.wait(timeout=10)
        second = executor.submit(_test_build_precomputed_cache, *args)
        time.sleep(0.1)
        release_compute.set()
        results = [first.result(timeout=20), second.result(timeout=20)]

    assert call_count == 1
    assert results[0].identity_sha256 == results[1].identity_sha256


def _multiprocess_cache_worker(
    output: Path,
    compute_count: Any,
    worker_ready_count: Any,
    compute_started: Any,
    release_compute: Any,
    connection: Connection,
) -> None:
    sana = SanaPilotConfig.load(SANA_CONFIG_PATH)
    subjects = SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH)

    def compute() -> dict[str, Tensor]:
        with compute_count.get_lock():
            compute_count.value += 1
        compute_started.set()
        if not release_compute.wait(timeout=90):
            raise RuntimeError("cross-process compute release timed out")
        return _valid_tensors()

    try:
        with worker_ready_count.get_lock():
            worker_ready_count.value += 1
        result = pilot_data._build_precomputed_cache_impl(
            _examples(subjects), output, sana, subjects, compute
        )
        receipt = result.receipt
        connection.send(
            (
                "ok",
                receipt.identity_sha256,
                receipt.manifest_sha256,
                receipt.manifest_byte_count,
                receipt.features_sha256,
                receipt.features_byte_count,
            )
        )
    except BaseException as error:
        connection.send(("error", type(error).__name__, str(error)))
    finally:
        connection.close()


def test_cross_process_build_lock_has_one_compute_winner_and_one_strict_cache(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    compute_count = context.Value("i", 0)
    worker_ready_count = context.Value("i", 0)
    compute_started = context.Event()
    release_compute = context.Event()
    first_parent, first_child = context.Pipe(duplex=False)
    second_parent, second_child = context.Pipe(duplex=False)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    worker_args = (
        output,
        compute_count,
        worker_ready_count,
        compute_started,
        release_compute,
    )
    first = context.Process(
        target=_multiprocess_cache_worker, args=(*worker_args, first_child)
    )
    second = context.Process(
        target=_multiprocess_cache_worker, args=(*worker_args, second_child)
    )
    first.start()
    try:
        assert compute_started.wait(timeout=90)
        second.start()
        deadline = time.monotonic() + 90
        while worker_ready_count.value != 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert worker_ready_count.value == 2
        time.sleep(0.5)
        release_compute.set()
        first.join(timeout=90)
        second.join(timeout=90)
        assert first.exitcode == second.exitcode == 0
    finally:
        release_compute.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    reports = [first_parent.recv(), second_parent.recv()]
    assert all(report[0] == "ok" for report in reports), reports
    assert reports[0][1:] == reports[1][1:]
    assert compute_count.value == 1
    assert list(parent.glob(".cache.staging-*")) == []
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in (
        output / "manifest.json",
        output / "features.safetensors",
        output / "complete",
        parent / "cache.lock",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    receipt = PilotCacheReceipt(
        identity_sha256=reports[0][1],
        manifest_sha256=reports[0][2],
        manifest_byte_count=reports[0][3],
        features_sha256=reports[0][4],
        features_byte_count=reports[0][5],
    )
    assert load_precomputed_cache(output, expected_receipt=receipt).receipt == receipt


def test_existing_cache_with_different_identity_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    output = parent / "cache"
    first = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    changed = list(_examples())
    changed_row = _row(0)
    changed_row["image"].putpixel((9, 9), (4, 3, 2))  # type: ignore[union-attr]
    changed[0] = build_example(
        0,
        changed_row,
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    with pytest.raises(FileExistsError, match="retained external receipt"):
        _test_build_precomputed_cache(
            tuple(changed),
            _fake_components(),
            output,
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
    assert load_precomputed_cache(
        output, expected_receipt=first.receipt
    ).identity_sha256 == first.identity_sha256


def test_cache_rejects_hardlinked_and_symlinked_component_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    parent = _secure_parent(tmp_path)
    hardlink_output = parent / "hardlink-cache"
    hardlinked = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        hardlink_output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    os.link(
        hardlink_output / "features.safetensors",
        parent / "features-hardlink.safetensors",
    )
    with pytest.raises(OSError, match="hard link|one"):
        load_precomputed_cache(
            hardlink_output, expected_receipt=hardlinked.receipt
        )

    symlink_output = parent / "symlink-cache"
    symlinked = _test_build_precomputed_cache(
        _examples(),
        _fake_components(),
        symlink_output,
        SanaPilotConfig.load(SANA_CONFIG_PATH),
        SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
    )
    features = symlink_output / "features.safetensors"
    real_features = parent / "real-features.safetensors"
    features.rename(real_features)
    features.symlink_to(real_features)
    with pytest.raises(OSError, match="symlink|regular"):
        load_precomputed_cache(
            symlink_output, expected_receipt=symlinked.receipt
        )


def test_cache_paths_reject_symlinked_ancestors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_compute(monkeypatch)
    real = _secure_parent(tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises((OSError, ValueError), match="symlink|safe"):
        _test_build_precomputed_cache(
            _examples(),
            _fake_components(),
            link / "cache",
            SanaPilotConfig.load(SANA_CONFIG_PATH),
            SubjectsPilotConfig.load(SUBJECTS_CONFIG_PATH),
        )
