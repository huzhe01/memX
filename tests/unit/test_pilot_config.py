from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any

import pytest

from ratemem.adapters.sana_layout import (
    ATTENTION_KINDS,
    SANA_LAYOUT_VERSION,
    TARGET_MODULES,
)
from ratemem.pilot.config import SanaPilotConfig

CONFIG_PATH = Path("configs/pilot/sana-1.5-1.6b.json")
SANA_REVISION = "b77948f2b4eed5c728e9b828ccff07f7427b43cc"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def _payload() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    if type(value) is dict:
        paths: list[tuple[object, ...]] = []
        for key, child in value.items():
            paths.extend(_leaf_paths(child, (*prefix, key)))
        return paths
    if type(value) is list:
        paths = []
        for index, child in enumerate(value):
            paths.extend(_leaf_paths(child, (*prefix, index)))
        return paths
    return [prefix]


def _replace_leaf(payload: dict[str, Any], path: tuple[object, ...]) -> None:
    parent: Any = payload
    for segment in path[:-1]:
        parent = parent[segment]
    leaf = parent[path[-1]]
    if type(leaf) is bool:
        replacement = not leaf
    elif type(leaf) is int:
        replacement = leaf + 1
    elif type(leaf) is float:
        replacement = leaf + 0.25
    elif type(leaf) is str:
        replacement = f"{leaf}-tampered"
    else:
        raise AssertionError(f"unsupported canonical leaf type: {type(leaf)}")
    parent[path[-1]] = replacement


def _equal_value_wrong_type_cases() -> list[tuple[tuple[object, ...], object]]:
    if not CONFIG_PATH.exists():
        return []
    payload = _payload()
    cases: list[tuple[tuple[object, ...], object]] = []
    for path in _leaf_paths(payload):
        parent: Any = payload
        for segment in path:
            parent = parent[segment]
        if type(parent) is bool:
            cases.append((path, int(parent)))
        elif type(parent) is int:
            cases.append((path, float(parent)))
        elif type(parent) is float and parent.is_integer():
            cases.append((path, int(parent)))
    return cases


def _direct_field_tampers() -> list[tuple[str, object]]:
    if not CONFIG_PATH.exists():
        return []
    config = SanaPilotConfig.load(CONFIG_PATH)
    cases: list[tuple[str, object]] = []
    for field in fields(config):
        value = getattr(config, field.name)
        if type(value) is bool:
            replacement = not value
        elif type(value) is int:
            replacement = value + 1
        elif type(value) is float:
            replacement = value + 0.25
        elif type(value) is str:
            replacement = f"{value}-tampered"
        elif type(value) is tuple:
            replacement = tuple(reversed(value))
        else:
            raise AssertionError(f"unsupported config field type: {type(value)}")
        cases.append((field.name, replacement))
    return cases


def _direct_wrong_type_tampers() -> list[tuple[str, object]]:
    field_names = {
        ("sana", "resolution"): "resolution",
        ("sana", "latent_channels"): "latent_channels",
        ("sana", "latent_size"): "latent_size",
        ("sana", "text_feature_dim"): "text_feature_dim",
        ("sana", "max_sequence_length"): "max_sequence_length",
        ("support_encoder", "feature_dim"): "support_feature_dim",
        ("adapter", "num_blocks"): "num_blocks",
        ("adapter", "width"): "width",
        ("adapter", "rank"): "rank",
        ("adapter", "atom_count"): "atom_count",
        ("adapter", "projection_count"): "projection_count",
        ("adapter", "code_dim"): "code_dim",
        ("adapter", "atom_tensor_count"): "atom_tensor_count",
        ("adapter", "atom_parameter_count"): "atom_parameter_count",
        ("training", "num_train_timesteps"): "num_train_timesteps",
        ("training", "flow_shift"): "flow_shift",
        ("training", "use_dynamic_shifting"): "use_dynamic_shifting",
        ("training", "gradient_checkpointing"): "gradient_checkpointing",
        ("training", "max_support_images"): "max_support_images",
        ("training", "query_passes_per_step"): "query_passes_per_step",
    }
    return [
        (field_names[path], replacement)
        for path, replacement in _equal_value_wrong_type_cases()
        if path in field_names
    ]


def test_committed_config_is_exact_immutable_and_has_all_derived_dimensions() -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)

    assert config.schema_version == "1.0.0"
    assert config.model_id == "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
    assert config.revision == SANA_REVISION
    assert config.support_model_id == "facebook/dinov2-small"
    assert config.support_revision == DINO_REVISION
    assert config.layout_version == SANA_LAYOUT_VERSION
    assert config.attention_kinds == ATTENTION_KINDS
    assert config.target_modules == TARGET_MODULES
    assert config.scheduler_class == "FlowMatchEulerDiscreteScheduler"
    assert config.num_train_timesteps == 1000
    assert config.flow_shift == 1.0
    assert config.use_dynamic_shifting is False
    assert config.optimizer_class == "AdamW"
    assert config.optimizer_kwargs == {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }
    assert config.code_shape == (20, 2, 3, 4)
    assert config.projection_count == 120
    assert config.code_dim == 480
    assert config.atom_tensor_count == 240
    assert config.atom_parameter_count == 8_601_600
    with pytest.raises(FrozenInstanceError):
        config.rank = 8  # type: ignore[misc]


@pytest.mark.parametrize("leaf_path", _leaf_paths(_payload()) if CONFIG_PATH.exists() else [])
def test_every_committed_leaf_is_immutable(
    leaf_path: tuple[object, ...], tmp_path: Path
) -> None:
    payload = _payload()
    _replace_leaf(payload, leaf_path)

    with pytest.raises(ValueError, match="canonical|exact|changed"):
        SanaPilotConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        (None, []),
        ("sana", []),
        ("support_encoder", "not-an-object"),
        ("adapter", None),
        ("training", 7),
    ],
)
def test_root_and_nested_sections_must_be_exact_objects(
    section: str | None,
    replacement: object,
    tmp_path: Path,
) -> None:
    payload: object = _payload()
    if section is None:
        payload = replacement
    else:
        assert isinstance(payload, dict)
        payload[section] = replacement

    with pytest.raises(ValueError, match="object|canonical|exact"):
        SanaPilotConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_duplicate_keys_and_nonfinite_constants_are_rejected(
    constant: str, tmp_path: Path
) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    duplicate_root = text.replace(
        '  "schema_version": "1.0.0",',
        '  "schema_version": "1.0.0",\n  "schema_version": "1.0.0",',
        1,
    )
    duplicate_nested = text.replace(
        '    "resolution": 1024,',
        '    "resolution": 1024,\n    "resolution": 1024,',
        1,
    )
    nonfinite = text.replace(
        '    "flow_shift": 1.0,', f'    "flow_shift": {constant},', 1
    )

    for index, invalid in enumerate((duplicate_root, duplicate_nested, nonfinite)):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate|finite|constant"):
            SanaPilotConfig.load(path)


@pytest.mark.parametrize("section", [None, "sana", "adapter", "training"])
def test_object_key_order_is_part_of_the_contract(
    section: str | None, tmp_path: Path
) -> None:
    payload = _payload()
    target = payload if section is None else payload[section]
    assert type(target) is dict
    first_key = next(iter(target))
    first_value = target.pop(first_key)
    target[first_key] = first_value

    with pytest.raises(ValueError, match="order|canonical|exact"):
        SanaPilotConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize(("leaf_path", "replacement"), _equal_value_wrong_type_cases())
def test_every_equal_python_value_with_the_wrong_exact_type_is_rejected(
    leaf_path: tuple[object, ...],
    replacement: object,
    tmp_path: Path,
) -> None:
    payload = _payload()
    parent: Any = payload
    for segment in leaf_path[:-1]:
        parent = parent[segment]
    parent[leaf_path[-1]] = replacement

    with pytest.raises(ValueError, match="type|canonical|exact"):
        SanaPilotConfig.load(_write(tmp_path, payload))


def test_plausible_but_wrong_ids_and_full_shas_are_rejected(tmp_path: Path) -> None:
    replacements = (
        ("sana", "model_id", "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"),
        ("sana", "revision", "0" * 40),
        ("support_encoder", "model_id", "facebook/dinov2-base"),
        ("support_encoder", "revision", "1" * 40),
    )
    for section, key, replacement in replacements:
        payload = _payload()
        payload[section][key] = replacement
        with pytest.raises(ValueError, match="canonical|exact|changed"):
            SanaPilotConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize("key", ["attention_kinds", "target_modules"])
@pytest.mark.parametrize("mutation", ["reorder", "duplicate"])
def test_layout_array_order_and_uniqueness_are_exact(
    key: str,
    mutation: str,
    tmp_path: Path,
) -> None:
    payload = _payload()
    values = payload["adapter"][key]
    assert type(values) is list
    if mutation == "reorder":
        values.reverse()
    else:
        values[-1] = values[0]

    with pytest.raises(ValueError, match="canonical|exact|changed"):
        SanaPilotConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "changes",
    [
        {
            "num_blocks": 10,
            "atom_count": 8,
            "projection_count": 60,
            "code_dim": 480,
            "atom_tensor_count": 120,
            "atom_parameter_count": 8_601_600,
        },
        {"width": 1120, "rank": 8},
    ],
)
def test_dimension_tradeoffs_that_preserve_a_derived_total_are_rejected(
    changes: dict[str, int], tmp_path: Path
) -> None:
    payload = deepcopy(_payload())
    payload["adapter"].update(changes)

    with pytest.raises(ValueError, match="canonical|exact|changed"):
        SanaPilotConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize(("field_name", "replacement"), _direct_field_tampers())
def test_every_direct_dataclass_field_tamper_is_rejected(
    field_name: str,
    replacement: object,
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)

    with pytest.raises(ValueError, match="canonical|exact|changed|type"):
        replace(config, **{field_name: replacement})


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    _direct_wrong_type_tampers(),
)
def test_direct_dataclass_equal_value_wrong_types_are_rejected(
    field_name: str, replacement: object
) -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)

    with pytest.raises(ValueError, match="canonical|exact|type"):
        replace(config, **{field_name: replacement})


def test_direct_reconstruction_with_all_canonical_fields_succeeds() -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)

    assert replace(config) == config


def test_public_validation_detects_forced_frozen_field_mutation() -> None:
    config = SanaPilotConfig.load(CONFIG_PATH)
    object.__setattr__(config, "revision", "0" * 40)

    with pytest.raises(ValueError, match="canonical|exact|changed"):
        config.validate()
