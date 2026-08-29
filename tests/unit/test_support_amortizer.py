from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from diffusers import SanaTransformer2DModel
from torch import Tensor, nn

from ratemem.adapters.sana_layout import install_sana_dynamic_atoms
from ratemem.support.amortizer import SupportAmortizer, SupportAmortizerArchitecture


def _amortizer(
    *,
    support_dim: int = 6,
    description_dim: int = 8,
    hidden_dim: int = 32,
    projection_count: int = 120,
    atom_count: int = 4,
    layers: int = 2,
    heads: int = 4,
) -> SupportAmortizer:
    torch.manual_seed(13)
    return SupportAmortizer(
        support_dim=support_dim,
        description_dim=description_dim,
        hidden_dim=hidden_dim,
        projection_count=projection_count,
        atom_count=atom_count,
        layers=layers,
        heads=heads,
    )


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("support_dim", 0),
        ("description_dim", -1),
        ("hidden_dim", True),
        ("projection_count", 0),
        ("atom_count", 0),
        ("layers", 0),
        ("heads", _IntSubclass(4)),
    ],
)
def test_constructor_requires_exact_positive_integers(name: str, value: object) -> None:
    arguments: dict[str, object] = {
        "support_dim": 6,
        "description_dim": 8,
        "hidden_dim": 32,
        "projection_count": 120,
        "atom_count": 4,
        "layers": 2,
        "heads": 4,
    }
    arguments[name] = value
    with pytest.raises((TypeError, ValueError), match=name):
        SupportAmortizer(**arguments)  # type: ignore[arg-type]


def test_constructor_requires_heads_to_divide_hidden_dimension() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be divisible by heads"):
        _amortizer(hidden_dim=30, heads=8)


def test_public_architecture_rejects_invalid_direct_construction() -> None:
    with pytest.raises(TypeError, match="support_dim"):
        SupportAmortizerArchitecture(True, 8, 32, 120, 4, 2, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_dim must be divisible by heads"):
        SupportAmortizerArchitecture(6, 8, 30, 120, 4, 2, 8)

    architecture = SupportAmortizerArchitecture(6, 8, 32, 120, 4, 2, 4)
    object.__setattr__(architecture, "heads", True)
    with pytest.raises(TypeError, match="heads"):
        architecture.validate()
    with pytest.raises(TypeError, match="heads"):
        _ = architecture.canonical
    with pytest.raises(TypeError, match="heads"):
        _ = architecture.signature


def test_architecture_identity_is_immutable_canonical_and_not_tensor_state() -> None:
    model = _amortizer()
    architecture = model.architecture
    assert isinstance(architecture, SupportAmortizerArchitecture)
    assert (
        architecture.support_dim,
        architecture.description_dim,
        architecture.hidden_dim,
        architecture.projection_count,
        architecture.atom_count,
        architecture.layers,
        architecture.heads,
    ) == (6, 8, 32, 120, 4, 2, 4)
    canonical = json.loads(model.architecture_canonical)
    assert canonical == {
        "atom_count": 4,
        "description_dim": 8,
        "heads": 4,
        "hidden_dim": 32,
        "layers": 2,
        "projection_count": 120,
        "schema_version": "ratemem-support-amortizer-v1",
        "support_dim": 6,
    }
    assert len(model.architecture_signature) == 64
    assert all(character in "0123456789abcdef" for character in model.architecture_signature)
    model.assert_architecture_signature(model.architecture_signature)
    assert all(isinstance(value, Tensor) for value in model.state_dict().values())
    assert all("architecture" not in key for key in model.state_dict())

    with pytest.raises(FrozenInstanceError):
        architecture.heads = 8  # type: ignore[misc]
    with pytest.raises((AttributeError, KeyError, TypeError)):
        model.heads = 8  # type: ignore[misc]


def test_same_shape_different_head_architecture_is_detectable_after_strict_load() -> None:
    heads_four = _amortizer(heads=4).eval()
    heads_eight = _amortizer(heads=8).eval()
    result = heads_eight.load_state_dict(heads_four.state_dict(), strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert heads_four.architecture_canonical != heads_eight.architecture_canonical
    assert heads_four.architecture_signature != heads_eight.architecture_signature
    with pytest.raises(ValueError, match="architecture signature"):
        heads_eight.assert_architecture_signature(heads_four.architecture_signature)


@pytest.mark.parametrize(
    "corruption",
    [
        lambda model: object.__setattr__(
            model, "_architecture", replace(model.architecture, heads=8)
        ),
        lambda model: setattr(model.encoder.layers[0].self_attn, "num_heads", 8),
        lambda model: setattr(model.encoder.layers[0].norm1, "eps", 0.5),
        lambda model: setattr(model.encoder.layers[0].self_attn, "add_zero_attn", True),
        lambda model: setattr(model.encoder, "mask_check", False),
        lambda model: setattr(model.encoder, "use_nested_tensor", True),
        lambda model: setattr(
            model,
            "head",
            nn.Linear(model.hidden_dim, model.projection_count * model.atom_count),
        ),
    ],
)
def test_mutated_architecture_or_topology_fails_closed_before_forward(
    corruption: Callable[[SupportAmortizer], None],
) -> None:
    model = _amortizer().eval()
    corruption(model)
    with pytest.raises(RuntimeError, match="amortizer architecture or topology was mutated"):
        model(*_valid_inputs())


def test_support_order_and_padding_are_permutation_invariant() -> None:
    model = _amortizer().eval()
    duplicate = torch.randn(2, 1, 6)
    support = torch.cat((duplicate, duplicate.clone(), torch.randn(2, 2, 6)), dim=1)
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    support[~mask] = torch.nan
    descriptions = torch.randn(2, 8)
    permutation = torch.tensor([3, 1, 0, 2])

    first = model(support, mask, descriptions).coefficients
    second = model(
        support[:, permutation], mask[:, permutation], descriptions
    ).coefficients
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    changed_padding = support.clone()
    changed_padding[~mask] = torch.inf
    third = model(changed_padding, mask, descriptions).coefficients
    torch.testing.assert_close(first, third, rtol=0.0, atol=0.0)


def test_padding_count_is_invariant_but_duplicate_observations_keep_multiplicity() -> None:
    model = _amortizer().eval()
    valid = torch.randn(2, 2, 6)
    descriptions = torch.randn(2, 8)
    compact = model(valid, torch.ones(2, 2, dtype=torch.bool), descriptions).coefficients

    padding = torch.full((2, 3, 6), torch.nan)
    padded = torch.cat((valid, padding), dim=1)
    padded_mask = torch.cat(
        (torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 3, dtype=torch.bool)), dim=1
    )
    extended = model(padded, padded_mask, descriptions).coefficients
    torch.testing.assert_close(compact, extended, rtol=1e-5, atol=1e-6)

    one_observation = model(
        valid[:, :1], torch.ones(2, 1, dtype=torch.bool), descriptions
    ).coefficients
    duplicated = model(
        valid[:, :1].expand(-1, 2, -1),
        torch.ones(2, 2, dtype=torch.bool),
        descriptions,
    ).coefficients
    assert not torch.allclose(one_observation, duplicated, rtol=1e-6, atol=1e-7)


def test_prediction_is_fp32_bounded_and_has_canonical_480_code() -> None:
    model = _amortizer(support_dim=384, description_dim=2304).eval()
    prediction = model(
        torch.randn(2, 3, 384),
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.randn(2, 2304),
    )
    assert prediction.logits.shape == prediction.coefficients.shape == (2, 120, 4)
    assert prediction.logits.dtype == prediction.coefficients.dtype == torch.float32
    assert prediction.scales.shape == (120, 1)
    assert prediction.scales.dtype == torch.float32
    assert prediction.coefficients.reshape(2, -1).shape == (2, 480)
    assert torch.all(torch.isfinite(prediction.logits))
    assert torch.all(torch.isfinite(prediction.scales))
    assert torch.all(prediction.scales > 0)
    assert torch.all(prediction.coefficients.abs() <= prediction.scales.unsqueeze(0))


def test_prediction_stays_fp32_under_cpu_autocast() -> None:
    model = _amortizer().eval()
    support = torch.randn(2, 2, 6)
    mask = torch.ones(2, 2, dtype=torch.bool)
    descriptions = torch.randn(2, 8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction = model(support, mask, descriptions)
    assert prediction.logits.dtype == torch.float32
    assert prediction.scales.dtype == torch.float32
    assert prediction.coefficients.dtype == torch.float32


def test_train_and_eval_paths_are_deterministic_without_dropout() -> None:
    model = _amortizer()
    support = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    descriptions = torch.randn(2, 8)
    for training in (True, False):
        model.train(training)
        first = model(support, mask, descriptions).coefficients
        second = model(support, mask, descriptions).coefficients
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_gradients_reach_every_parameter_and_only_valid_input_positions() -> None:
    model = _amortizer(projection_count=6, layers=1).train()
    support = torch.randn(2, 3, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    descriptions = torch.randn(2, 8, requires_grad=True)
    prediction = model(support, mask, descriptions)
    prediction.coefficients.square().mean().backward()

    assert support.grad is not None
    assert descriptions.grad is not None
    assert torch.all(support.grad[~mask] == 0)
    assert torch.all(torch.isfinite(support.grad[mask]))
    assert torch.all(torch.isfinite(descriptions.grad))
    assert torch.all(support.grad[mask].abs().sum(dim=-1) > 0)
    assert torch.all(descriptions.grad.abs().sum(dim=-1) > 0)
    assert all(parameter.requires_grad for parameter in model.parameters())
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None
        or not bool(torch.isfinite(parameter.grad).all())
        or not bool((parameter.grad != 0).any())
    ]
    assert invalid == []


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


def test_flattened_coefficients_reach_every_tiny_sana_atom_and_amortizer_parameter() -> None:
    torch.manual_seed(41)
    transformer = _tiny_sana().requires_grad_(False).eval()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=1
    )
    amortizer = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=16,
        projection_count=6,
        atom_count=4,
        layers=1,
        heads=4,
    ).train()
    prediction = amortizer(
        torch.randn(2, 2, 6),
        torch.ones(2, 2, dtype=torch.bool),
        torch.randn(2, 8),
    )
    flattened = prediction.coefficients.reshape(2, -1)
    assert flattened.shape == (2, 24)

    with bank.activate(flattened):
        loss = sum(
            wrapper(torch.randn(2, 3, wrapper.base.in_features)).square().mean()
            for wrapper in bank.wrappers
        )
        loss.backward()

    invalid_atoms = [
        name
        for name, parameter in transformer.named_parameters()
        if ("atom_down" in name or "atom_up" in name)
        and (
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            or not bool((parameter.grad != 0).any())
        )
    ]
    atom_parameters = [
        parameter
        for name, parameter in transformer.named_parameters()
        if "atom_down" in name or "atom_up" in name
    ]
    assert len(atom_parameters) == 12
    assert invalid_atoms == []
    base_parameters = [
        parameter
        for name, parameter in transformer.named_parameters()
        if ".base." in name
    ]
    assert base_parameters
    assert all(parameter.grad is None for parameter in base_parameters)
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool((parameter.grad != 0).any())
        for parameter in amortizer.parameters()
    )


def test_empty_support_slots_and_all_masked_rows_are_rejected() -> None:
    model = _amortizer().eval()
    with pytest.raises(ValueError, match="positive batch and support dimensions"):
        model(torch.empty(0, 2, 6), torch.empty(0, 2, dtype=torch.bool), torch.empty(0, 8))
    with pytest.raises(ValueError, match="positive batch and support dimensions"):
        model(torch.empty(1, 0, 6), torch.empty(1, 0, dtype=torch.bool), torch.ones(1, 8))
    with pytest.raises(ValueError, match="each concept requires at least one support image"):
        model(
            torch.randn(2, 2, 6),
            torch.tensor([[True, False], [False, False]]),
            torch.randn(2, 8),
        )


def _valid_inputs() -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.randn(2, 3, 6),
        torch.tensor([[True, True, False], [True, True, True]]),
        torch.randn(2, 8),
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda support, mask, description: (support[0], mask, description), "rank-3"),
        (
            lambda support, mask, description: (support, mask[:, :2], description),
            "mask shape",
        ),
        (
            lambda support, mask, description: (support[:, :, :5], mask, description),
            "support feature dimension must be 6",
        ),
        (
            lambda support, mask, description: (support, mask, description[0]),
            "description features must be rank-2",
        ),
        (
            lambda support, mask, description: (support, mask, description[:, :7]),
            "description feature dimension must be 8",
        ),
        (
            lambda support, mask, description: (support, mask, description[:1]),
            "batch sizes must match",
        ),
        (
            lambda support, mask, description: (support.double(), mask, description),
            "support features must have dtype torch.float32",
        ),
        (
            lambda support, mask, description: (support, mask.to(torch.int64), description),
            "support mask must have dtype torch.bool",
        ),
        (
            lambda support, mask, description: (support, mask, description.double()),
            "description features must have dtype torch.float32",
        ),
    ],
)
def test_forward_validates_shapes_and_dtypes(
    mutate: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]], message: str
) -> None:
    inputs = mutate(*_valid_inputs())
    with pytest.raises((TypeError, ValueError), match=message):
        _amortizer().eval()(*inputs)


def test_forward_rejects_non_tensors_and_cross_device_inputs() -> None:
    support, mask, descriptions = _valid_inputs()
    with pytest.raises(TypeError, match="support_features must be a Tensor"):
        _amortizer().eval()(object(), mask, descriptions)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same device"):
        _amortizer().eval()(support, mask.to(device="meta"), descriptions)


def test_forward_rejects_nonfinite_valid_features_but_ignores_masked_padding() -> None:
    model = _amortizer().eval()
    support, mask, descriptions = _valid_inputs()
    support[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="unmasked support features must be finite"):
        model(support, mask, descriptions)

    support, mask, descriptions = _valid_inputs()
    descriptions[0, 0] = torch.inf
    with pytest.raises(ValueError, match="description features must be finite"):
        model(support, mask, descriptions)


def test_forward_rejects_non_fp32_amortizer_parameters() -> None:
    support, mask, descriptions = _valid_inputs()
    with pytest.raises(ValueError, match="amortizer parameters must have dtype torch.float32"):
        _amortizer().double().eval()(support, mask, descriptions)


def test_forward_requires_trainable_noninference_unaliased_parameters() -> None:
    support, mask, descriptions = _valid_inputs()
    frozen = _amortizer().eval()
    frozen.support_type.requires_grad_(False)
    with pytest.raises(ValueError, match="every amortizer parameter must require gradients"):
        frozen(support, mask, descriptions)

    with torch.inference_mode():
        inference_parameters = _amortizer().eval()
    with pytest.raises(ValueError, match="amortizer parameters must not be inference tensors"):
        inference_parameters(support, mask, descriptions)

    duplicate_object = _amortizer().eval()
    duplicate_object.description_type = duplicate_object.support_type
    with pytest.raises(ValueError, match="duplicate parameter objects"):
        duplicate_object(support, mask, descriptions)

    aliased_storage = _amortizer().eval()
    aliased_storage.description_type.data = aliased_storage.support_type.data
    with pytest.raises(ValueError, match="parameter storage aliases"):
        aliased_storage(support, mask, descriptions)


def test_inference_features_are_rejected_for_training_but_allowed_for_inference() -> None:
    model = _amortizer().eval()
    with torch.inference_mode():
        support = torch.randn(2, 3, 6)
        mask = torch.ones(2, 3, dtype=torch.bool)
        descriptions = torch.randn(2, 8)
    with pytest.raises(ValueError, match="inference support or description features"):
        model(support, mask, descriptions)
    with torch.inference_mode():
        prediction = model(support, mask, descriptions)
    assert prediction.coefficients.shape == (2, 120, 4)
    with torch.no_grad():
        no_grad_prediction = model(support, mask, descriptions)
    assert no_grad_prediction.coefficients.shape == (2, 120, 4)


def test_inference_boolean_mask_is_safe_during_gradient_enabled_training() -> None:
    model = _amortizer(projection_count=6, layers=1).train()
    support = torch.randn(2, 3, 6, requires_grad=True)
    descriptions = torch.randn(2, 8, requires_grad=True)
    with torch.inference_mode():
        mask = torch.tensor([[True, True, False], [True, False, True]])
    prediction = model(support, mask, descriptions)
    prediction.coefficients.square().mean().backward()
    assert support.grad is not None and descriptions.grad is not None


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda model: model.raw_projection_scale.fill_(float("-inf")),
        lambda model: model.support_projection.weight.fill_(float("nan")),
        lambda model: model.register_buffer("poison", torch.tensor(float("nan"))),
    ],
)
def test_forward_rejects_nonfinite_amortizer_parameter_or_buffer(
    corrupt: Callable[[SupportAmortizer], object],
) -> None:
    model = _amortizer().eval()
    with torch.no_grad():
        corrupt(model)
    with pytest.raises(ValueError, match="amortizer floating tensors must be finite"):
        model(*_valid_inputs())


def test_strict_state_roundtrip_preserves_prediction_exactly() -> None:
    first = _amortizer().eval()
    support, mask, descriptions = _valid_inputs()
    expected = first(support, mask, descriptions)
    state = {name: value.detach().clone() for name, value in first.state_dict().items()}

    torch.manual_seed(99)
    second = SupportAmortizer(
        support_dim=6,
        description_dim=8,
        hidden_dim=32,
        projection_count=120,
        atom_count=4,
        layers=2,
        heads=4,
    ).eval()
    result = second.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    actual = second(support, mask, descriptions)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.scales, expected.scales, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.coefficients, expected.coefficients, rtol=0.0, atol=0.0)

    incomplete = dict(state)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(RuntimeError, match="Missing key"):
        second.load_state_dict(incomplete, strict=True)
