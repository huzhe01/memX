from __future__ import annotations

import gc
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from inspect import Parameter, signature
from typing import Any
from weakref import ref

import pytest
import torch
from torch import nn

import ratemem.adapters.sana_layout as sana_layout_module
from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear
from ratemem.adapters.sana_layout import (
    ATTENTION_KINDS,
    PRODUCTION_ATOM_COUNT,
    PRODUCTION_BLOCK_COUNT,
    PRODUCTION_RANK,
    PRODUCTION_WIDTH,
    SANA_LAYOUT_VERSION,
    TARGET_MODULES,
    SanaAdapterLayout,
    SanaDynamicAdapterBank,
    install_sana_dynamic_atoms,
    validate_production_sana_layout,
)


class _IntSubclass(int):
    pass


class _LinearSubclass(nn.Linear):
    pass


class _ToyAttention(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        options: dict[str, Any] = {"device": device, "dtype": dtype}
        self.to_q = nn.Linear(width, width, bias=bias, **options)
        self.to_k = nn.Linear(width, width, bias=bias, **options)
        self.to_v = nn.Linear(width, width, bias=bias, **options)
        self.to_out = nn.ModuleList(
            [nn.Linear(width, width, bias=True, **options), nn.Dropout(0.0)]
        )


class _AdversarialSetterAttention(_ToyAttention):
    _armed: bool
    setter_calls: int

    def __init__(self, width: int, *, bias: bool) -> None:
        object.__setattr__(self, "_armed", False)
        object.__setattr__(self, "setter_calls", 0)
        super().__init__(width, bias=bias)
        self.register_buffer("side_effect_buffer", torch.zeros(1))

    def arm(self) -> None:
        object.__setattr__(self, "_armed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if (
            getattr(self, "_armed", False)
            and name in TARGET_MODULES
            and isinstance(value, DynamicAtomLinear)
        ):
            object.__setattr__(self, "setter_calls", self.setter_calls + 1)
            with torch.no_grad():
                self.to_out[0].weight.add_(100.0)
                self.side_effect_buffer.add_(1.0)
            self.train(True)
            raise RuntimeError("adversarial setter ran")
        super().__setattr__(name, value)


class _AdversarialGetterAttention(_ToyAttention):
    _armed: bool
    getter_calls: int

    def __init__(self, width: int, *, bias: bool) -> None:
        object.__setattr__(self, "_armed", False)
        object.__setattr__(self, "getter_calls", 0)
        super().__init__(width, bias=bias)
        self.register_buffer("getter_probe", torch.zeros(1))

    def arm(self) -> None:
        object.__setattr__(self, "_armed", True)

    def __getattribute__(self, name: str) -> Any:
        if (
            name in TARGET_MODULES
            and object.__getattribute__(self, "_armed")
        ):
            object.__setattr__(
                self,
                "getter_calls",
                object.__getattribute__(self, "getter_calls") + 1,
            )
            buffers = object.__getattribute__(self, "_buffers")
            with torch.no_grad():
                buffers["getter_probe"].add_(1.0)
            self.train(True)
        return super().__getattribute__(name)


class _ToyBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn1 = _ToyAttention(
            width, bias=False, device=device, dtype=dtype
        )
        self.attn2 = _ToyAttention(width, bias=True, device=device, dtype=dtype)
        self.ff = nn.Linear(width, width, device=device, dtype=dtype)


class _ToyTransformer(nn.Module):
    def __init__(
        self,
        blocks: int = 2,
        width: int = 8,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                _ToyBlock(width, device=device, dtype=dtype)
                for _ in range(blocks)
            ]
        )
        self.final = nn.Linear(width, width, device=device, dtype=dtype)


def _toy(
    *, blocks: int = 2, width: int = 8, training: bool = False
) -> _ToyTransformer:
    transformer = _ToyTransformer(blocks=blocks, width=width)
    transformer.requires_grad_(False)
    transformer.train(training)
    return transformer


def _targets(transformer: _ToyTransformer) -> tuple[nn.Module, ...]:
    return tuple(
        getattr(getattr(block, attention), projection)
        for block in transformer.transformer_blocks
        for attention in ATTENTION_KINDS
        for projection in TARGET_MODULES
    )


def _snapshot(transformer: nn.Module) -> dict[str, object]:
    return {
        "modules": tuple(
            (name, id(module), module.training)
            for name, module in transformer.named_modules(remove_duplicate=False)
        ),
        "parameters": tuple(
            (
                name,
                id(parameter),
                parameter.requires_grad,
                parameter.device,
                parameter.dtype,
            )
            for name, parameter in transformer.named_parameters(
                remove_duplicate=False
            )
        ),
        "state": OrderedDict(
            (
                name,
                value.detach().clone()
                if value.device.type != "meta"
                else (tuple(value.shape), value.device, value.dtype),
            )
            for name, value in transformer.state_dict().items()
        ),
    }


def _assert_snapshot(transformer: nn.Module, expected: dict[str, object]) -> None:
    actual = _snapshot(transformer)
    assert actual["modules"] == expected["modules"]
    assert actual["parameters"] == expected["parameters"]
    expected_state = expected["state"]
    actual_state = actual["state"]
    assert isinstance(expected_state, OrderedDict)
    assert isinstance(actual_state, OrderedDict)
    assert tuple(actual_state) == tuple(expected_state)
    for name in expected_state:
        actual_value = actual_state[name]
        expected_value = expected_state[name]
        if isinstance(expected_value, torch.Tensor):
            assert isinstance(actual_value, torch.Tensor)
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_production_layout_order_formula_and_config_constants_are_canonical() -> None:
    layout = SanaAdapterLayout(
        num_blocks=PRODUCTION_BLOCK_COUNT,
        atom_count=PRODUCTION_ATOM_COUNT,
    )

    assert SANA_LAYOUT_VERSION == "sana-qkv-v1"
    assert ATTENTION_KINDS == ("attn1", "attn2")
    assert TARGET_MODULES == ("to_q", "to_k", "to_v")
    assert layout.code_shape == (20, 2, 3, 4)
    assert layout.projection_count == 120
    assert layout.code_dim == 480
    assert layout.atom_tensor_count == 240
    assert len(layout.projection_names) == 120
    assert layout.projection_names[:4] == (
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.attn1.to_k",
        "transformer_blocks.0.attn1.to_v",
        "transformer_blocks.0.attn2.to_q",
    )
    assert layout.projection_names[-1] == "transformer_blocks.19.attn2.to_v"
    assert layout.trainable_parameter_count(
        width=PRODUCTION_WIDTH, rank=PRODUCTION_RANK
    ) == 8_601_600


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"num_blocks": True, "atom_count": 4}, TypeError, "num_blocks"),
        ({"num_blocks": _IntSubclass(2), "atom_count": 4}, TypeError, "num_blocks"),
        ({"num_blocks": 0, "atom_count": 4}, ValueError, "num_blocks"),
        ({"num_blocks": 2, "atom_count": False}, TypeError, "atom_count"),
        ({"num_blocks": 2, "atom_count": 0}, ValueError, "atom_count"),
    ],
)
def test_layout_requires_exact_positive_integers(
    arguments: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        SanaAdapterLayout(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("width", True, TypeError),
        ("width", 0, ValueError),
        ("rank", _IntSubclass(4), TypeError),
        ("rank", 0, ValueError),
    ],
)
def test_parameter_formula_requires_exact_positive_integers(
    field: str, value: object, error: type[Exception]
) -> None:
    arguments: dict[str, object] = {"width": 8, "rank": 2}
    arguments[field] = value
    with pytest.raises(error, match=field):
        SanaAdapterLayout(2, 4).trainable_parameter_count(**arguments)  # type: ignore[arg-type]


def test_install_is_transactional_for_a_missing_or_wrong_last_target() -> None:
    for replacement in (None, nn.Identity()):
        transformer = _toy()
        attention = transformer.transformer_blocks[-1].attn2
        if replacement is None:
            delattr(attention, "to_v")
        else:
            attention.to_v = replacement  # type: ignore[assignment]
        before = _snapshot(transformer)

        with pytest.raises((TypeError, ValueError), match="to_v"):
            install_sana_dynamic_atoms(
                transformer, rank=2, atom_count=4, expected_blocks=2
            )

        _assert_snapshot(transformer, before)


def test_nth_constructor_failure_leaves_the_transformer_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    before = _snapshot(transformer)
    original = sana_layout_module.DynamicAtomLinear
    calls = 0

    def fail_fourth(
        base: nn.Linear, *, rank: int, atom_count: int
    ) -> DynamicAtomLinear:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("constructor sentinel")
        return original(base, rank=rank, atom_count=atom_count)

    monkeypatch.setattr(sana_layout_module, "DynamicAtomLinear", fail_fourth)

    with pytest.raises(RuntimeError, match="constructor sentinel"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    assert calls == 4
    _assert_snapshot(transformer, before)


def test_direct_commit_failure_after_current_write_rolls_back_every_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    before = _snapshot(transformer)
    calls = 0

    def fail_after_write(
        owner: nn.Module,
        attribute: str,
        expected: nn.Module,
        replacement: nn.Module,
    ) -> None:
        nonlocal calls
        assert owner._modules.get(attribute) is expected
        owner._modules[attribute] = replacement
        calls += 1
        if calls == 6:
            raise RuntimeError("commit sentinel")

    monkeypatch.setattr(
        sana_layout_module,
        "_commit_target_module",
        fail_after_write,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="commit sentinel"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    assert calls == 6
    _assert_snapshot(transformer, before)


def test_install_never_invokes_an_adversarial_target_setter() -> None:
    transformer = _toy()
    attention = _AdversarialSetterAttention(8, bias=True)
    attention.requires_grad_(False)
    attention.eval()
    transformer.transformer_blocks[-1].attn2 = attention
    unrelated_weight = attention.to_out[0].weight.detach().clone()
    unrelated_buffer = attention.side_effect_buffer.detach().clone()
    attention.arm()

    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )

    assert len(bank.wrappers) == 12
    assert attention.setter_calls == 0
    assert torch.equal(attention.to_out[0].weight, unrelated_weight)
    assert torch.equal(attention.side_effect_buffer, unrelated_buffer)
    assert all(not module.training for module in transformer.modules())


def test_inventory_never_invokes_an_adversarial_target_getter() -> None:
    transformer = _toy()
    attention = _AdversarialGetterAttention(8, bias=True)
    attention.requires_grad_(False)
    attention.eval()
    transformer.transformer_blocks[-1]._modules["attn2"] = attention
    probe_before = attention.getter_probe.detach().clone()
    attention.arm()

    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )

    assert len(bank.wrappers) == 12
    assert attention.getter_calls == 0
    assert torch.equal(attention.getter_probe, probe_before)
    assert all(not module.training for module in transformer.modules())


def test_install_rejects_a_custom_target_module_registry_without_mutation() -> None:
    class _CustomModuleRegistry(dict[str, nn.Module | None]):
        pass

    transformer = _toy()
    attention = transformer.transformer_blocks[-1].attn2
    attention._modules = _CustomModuleRegistry(attention._modules)
    before = _snapshot(transformer)

    with pytest.raises(TypeError, match="exact built-in dict"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


@pytest.mark.parametrize(
    "container",
    ["root", "module-list", "block"],
)
def test_install_rejects_noncanonical_container_registration_without_mutation(
    container: str,
) -> None:
    transformer = _toy()
    blocks = transformer.transformer_blocks
    if container == "root":
        registered = transformer._modules.pop("transformer_blocks")
        transformer._modules["noncanonical"] = registered
        object.__setattr__(transformer, "transformer_blocks", registered)
    elif container == "module-list":
        registered = blocks._modules.pop("0")
        blocks._modules["noncanonical"] = registered
    else:
        block = blocks[0]
        registered = block._modules.pop("attn1")
        block._modules["noncanonical"] = registered
        object.__setattr__(block, "attn1", registered)
    before = _snapshot(transformer)

    with pytest.raises(TypeError, match="canonically registered"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


def test_install_preserves_mode_placement_and_only_atoms_are_trainable(
) -> None:
    transformer = _toy(training=False)
    original_targets = _targets(transformer)
    original_non_qkv = tuple(
        block.attn1.to_out[0] for block in transformer.transformer_blocks
    )

    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )

    assert not transformer.training
    assert all(not module.training for module in transformer.modules())
    assert len(bank.wrappers) == 12
    for original, wrapper in zip(original_targets, bank.wrappers, strict=True):
        assert isinstance(wrapper, DynamicAtomLinear)
        assert wrapper.base is original
        assert not wrapper.training
        assert not wrapper.base.training
        assert not wrapper.base.weight.requires_grad
        assert wrapper.atom_down.requires_grad and wrapper.atom_up.requires_grad
        assert wrapper.atom_down.device == wrapper.base.weight.device
        assert wrapper.atom_down.dtype == wrapper.base.weight.dtype
        assert wrapper.atom_up.device == wrapper.base.weight.device
        assert wrapper.atom_up.dtype == wrapper.base.weight.dtype
    assert original_non_qkv == tuple(
        block.attn1.to_out[0] for block in transformer.transformer_blocks
    )
    trainable_names = {
        name for name, parameter in transformer.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        f"{path}.{atom}"
        for path in bank.layout.projection_names
        for atom in ("atom_down", "atom_up")
    }


def test_install_requires_the_entire_transformer_to_be_frozen() -> None:
    transformer = _toy()
    transformer.final.weight.requires_grad_(True)
    before = _snapshot(transformer)

    with pytest.raises(ValueError, match="entire transformer must be frozen"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


@pytest.mark.parametrize("training_scope", ["root", "deep-child"])
def test_install_requires_every_submodule_to_already_be_in_eval(
    training_scope: str,
) -> None:
    transformer = _toy()
    if training_scope == "root":
        transformer.train()
    else:
        transformer.transformer_blocks[0].attn1.to_out[1].train()
    before = _snapshot(transformer)

    with pytest.raises(ValueError, match="every transformer module must be in eval"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


def _apply_alias_case(transformer: _ToyTransformer, case: str) -> None:
    first = transformer.transformer_blocks[0]
    second = transformer.transformer_blocks[1]
    if case == "block":
        transformer.transformer_blocks[1] = first
    elif case == "attention":
        second.attn1 = first.attn1
    elif case == "linear":
        second.attn1.to_q = first.attn1.to_q
    elif case == "parameter":
        second.attn1.to_q.weight = first.attn1.to_q.weight
    elif case == "storage":
        second.attn1.to_q.weight = nn.Parameter(
            first.attn1.to_q.weight.detach(), requires_grad=False
        )
    elif case == "fused":
        second.attn1.to_q = _LinearSubclass(8, 8, bias=False)
        second.attn1.to_q.requires_grad_(False)
    elif case == "wrapped":
        second.attn1.to_q = DynamicAtomLinear(
            second.attn1.to_q, rank=2, atom_count=4
        )  # type: ignore[assignment]
        second.attn1.to_q.requires_grad_(False)
    elif case == "non-target-module":
        transformer.final = first.attn1.to_q
    elif case == "non-target-parameter":
        transformer.final.weight = first.attn1.to_q.weight
    elif case == "non-target-storage":
        first.ff.weight = nn.Parameter(
            first.attn1.to_q.weight.detach(), requires_grad=False
        )
    elif case == "non-target-buffer-storage":
        transformer.register_buffer(
            "target_storage_alias", first.attn1.to_q.weight.detach()
        )
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "block",
        "attention",
        "linear",
        "parameter",
        "storage",
        "fused",
        "wrapped",
        "non-target-module",
        "non-target-parameter",
        "non-target-storage",
        "non-target-buffer-storage",
    ],
)
def test_alias_fused_and_wrapped_targets_are_rejected_without_mutation(
    case: str,
) -> None:
    transformer = _toy()
    _apply_alias_case(transformer, case)
    before = _snapshot(transformer)

    with pytest.raises((TypeError, ValueError), match="alias|exact nn.Linear|wrapped"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


@pytest.mark.parametrize(
    "case",
    [
        "meta",
        "heterogeneous-dtype",
        "bias-placement",
        "weight-shape",
        "bias-shape",
    ],
)
def test_generic_install_rejects_unmaterialized_heterogeneous_or_malformed_targets(
    case: str,
) -> None:
    if case == "meta":
        transformer = _ToyTransformer(blocks=2, width=8, device="meta")
    else:
        transformer = _toy()
        attention = transformer.transformer_blocks[-1].attn2
        if case == "heterogeneous-dtype":
            attention.to_v = nn.Linear(8, 8, dtype=torch.float64)
        elif case == "bias-placement":
            attention.to_v.bias = nn.Parameter(torch.empty(8, device="meta"))
        elif case == "weight-shape":
            attention.to_v.weight = nn.Parameter(torch.empty(7, 8))
        else:
            attention.to_v.bias = nn.Parameter(torch.empty(1))
    transformer.requires_grad_(False)
    transformer.eval()
    before = _snapshot(transformer)

    with pytest.raises(
        ValueError,
        match="materialized|placement|device and dtype|weight shape|bias shape|numel",
    ):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


@pytest.mark.parametrize("field", ["weight", "bias", "none-bias"])
def test_generic_install_requires_direct_parameter_registration(field: str) -> None:
    transformer = _toy()
    attention = transformer.transformer_blocks[-1].attn2
    target = attention.to_v
    if field == "none-bias":
        target = transformer.transformer_blocks[-1].attn1.to_v
        target._parameters.pop("bias")
        object.__setattr__(target, "bias", None)
        escaped: nn.Parameter | None = None
    else:
        escaped = target._parameters.pop(field)
        assert isinstance(escaped, nn.Parameter)
        object.__setattr__(target, field, escaped)
        escaped.requires_grad_(True)
    transformer.requires_grad_(False)
    transformer.eval()
    before = _snapshot(transformer)

    with pytest.raises(ValueError, match="directly registered.*weight|bias"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)
    assert field not in target._parameters
    if escaped is not None:
        assert getattr(target, field) is escaped
        assert escaped.requires_grad


def test_second_install_is_rejected_without_mutation() -> None:
    transformer = _toy()
    install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    before = _snapshot(transformer)

    with pytest.raises((TypeError, ValueError), match="wrapped|already installed"):
        install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )

    _assert_snapshot(transformer, before)


def test_unrelated_scalar_and_empty_buffers_do_not_create_false_storage_aliases() -> None:
    transformer = _toy()
    transformer.register_buffer("scalar_metadata", torch.tensor(1.0))
    transformer.register_buffer("empty_metadata_a", torch.empty(0))
    transformer.register_buffer("empty_metadata_b", torch.empty(0))

    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )

    assert len(bank.wrappers) == 12


def test_bank_is_non_owning_and_exposes_unique_canonical_trainables() -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )

    assert not isinstance(bank, nn.Module)
    assert isinstance(bank, SanaDynamicAdapterBank)
    assert len(bank.wrappers) == bank.layout.projection_count
    parameters = list(bank.parameters())
    named_parameters = list(bank.named_parameters())
    assert len(parameters) == 24
    assert len({id(parameter) for parameter in parameters}) == 24
    assert [parameter for _, parameter in named_parameters] == parameters
    assert [name for name, _ in named_parameters] == [
        f"{path}.{atom}"
        for path in bank.layout.projection_names
        for atom in ("atom_down", "atom_up")
    ]
    assert not any(
        isinstance(value, nn.Module) for value in vars(bank).values()
    )


def test_public_bank_constructor_keeps_exact_two_argument_api() -> None:
    parameters = tuple(signature(SanaDynamicAdapterBank).parameters.values())

    assert tuple(parameter.name for parameter in parameters) == (
        "layout",
        "wrappers",
    )
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    )
    assert all(parameter.default is Parameter.empty for parameter in parameters)


def test_direct_bank_constructor_supports_state_and_activation() -> None:
    layout = SanaAdapterLayout(num_blocks=1, atom_count=3)
    wrappers = tuple(
        DynamicAtomLinear(nn.Linear(4, 4), rank=2, atom_count=3)
        for _path in layout.projection_names
    )
    bank = SanaDynamicAdapterBank(layout, wrappers)

    assert bank.wrappers == wrappers
    expected_keys = tuple(
        f"{path}.{atom}"
        for path in layout.projection_names
        for atom in ("atom_down", "atom_up")
    )
    state = bank.state_dict()
    assert tuple(state) == expected_keys
    replacement = OrderedDict(
        (name, torch.full_like(value, float(index + 1)))
        for index, (name, value) in enumerate(state.items())
    )
    result = bank.load_state_dict(replacement, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for name, value in bank.state_dict().items():
        torch.testing.assert_close(value, replacement[name], rtol=0.0, atol=0.0)

    code = torch.arange(layout.code_dim, dtype=torch.float32)
    with bank.activate(code):
        for index, wrapper in enumerate(wrappers):
            start = index * layout.atom_count
            torch.testing.assert_close(
                wrapper._coefficients,
                code[start : start + layout.atom_count],
                rtol=0.0,
                atol=0.0,
            )
    assert all(wrapper._coefficients is None for wrapper in wrappers)


def test_direct_bank_does_not_keep_wrappers_alive() -> None:
    def construct() -> tuple[SanaDynamicAdapterBank, tuple[Any, ...]]:
        layout = SanaAdapterLayout(num_blocks=1, atom_count=3)
        wrappers = tuple(
            DynamicAtomLinear(nn.Linear(4, 4), rank=2, atom_count=3)
            for _path in layout.projection_names
        )
        return SanaDynamicAdapterBank(layout, wrappers), tuple(
            ref(wrapper) for wrapper in wrappers
        )

    bank, wrapper_refs = construct()
    gc.collect()

    assert all(wrapper_ref() is None for wrapper_ref in wrapper_refs)
    with pytest.raises(RuntimeError, match="released"):
        _ = bank.wrappers


def test_bank_does_not_keep_transformer_owner_or_wrapper_alive() -> None:
    def install() -> tuple[
        SanaDynamicAdapterBank,
        Any,
        Any,
        Any,
    ]:
        transformer = _toy()
        bank = install_sana_dynamic_atoms(
            transformer, rank=2, atom_count=4, expected_blocks=2
        )
        return (
            bank,
            ref(transformer),
            ref(transformer.transformer_blocks[0].attn1),
            ref(bank.wrappers[0]),
        )

    bank, transformer_ref, owner_ref, wrapper_ref = install()
    gc.collect()

    assert transformer_ref() is None
    assert owner_ref() is None
    assert wrapper_ref() is None
    with pytest.raises(RuntimeError, match="released|canonical"):
        _ = bank.wrappers


@pytest.mark.parametrize(
    "entrypoint",
    ["wrappers", "named_parameters", "parameters", "state_dict", "load", "activate"],
)
@pytest.mark.parametrize(
    "drift",
    ["target-replaced", "qk-swapped", "attentions-swapped", "blocks-swapped"],
)
def test_bank_entrypoints_fail_closed_after_canonical_path_drift(
    entrypoint: str,
    drift: str,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    original_wrappers = bank.wrappers
    saved_state = OrderedDict(
        (name, value.detach().clone()) for name, value in bank.state_dict().items()
    )
    first_owner = transformer.transformer_blocks[0].attn1
    if drift == "target-replaced":
        first_owner._modules["to_q"] = original_wrappers[0].base
    elif drift == "qk-swapped":
        first_owner._modules["to_q"], first_owner._modules["to_k"] = (
            first_owner._modules["to_k"],
            first_owner._modules["to_q"],
        )
    elif drift == "attentions-swapped":
        first_block = transformer.transformer_blocks[0]
        second_block = transformer.transformer_blocks[1]
        first_block._modules["attn1"], second_block._modules["attn1"] = (
            second_block._modules["attn1"],
            first_block._modules["attn1"],
        )
    else:
        blocks = transformer.transformer_blocks
        blocks._modules["0"], blocks._modules["1"] = (
            blocks._modules["1"],
            blocks._modules["0"],
        )

    def invoke() -> None:
        if entrypoint == "wrappers":
            _ = bank.wrappers
        elif entrypoint == "named_parameters":
            list(bank.named_parameters())
        elif entrypoint == "parameters":
            list(bank.parameters())
        elif entrypoint == "state_dict":
            bank.state_dict()
        elif entrypoint == "load":
            bank.load_state_dict(saved_state, strict=True)
        else:
            with bank.activate(torch.zeros(bank.layout.code_dim)):
                raise AssertionError("drifted bank became active")

    with pytest.raises(RuntimeError, match="canonical"):
        invoke()

    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in original_wrappers
    )


@pytest.mark.parametrize("batch", [None, 3])
def test_activation_maps_flat_arange_codes_exhaustively_and_preserves_gradients(
    batch: int | None,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    shape = (bank.layout.code_dim,) if batch is None else (batch, bank.layout.code_dim)
    code = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(
        shape
    )
    code.requires_grad_(True)

    with bank.activate(code):
        reshaped = (
            code.reshape(bank.layout.projection_count, bank.layout.atom_count)
            if batch is None
            else code.reshape(
                batch, bank.layout.projection_count, bank.layout.atom_count
            )
        )
        for index, wrapper in enumerate(bank.wrappers):
            expected = reshaped[index] if batch is None else reshaped[:, index]
            assert wrapper._coefficients is not None
            assert torch.equal(wrapper._coefficients, expected)
        active_sum = sum(
            wrapper._coefficients.sum()  # type: ignore[union-attr]
            for wrapper in bank.wrappers
        )
        active_sum.backward()

    assert code.grad is not None
    assert torch.equal(code.grad, torch.ones_like(code))
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (object(), TypeError, "Tensor"),
        (torch.tensor(1.0), ValueError, "1D or 2D"),
        (torch.zeros(2, 3, 4), ValueError, "1D or 2D"),
        (torch.zeros(47), ValueError, "code dimension must be 48"),
        (torch.zeros(2, 49), ValueError, "code dimension must be 48"),
    ],
)
def test_activation_prevalidates_the_whole_code_before_entering_any_wrapper(
    value: object, error: type[Exception], message: str
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )

    with pytest.raises(error, match=message):
        with bank.activate(value):  # type: ignore[arg-type]
            raise AssertionError("invalid code became active")

    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)


def test_nested_activation_preserves_the_outer_coefficients_and_cleans_up() -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    outer = torch.arange(bank.layout.code_dim, dtype=torch.float32)
    inner = torch.zeros_like(outer)

    with bank.activate(outer):
        active = tuple(wrapper._coefficients for wrapper in bank.wrappers)
        with pytest.raises(RuntimeError, match="already active"):
            with bank.activate(inner):
                raise AssertionError("nested activation entered")
        assert tuple(wrapper._coefficients for wrapper in bank.wrappers) == active
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)


def test_body_exception_and_partial_enter_failure_both_rollback_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)

    with pytest.raises(RuntimeError, match="body sentinel"):
        with bank.activate(code):
            raise RuntimeError("body sentinel")
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)

    external = torch.full((bank.layout.atom_count,), 7.0)
    with bank.wrappers[5].use_coefficients(external):
        external_state = (
            bank.wrappers[5]._coefficients,
            bank.wrappers[5]._activation_token,
            bank.wrappers[5]._coefficient_version,
        )
        with pytest.raises(RuntimeError, match="already active"):
            with bank.activate(code):
                raise AssertionError("externally occupied activation entered")
        assert all(
            wrapper._coefficients is None for wrapper in bank.wrappers[:5]
        )
        assert (
            bank.wrappers[5]._coefficients,
            bank.wrappers[5]._activation_token,
            bank.wrappers[5]._coefficient_version,
        ) == external_state
        assert all(
            wrapper._coefficients is None for wrapper in bank.wrappers[6:]
        )
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)

    @contextmanager
    def fail_enter(_coefficients: torch.Tensor) -> Any:
        raise RuntimeError("enter sentinel")
        yield

    monkeypatch.setattr(bank.wrappers[5], "use_coefficients", fail_enter)
    with pytest.raises(RuntimeError, match="enter sentinel"):
        with bank.activate(code):
            raise AssertionError("partial activation entered")
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)


def test_mutate_then_raise_during_enter_restores_exact_transient_state_and_reuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)
    failing_wrapper = bank.wrappers[5]
    original_use_coefficients = failing_wrapper.use_coefficients

    @contextmanager
    def mutate_then_fail(coefficients: torch.Tensor) -> Any:
        object.__setattr__(failing_wrapper, "_coefficients", coefficients)
        object.__setattr__(failing_wrapper, "_activation_token", object())
        object.__setattr__(failing_wrapper, "_coefficient_version", 73)
        raise RuntimeError("mutate-then-enter sentinel")
        yield

    monkeypatch.setattr(
        failing_wrapper, "use_coefficients", mutate_then_fail
    )
    with pytest.raises(RuntimeError, match="mutate-then-enter sentinel"):
        with bank.activate(code):
            raise AssertionError("partial activation entered")

    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in bank.wrappers
    )

    monkeypatch.setattr(
        failing_wrapper, "use_coefficients", original_use_coefficients
    )
    with bank.activate(code):
        assert all(wrapper._coefficients is not None for wrapper in bank.wrappers)
    assert all(wrapper._coefficients is None for wrapper in bank.wrappers)


@pytest.mark.parametrize("corruption", ["coefficient-version", "activation-token"])
def test_activation_fails_closed_instead_of_hiding_body_corruption(
    corruption: str,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)

    with pytest.raises(RuntimeError, match="modified|activation state"):
        with bank.activate(code):
            if corruption == "coefficient-version":
                code.add_(1.0)
            else:
                object.__setattr__(
                    bank.wrappers[5], "_activation_token", object()
                )

    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in bank.wrappers
    )
    with bank.activate(torch.zeros_like(code)):
        assert all(wrapper._coefficients is not None for wrapper in bank.wrappers)


def test_activation_validates_each_successful_enter_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)
    failing_wrapper = bank.wrappers[2]
    original_use_coefficients = failing_wrapper.use_coefficients

    @contextmanager
    def yield_without_activation(_coefficients: torch.Tensor) -> Any:
        yield

    monkeypatch.setattr(
        failing_wrapper,
        "use_coefficients",
        yield_without_activation,
    )
    with pytest.raises(RuntimeError, match="failed to activate"):
        with bank.activate(code):
            raise AssertionError("invalid enter reached the body")

    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in bank.wrappers
    )
    monkeypatch.setattr(
        failing_wrapper, "use_coefficients", original_use_coefficients
    )
    with bank.activate(code):
        assert all(wrapper._coefficients is not None for wrapper in bank.wrappers)


def test_activation_rejects_a_forged_untracked_version_in_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)
    failing_wrapper = bank.wrappers[2]

    @contextmanager
    def forge_version(coefficients: torch.Tensor) -> Any:
        object.__setattr__(failing_wrapper, "_coefficients", coefficients)
        object.__setattr__(failing_wrapper, "_activation_token", object())
        object.__setattr__(failing_wrapper, "_coefficient_version", -1)
        try:
            yield
        finally:
            object.__setattr__(failing_wrapper, "_coefficients", None)
            object.__setattr__(failing_wrapper, "_activation_token", None)
            object.__setattr__(failing_wrapper, "_coefficient_version", None)

    monkeypatch.setattr(failing_wrapper, "use_coefficients", forge_version)
    with torch.no_grad(), pytest.raises(RuntimeError, match="version"):
        with bank.activate(code):
            raise AssertionError("forged version reached the body")

    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in bank.wrappers
    )


def test_activation_restores_transients_when_successful_context_exit_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)
    failing_wrapper = bank.wrappers[2]
    original_use_coefficients = failing_wrapper.use_coefficients

    @contextmanager
    def poison_then_fail(coefficients: torch.Tensor) -> Any:
        with original_use_coefficients(coefficients):
            yield
        object.__setattr__(failing_wrapper, "_coefficients", coefficients)
        object.__setattr__(failing_wrapper, "_activation_token", object())
        object.__setattr__(failing_wrapper, "_coefficient_version", 123)
        raise RuntimeError("exit poison sentinel")

    monkeypatch.setattr(
        failing_wrapper,
        "use_coefficients",
        poison_then_fail,
    )
    with pytest.raises(RuntimeError, match="exit poison sentinel"):
        with bank.activate(code):
            pass

    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in bank.wrappers
    )
    monkeypatch.setattr(
        failing_wrapper, "use_coefficients", original_use_coefficients
    )
    with bank.activate(code):
        assert all(wrapper._coefficients is not None for wrapper in bank.wrappers)


def test_body_exception_cannot_hide_in_place_coefficient_corruption() -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    code = torch.zeros(bank.layout.code_dim)

    with pytest.raises(RuntimeError, match="modified") as raised:
        with bank.activate(code):
            code.add_(1.0)
            raise ValueError("body sentinel")

    assert isinstance(raised.value.__context__, ValueError)
    assert all(
        (
            wrapper._coefficients,
            wrapper._activation_token,
            wrapper._coefficient_version,
        )
        == (None, None, None)
        for wrapper in bank.wrappers
    )


def test_state_load_stages_aliasing_values_before_an_exact_swap() -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    named = list(bank.named_parameters())
    (first_name, first), (second_name, second) = named[0], named[2]
    assert first.shape == second.shape
    with torch.no_grad():
        first.fill_(1.0)
        second.fill_(2.0)

    result = bank.load_state_dict(
        OrderedDict(
            (
                (first_name, second.detach()),
                (second_name, first.detach()),
            )
        ),
        strict=False,
    )

    assert first_name not in result.missing_keys
    assert second_name not in result.missing_keys
    assert torch.equal(first, torch.full_like(first, 2.0))
    assert torch.equal(second, torch.full_like(second, 1.0))


def test_state_load_casts_independent_sources_to_target_placement_without_grad() -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    name, parameter = next(bank.named_parameters())
    source = torch.full(
        parameter.shape,
        3.25,
        dtype=torch.float64,
        requires_grad=True,
    )

    bank.load_state_dict({name: source}, strict=False)

    assert parameter.is_leaf
    assert parameter.requires_grad
    assert parameter.device.type == "cpu"
    assert parameter.dtype == torch.float32
    assert torch.equal(parameter, torch.full_like(parameter, 3.25))
    assert source.dtype == torch.float64
    assert source.requires_grad
    assert source.grad is None


def test_later_unmaterialized_state_value_fails_before_any_target_mutation() -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    named = list(bank.named_parameters())
    (first_name, first), (second_name, second) = named[:2]
    first_before = first.detach().clone()
    second_before = second.detach().clone()
    first_version = first._version
    second_version = second._version
    supplied = OrderedDict(
        (
            (first_name, torch.full_like(first, 99.0)),
            (second_name, torch.empty_like(second, device="meta")),
        )
    )

    with pytest.raises((NotImplementedError, RuntimeError), match="meta tensor"):
        bank.load_state_dict(supplied, strict=False)

    assert torch.equal(first, first_before)
    assert torch.equal(second, second_before)
    assert first._version == first_version
    assert second._version == second_version


@pytest.mark.parametrize("failure_phase", ["stage", "commit"])
def test_state_load_failure_is_transactional_and_does_not_mutate_sources(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    transformer = _toy()
    bank = install_sana_dynamic_atoms(
        transformer, rank=2, atom_count=4, expected_blocks=2
    )
    supplied = OrderedDict(
        (name, torch.randn_like(parameter, dtype=torch.float64, requires_grad=True))
        for name, parameter in bank.named_parameters()
    )
    sources_before = OrderedDict(
        (name, value.detach().clone()) for name, value in supplied.items()
    )
    targets_before = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in bank.named_parameters()
    )
    versions_before = OrderedDict(
        (name, parameter._version) for name, parameter in bank.named_parameters()
    )
    calls = 0

    if failure_phase == "stage":

        def fail_second_stage(
            parameter: nn.Parameter, value: torch.Tensor
        ) -> torch.Tensor:
            nonlocal calls
            calls += 1
            staged = (
                value.detach()
                .to(device=parameter.device, dtype=parameter.dtype)
                .clone()
            )
            if calls == 2:
                raise RuntimeError("stage sentinel")
            return staged

        monkeypatch.setattr(
            sana_layout_module,
            "_stage_state_value",
            fail_second_stage,
            raising=False,
        )
        expected_message = "stage sentinel"
    else:

        def fail_second_commit(
            parameter: nn.Parameter, value: torch.Tensor
        ) -> None:
            nonlocal calls
            calls += 1
            parameter.copy_(value)
            if calls == 2:
                raise RuntimeError("copy sentinel")

        monkeypatch.setattr(
            sana_layout_module,
            "_copy_staged_state_value",
            fail_second_commit,
            raising=False,
        )
        expected_message = "copy sentinel"

    with pytest.raises(RuntimeError, match=expected_message):
        bank.load_state_dict(supplied, strict=True)

    assert calls == 2
    for name, parameter in bank.named_parameters():
        assert torch.equal(parameter, targets_before[name])
        if failure_phase == "stage":
            assert parameter._version == versions_before[name]
        else:
            assert parameter._version > versions_before[name]
    for name, source in supplied.items():
        assert source.dtype == torch.float64
        assert source.requires_grad
        assert source.grad is None
        assert torch.equal(source, sources_before[name])


def test_full_and_trainable_only_state_dicts_have_strict_canonical_boundaries() -> None:
    first = _toy()
    first_bank = install_sana_dynamic_atoms(
        first, rank=2, atom_count=4, expected_blocks=2
    )
    first_state = OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in first.state_dict().items()
    )
    bank_state = first_bank.state_dict()

    expected_bank_keys = tuple(
        f"{path}.{atom}"
        for path in first_bank.layout.projection_names
        for atom in ("atom_down", "atom_up")
    )
    assert tuple(bank_state) == expected_bank_keys
    assert not any(".base." in name or name.startswith("wrappers.") for name in bank_state)

    second = _toy()
    second_bank = install_sana_dynamic_atoms(
        second, rank=2, atom_count=4, expected_blocks=2
    )
    incompatible = second.load_state_dict(first_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    result = second_bank.load_state_dict(bank_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert set(second_bank.state_dict()) == set(expected_bank_keys)
    assert all(wrapper.atom_down.requires_grad for wrapper in second_bank.wrappers)
    assert all(wrapper.atom_up.requires_grad for wrapper in second_bank.wrappers)

    x = torch.randn(3, 8)
    code = torch.randn(3, first_bank.layout.code_dim)
    with first_bank.activate(code), second_bank.activate(code):
        for first_wrapper, second_wrapper in zip(
            first_bank.wrappers, second_bank.wrappers, strict=True
        ):
            torch.testing.assert_close(
                second_wrapper(x), first_wrapper(x), rtol=0.0, atol=0.0
            )


def _production_meta_transformer() -> _ToyTransformer:
    transformer = _ToyTransformer(
        blocks=PRODUCTION_BLOCK_COUNT,
        width=PRODUCTION_WIDTH,
        device="meta",
        dtype=torch.bfloat16,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def test_production_validator_separates_structure_from_real_cuda_placement() -> None:
    transformer = _production_meta_transformer()

    layout = validate_production_sana_layout(
        transformer,
        rank=PRODUCTION_RANK,
        atom_count=PRODUCTION_ATOM_COUNT,
        require_cuda_bfloat16=False,
    )

    assert layout.projection_count == 120
    assert layout.code_dim == 480
    assert layout.atom_tensor_count == 240
    assert layout.trainable_parameter_count(
        width=PRODUCTION_WIDTH, rank=PRODUCTION_RANK
    ) == 8_601_600
    with pytest.raises(ValueError, match="CUDA bfloat16"):
        validate_production_sana_layout(
            transformer,
            rank=PRODUCTION_RANK,
            atom_count=PRODUCTION_ATOM_COUNT,
        )


def test_production_validator_reports_exact_shape_bias_and_hyperparameter_errors() -> None:
    def corrupt_feature_metadata(model: _ToyTransformer) -> None:
        target = model.transformer_blocks[-1].attn2.to_v
        target.in_features = PRODUCTION_WIDTH - 1
        target.out_features = PRODUCTION_WIDTH + 1

    factories: tuple[tuple[str, Callable[[_ToyTransformer], None], str], ...] = (
        (
            "shape",
            lambda model: setattr(
                model.transformer_blocks[-1].attn2,
                "to_v",
                nn.Linear(
                    PRODUCTION_WIDTH,
                    PRODUCTION_WIDTH - 1,
                    bias=True,
                    device="meta",
                    dtype=torch.bfloat16,
                ),
            ),
            "in_features.*2240.*out_features.*2240",
        ),
        (
            "attn1 bias",
            lambda model: setattr(
                model.transformer_blocks[-1].attn1,
                "to_v",
                nn.Linear(
                    PRODUCTION_WIDTH,
                    PRODUCTION_WIDTH,
                    bias=True,
                    device="meta",
                    dtype=torch.bfloat16,
                ),
            ),
            "attn1.*without bias",
        ),
        (
            "attn2 bias",
            lambda model: setattr(
                model.transformer_blocks[-1].attn2,
                "to_v",
                nn.Linear(
                    PRODUCTION_WIDTH,
                    PRODUCTION_WIDTH,
                    bias=False,
                    device="meta",
                    dtype=torch.bfloat16,
                ),
            ),
            "attn2.*with bias",
        ),
        (
            "attn2 bias shape",
            lambda model: setattr(
                model.transformer_blocks[-1].attn2.to_v,
                "bias",
                nn.Parameter(
                    torch.empty(1, device="meta", dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            ),
            "bias shape.*2240.*numel.*2240",
        ),
        (
            "feature metadata",
            corrupt_feature_metadata,
            "weight shape.*in_features.*2239.*out_features.*2241",
        ),
    )
    for _name, mutate, message in factories:
        transformer = _production_meta_transformer()
        mutate(transformer)
        transformer.requires_grad_(False)
        transformer.eval()
        with pytest.raises(ValueError, match=message):
            validate_production_sana_layout(
                transformer,
                rank=PRODUCTION_RANK,
                atom_count=PRODUCTION_ATOM_COUNT,
                require_cuda_bfloat16=False,
            )

    transformer = _production_meta_transformer()
    with pytest.raises(ValueError, match="production rank must be 4"):
        validate_production_sana_layout(
            transformer,
            rank=3,
            atom_count=PRODUCTION_ATOM_COUNT,
            require_cuda_bfloat16=False,
        )
    with pytest.raises(ValueError, match="production atom_count must be 4"):
        validate_production_sana_layout(
            transformer,
            rank=PRODUCTION_RANK,
            atom_count=3,
            require_cuda_bfloat16=False,
        )


def test_production_validator_rejects_bias_placement_corruption() -> None:
    transformer = _production_meta_transformer()
    target = transformer.transformer_blocks[-1].attn2.to_v
    target.bias = nn.Parameter(torch.empty(PRODUCTION_WIDTH, dtype=torch.float32))
    transformer.requires_grad_(False)
    transformer.eval()

    with pytest.raises(ValueError, match="bias placement"):
        validate_production_sana_layout(
            transformer,
            rank=PRODUCTION_RANK,
            atom_count=PRODUCTION_ATOM_COUNT,
            require_cuda_bfloat16=False,
        )
