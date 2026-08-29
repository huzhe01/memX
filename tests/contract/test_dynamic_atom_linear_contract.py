from inspect import getsource

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear


def _layer(*, bias: bool = True) -> DynamicAtomLinear:
    torch.manual_seed(17)
    return DynamicAtomLinear(nn.Linear(5, 7, bias=bias), rank=2, atom_count=3)


def test_atom_storage_is_created_from_the_base_weight() -> None:
    source = getsource(DynamicAtomLinear.__init__)
    assert source.count("base.weight.new_empty") == 2


def test_active_parameter_is_never_registered_or_serialized() -> None:
    layer = _layer()
    coefficients = nn.Parameter(torch.randn(2, layer.atom_count))
    expected_parameters = {"base.weight", "base.bias", "atom_down", "atom_up"}
    expected_state = expected_parameters

    assert set(dict(layer.named_parameters())) == expected_parameters
    assert dict(layer.named_buffers()) == {}
    assert set(layer.state_dict()) == expected_state
    with layer.use_coefficients(coefficients):
        assert layer._coefficients is coefficients
        assert set(dict(layer.named_parameters())) == expected_parameters
        assert dict(layer.named_buffers()) == {}
        assert set(layer.state_dict()) == expected_state
    assert set(dict(layer.named_parameters())) == expected_parameters
    assert dict(layer.named_buffers()) == {}
    assert set(layer.state_dict()) == expected_state


@pytest.mark.parametrize("bias", [False, True])
def test_state_dict_has_exact_keys_and_strictly_round_trips(bias: bool) -> None:
    layer = _layer(bias=bias)
    expected_keys = {"base.weight", "atom_down", "atom_up"}
    if bias:
        expected_keys.add("base.bias")
    state = {key: value.detach().clone() for key, value in layer.state_dict().items()}
    restored = _layer(bias=bias)

    incompatible = restored.load_state_dict(state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert set(restored.state_dict()) == expected_keys
    for key, expected in state.items():
        assert torch.equal(restored.state_dict()[key], expected)
    assert all(not parameter.requires_grad for parameter in restored.base.parameters())
    assert restored.atom_down.requires_grad
    assert restored.atom_up.requires_grad

    x = torch.randn(2, 4, 5)
    coefficients = torch.randn(2, layer.atom_count)
    with layer.use_coefficients(coefficients):
        expected_output = layer(x)
    with restored.use_coefficients(coefficients):
        actual_output = restored(x)
    torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)


def test_gradients_reach_input_code_and_both_atom_factors_but_not_base() -> None:
    layer = _layer()
    x = torch.randn(2, 4, 5, requires_grad=True)
    coefficients = torch.randn(2, layer.atom_count, requires_grad=True)

    with layer.use_coefficients(coefficients):
        layer(x).square().mean().backward()

    for gradient in (
        x.grad,
        coefficients.grad,
        layer.atom_down.grad,
        layer.atom_up.grad,
    ):
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0
    assert all(parameter.grad is None for parameter in layer.base.parameters())


def test_autocast_uses_atom_output_dtype_without_promoting_and_preserves_leaf_gradient() -> None:
    layer = _layer()
    x = torch.randn(2, 4, 5, dtype=torch.float32, requires_grad=True)
    coefficients = torch.randn(
        2, layer.atom_count, dtype=torch.float32, requires_grad=True
    )

    with layer.use_coefficients(coefficients):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = layer(x)
        assert output.dtype == torch.bfloat16
        output.float().square().mean().backward()

    assert coefficients.grad is not None
    assert coefficients.grad.dtype == torch.float32
    assert torch.count_nonzero(coefficients.grad) > 0
    assert x.grad is not None and x.grad.dtype == torch.float32


def _run_with_gradients(
    layer: DynamicAtomLinear,
    x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    checkpointed: bool,
) -> tuple[torch.Tensor, ...]:
    with layer.use_coefficients(coefficients):
        output = (
            checkpoint(layer, x, use_reentrant=False)
            if checkpointed
            else layer(x)
        )
        output.square().mean().backward()
    assert x.grad is not None
    assert coefficients.grad is not None
    assert layer.atom_down.grad is not None
    assert layer.atom_up.grad is not None
    return (
        output.detach(),
        x.grad.detach().clone(),
        coefficients.grad.detach().clone(),
        layer.atom_down.grad.detach().clone(),
        layer.atom_up.grad.detach().clone(),
    )


def test_non_reentrant_checkpoint_matches_output_and_gradients_and_reuses_coefficients() -> None:
    eager_layer = _layer()
    checkpoint_layer = _layer()
    checkpoint_layer.load_state_dict(eager_layer.state_dict(), strict=True)
    eager_x = torch.randn(2, 4, 5, requires_grad=True)
    checkpoint_x = eager_x.detach().clone().requires_grad_(True)
    eager_coefficients = torch.randn(2, 3, requires_grad=True)
    checkpoint_coefficients = eager_coefficients.detach().clone().requires_grad_(True)
    observed_coefficients: list[torch.Tensor | None] = []
    hook = checkpoint_layer.register_forward_pre_hook(
        lambda module, _inputs: observed_coefficients.append(module._coefficients)
    )
    try:
        eager_results = _run_with_gradients(
            eager_layer, eager_x, eager_coefficients, checkpointed=False
        )
        checkpoint_results = _run_with_gradients(
            checkpoint_layer,
            checkpoint_x,
            checkpoint_coefficients,
            checkpointed=True,
        )
    finally:
        hook.remove()

    assert len(observed_coefficients) >= 2
    assert all(value is checkpoint_coefficients for value in observed_coefficients)
    for eager, recomputed in zip(eager_results, checkpoint_results, strict=True):
        torch.testing.assert_close(recomputed, eager, rtol=1e-6, atol=1e-7)


def test_checkpoint_backward_after_context_exit_fails_closed_and_layer_is_reusable() -> None:
    layer = _layer()
    x = torch.randn(2, 4, 5, requires_grad=True)
    coefficients = torch.randn(2, layer.atom_count, requires_grad=True)

    with layer.use_coefficients(coefficients):
        output = checkpoint(layer, x, use_reentrant=False)

    with pytest.raises(
        RuntimeError, match="coefficient context must remain active through backward"
    ):
        output.square().mean().backward()
    assert coefficients.grad is None
    assert layer._coefficients is None

    replacement_x = torch.randn(2, 4, 5, requires_grad=True)
    replacement_coefficients = torch.randn(
        2, layer.atom_count, requires_grad=True
    )
    with layer.use_coefficients(replacement_coefficients):
        checkpoint(layer, replacement_x, use_reentrant=False).sum().backward()
    assert replacement_coefficients.grad is not None


def test_dynamic_path_never_passes_a_dense_delta_weight_to_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.nn import functional as functional

    layer = _layer()
    x = torch.randn(2, 4, 5)
    observed_shapes: list[tuple[int, ...]] = []
    original_linear = functional.linear

    def traced_linear(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observed_shapes.append(tuple(weight.shape))
        return original_linear(input_, weight, bias)

    monkeypatch.setattr(functional, "linear", traced_linear)
    with layer.use_coefficients(torch.ones(2, layer.atom_count)):
        layer(x)

    assert observed_shapes.count((7, 5)) == 1
    assert observed_shapes.count((2, 5)) == layer.atom_count
    assert observed_shapes.count((7, 2)) == layer.atom_count
    assert len(observed_shapes) == 1 + 2 * layer.atom_count
