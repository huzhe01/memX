import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear


class _IntSubclass(int):
    pass


class _LinearSubclass(nn.Linear):
    pass


def _layer(*, bias: bool = True, dtype: torch.dtype = torch.float32) -> DynamicAtomLinear:
    torch.manual_seed(7)
    return DynamicAtomLinear(
        nn.Linear(5, 7, bias=bias, dtype=dtype), rank=2, atom_count=3
    )


def _input_shape(ndim: int) -> tuple[int, ...]:
    return {
        1: (5,),
        2: (3, 5),
        3: (3, 4, 5),
        4: (3, 2, 4, 5),
    }[ndim]


def _dense_reference(
    layer: DynamicAtomLinear, x: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    if coefficients.ndim == 1:
        delta = torch.einsum(
            "a,aor,ari->oi", coefficients, layer.atom_up, layer.atom_down
        )
        return F.linear(x, layer.base.weight + delta, layer.base.bias)

    outputs = []
    for sample, sample_coefficients in zip(x, coefficients, strict=True):
        delta = torch.einsum(
            "a,aor,ari->oi",
            sample_coefficients,
            layer.atom_up,
            layer.atom_down,
        )
        outputs.append(F.linear(sample, layer.base.weight + delta, layer.base.bias))
    return torch.stack(outputs)


def test_constructor_rejects_a_non_linear_base() -> None:
    with pytest.raises(TypeError, match="base must be an exact nn.Linear"):
        DynamicAtomLinear(nn.Identity(), rank=2, atom_count=3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "base",
    [
        _LinearSubclass(5, 7),
        nn.LazyLinear(7),
    ],
    ids=["linear-subclass", "lazy-linear"],
)
def test_constructor_rejects_linear_subclasses(base: nn.Module) -> None:
    with pytest.raises(TypeError, match="base must be an exact nn.Linear"):
        DynamicAtomLinear(base, rank=2, atom_count=3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("rank", True, TypeError, "rank must be an int"),
        ("rank", 2.0, TypeError, "rank must be an int"),
        ("rank", _IntSubclass(2), TypeError, "rank must be an int"),
        ("rank", 0, ValueError, "rank must be positive"),
        ("rank", -1, ValueError, "rank must be positive"),
        ("atom_count", False, TypeError, "atom_count must be an int"),
        ("atom_count", "3", TypeError, "atom_count must be an int"),
        (
            "atom_count",
            _IntSubclass(3),
            TypeError,
            "atom_count must be an int",
        ),
        ("atom_count", 0, ValueError, "atom_count must be positive"),
        ("atom_count", -1, ValueError, "atom_count must be positive"),
    ],
)
def test_constructor_strictly_validates_rank_and_atom_count(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {"rank": 2, "atom_count": 3}
    arguments[field] = value
    with pytest.raises(error, match=message):
        DynamicAtomLinear(nn.Linear(5, 7), **arguments)  # type: ignore[arg-type]


def test_constructor_preserves_and_freezes_base_and_inherits_weight_placement() -> None:
    base = nn.Linear(5, 7, bias=True, dtype=torch.float64)
    weight_before = base.weight.detach().clone()
    bias_before = base.bias.detach().clone() if base.bias is not None else None

    layer = DynamicAtomLinear(base, rank=2, atom_count=3)

    assert layer.base is base
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert torch.equal(base.weight, weight_before)
    assert base.bias is not None and bias_before is not None
    assert torch.equal(base.bias, bias_before)
    assert layer.atom_down.device == base.weight.device
    assert layer.atom_up.device == base.weight.device
    assert layer.atom_down.dtype == base.weight.dtype
    assert layer.atom_up.dtype == base.weight.dtype


@pytest.mark.parametrize("input_ndim", [2, 3, 4])
@pytest.mark.parametrize("batched_coefficients", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_global_and_batched_coefficients_match_explicit_dense_weights(
    input_ndim: int, batched_coefficients: bool, bias: bool
) -> None:
    layer = _layer(bias=bias)
    x = torch.randn(_input_shape(input_ndim))
    coefficients = (
        torch.randn(3, layer.atom_count)
        if batched_coefficients
        else torch.randn(layer.atom_count)
    )
    expected = _dense_reference(layer, x, coefficients)

    with layer.use_coefficients(coefficients):
        actual = layer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("input_ndim", [1, 4])
def test_global_coefficients_preserve_linear_leading_dimension_semantics(
    input_ndim: int,
) -> None:
    layer = _layer()
    x = torch.randn(_input_shape(input_ndim))
    coefficients = torch.randn(layer.atom_count)
    expected = _dense_reference(layer, x, coefficients)

    with layer.use_coefficients(coefficients):
        actual = layer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("input_ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_without_coefficients_exactly_preserves_linear_semantics(
    input_ndim: int, bias: bool
) -> None:
    layer = _layer(bias=bias)
    x = torch.randn(_input_shape(input_ndim))
    assert torch.equal(layer(x), layer.base(x))


@pytest.mark.parametrize("input_ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_zero_coefficients_exactly_match_the_frozen_base(
    input_ndim: int, bias: bool
) -> None:
    layer = _layer(bias=bias)
    x = torch.randn(_input_shape(input_ndim))
    expected = layer.base(x)

    with layer.use_coefficients(torch.zeros(layer.atom_count)):
        actual = layer(x)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("coefficients", "message"),
    [
        (torch.tensor(1.0), "coefficients must be 1D or 2D"),
        (torch.ones(1, 1, 3), "coefficients must be 1D or 2D"),
        (torch.ones(2), "coefficient atom dimension must be 3"),
        (torch.ones(2, 2), "coefficient atom dimension must be 3"),
    ],
)
def test_coefficient_shape_is_validated_before_activation(
    coefficients: torch.Tensor, message: str
) -> None:
    layer = _layer()
    with pytest.raises(ValueError, match=message):
        with layer.use_coefficients(coefficients):
            raise AssertionError("invalid coefficients became active")
    assert layer._coefficients is None


def test_non_tensor_coefficients_are_rejected_before_activation() -> None:
    layer = _layer()
    with pytest.raises(TypeError, match="coefficients must be a Tensor"):
        with layer.use_coefficients([1.0, 2.0, 3.0]):  # type: ignore[arg-type]
            raise AssertionError("invalid coefficients became active")
    assert layer._coefficients is None


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((), "input must have at least one dimension"),
        ((2, 4, 6), "input feature dimension must be 5"),
    ],
)
def test_input_shape_is_rejected_before_the_base_runs(
    shape: tuple[int, ...], message: str
) -> None:
    layer = _layer()
    base_calls: list[bool] = []
    hook = layer.base.register_forward_pre_hook(
        lambda _module, _inputs: base_calls.append(True)
    )
    try:
        with layer.use_coefficients(torch.ones(layer.atom_count)):
            with pytest.raises(ValueError, match=message):
                layer(torch.randn(shape))
    finally:
        hook.remove()
    assert base_calls == []


def test_batched_coefficients_require_a_batch_dimension_before_the_base_runs() -> None:
    layer = _layer()
    base_calls: list[bool] = []
    hook = layer.base.register_forward_pre_hook(
        lambda _module, _inputs: base_calls.append(True)
    )
    try:
        with layer.use_coefficients(torch.ones(1, layer.atom_count)):
            with pytest.raises(
                ValueError,
                match="batched coefficients require input with a batch dimension",
            ):
                layer(torch.randn(5))
    finally:
        hook.remove()
    assert base_calls == []


def test_coefficient_batch_mismatch_is_rejected_before_the_base_runs() -> None:
    layer = _layer()
    base_calls: list[bool] = []
    hook = layer.base.register_forward_pre_hook(
        lambda _module, _inputs: base_calls.append(True)
    )
    try:
        with layer.use_coefficients(torch.ones(3, layer.atom_count)):
            with pytest.raises(
                ValueError, match="coefficient batch 3 does not match input batch 2"
            ):
                layer(torch.randn(2, 4, 5))
    finally:
        hook.remove()
    assert base_calls == []


def test_nested_activation_is_rejected_without_disturbing_the_outer_context() -> None:
    layer = _layer()
    x = torch.randn(2, 5)
    outer = torch.tensor([0.25, -0.5, 0.75])
    inner = torch.tensor([-1.0, 0.0, 1.0])

    with layer.use_coefficients(outer):
        expected = layer(x)
        with pytest.raises(RuntimeError, match="coefficients are already active"):
            with layer.use_coefficients(inner):
                raise AssertionError("nested coefficients became active")
        assert layer._coefficients is outer
        actual = layer(x)
        torch.testing.assert_close(actual, expected)

    assert layer._coefficients is None


def test_nested_activation_error_takes_priority_over_inner_shape_validation() -> None:
    layer = _layer()
    outer = torch.ones(layer.atom_count)
    with layer.use_coefficients(outer):
        with pytest.raises(RuntimeError, match="coefficients are already active"):
            with layer.use_coefficients(torch.ones(1)):
                raise AssertionError("nested coefficients became active")
        assert layer._coefficients is outer
    assert layer._coefficients is None


def test_exception_cleanup_leaves_the_layer_reusable() -> None:
    layer = _layer()
    first = torch.ones(layer.atom_count)
    second = torch.zeros(layer.atom_count)

    with pytest.raises(RuntimeError, match="contract sentinel"):
        with layer.use_coefficients(first):
            raise RuntimeError("contract sentinel")
    assert layer._coefficients is None

    with layer.use_coefficients(second):
        assert layer._coefficients is second
        layer(torch.randn(2, 5))
    assert layer._coefficients is None
