from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear

WIDTH = 2240
DENSE_WEIGHT_BYTES = WIDTH * WIDTH * torch.bfloat16.itemsize
MINIMUM_GAP_BYTES = DENSE_WEIGHT_BYTES // 2


def _peak_bytes(callable_: Callable[[], torch.Tensor]) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = callable_()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    return peak


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA contract runs only in the Modal pilot"
)
def test_cpu_coefficient_leaf_casts_to_cuda_and_receives_its_gradient() -> None:
    layer = DynamicAtomLinear(
        nn.Linear(5, 7, device="cuda", dtype=torch.bfloat16),
        rank=2,
        atom_count=3,
    )
    x = torch.randn(2, 4, 5, device="cuda", dtype=torch.bfloat16)
    coefficients = torch.randn(2, 3, device="cpu", requires_grad=True)

    with layer.use_coefficients(coefficients):
        output = layer(x)
        assert output.device.type == "cuda"
        assert output.dtype == torch.bfloat16
        output.float().square().mean().backward()

    assert coefficients.grad is not None
    assert coefficients.grad.device.type == "cpu"
    assert torch.count_nonzero(coefficients.grad) > 0


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA contract runs only in the Modal pilot"
)
def test_dynamic_path_has_a_repeatable_gap_below_explicit_dense_delta() -> None:
    base = nn.Linear(
        WIDTH, WIDTH, bias=False, device="cuda", dtype=torch.bfloat16
    )
    layer = DynamicAtomLinear(base, rank=4, atom_count=4)
    x = torch.randn(1, 128, WIDTH, device="cuda", dtype=torch.bfloat16)
    coefficients = torch.randn(4, device="cuda", dtype=torch.bfloat16)

    def dynamic() -> torch.Tensor:
        with layer.use_coefficients(coefficients):
            return layer(x)

    def explicit() -> torch.Tensor:
        delta = torch.einsum(
            "a,aor,ari->oi", coefficients, layer.atom_up, layer.atom_down
        )
        return F.linear(x, layer.base.weight + delta)

    dynamic()
    explicit()
    torch.cuda.synchronize()

    dynamic_peaks = [_peak_bytes(dynamic)]
    explicit_peaks = [_peak_bytes(explicit)]
    explicit_peaks.append(_peak_bytes(explicit))
    dynamic_peaks.append(_peak_bytes(dynamic))

    assert max(dynamic_peaks) + MINIMUM_GAP_BYTES <= min(explicit_peaks)
