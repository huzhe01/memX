from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from ratemem.adapters.sana_layout import (
    PRODUCTION_ATOM_COUNT,
    PRODUCTION_BLOCK_COUNT,
    PRODUCTION_RANK,
    PRODUCTION_WIDTH,
    validate_production_sana_layout,
)


class _ProductionAttention(nn.Module):
    def __init__(self, *, bias: bool) -> None:
        super().__init__()
        options = {"device": "cuda", "dtype": torch.bfloat16}
        self.to_q = nn.Linear(PRODUCTION_WIDTH, PRODUCTION_WIDTH, bias=bias, **options)
        self.to_k = nn.Linear(PRODUCTION_WIDTH, PRODUCTION_WIDTH, bias=bias, **options)
        self.to_v = nn.Linear(PRODUCTION_WIDTH, PRODUCTION_WIDTH, bias=bias, **options)


class _ProductionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _ProductionAttention(bias=False)
        self.attn2 = _ProductionAttention(bias=True)


class _ProductionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_ProductionBlock() for _ in range(PRODUCTION_BLOCK_COUNT)]
        )


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("RATEMEM_RUN_PRODUCTION_SANA_LAYOUT") != "1",
    reason="explicit CUDA BF16 production-layout opt-in is required",
)
def test_randomized_production_layout_is_exactly_cuda_bfloat16() -> None:
    transformer = _ProductionTransformer().requires_grad_(False).eval()

    layout = validate_production_sana_layout(
        transformer,
        rank=PRODUCTION_RANK,
        atom_count=PRODUCTION_ATOM_COUNT,
    )

    assert layout.projection_count == 120
    assert layout.code_dim == 480
    assert layout.atom_tensor_count == 240
    assert layout.trainable_parameter_count(
        width=PRODUCTION_WIDTH, rank=PRODUCTION_RANK
    ) == 8_601_600
