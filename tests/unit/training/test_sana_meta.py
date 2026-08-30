from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from ratemem.sana.flow import FlowBatch
from ratemem.support.amortizer import SupportAmortizer
from ratemem.training.sana_meta import SanaMetaResolver


class FakeBank:
    def __init__(self) -> None:
        self.active: torch.Tensor | None = None

    @contextmanager
    def activate(self, coefficients: torch.Tensor) -> Iterator[None]:
        self.active = coefficients
        try:
            yield
        finally:
            self.active = None


class FakeTransformer(nn.Module):
    def __init__(self, bank: FakeBank) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))
        self.bank = bank

    def forward(self, *, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        del kwargs
        if self.bank.active is None:
            raise RuntimeError("adapter bank was not active")
        adjustment = self.bank.active.mean() + self.anchor
        return (hidden_states * 0.0 + adjustment,)


def test_sana_meta_resolver_binds_target_flow_and_utility_without_extra_passes() -> None:
    amortizer = SupportAmortizer(
        support_dim=4,
        description_dim=6,
        hidden_dim=8,
        projection_count=2,
        atom_count=2,
        layers=1,
        heads=2,
    )
    bank = FakeBank()
    transformer = FakeTransformer(bank)
    resolver = SanaMetaResolver(
        transformer,
        bank,
        amortizer,
        tuple(float(1000 - index) for index in range(1000)),
        tuple(float(1000 - index) / 1000 for index in range(1000)),
        seed=17,
        group_size=2,
        autocast_dtype=None,
    )
    batch = FlowBatch(
        clean_latents=torch.zeros(1, 1, 2, 2),
        prompt_embeddings=torch.zeros(1, 3, 5),
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.int64),
        support_features=torch.ones(1, 1, 4),
        support_mask=torch.ones(1, 1, dtype=torch.bool),
        description_features=torch.ones(1, 6),
    )
    trace_id = "a" * 64
    resolver.bind(trace_id, 0, batch)

    target = resolver.target_code(trace_id, 0)
    flow_loss = resolver.one_timestep_flow_loss(trace_id, 1, target)
    features, observed, mask = resolver.utility_supervision(trace_id, 1)
    resolver.backward(flow_loss)

    assert target.shape == (1, 4)
    assert flow_loss.shape == ()
    assert flow_loss.requires_grad
    assert features.concept.shape == (1, 4)
    assert features.incidence.shape == (1, 2, 4)
    assert observed.shape == mask.shape == (1, 2, 2)
    assert bank.active is None
