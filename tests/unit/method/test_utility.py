from __future__ import annotations

import pytest
import torch

from ratemem.method.utility import (
    CausalFeatureBatch,
    CausalRequestHistory,
    NonnegativeUtilityCalibrator,
)


def test_request_weight_uses_only_operational_reads_before_allocation() -> None:
    history = CausalRequestHistory(decay=0.97)
    history = history.observe_read("a", event_index=2, operational=True)
    unchanged = history.observe_read("a", event_index=3, operational=False)
    assert unchanged is history
    assert history.weight("a", allocation_event_index=4) == pytest.approx(0.97)
    with pytest.raises(ValueError, match="future"):
        history.weight("a", allocation_event_index=2)


def test_beta_and_packet_gain_outputs_are_finite_and_nonnegative() -> None:
    model = NonnegativeUtilityCalibrator(
        concept_features=4,
        incidence_features=5,
        hidden=8,
        groups=3,
    )
    batch = CausalFeatureBatch(
        concept=torch.randn(2, 4),
        incidence=torch.randn(2, 4, 5),
        incidence_mask=torch.tensor(
            [[True, False, True, False], [False, True, True, False]]
        ),
        maximum_source_event_index=torch.tensor([5, 7]),
        allocation_event_index=torch.tensor([5, 8]),
    )
    result = model(batch)
    assert torch.isfinite(result.beta).all() and torch.all(result.beta >= 0)
    assert torch.isfinite(result.value).all() and torch.all(result.value >= 0)
    assert torch.all(result.value[~batch.incidence_mask] == 0)


def test_future_features_are_rejected() -> None:
    model = NonnegativeUtilityCalibrator(2, 2, 4, 1)
    batch = CausalFeatureBatch(
        concept=torch.zeros(1, 2),
        incidence=torch.zeros(1, 1, 2),
        incidence_mask=torch.ones(1, 1, dtype=torch.bool),
        maximum_source_event_index=torch.tensor([4]),
        allocation_event_index=torch.tensor([3]),
    )
    with pytest.raises(ValueError, match="future"):
        model(batch)
