from __future__ import annotations

import torch

from ratemem.training.losses import (
    dictionary_balance_loss,
    expected_rate_loss,
    reuse_affinity_loss,
)


def test_reuse_affinity_rewards_matching_assignments_for_similar_residuals() -> None:
    probabilities = torch.tensor(
        [
            [[[[0.9, 0.1]]]],
            [[[[0.9, 0.1]]]],
            [[[[0.1, 0.9]]]],
        ]
    )
    residuals = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    )
    aligned = reuse_affinity_loss(
        probabilities,
        residuals,
        similarity_center=0.8,
        similarity_width=0.1,
    )
    permuted = reuse_affinity_loss(
        probabilities[[0, 2, 1]],
        residuals,
        0.8,
        0.1,
    )
    assert aligned < permuted


def test_balance_loss_penalizes_single_entry_collapse() -> None:
    uniform = torch.full((4, 2, 1, 4), 0.25)
    collapsed = torch.zeros_like(uniform)
    collapsed[:, :, :, 0] = 1.0
    assert dictionary_balance_loss(uniform) < dictionary_balance_loss(collapsed)


def test_rate_loss_charges_exactly_eight_selected_candidate_costs() -> None:
    selected = torch.zeros(1, 30, 2)
    selected.reshape(-1)[:8] = 1.0
    costs = torch.arange(1, 61, dtype=torch.float32).reshape(30, 2)
    actual = expected_rate_loss(selected, costs, budget_bytes=1000)
    torch.testing.assert_close(actual, torch.tensor(sum(range(1, 9)) / 1000))
