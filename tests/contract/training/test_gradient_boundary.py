from __future__ import annotations

import torch

from ratemem.training.functional_state import FunctionalMemoryState
from tests.unit.training.test_meta_trainer import make_trainer, two_query_segment


def test_codec_and_utility_receive_gradients_and_state_is_detached() -> None:
    trainer, _resolver = make_trainer()
    receipt = trainer.train_segment(
        two_query_segment(),
        FunctionalMemoryState(),
        temperature=0.5,
        candidate_cost_bytes=torch.ones(2, 1),
        budget_bytes=2048,
    )
    assert any(
        parameter.grad is not None for parameter in trainer.codec.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in trainer.utility.parameters()
    )
    assert all(
        not code.requires_grad and code.grad_fn is None
        for code in receipt.detached_state.codes.values()
    )
