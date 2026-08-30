from __future__ import annotations

import pytest
import torch

from ratemem.training.functional_state import FunctionalMemoryState
from ratemem.training.segments import FrozenTrainingEvent, TrainingSegment
from tests.unit.training.test_meta_trainer import make_trainer


def _query_events(count: int) -> tuple[FrozenTrainingEvent, ...]:
    rows = [FrozenTrainingEvent(0, "create", "a")]
    rows.extend(
        FrozenTrainingEvent(
            index,
            "read",
            "a",
            prompt_id=f"p-{index}",
            has_training_query=True,
        )
        for index in range(1, count + 1)
    )
    return tuple(rows)


def test_two_queries_use_exactly_two_transformer_passes() -> None:
    trainer, resolver = make_trainer()
    segment = TrainingSegment("a" * 64, 0, _query_events(2))
    receipt = trainer.train_segment(
        segment,
        FunctionalMemoryState(),
        temperature=0.5,
        candidate_cost_bytes=torch.ones(2, 1),
        budget_bytes=2048,
    )
    assert receipt.transformer_passes == 2
    assert resolver.transformer_calls == 2
    assert resolver.full_denoising_calls == 0


def test_third_query_is_rejected_before_transformer_call() -> None:
    trainer, resolver = make_trainer()
    segment = TrainingSegment("a" * 64, 0, _query_events(3))
    with pytest.raises(RuntimeError, match="pass cap"):
        trainer.train_segment(
            segment,
            FunctionalMemoryState(),
            temperature=0.5,
            candidate_cost_bytes=torch.ones(2, 1),
            budget_bytes=2048,
        )
    assert resolver.transformer_calls == 2
    assert resolver.full_denoising_calls == 0
