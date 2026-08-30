from __future__ import annotations

import torch

from ratemem.method.codec import RateMemDifferentiableCodec
from ratemem.method.dictionary import GroupRVQDictionary
from ratemem.method.utility import CausalFeatureBatch, NonnegativeUtilityCalibrator
from ratemem.training.functional_state import FunctionalMemoryState
from ratemem.training.losses import LossWeights
from ratemem.training.meta_trainer import SequentialMetaTrainer
from ratemem.training.segments import FrozenTrainingEvent, TrainingSegment


class FakeResolver:
    def __init__(self, width: int) -> None:
        self.width = width
        self.transformer_calls = 0
        self.full_denoising_calls = 0

    def target_code(self, trace_id: str, event_index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(event_index + 11)
        return torch.randn(1, self.width, generator=generator)

    def one_timestep_flow_loss(
        self,
        trace_id: str,
        event_index: int,
        adapter_code: torch.Tensor,
    ) -> torch.Tensor:
        self.transformer_calls += 1
        return adapter_code.float().square().mean()

    def utility_supervision(
        self,
        trace_id: str,
        event_index: int,
    ) -> tuple[CausalFeatureBatch, torch.Tensor, torch.Tensor]:
        features = CausalFeatureBatch(
            concept=torch.ones(1, 3),
            incidence=torch.ones(1, 1, 4),
            incidence_mask=torch.ones(1, 1, dtype=torch.bool),
            maximum_source_event_index=torch.tensor([event_index]),
            allocation_event_index=torch.tensor([event_index]),
        )
        return features, torch.ones(1, 1, 2), torch.ones(1, 1, 2, dtype=torch.bool)


def make_trainer() -> tuple[SequentialMetaTrainer, FakeResolver]:
    dictionary = GroupRVQDictionary(2, 4, 1, 4)
    codec = RateMemDifferentiableCodec(
        dictionary,
        group_size=4,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=2,
    )
    utility = NonnegativeUtilityCalibrator(3, 4, 6, 2)
    resolver = FakeResolver(width=8)
    optimizer = torch.optim.AdamW(
        [*codec.parameters(), *utility.parameters()],
        lr=1e-3,
    )
    weights = LossWeights(1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1)
    return SequentialMetaTrainer(codec, utility, optimizer, resolver, weights), resolver


def two_query_segment() -> TrainingSegment:
    return TrainingSegment(
        "a" * 64,
        0,
        (
            FrozenTrainingEvent(0, "create", "a"),
            FrozenTrainingEvent(
                1,
                "update",
                "a",
                support_image_ids=("s",),
                has_training_query=True,
            ),
        ),
    )


def test_meta_training_updates_parameters_and_detaches_boundary() -> None:
    trainer, resolver = make_trainer()
    before = trainer.codec.dictionary.codebooks.detach().clone()
    receipt = trainer.train_segment(
        two_query_segment(),
        FunctionalMemoryState(),
        temperature=0.5,
        candidate_cost_bytes=torch.ones(2, 1),
        budget_bytes=2048,
    )
    assert resolver.transformer_calls == 1
    assert receipt.transformer_passes == 1
    assert receipt.detached_state.codes
    assert all(
        not code.requires_grad and code.grad_fn is None
        for code in receipt.detached_state.codes.values()
    )
    assert not torch.equal(before, trainer.codec.dictionary.codebooks.detach())
