from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ratemem.baselines.lora_reference import (
    LoRAFitResult,
    LoRAOptimizationAdapter,
    LoRAReferenceConfig,
    LoRAState,
    SelectedLoRAHyperparameters,
    load_lora_reference_config,
)
from ratemem.baselines.protocol import CausalEventView, FrozenComparisonContract
from ratemem.evaluation.traces import CreateEvent, ReadEvent, UpdateEvent


@dataclass(frozen=True)
class FitCall:
    support_ids: tuple[str, ...]
    description_id: str
    initial_sha256: str | None
    seed: int


class RecordingTrainer:
    backbone_id = "sana_1_5_1_6b"
    backbone_revision = "4" * 40

    def __init__(self) -> None:
        self.calls: list[FitCall] = []

    def fit(
        self,
        support_image_ids: Sequence[str],
        description_id: str,
        initial_lora: LoRAState | None,
        config: LoRAReferenceConfig,
        hyperparameters: SelectedLoRAHyperparameters,
        seed: int,
    ) -> LoRAFitResult:
        self.calls.append(
            FitCall(
                tuple(support_image_ids),
                description_id,
                None if initial_lora is None else initial_lora.sha256,
                seed,
            )
        )
        offset = len(self.calls) + (0 if initial_lora is None else 10)
        tensors = {
            "transformer.blocks.0.attn1.to_q.lora_A": torch.full(
                (hyperparameters.rank, 4), float(offset)
            ),
            "transformer.blocks.0.attn1.to_q.lora_B": torch.full(
                (4, hyperparameters.rank), float(offset + 1)
            ),
        }
        state = LoRAState.from_tensors(tensors)
        frozen_sha = hashlib.sha256(b"frozen-sana").hexdigest()
        return LoRAFitResult(
            state=state,
            frozen_parameter_sha256_before=frozen_sha,
            frozen_parameter_sha256_after=frozen_sha,
            changed_parameter_names=tuple(sorted(tensors)),
            optimizer_state_discarded=True,
        )

    def generate(
        self,
        state: LoRAState,
        prompt_id: str,
        seed: int,
        contract: FrozenComparisonContract,
    ) -> Tensor:
        value = int(state.sha256[:8], 16) ^ seed ^ len(prompt_id)
        return torch.tensor([value % 997], dtype=torch.float32)


def _contract(*, budget: int = 100_000) -> FrozenComparisonContract:
    return FrozenComparisonContract(
        trace_id="6" * 64,
        dataset_lock_sha256="1" * 64,
        evaluation_lock_sha256="2" * 64,
        baseline_requirements_sha256="3" * 64,
        backbone_id="sana_1_5_1_6b",
        backbone_revision="4" * 40,
        adapter_layout_sha256="5" * 64,
        amortizer_sha256="3" * 64,
        adapter_basis_sha256="4" * 64,
        codec_dictionary_sha256="d" * 64,
        candidate_stream_sha256="a" * 64,
        prompt_pool_sha256="7" * 64,
        support_pool_sha256="5" * 64,
        noise_seed_manifest_sha256="8" * 64,
        sampler_id="flow-dpm",
        scheduler_revision="scheduler-v1",
        cfg_scale=4.5,
        resolution=(1024, 1024),
        denoising_steps=20,
        byte_budget=budget,
        request_regime="uniform",
        search_budget_sha256="9" * 64,
    )


def _adapter(trainer: RecordingTrainer) -> LoRAOptimizationAdapter:
    config = load_lora_reference_config(Path("configs/baselines/lora-reference.yaml"))
    return LoRAOptimizationAdapter(
        config,
        trainer,
        SelectedLoRAHyperparameters(rank=2, learning_rate=0.00001, steps=50),
    )


def _events() -> tuple[CreateEvent, UpdateEvent, ReadEvent]:
    return (
        CreateEvent(
            event_index=0,
            handle="h0",
            concept_token="<concept_000000>",
            support_image_ids=("support-0",),
            description_id="description-0",
        ),
        UpdateEvent(event_index=1, handle="h0", support_image_ids=("support-new",)),
        ReadEvent(
            event_index=2,
            handle="h0",
            prompt_id="prompt-0",
            generation_seed=17,
        ),
    )


def test_create_update_and_restore_preserve_the_matched_lora_contract() -> None:
    trainer = RecordingTrainer()
    adapter = _adapter(trainer)
    adapter.initialize(_contract())
    events = _events()
    create = adapter.apply_event(events[0], CausalEventView(events, 0))
    first_sha = adapter.inspect_state().records["h0"].state.sha256
    update = adapter.apply_event(events[1], CausalEventView(events, 1))
    assert create.outcome == "created"
    assert update.outcome == "updated"
    assert trainer.calls[0].initial_sha256 is None
    assert trainer.calls[1].initial_sha256 == first_sha
    assert trainer.calls[1].support_ids == ("support-new",)
    assert adapter.inspect_state().optimizer_state_present is False
    assert all(
        tensor.name.endswith(("to_q.lora_A", "to_q.lora_B"))
        for tensor in adapter.inspect_state().records["h0"].state.tensors
    )

    payload = adapter.export_online_state()
    restored = _adapter(RecordingTrainer())
    restored.initialize(_contract())
    restored.import_online_state(payload)
    assert restored.export_online_state() == payload
    assert restored.apply_event(events[2], CausalEventView(events, 2)) == adapter.apply_event(
        events[2], CausalEventView(events, 2)
    )


def test_config_freezes_exactly_twenty_four_validation_cells() -> None:
    config = load_lora_reference_config(Path("configs/baselines/lora-reference.yaml"))
    assert (
        len(config.search_space.rank)
        * len(config.search_space.learning_rate)
        * len(config.search_space.steps)
        * len(config.search_space.prior_preservation_weight)
        == 24
    )
