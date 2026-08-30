from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ratemem.baselines.feature_cache import CachedFeature, FeatureCacheAdapter
from ratemem.baselines.lora_reference import (
    LoRAFitResult,
    LoRAOptimizationAdapter,
    LoRAReferenceConfig,
    LoRAState,
    SelectedLoRAHyperparameters,
    load_lora_reference_config,
)
from ratemem.baselines.protocol import FrozenComparisonContract
from ratemem.baselines.replay import replay_paired, write_paired_replay
from ratemem.evaluation.traces import CreateEvent, ProbeEvent, ReadEvent, Trace


class FixtureFeatureBackend:
    backbone_id = "sana_1_5_1_6b"
    source_revision = "4" * 40
    shared_trained_bytes = 0

    def encode_support(
        self,
        support_image_ids: Sequence[str],
        description_id: str,
    ) -> CachedFeature:
        value = float(len(support_image_ids) + len(description_id))
        return CachedFeature(
            tensor=np.full((2, 2), value, dtype=np.float32),
            tap_path="transformer.block.0",
            injection_path="transformer.block.1",
            encoding_timestep=1,
            scale=1.0,
        )

    def generate(self, feature: CachedFeature, prompt_id: str, seed: int) -> Tensor:
        return torch.tensor([float(feature.tensor.sum()) + len(prompt_id) + seed])

    def one_step_latent(
        self,
        feature: CachedFeature,
        prompt_id: str,
        seed: int,
        timestep: int,
    ) -> Tensor:
        return self.generate(feature, prompt_id, seed) + timestep


class FixtureLoRATrainer:
    backbone_id = "sana_1_5_1_6b"
    backbone_revision = "4" * 40

    def fit(
        self,
        support_image_ids: Sequence[str],
        description_id: str,
        initial_lora: LoRAState | None,
        config: LoRAReferenceConfig,
        hyperparameters: SelectedLoRAHyperparameters,
        seed: int,
    ) -> LoRAFitResult:
        del support_image_ids, description_id, initial_lora, config, seed
        tensors = {
            "transformer.blocks.0.attn1.to_q.lora_A": torch.ones(
                (hyperparameters.rank, 2)
            ),
            "transformer.blocks.0.attn1.to_q.lora_B": torch.ones(
                (2, hyperparameters.rank)
            ),
        }
        state = LoRAState.from_tensors(tensors)
        frozen = hashlib.sha256(b"frozen").hexdigest()
        return LoRAFitResult(
            state=state,
            frozen_parameter_sha256_before=frozen,
            frozen_parameter_sha256_after=frozen,
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
        del contract
        return torch.tensor([int(state.sha256[:8], 16) % 1000 + len(prompt_id) + seed])


def _trace() -> Trace:
    handle = "h_aaaaaaaaaaaa_000000"
    events = (
        CreateEvent(
            event_index=0,
            handle=handle,
            concept_token="<concept_000000>",
            support_image_ids=("support-0",),
            description_id="description-0",
        ),
        ProbeEvent(
            event_index=1,
            snapshot_event_index=0,
            handle=handle,
            prompt_id="probe-0",
            generation_seed=19,
        ),
        ReadEvent(
            event_index=2,
            handle=handle,
            prompt_id="prompt-0",
            generation_seed=17,
        ),
    )
    return Trace(
        schema_version="1.0",
        trace_id="6" * 64,
        split="train",
        dataset_lock_id="1" * 64,
        trace_builder_revision="lifecycle_trace_v1",
        trace_seed=1,
        seed_namespace="fixture",
        request_regime="uniform",
        protocol="no_pressure",
        concept_pool_sha256="2" * 64,
        prompt_pool_sha256="7" * 64,
        concept_ids=("<concept_000000>",),
        generation_seeds=(17, 19),
        events=events,
    )


def _contract(method_candidate_sha: str) -> FrozenComparisonContract:
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
        candidate_stream_sha256=method_candidate_sha,
        prompt_pool_sha256="7" * 64,
        support_pool_sha256="5" * 64,
        noise_seed_manifest_sha256="8" * 64,
        sampler_id="flow-dpm",
        scheduler_revision="scheduler-v1",
        cfg_scale=4.5,
        resolution=(1024, 1024),
        denoising_steps=20,
        byte_budget=100_000,
        request_regime="uniform",
        search_budget_sha256="9" * 64,
    )


def test_paired_replay_handles_read_only_probe_gaps_and_exact_ledgers(
    tmp_path: Path,
) -> None:
    lora = LoRAOptimizationAdapter(
        load_lora_reference_config(Path("configs/baselines/lora-reference.yaml")),
        FixtureLoRATrainer(),
        SelectedLoRAHyperparameters(rank=2, learning_rate=0.00001, steps=50),
    )
    adapters = {
        "dreamcache_feature_cache": FeatureCacheAdapter(
            "dreamcache_feature_cache",
            FixtureFeatureBackend(),
        ),
        "per_concept_lora": lora,
    }
    contracts = {
        "dreamcache_feature_cache": _contract("a" * 64),
        "per_concept_lora": _contract("b" * 64),
    }
    result = replay_paired(_trace(), adapters, contracts)
    assert tuple(result.artifacts) == (
        "dreamcache_feature_cache",
        "per_concept_lora",
    )
    for artifact in result.artifacts.values():
        assert [row.event_index for row in artifact.receipts] == [0, 2]
        assert [row.probe_event_index for row in artifact.probes] == [1]
        assert all(
            row.ledger.online_state_bytes <= artifact.byte_budget
            for row in artifact.receipts
        )
        assert all(row.access_mode == "causal_prefix" for row in artifact.access_audit)
    manifest = write_paired_replay(tmp_path / "replay", result)
    assert manifest.is_file()
    assert (manifest.parent / "per_concept_lora/events.jsonl").is_file()
