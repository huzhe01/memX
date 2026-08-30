from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.baselines.protocol import CausalEventView, FrozenComparisonContract
from ratemem.evaluation.traces import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.method.adapter import RateMemAdapter
from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import RateMemHardCodec
from ratemem.method.controller import RateMemController
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary
from ratemem.method.proposal import ImmutableBundleProposal


class _Predictor:
    def predict(
        self,
        support_image_ids: Sequence[str],
        description_id: str | None,
    ) -> NDArray[np.float32]:
        identity = "\0".join((*support_image_ids, description_id or ""))
        offset = int(hashlib.sha256(identity.encode()).hexdigest()[:4], 16) / 65535
        return np.array([0.9 + offset * 0.01, -0.5, 0.3, -1.0], dtype=np.float32)


class _Generator:
    def generate(
        self,
        adapter_code: NDArray[np.float32],
        prompt_id: str,
        seed: int,
    ) -> bytes:
        return hashlib.sha256(
            adapter_code.astype("<f4").tobytes()
            + prompt_id.encode()
            + seed.to_bytes(8, "little")
        ).digest()


def _oracle(
    cohort: Sequence[str],
    bundles: Sequence[ImmutableBundleProposal],
) -> CoverageOracle:
    return CoverageOracle(
        bundles={
            bundle.packet.packet_id: PacketBundle(
                bundle.packet.packet_id,
                bundle.cost_bytes,
                {edge.handle: (1.0,) for edge in bundle.incidences},
            )
            for bundle in bundles
        },
        request_weights={handle: 1.0 for handle in cohort},
        group_weights={handle: (1.0,) for handle in cohort},
    )


def adapter_factory() -> RateMemAdapter:
    dictionary = GroupRVQDictionary(1, 4, 1, 2)
    with torch.no_grad():
        dictionary.codebooks.copy_(
            torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
        )
        dictionary.normalize_codebooks_()
    codec = RateMemHardCodec(
        BlockwiseBaseQuantizer(4, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=1,
    )
    return RateMemAdapter(
        _Predictor(),
        _Generator(),
        codec,
        lambda budget: RateMemController(budget, _oracle),
        shared_trained_bytes=128,
    )


def comparison_contract() -> FrozenComparisonContract:
    return FrozenComparisonContract(
        trace_id="1" * 64,
        dataset_lock_sha256="2" * 64,
        evaluation_lock_sha256="3" * 64,
        baseline_requirements_sha256="4" * 64,
        backbone_id="sana_1_5_1_6b",
        backbone_revision="5" * 40,
        adapter_layout_sha256="6" * 64,
        amortizer_sha256="7" * 64,
        adapter_basis_sha256="8" * 64,
        codec_dictionary_sha256="9" * 64,
        candidate_stream_sha256="a" * 64,
        prompt_pool_sha256="b" * 64,
        support_pool_sha256="c" * 64,
        noise_seed_manifest_sha256="d" * 64,
        sampler_id="flow-euler",
        scheduler_revision="e" * 40,
        cfg_scale=4.5,
        resolution=(1024, 1024),
        denoising_steps=20,
        byte_budget=65536,
        request_regime="uniform",
        search_budget_sha256="f" * 64,
    )


def lifecycle_events() -> tuple[LifecycleEvent, ...]:
    handle = "h_111111111111_000000"
    return (
        CreateEvent(
            event_index=0,
            handle=handle,
            concept_token="<concept_000001>",
            support_image_ids=("support-a",),
            description_id="description-a",
        ),
        ReadEvent(
            event_index=1,
            handle=handle,
            prompt_id="prompt-a",
            generation_seed=7,
        ),
        UpdateEvent(
            event_index=2,
            handle=handle,
            support_image_ids=("support-b",),
        ),
        DeleteEvent(event_index=3, handle=handle),
        ReadEvent(
            event_index=4,
            handle=handle,
            prompt_id="prompt-b",
            generation_seed=8,
        ),
    )


def test_create_read_update_delete_and_stale_handle() -> None:
    adapter = adapter_factory()
    contract = comparison_contract()
    events = lifecycle_events()
    adapter.initialize(contract)
    receipts = [
        adapter.apply_event(event, CausalEventView(events, current_index=index))
        for index, event in enumerate(events)
    ]
    assert [row.outcome for row in receipts] == [
        "created",
        "read",
        "updated",
        "deleted",
        "stale_handle",
    ]
    assert all(
        row.ledger.online_state_bytes <= contract.byte_budget for row in receipts
    )
    assert all(row.method_id == "ratemem_v1" for row in receipts)
    assert all(row.trace_id == contract.trace_id for row in receipts)
    assert all(
        row.candidate_stream_sha256 == contract.candidate_stream_sha256
        for row in receipts
    )
