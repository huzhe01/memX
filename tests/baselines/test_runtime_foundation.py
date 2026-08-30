from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from ratemem.baselines.independent import IndependentCodeCacheAdapter
from ratemem.baselines.ledger import (
    decode_state,
    empty_components,
    export_state,
    ledger_from_export,
)
from ratemem.baselines.online_share import OnlineShareAdapter, update_subspace
from ratemem.baselines.private_progressive import (
    PrivateProgressiveAdapter,
    RateChoice,
    exact_separable_allocation,
)
from ratemem.baselines.protocol import CausalEventView, FrozenComparisonContract, FutureAccessError
from ratemem.baselines.shared_greedy import SharedPacketGreedyAdapter
from ratemem.baselines.shared_inputs import (
    CandidateAccessError,
    SharedInputReader,
    materialize_fixture_bundle,
)
from ratemem.baselines.static_shared import CtsCodebook, StaticSharedAdapter, VbCodebook
from ratemem.evaluation.traces import CreateEvent


def _events() -> tuple[CreateEvent, CreateEvent]:
    return (
        CreateEvent(
            event_index=0,
            handle="fixture-handle-0",
            concept_token="<concept_000000>",
            support_image_ids=("image-0",),
            description_id="description-0",
        ),
        CreateEvent(
            event_index=1,
            handle="fixture-handle-1",
            concept_token="<concept_000001>",
            support_image_ids=("image-1",),
            description_id="description-1",
        ),
    )


def _contract(candidate_stream_sha256: str) -> FrozenComparisonContract:
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
        candidate_stream_sha256=candidate_stream_sha256,
        prompt_pool_sha256="7" * 64,
        support_pool_sha256="5" * 64,
        noise_seed_manifest_sha256="8" * 64,
        sampler_id="flow-dpm",
        scheduler_revision="scheduler-v1",
        cfg_scale=4.5,
        resolution=(1024, 1024),
        denoising_steps=20,
        byte_budget=10_000,
        request_regime="uniform",
        search_budget_sha256="9" * 64,
    )


def test_causal_view_and_shared_reader_reject_future_access(tmp_path: Path) -> None:
    events = _events()
    view = CausalEventView(events, 0)
    assert view.history() == events[:1]
    with pytest.raises(FutureAccessError, match="causal adapter requested event 1"):
        view.at(1)
    bundle = materialize_fixture_bundle(tmp_path / "bundle")
    reader = SharedInputReader(bundle, "independent_fifo")
    with pytest.raises(CandidateAccessError, match="future shared input event 1"):
        reader.for_event(1, 0)


def test_canonical_ledger_accounts_for_every_byte_and_component() -> None:
    components = empty_components()
    components["base_codes"].append({"handle": "h0", "data": b"abc"})
    payload = export_state(components)
    assert decode_state(payload)["base_codes"] == ({"handle": "h0", "data": b"abc"},)
    ledger = ledger_from_export(payload, shared_trained_bytes=17, external_support_bytes=23)
    assert ledger.online_state_bytes == len(payload)
    assert sum(ledger.component_bytes.values()) == len(payload)
    assert ledger.shared_trained_bytes == 17
    assert ledger.external_support_bytes == 23


def test_independent_cache_export_restores_the_next_exact_receipt(tmp_path: Path) -> None:
    bundle = materialize_fixture_bundle(tmp_path / "bundle")
    reader = SharedInputReader(bundle, "independent_fifo")
    contract = _contract(bundle.manifest.candidate_stream_sha256)
    events = _events()
    original = IndependentCodeCacheAdapter(
        "independent_fifo",
        "fifo",
        shared_inputs=reader,
    )
    original.initialize(contract)
    original.apply_event(events[0], CausalEventView(events, 0))
    exported = original.export_online_state()
    restored = IndependentCodeCacheAdapter(
        "independent_fifo",
        "fifo",
        shared_inputs=reader,
    )
    restored.initialize(contract)
    restored.import_online_state(exported)
    assert restored.export_online_state() == exported
    assert restored.apply_event(events[1], CausalEventView(events, 1)) == original.apply_event(
        events[1],
        CausalEventView(events, 1),
    )


def test_all_native_code_controls_restore_the_next_exact_receipt(tmp_path: Path) -> None:
    bundle = materialize_fixture_bundle(tmp_path / "bundle")
    contract = _contract(bundle.manifest.candidate_stream_sha256)
    events = _events()
    codebook_file = tmp_path / "codebook.bin"
    codebook_file.write_bytes(b"locked-codebook-fixture")
    cts = CtsCodebook.from_fixture(
        group_bases=(np.eye(4, 480, dtype=np.float32),),
        quantization_bits=16,
    )
    vb = VbCodebook.from_fixture(
        bank=np.tile(np.eye(16, dtype=np.float32), (2, 1)),
        subvector_size=16,
        top_k=2,
        weight_bits=16,
    )
    factories = {
        "private_progressive_size_aware": lambda reader: PrivateProgressiveAdapter(
            "private_progressive_size_aware",
            policy="size_aware",
            shared_inputs=reader,
        ),
        "private_progressive_separable_rate": lambda reader: PrivateProgressiveAdapter(
            "private_progressive_separable_rate",
            policy="separable_rate",
            shared_inputs=reader,
        ),
        "shared_packet_plain_greedy": lambda reader: SharedPacketGreedyAdapter(
            shared_inputs=reader
        ),
        "cts_style_static": lambda reader: StaticSharedAdapter(
            "cts_style_static",
            codebook=cts,
            codebook_file=codebook_file,
            shared_inputs=reader,
        ),
        "vb_lora_style_static": lambda reader: StaticSharedAdapter(
            "vb_lora_style_static",
            codebook=vb,
            codebook_file=codebook_file,
            shared_inputs=reader,
        ),
        "share_style_online": lambda reader: OnlineShareAdapter(
            rank=2,
            shared_inputs=reader,
        ),
    }
    for method_id, factory in factories.items():
        reader = SharedInputReader(bundle, method_id)
        original = factory(reader)
        original.initialize(contract)
        original.apply_event(events[0], CausalEventView(events, 0))
        exported = original.export_online_state()
        restored = factory(reader)
        restored.initialize(contract)
        restored.import_online_state(exported)
        assert restored.export_online_state() == exported
        assert restored.apply_event(
            events[1],
            CausalEventView(events, 1),
        ) == original.apply_event(events[1], CausalEventView(events, 1))


def test_rate_static_and_online_controls_have_distinct_exact_algorithms() -> None:
    choices = {
        "h0": (
            RateChoice("h0", 0, 0, Decimal("0")),
            RateChoice("h0", 1, 3, Decimal("5")),
        ),
        "h1": (
            RateChoice("h1", 0, 0, Decimal("0")),
            RateChoice("h1", 1, 2, Decimal("4")),
        ),
    }
    allocation = exact_separable_allocation(choices, 3)
    assert allocation.prefix_by_handle == {"h0": 1, "h1": 0}

    cts = CtsCodebook.from_fixture(
        group_bases=(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),),
        quantization_bits=16,
    )
    np.testing.assert_allclose(
        cts.decode(cts.encode(np.array([3, -2, 7], dtype=np.float32))),
        np.array([3, -2, 0], dtype=np.float32),
    )

    vb = VbCodebook.from_fixture(
        bank=np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32),
        subvector_size=2,
        top_k=2,
        weight_bits=16,
    )
    encoded = vb.encode(np.array([0.75, 0.25], dtype=np.float32))
    assert encoded.indices == ((0, 1),)
    np.testing.assert_allclose(vb.decode(encoded), [0.75, 0.25], atol=1e-3)

    basis, coefficients = update_subspace(
        np.array([[1, 0, 0]], dtype=np.float32),
        {"h0": np.array([2], dtype=np.float32)},
        np.array([0, 3, 0], dtype=np.float32),
        "h1",
        1,
    )
    assert basis.shape == (1, 3)
    assert set(coefficients) == {"h0", "h1"}
