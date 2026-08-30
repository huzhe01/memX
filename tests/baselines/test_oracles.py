from __future__ import annotations

from pathlib import Path

import pytest

from ratemem.baselines.oracles import (
    ExactAppendOnlyAdapter,
    FutureHandle,
    FuturePacket,
    FutureTracePacketAdapter,
    FutureTraceProblem,
    FutureUtility,
    SymmetricTeacherQuantizer,
    exhaustive_future_trace,
    solve_future_trace,
)
from ratemem.baselines.protocol import (
    CausalEventView,
    FrozenComparisonContract,
    FutureAccessError,
)
from ratemem.baselines.shared_inputs import SharedInputReader, materialize_fixture_bundle
from ratemem.evaluation.traces import CreateEvent


def _contract(candidate_stream_sha256: str, *, budget: int = 10_000) -> FrozenComparisonContract:
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
        byte_budget=budget,
        request_regime="uniform",
        search_budget_sha256="9" * 64,
    )


def test_append_only_roundtrips_and_never_evicts(tmp_path: Path) -> None:
    bundle = materialize_fixture_bundle(tmp_path / "bundle")
    events = (
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
    reader = SharedInputReader(bundle, "exact_append_only_quantized")
    adapter = ExactAppendOnlyAdapter(
        (SymmetricTeacherQuantizer(8), SymmetricTeacherQuantizer(16)),
        shared_inputs=reader,
    )
    adapter.initialize(_contract(bundle.manifest.candidate_stream_sha256))
    receipt = adapter.apply_event(events[0], CausalEventView(events, 0))
    assert receipt.outcome == "created"
    assert receipt.evicted_handles == ()
    assert adapter.inspect_state().quantizer_id["fixture-handle-0"] == "symmetric_int16_v1"
    payload = adapter.export_online_state()
    restored = ExactAppendOnlyAdapter(
        (SymmetricTeacherQuantizer(8), SymmetricTeacherQuantizer(16)),
        shared_inputs=reader,
    )
    restored.initialize(_contract(bundle.manifest.candidate_stream_sha256))
    restored.import_online_state(payload)
    assert restored.export_online_state() == payload
    assert restored.apply_event(events[1], CausalEventView(events, 1)) == adapter.apply_event(
        events[1], CausalEventView(events, 1)
    )


def _future_problem() -> FutureTraceProblem:
    return FutureTraceProblem(
        byte_budget=7,
        handles=(
            FutureHandle(handle="h0", base_bytes=2, create_event=0),
            FutureHandle(handle="h1", base_bytes=2, create_event=1),
        ),
        packets=(
            FuturePacket(
                packet_id="p0",
                cost_bytes=3,
                proposal_event=0,
                dependent_handles=("h0",),
                gain_by_handle={"h0": 7},
            ),
            FuturePacket(
                packet_id="p1",
                cost_bytes=3,
                proposal_event=1,
                dependent_handles=("h1",),
                gain_by_handle={"h1": 9},
            ),
        ),
        utilities=(
            FutureUtility(
                event_index=0,
                request_weight_by_handle={"h0": 1, "h1": 0},
                coverage_cap_by_handle={"h0": 10, "h1": 10},
                base_gain_by_handle={"h0": 1, "h1": 1},
            ),
            FutureUtility(
                event_index=1,
                request_weight_by_handle={"h0": 1, "h1": 4},
                coverage_cap_by_handle={"h0": 10, "h1": 10},
                base_gain_by_handle={"h0": 1, "h1": 1},
            ),
        ),
    )


def test_future_milp_matches_exhaustive_integer_optimum() -> None:
    problem = _future_problem()
    solved = solve_future_trace(problem)
    exhaustive = exhaustive_future_trace(problem)
    assert solved.status == "optimal"
    assert solved.objective_integer == exhaustive.objective_integer
    assert all(row.serialized_bytes <= problem.byte_budget for row in solved.allocations)
    assert solved.certificate.problem_sha256 == problem.sha256


def test_future_adapter_requires_upper_role_and_restores_certified_state() -> None:
    problem = _future_problem()
    result = solve_future_trace(problem)
    with pytest.raises(FutureAccessError, match="full trace requires upper_reference role"):
        FutureTracePacketAdapter(
            problem,
            result,
            base_payload_by_handle={"h0": b"b0", "h1": b"b1"},
            packet_payload_by_id={"p0": b"000", "p1": b"111"},
            requesting_role="causal",
        )
    events = (
        CreateEvent(
            event_index=0,
            handle="h0",
            concept_token="<concept_000000>",
            support_image_ids=("image-0",),
            description_id="description-0",
        ),
        CreateEvent(
            event_index=1,
            handle="h1",
            concept_token="<concept_000001>",
            support_image_ids=("image-1",),
            description_id="description-1",
        ),
    )

    def make() -> FutureTracePacketAdapter:
        return FutureTracePacketAdapter(
            problem,
            result,
            base_payload_by_handle={"h0": b"b0", "h1": b"b1"},
            packet_payload_by_id={"p0": b"000", "p1": b"111"},
            requesting_role="upper_reference",
        )

    adapter = make()
    adapter.initialize(_contract("a" * 64))
    adapter.apply_event(events[0], CausalEventView(events, 0))
    payload = adapter.export_online_state()
    restored = make()
    restored.initialize(_contract("a" * 64))
    restored.import_online_state(payload)
    assert restored.export_online_state() == payload
    assert restored.apply_event(events[1], CausalEventView(events, 1)) == adapter.apply_event(
        events[1], CausalEventView(events, 1)
    )
