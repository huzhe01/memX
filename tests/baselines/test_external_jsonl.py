from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ratemem.baselines.external_jsonl import (
    ExternalJsonlAdapter,
    ExternalProtocolError,
    ExternalWorkerManifest,
)
from ratemem.baselines.protocol import CausalEventView, FrozenComparisonContract
from ratemem.evaluation.traces import CreateEvent, ProbeEvent, ReadEvent

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests/fixtures/baselines/external_worker.py"


def _manifest(mode: str, *, timeout: float = 2.0) -> ExternalWorkerManifest:
    return ExternalWorkerManifest(
        method_id="dreamcache_feature_cache",
        command=(sys.executable, str(WORKER), "--mode", mode),
        checkout=ROOT,
        source_revision="1" * 40,
        source_archive_sha256="2" * 64,
        environment={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT / "src"),
        },
        timeout_seconds=timeout,
        termination_grace_seconds=0.1,
    )


def _contract() -> FrozenComparisonContract:
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
        byte_budget=100_000,
        request_regime="uniform",
        search_budget_sha256="9" * 64,
    )


def _events() -> tuple[CreateEvent, ReadEvent]:
    return (
        CreateEvent(
            event_index=0,
            handle="h0",
            concept_token="<concept_000000>",
            support_image_ids=("support-0",),
            description_id="description-0",
        ),
        ReadEvent(
            event_index=1,
            handle="h0",
            prompt_id="prompt-0",
            generation_seed=17,
        ),
    )


def test_external_worker_roundtrips_state_and_probe_is_read_only() -> None:
    adapter = ExternalJsonlAdapter(_manifest("valid"))
    adapter.initialize(_contract())
    events = _events()
    create = adapter.apply_event(events[0], CausalEventView(events, 0))
    assert create.ledger.online_state_bytes == len(adapter.export_online_state())
    snapshot = adapter.copy_snapshot()
    before = adapter.export_online_state()
    probe = ProbeEvent(
        event_index=1,
        snapshot_event_index=0,
        handle="h0",
        prompt_id="probe-0",
        generation_seed=19,
    )
    result = adapter.score_probe(snapshot, probe)
    assert result.update_usage is False
    assert adapter.export_online_state() == before
    adapter.close()


def test_external_worker_state_restore_reproduces_next_receipt() -> None:
    events = _events()
    original = ExternalJsonlAdapter(_manifest("valid"))
    original.initialize(_contract())
    original.apply_event(events[0], CausalEventView(events, 0))
    exported = original.export_online_state()
    restored = ExternalJsonlAdapter(_manifest("valid"))
    restored.initialize(_contract())
    restored.import_online_state(exported)
    assert restored.apply_event(events[1], CausalEventView(events, 1)) == original.apply_event(
        events[1], CausalEventView(events, 1)
    )
    assert restored.inspect_launch().shell is False
    assert set(restored.inspect_launch().environment) <= {
        "PATH",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    }
    original.close()
    restored.close()


@pytest.mark.parametrize(
    "mode",
    ["stdout_log", "extra_field", "wrong_index", "invalid_base64"],
)
def test_noncanonical_worker_output_fails_closed(mode: str) -> None:
    adapter = ExternalJsonlAdapter(_manifest(mode))
    with pytest.raises(ExternalProtocolError):
        adapter.initialize(_contract())


def test_worker_timeout_is_failure_not_partial_result() -> None:
    adapter = ExternalJsonlAdapter(_manifest("hang", timeout=0.05))
    with pytest.raises(ExternalProtocolError, match="deadline"):
        adapter.initialize(_contract())


def test_source_identity_is_not_a_mutable_branch_name() -> None:
    manifest = _manifest("valid")
    assert len(manifest.source_revision) == 40
    assert manifest.source_archive_sha256 == "2" * 64
