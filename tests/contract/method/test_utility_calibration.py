from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.method.proposal import ImmutableBundleProposal
from ratemem.method.utility import (
    CalibrationReceipt,
    CausalRequestHistory,
    build_coverage_oracle,
    calibration_receipt,
    enforce_calibration,
)
from ratemem.state.model import Incidence
from ratemem.state.serialization import bundle_cost_bytes, packet_from_payload


def _receipt(maximum_allowed_ece: float = 1.0) -> CalibrationReceipt:
    return calibration_receipt(
        predicted=np.array([0.0, 1.0]),
        observed=np.array([0.0, 1.0]),
        bins=2,
        method_lock_sha256="a" * 64,
        feature_manifest_sha256="b" * 64,
        label_artifact_sha256="c" * 64,
        maximum_allowed_ece=maximum_allowed_ece,
    )


def test_calibration_schema_is_current_and_threshold_is_fail_closed() -> None:
    path = Path("schemas/ratemem-utility-calibration-v1.schema.json")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert path.read_bytes() == canonical_json_bytes(
        CalibrationReceipt.model_json_schema()
    )
    receipt = calibration_receipt(
        predicted=[0.0, 1.0],
        observed=[1.0, 0.0],
        bins=2,
        method_lock_sha256="a" * 64,
        feature_manifest_sha256="b" * 64,
        label_artifact_sha256="c" * 64,
        maximum_allowed_ece=0.05,
    )
    with pytest.raises(RuntimeError, match="ECE"):
        enforce_calibration(receipt)


def test_oracle_uses_causal_history_cold_start_and_zero_absent_groups() -> None:
    packet = packet_from_payload(b"shared")
    incidences = (
        Incidence("a", packet.packet_id, 1),
        Incidence("b", packet.packet_id, 1),
    )
    proposal = ImmutableBundleProposal(
        packet,
        incidences,
        bundle_cost_bytes(packet, incidences),
    )
    history = CausalRequestHistory(0.5).observe_read("a", 1, True)
    oracle, audit = build_coverage_oracle(
        ("a", "b", "c"),
        (proposal,),
        history,
        allocation_event_index=3,
        incidence_predictions={
            ("a", packet.packet_id): (0.2, 0.3),
            ("b", packet.packet_id): (0.4, 0.5),
        },
        concept_betas={"a": (1.0, 1.0), "b": (1.0, 1.0), "c": (1.0, 1.0)},
        calibration=_receipt(),
        cold_start_handles=("c",),
        maximum_feature_event_index=3,
    )
    assert oracle.request_weights == {"a": 0.5, "b": 0.0, "c": 1.0}
    assert oracle.bundles[packet.packet_id].gains["c"] == (0.0, 0.0)
    assert audit.request_weights == oracle.request_weights
    assert audit.cold_start_handles == ("c",)
