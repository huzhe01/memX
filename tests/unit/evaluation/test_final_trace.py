from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from ratemem.evaluation.final_trace import (
    AccessPurpose,
    FinalEvaluationContext,
    FinalTraceAccessDenied,
    acquire_final_trace,
    generate_x25519_keypair,
    open_final_trace,
    seal_final_trace,
)


def _context_kwargs() -> dict[str, object]:
    return {
        "git_commit": "1" * 40,
        "clean_diff_sha256": "2" * 64,
        "dataset_lock_sha256": "3" * 64,
        "evaluation_lock_sha256": "4" * 64,
        "baseline_lock_sha256": "5" * 64,
        "method_lock_sha256": "6" * 64,
        "method_cpu_gate_sha256": "7" * 64,
        "comparative_execution_freeze_sha256": "8" * 64,
        "final_envelope_sha256": "9" * 64,
    }


def test_final_trace_round_trip_requires_private_key_and_binds_manifest() -> None:
    private_key, public_key = generate_x25519_keypair()
    envelope = seal_final_trace(
        b'{"kind":"read"}\n',
        public_key,
        associated_manifest=b"manifest",
    )

    with pytest.raises(InvalidTag):
        open_final_trace(
            envelope,
            generate_x25519_keypair()[0],
            associated_manifest=b"manifest",
        )
    with pytest.raises(ValueError, match="associated manifest"):
        open_final_trace(
            envelope,
            private_key,
            associated_manifest=b"changed-manifest",
        )
    stream = open_final_trace(
        envelope,
        private_key,
        associated_manifest=b"manifest",
    )
    assert stream.read() == b'{"kind":"read"}\n'
    stream.close()


def test_paid_context_requires_all_three_consumed_compute_records() -> None:
    with pytest.raises(ValueError, match="requires all compute approvals"):
        FinalEvaluationContext(
            **_context_kwargs(),
            paid_compute=True,
            scientific_compute_authorization_sha256="a" * 64,
            scientific_cost_reservation_sha256="b" * 64,
            scientific_phase_launch_receipt_sha256=None,
        )
    with pytest.raises(ValueError, match="cannot name paid-compute approvals"):
        FinalEvaluationContext(
            **_context_kwargs(),
            paid_compute=False,
            scientific_compute_authorization_sha256="a" * 64,
            scientific_cost_reservation_sha256=None,
            scientific_phase_launch_receipt_sha256=None,
        )


@pytest.mark.parametrize(
    "purpose",
    [
        AccessPurpose.TRAINING,
        AccessPurpose.MODEL_SELECTION,
        AccessPurpose.COMPARATIVE_VALIDATION,
    ],
)
def test_nonfinal_purposes_are_denied_before_creating_a_ledger(
    tmp_path: Path,
    purpose: AccessPurpose,
) -> None:
    _private_key, public_key = generate_x25519_keypair()
    envelope = seal_final_trace(
        b'{"kind":"read"}\n',
        public_key,
        associated_manifest=b"manifest",
    )
    ledger = tmp_path / "unused.json"

    with pytest.raises(FinalTraceAccessDenied):
        with acquire_final_trace(
            envelope,
            purpose=purpose,
            permit=None,
            ledger_path=ledger,
        ):
            pass
    assert not ledger.exists()
