from __future__ import annotations

import base64
import hashlib
import json
import re
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from ratemem.evaluation.canonical import write_json_atomic
from ratemem.evaluation.dataset_lock import (
    load_inventory,
    seal_dataset_lock,
    write_dataset_lock_and_card,
)
from ratemem.evaluation.final_trace import (
    AccessPurpose,
    FinalEvaluationContext,
    FinalTraceAccessDenied,
    FinalTraceAlreadyOpened,
    FinalTraceEnvelope,
    acquire_final_trace,
    create_final_evaluation_permit,
    generate_x25519_keypair,
    open_final_trace,
    seal_final_trace,
)
from ratemem.evaluation.statistics import PowerSearchPoint, RequiredUnits
from ratemem.evaluation.traces import AllPools

POLICY = Path("configs/scientific/trace-policy.yaml")
POOLS = Path("tests/fixtures/scientific/concept-pools.json")


def _context(envelope_sha256: str) -> FinalEvaluationContext:
    return FinalEvaluationContext(
        git_commit="1" * 40,
        clean_diff_sha256="2" * 64,
        dataset_lock_sha256="3" * 64,
        evaluation_lock_sha256="4" * 64,
        baseline_lock_sha256="5" * 64,
        method_lock_sha256="6" * 64,
        method_cpu_gate_sha256="7" * 64,
        comparative_execution_freeze_sha256="8" * 64,
        final_envelope_sha256=envelope_sha256,
        paid_compute=False,
        scientific_compute_authorization_sha256=None,
        scientific_cost_reservation_sha256=None,
        scientific_phase_launch_receipt_sha256=None,
    )


def test_open_ledger_is_created_before_decryption_and_blocks_second_attempt(
    tmp_path: Path,
) -> None:
    private_key, public_key = generate_x25519_keypair()
    envelope = seal_final_trace(
        b'{"kind":"read"}\n',
        public_key,
        associated_manifest=b"manifest",
    )
    signing_key = Ed25519PrivateKey.generate()
    context = _context(envelope.sha256)
    permit = create_final_evaluation_permit(
        freeze_id="final_freeze_001",
        context=context,
        signing_key=signing_key,
        approved_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    ledger = tmp_path / "final-open-ledger.json"
    with acquire_final_trace(
        envelope,
        purpose=AccessPurpose.FINAL_EVALUATION,
        permit=permit,
        ledger_path=ledger,
        private_key=private_key,
        approver_public_key=signing_key.public_key(),
        current_context=context,
        associated_manifest=b"manifest",
    ) as stream:
        assert stream.read(1) == b"{"

    with pytest.raises(FinalTraceAlreadyOpened):
        with acquire_final_trace(
            envelope,
            purpose=AccessPurpose.FINAL_EVALUATION,
            permit=permit,
            ledger_path=ledger,
            private_key=private_key,
            approver_public_key=signing_key.public_key(),
            current_context=context,
            associated_manifest=b"manifest",
        ):
            pass
    ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert ledger_payload["status"] == "opened"
    assert ledger_payload["exit_status"] == "success"
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600


def test_tampered_permit_is_rejected_without_consuming_open_attempt(
    tmp_path: Path,
) -> None:
    private_key, public_key = generate_x25519_keypair()
    envelope = seal_final_trace(
        b'{"kind":"read"}\n',
        public_key,
        associated_manifest=b"manifest",
    )
    signing_key = Ed25519PrivateKey.generate()
    context = _context(envelope.sha256)
    permit = create_final_evaluation_permit(
        freeze_id="final_freeze_original",
        context=context,
        signing_key=signing_key,
        approved_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    ).model_copy(update={"freeze_id": "final_freeze_tampered"})
    ledger = tmp_path / "must-not-exist.json"

    with pytest.raises(FinalTraceAccessDenied, match="signature is invalid"):
        with acquire_final_trace(
            envelope,
            purpose=AccessPurpose.FINAL_EVALUATION,
            permit=permit,
            ledger_path=ledger,
            private_key=private_key,
            approver_public_key=signing_key.public_key(),
            current_context=context,
            associated_manifest=b"manifest",
        ):
            pass
    assert not ledger.exists()


def test_failed_decryption_consumes_the_open_attempt(tmp_path: Path) -> None:
    private_key, public_key = generate_x25519_keypair()
    envelope = seal_final_trace(
        b'{"kind":"read"}\n',
        public_key,
        associated_manifest=b"manifest",
    )
    ciphertext = bytearray(base64.b64decode(envelope.ciphertext_base64, validate=True))
    ciphertext[-1] ^= 1
    corrupted = envelope.model_copy(
        update={"ciphertext_base64": base64.b64encode(ciphertext).decode("ascii")}
    )
    signing_key = Ed25519PrivateKey.generate()
    context = _context(corrupted.sha256)
    permit = create_final_evaluation_permit(
        freeze_id="final_freeze_failed",
        context=context,
        signing_key=signing_key,
        approved_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
    )
    ledger = tmp_path / "failed-open.json"

    with pytest.raises(InvalidTag):
        with acquire_final_trace(
            corrupted,
            purpose=AccessPurpose.FINAL_EVALUATION,
            permit=permit,
            ledger_path=ledger,
            private_key=private_key,
            approver_public_key=signing_key.public_key(),
            current_context=context,
            associated_manifest=b"manifest",
        ):
            pass
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["status"] == "opened"
    assert payload["exit_status"] == "error"
    assert payload["error_class"] == "InvalidTag"


def _sealed_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_lock = seal_dataset_lock(
        load_inventory(Path("tests/fixtures/scientific/source-inventory.json")),
        policy_path=Path("configs/scientific/dataset-policy.yaml"),
        mode="synthetic",
    )
    dataset_lock_path = tmp_path / "dataset-lock.yaml"
    write_dataset_lock_and_card(
        dataset_lock,
        dataset_lock_path,
        tmp_path / "data-card.md",
    )
    pools = AllPools.load(POOLS).model_copy(
        update={"dataset_lock_id": dataset_lock.lock_id}
    )
    pools_path = tmp_path / "concept-pools.json"
    write_json_atomic(pools_path, pools.model_dump(mode="json"))
    provisional = RequiredUnits(
        schema_version="1.0",
        calibration_record_sha256="1" * 64,
        calibration_pool_sha256="2" * 64,
        maximum_half_width=0.02,
        minimum_effect=0.03,
        alpha=0.05,
        target_power=0.80,
        minimum_units=1,
        simulation_seed=314159,
        monte_carlo_draws=128,
        ci_required_units=1,
        power_required_units=1,
        required_units=1,
        search_curve=(
            PowerSearchPoint(units=1, ci_half_width=0.01, simulated_power=0.9),
        ),
        record_sha256="0" * 64,
    )
    required = provisional.model_copy(
        update={"record_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )
    power_path = tmp_path / "required-units.json"
    write_json_atomic(power_path, required.model_dump(mode="json"))
    return dataset_lock_path, pools_path, power_path


def test_keygen_and_seal_final_cli_never_write_plaintext(tmp_path: Path) -> None:
    dataset_lock, pools, power = _sealed_inputs(tmp_path)
    private_key_path = tmp_path / "secrets" / "final-trace.key"
    public_key_path = tmp_path / "public" / "final-trace-recipient.pem"
    generated = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "keygen",
            "--private-key",
            str(private_key_path),
            "--public-key",
            str(public_key_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.stdout == "PASS final-trace keypair generated: private-key-disclosed=false\n"
    assert stat.S_IMODE(private_key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_key_path.stat().st_mode) == 0o644
    assert "PRIVATE KEY" not in generated.stdout + generated.stderr

    manifest_path = tmp_path / "release" / "final-test-manifest.json"
    envelope_path = tmp_path / "release" / "final-test-envelope.json"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "seal-final",
            "--dataset-lock",
            str(dataset_lock),
            "--policy",
            str(POLICY),
            "--power-record",
            str(power),
            "--concept-pools",
            str(pools),
            "--recipient",
            str(public_key_path),
            "--manifest-output",
            str(manifest_path),
            "--envelope-output",
            str(envelope_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert re.fullmatch(
        r"PASS final-trace sealed: plaintext retained=false envelope=[0-9a-f]{64}\n",
        completed.stdout,
    )
    assert not list(tmp_path.rglob("*.plaintext"))
    assert '"kind"' not in manifest_path.read_text(encoding="utf-8")
    assert '"kind"' not in envelope_path.read_text(encoding="utf-8")

    private_key = serialization.load_pem_private_key(
        private_key_path.read_bytes(),
        password=None,
    )
    assert isinstance(private_key, X25519PrivateKey)
    envelope = FinalTraceEnvelope.model_validate_json(
        envelope_path.read_text(encoding="utf-8")
    )
    stream = open_final_trace(
        envelope,
        private_key,
        associated_manifest=manifest_path.read_bytes(),
    )
    rows = stream.read().splitlines()
    stream.close()
    assert len(rows) == envelope.event_count
    assert all(set(json.loads(row)) == {"event", "trace_id"} for row in rows)

    repeated = subprocess.run(
        [
            "uv",
            "run",
            "ratemem-eval",
            "traces",
            "seal-final",
            "--dataset-lock",
            str(dataset_lock),
            "--policy",
            str(POLICY),
            "--power-record",
            str(power),
            "--concept-pools",
            str(pools),
            "--recipient",
            str(public_key_path),
            "--manifest-output",
            str(manifest_path),
            "--envelope-output",
            str(envelope_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode == 2
    assert repeated.stderr.endswith("BLOCKED final-trace: output path already exists\n")
