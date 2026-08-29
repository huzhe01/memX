from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import ratemem.pilot.costs as costs_module
import ratemem.pilot.private_io as private_io
from ratemem.pilot.costs import CostLedger, CostRates, ResourceContract, conservative_bound


def _rates() -> CostRates:
    return CostRates.normalize(
        {
            "gpu_l40s_per_second": "0.000542",
            "cpu_core_per_second": "0.0000131",
            "memory_gib_per_second": "0.00000222",
            "volume_gib_month": "0.09",
        }
    )


def _resources() -> ResourceContract:
    return ResourceContract(
        gpu_count=1,
        cpu_cores=4,
        memory_gib=32,
        timeout_seconds=7200,
        startup_timeout_seconds=1800,
        storage_gib_bound=24,
        non_gpu_setup_allowance_usd=Decimal("2.00"),
    )


def test_bound_includes_gpu_cpu_requested_ram_startup_and_storage() -> None:
    assert conservative_bound(_rates(), _resources()) == Decimal("10.15")


@pytest.mark.parametrize(
    "raw",
    [
        {"gpu_l40s_per_second": "0.1"},
        {
            "gpu_l40s_per_second": "NaN",
            "cpu_core_per_second": "0.1",
            "memory_gib_per_second": "0.1",
            "volume_gib_month": "0.1",
        },
        {
            "gpu_l40s_per_second": "-0.1",
            "cpu_core_per_second": "0.1",
            "memory_gib_per_second": "0.1",
            "volume_gib_month": "0.1",
        },
    ],
)
def test_rates_reject_missing_nonfinite_and_negative_values(raw: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="rate|finite|positive|keys"):
        CostRates.normalize(raw)


def test_resource_contract_rejects_bool_zero_and_nonfinite_allowance() -> None:
    with pytest.raises((TypeError, ValueError)):
        ResourceContract(**(_resources().__dict__ | {"gpu_count": True}))
    with pytest.raises((TypeError, ValueError)):
        ResourceContract(**(_resources().__dict__ | {"timeout_seconds": 0}))
    with pytest.raises((TypeError, ValueError)):
        ResourceContract(
            **(_resources().__dict__ | {"non_gpu_setup_allowance_usd": Decimal("NaN")})
        )


def test_reservation_uses_known_plus_all_pending_plus_new_at_27(tmp_path: Path) -> None:
    ledger = CostLedger(
        tmp_path / "private" / "cost-ledger.jsonl",
        internal_limit_usd=Decimal("27.00"),
    )
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("20.00"),
        rates_sha256="1" * 64,
    )
    ledger.reserve(
        "attempt-two",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("6.00"),
        rates_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="USD 27.00"):
        ledger.reserve(
            "attempt-three",
            known_usage=Decimal("1.00"),
            phase_bound=Decimal("0.01"),
            rates_sha256="3" * 64,
        )
    ledger.verify_hash_chain()


def test_duplicate_attempt_and_decreasing_known_usage_fail_closed(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("2.00"),
        phase_bound=Decimal("3.00"),
        rates_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="already"):
        ledger.reserve(
            "attempt-one",
            known_usage=Decimal("2.00"),
            phase_bound=Decimal("3.00"),
            rates_sha256="1" * 64,
        )
    with pytest.raises(ValueError, match="decrease|monotonic"):
        ledger.reserve(
            "attempt-two",
            known_usage=Decimal("1.99"),
            phase_bound=Decimal("1.00"),
            rates_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("known", "bound"),
    [
        (Decimal("NaN"), Decimal("1")),
        (Decimal("-1"), Decimal("1")),
        (Decimal("1"), Decimal("Infinity")),
        (Decimal("1"), Decimal("0")),
    ],
)
def test_reservation_rejects_nonfinite_negative_and_zero_amounts(
    tmp_path: Path,
    known: Decimal,
    bound: Decimal,
) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    with pytest.raises(ValueError, match="finite|nonnegative|positive"):
        ledger.reserve(
            "attempt-one",
            known_usage=known,
            phase_bound=bound,
            rates_sha256="1" * 64,
        )


def test_reconcile_requires_open_reservation_and_fresh_metered_usage(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    with pytest.raises(ValueError, match="open reservation"):
        ledger.reconcile(
            "missing",
            reconciled_cost=Decimal("1.00"),
            known_usage_after=Decimal("2.00"),
        )
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("2.00"),
        phase_bound=Decimal("3.00"),
        rates_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="PENDING"):
        ledger.reconcile(
            "attempt-one",
            reconciled_cost=Decimal("0.00"),
            known_usage_after=Decimal("1.99"),
        )
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("2.50"),
    )
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("2.50"),
    )
    ledger.verify_hash_chain()


@pytest.mark.parametrize("mutation", ["tamper", "truncate", "reorder", "duplicate_json"])
def test_hash_chain_tamper_truncate_reorder_and_duplicate_json_are_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("2.00"),
        rates_sha256="1" * 64,
    )
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("1.50"),
    )
    lines = ledger.path.read_text().splitlines(keepends=True)
    if mutation == "tamper":
        lines[0] = lines[0].replace('"phase_bound_usd":"2.00"', '"phase_bound_usd":"1.00"')
    elif mutation == "truncate":
        lines[-1] = lines[-1][:-2]
    elif mutation == "reorder":
        lines.reverse()
    else:
        lines[0] = lines[0].replace(
            '"kind":"reserve"',
            '"kind":"reserve","kind":"reserve"',
        )
    ledger.path.write_text("".join(lines))
    ledger.path.chmod(0o600)

    with pytest.raises(ValueError, match="ledger|hash|sequence|newline|duplicate"):
        ledger.verify_hash_chain()
    with pytest.raises(ValueError, match="ledger|hash|sequence|newline|duplicate"):
        ledger.reserve(
            "attempt-two",
            known_usage=Decimal("1.50"),
            phase_bound=Decimal("1.00"),
            rates_sha256="2" * 64,
        )


def test_ledger_rejects_replacement_with_a_valid_old_prefix(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("2.00"),
        rates_sha256="1" * 64,
    )
    valid_old_prefix = ledger.path.read_bytes()
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("1.50"),
    )
    ledger.path.write_bytes(valid_old_prefix)
    ledger.path.chmod(0o600)

    with pytest.raises(ValueError, match="receipt|rollback|truncat"):
        ledger.verify_hash_chain()
    with pytest.raises(ValueError, match="receipt|rollback|truncat"):
        ledger.reserve(
            "attempt-two",
            known_usage=Decimal("1.50"),
            phase_bound=Decimal("1.00"),
            rates_sha256="2" * 64,
        )


def test_receipt_publication_crash_poisons_future_ledger_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))

    def fail_receipt(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected receipt publication crash")

    monkeypatch.setattr(costs_module, "write_exclusive_private_json", fail_receipt)
    with pytest.raises(OSError, match="receipt publication"):
        ledger.reserve(
            "attempt-one",
            known_usage=Decimal("1.00"),
            phase_bound=Decimal("2.00"),
            rates_sha256="1" * 64,
        )
    monkeypatch.undo()
    with pytest.raises(ValueError, match="receipt.*missing|interrupted"):
        ledger.verify_hash_chain()


def test_append_failure_does_not_claim_a_successful_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))

    def fail_write(_descriptor: int, _content: bytes) -> None:
        raise OSError("injected partial write")

    monkeypatch.setattr(private_io, "_write_all", fail_write)
    with pytest.raises(OSError, match="partial write"):
        ledger.reserve(
            "attempt-one",
            known_usage=Decimal("1.00"),
            phase_bound=Decimal("2.00"),
            rates_sha256="1" * 64,
        )
    assert not ledger.path.exists() or ledger.path.read_bytes() == b""


def test_ledger_file_and_lock_are_private(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("2.00"),
        rates_sha256="1" * 64,
    )
    assert ledger.path.stat().st_mode & 0o777 == 0o600
    assert ledger.lock_path.stat().st_mode & 0o777 == 0o600
    json.loads(ledger.path.read_text().splitlines()[0])


def test_reservation_preview_is_read_only_and_reports_existing_pending(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    assert ledger.preview_reservation(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("20.00"),
        rates_sha256="1" * 64,
    ) == Decimal("0")
    assert not ledger.path.exists()
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("20.00"),
        rates_sha256="1" * 64,
    )
    assert ledger.preview_reservation(
        "attempt-two",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("6.00"),
        rates_sha256="2" * 64,
    ) == Decimal("20.00")
    with pytest.raises(ValueError, match="USD 27.00"):
        ledger.preview_reservation(
            "attempt-three",
            known_usage=Decimal("1.00"),
            phase_bound=Decimal("6.01"),
            rates_sha256="3" * 64,
        )


def test_reconciliation_record_is_readable_and_exact_retry_is_idempotent(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("2.00"),
        phase_bound=Decimal("3.00"),
        rates_sha256="1" * 64,
    )
    reservation = ledger.attempt_cost("attempt-one")
    assert reservation is not None
    assert reservation.known_usage_before == Decimal("2.00")
    assert reservation.phase_bound == Decimal("3.00")
    assert reservation.reconciled_cost is None
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("2.50"),
    )
    before = ledger.path.read_bytes()
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("2.50"),
    )
    assert ledger.path.read_bytes() == before
    reconciled = ledger.attempt_cost("attempt-one")
    assert reconciled is not None
    assert reconciled.reconciled_cost == Decimal("0.50")
    assert reconciled.known_usage_after == Decimal("2.50")
    with pytest.raises(ValueError, match="different reconciliation"):
        ledger.reconcile(
            "attempt-one",
            reconciled_cost=Decimal("0.51"),
            known_usage_after=Decimal("2.51"),
        )


def test_pristine_guard_rejects_any_prior_reservation_even_after_reconciliation(
    tmp_path: Path,
) -> None:
    ledger = CostLedger(tmp_path / "private" / "ledger.jsonl", internal_limit_usd=Decimal("27.00"))
    ledger.require_pristine()
    ledger.reserve(
        "attempt-one",
        known_usage=Decimal("1.00"),
        phase_bound=Decimal("2.00"),
        rates_sha256="1" * 64,
    )
    ledger.reconcile(
        "attempt-one",
        reconciled_cost=Decimal("0.50"),
        known_usage_after=Decimal("1.50"),
    )

    with pytest.raises(ValueError, match="prior reservation"):
        ledger.require_pristine()
