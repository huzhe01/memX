from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Never, cast

from ratemem.pilot.private_io import (
    private_lock,
    read_private_bytes,
    read_private_json,
    write_atomic_private_bytes,
    write_exclusive_private_bytes,
    write_exclusive_private_json,
)

_RATE_KEYS = {
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
}
_ZERO_HASH = "0" * 64


def _require_decimal(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{name} must be an exact Decimal")
    checked = value
    if not checked.is_finite() or checked < 0 or (positive and checked <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return checked


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    checked = value
    if checked <= 0:
        raise ValueError(f"{name} must be positive")
    return checked


@dataclass(frozen=True)
class CostRates:
    gpu_l40s_per_second: Decimal
    cpu_core_per_second: Decimal
    memory_gib_per_second: Decimal
    volume_gib_month: Decimal

    def __post_init__(self) -> None:
        for name in _RATE_KEYS:
            _require_decimal(getattr(self, name), name, positive=True)

    @classmethod
    def normalize(cls, raw: dict[str, str]) -> CostRates:
        if type(raw) is not dict or set(raw) != _RATE_KEYS:
            raise ValueError("unexpected Modal rate keys")
        values: dict[str, Decimal] = {}
        for key, value in raw.items():
            if type(key) is not str or type(value) is not str:
                raise TypeError("Modal rate keys and values must be exact strings")
            try:
                parsed = Decimal(value)
            except InvalidOperation as error:
                raise ValueError(f"rate {key} must be decimal") from error
            if not parsed.is_finite() or parsed <= 0:
                raise ValueError(f"rate {key} must be finite and positive")
            values[key] = parsed
        return cls(**values)


@dataclass(frozen=True)
class ResourceContract:
    gpu_count: int
    cpu_cores: int
    memory_gib: int
    timeout_seconds: int
    startup_timeout_seconds: int
    storage_gib_bound: int
    non_gpu_setup_allowance_usd: Decimal

    def __post_init__(self) -> None:
        for name in (
            "gpu_count",
            "cpu_cores",
            "memory_gib",
            "timeout_seconds",
            "startup_timeout_seconds",
            "storage_gib_bound",
        ):
            _require_positive_int(getattr(self, name), name)
        _require_decimal(
            self.non_gpu_setup_allowance_usd,
            "non_gpu_setup_allowance_usd",
        )


def conservative_bound(rates: CostRates, resources: ResourceContract) -> Decimal:
    if type(rates) is not CostRates or type(resources) is not ResourceContract:
        raise TypeError("cost bound requires exact rates and resources")
    rates.__post_init__()
    resources.__post_init__()
    billed_seconds = Decimal(resources.timeout_seconds + resources.startup_timeout_seconds)
    compute = billed_seconds * (
        rates.gpu_l40s_per_second * resources.gpu_count
        + rates.cpu_core_per_second * resources.cpu_cores
        + rates.memory_gib_per_second * resources.memory_gib
    )
    storage = rates.volume_gib_month * resources.storage_gib_bound
    total = compute + storage + resources.non_gpu_setup_allowance_usd
    return total.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate ledger JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"non-finite ledger JSON constant: {value}")


def _canonical(value: dict[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ledger entry must be finite canonical JSON") from error


def _lower_sha(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    checked = value
    if len(checked) != 64 or any(character not in "0123456789abcdef" for character in checked):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return checked


def _amount_text(value: object, name: str, *, positive: bool = False) -> Decimal:
    if type(value) is not str:
        raise TypeError(f"ledger {name} must be an exact decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"ledger {name} must be decimal") from error
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        raise ValueError(f"ledger {name} must be finite and nonnegative")
    return parsed


@dataclass(frozen=True, slots=True)
class _LedgerState:
    entries: tuple[dict[str, Any], ...]
    reserved_ids: frozenset[str]
    open_bounds: dict[str, Decimal]
    reservation_known: dict[str, Decimal]
    latest_known_usage: Decimal
    reservation_bounds: dict[str, Decimal]
    all_reservation_known: dict[str, Decimal]
    reconciliations: dict[str, tuple[Decimal, Decimal]]


@dataclass(frozen=True, slots=True)
class AttemptCost:
    """Authoritative ledger facts for one attempt, including a completed retry."""

    known_usage_before: Decimal
    phase_bound: Decimal
    reconciled_cost: Decimal | None
    known_usage_after: Decimal | None


class CostLedger:
    def __init__(self, path: Path, *, internal_limit_usd: Decimal) -> None:
        if type(path) is not type(Path()):
            raise TypeError("ledger path must be an exact Path")
        if _require_decimal(internal_limit_usd, "internal_limit_usd", positive=True) != Decimal(
            "27.00"
        ):
            raise ValueError("cost ledger internal limit must be exactly USD 27.00")
        self.path = path
        self.internal_limit_usd = internal_limit_usd
        self.lock_path = path.with_name(f"{path.name}.lock")

    def _receipt_path(self, sequence: int) -> Path:
        return self.path.with_name(f"{self.path.name}.receipt-{sequence:020d}.json")

    def _receipt_exists(self, sequence: int) -> bool:
        try:
            self._receipt_path(sequence).lstat()
        except FileNotFoundError:
            return False
        return True

    def _validate_receipts(self, entries: list[dict[str, Any]]) -> None:
        previous_receipt = _ZERO_HASH
        for sequence, entry in enumerate(entries):
            path = self._receipt_path(sequence)
            if not self._receipt_exists(sequence):
                raise ValueError("ledger receipt is missing after an interrupted append")
            receipt = read_private_json(path)
            expected = {
                "sequence",
                "entry_sha256",
                "previous_receipt_sha256",
                "receipt_sha256",
            }
            if set(receipt) != expected or type(receipt["sequence"]) is not int:
                raise ValueError("ledger receipt schema is invalid")
            body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            claimed = _lower_sha(receipt["receipt_sha256"], "receipt_sha256")
            if (
                receipt["sequence"] != sequence
                or receipt["entry_sha256"] != entry["entry_sha256"]
                or receipt["previous_receipt_sha256"] != previous_receipt
                or hashlib.sha256(_canonical(body).encode("ascii")).hexdigest() != claimed
            ):
                raise ValueError("ledger receipt detects rollback or truncation")
            previous_receipt = claimed
        if self._receipt_exists(len(entries)):
            raise ValueError("ledger receipt detects rollback or truncation")

    def _publish_receipt(self, sequence: int, entry_sha256: str) -> None:
        previous_receipt = _ZERO_HASH
        if sequence:
            prior = read_private_json(self._receipt_path(sequence - 1))
            previous_receipt = _lower_sha(prior.get("receipt_sha256"), "receipt_sha256")
        body: dict[str, object] = {
            "sequence": sequence,
            "entry_sha256": entry_sha256,
            "previous_receipt_sha256": previous_receipt,
        }
        receipt = body | {
            "receipt_sha256": hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()
        }
        write_exclusive_private_json(self._receipt_path(sequence), receipt)

    def _state_unlocked(self) -> _LedgerState:
        try:
            self.path.lstat()
        except FileNotFoundError:
            self._validate_receipts([])
            return _LedgerState((), frozenset(), {}, {}, Decimal("0"), {}, {}, {})
        content = read_private_bytes(self.path)
        if content and not content.endswith(b"\n"):
            raise ValueError("ledger is truncated and lacks a final newline")
        entries: list[dict[str, Any]] = []
        previous = _ZERO_HASH
        reserved_ids: set[str] = set()
        open_bounds: dict[str, Decimal] = {}
        reservation_known: dict[str, Decimal] = {}
        reservation_bounds: dict[str, Decimal] = {}
        all_reservation_known: dict[str, Decimal] = {}
        reconciliations: dict[str, tuple[Decimal, Decimal]] = {}
        latest_known = Decimal("0")
        for sequence, raw_line in enumerate(content.splitlines()):
            try:
                text = raw_line.decode("utf-8")
                decoded = json.loads(
                    text,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_nonfinite,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("ledger contains invalid JSON") from error
            if type(decoded) is not dict:
                raise ValueError("ledger entry must be an exact object")
            entry = cast(dict[str, Any], decoded)
            if _canonical(entry) != text:
                raise ValueError("ledger entry is not canonical JSON")
            claimed = entry.get("entry_sha256")
            _lower_sha(claimed, "entry_sha256")
            body = {key: value for key, value in entry.items() if key != "entry_sha256"}
            if type(body.get("sequence")) is not int or body["sequence"] != sequence:
                raise ValueError("ledger sequence is invalid")
            if body.get("previous_sha256") != previous:
                raise ValueError("ledger previous hash is invalid")
            actual = hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()
            if actual != claimed:
                raise ValueError("ledger entry hash is invalid")
            kind = body.get("kind")
            attempt = body.get("attempt_id")
            if type(attempt) is not str or not attempt:
                raise ValueError("ledger attempt_id must be a nonempty exact string")
            if kind == "reserve":
                expected = {
                    "kind",
                    "attempt_id",
                    "at",
                    "known_usage_usd",
                    "pending_before_usd",
                    "phase_bound_usd",
                    "rates_sha256",
                    "sequence",
                    "previous_sha256",
                }
                if set(body) != expected or attempt in reserved_ids:
                    raise ValueError("ledger reserve schema or duplicate attempt is invalid")
                known = _amount_text(body["known_usage_usd"], "known usage")
                pending = _amount_text(body["pending_before_usd"], "pending")
                bound = _amount_text(body["phase_bound_usd"], "phase bound", positive=True)
                _lower_sha(body["rates_sha256"], "rates_sha256")
                if pending != sum(open_bounds.values(), Decimal("0")):
                    raise ValueError("ledger pending amount does not match open reservations")
                if known < latest_known:
                    raise ValueError("ledger known usage decreased")
                reserved_ids.add(attempt)
                open_bounds[attempt] = bound
                reservation_known[attempt] = known
                reservation_bounds[attempt] = bound
                all_reservation_known[attempt] = known
                latest_known = known
            elif kind == "reconcile":
                expected = {
                    "kind",
                    "attempt_id",
                    "at",
                    "reconciled_cost_usd",
                    "known_usage_after_usd",
                    "sequence",
                    "previous_sha256",
                }
                if set(body) != expected or attempt not in open_bounds:
                    raise ValueError("ledger reconcile lacks exactly one open reservation")
                cost = _amount_text(body["reconciled_cost_usd"], "reconciled cost")
                after = _amount_text(body["known_usage_after_usd"], "known usage after")
                if after < reservation_known[attempt] or after - reservation_known[attempt] != cost:
                    raise ValueError("ledger reconciliation is nonmonotonic or inconsistent")
                open_bounds.pop(attempt)
                reservation_known.pop(attempt)
                reconciliations[attempt] = (cost, after)
                latest_known = max(latest_known, after)
            else:
                raise ValueError("ledger entry kind is invalid")
            datetime.fromisoformat(cast(str, body.get("at")))
            previous = cast(str, claimed)
            entries.append(entry)
        self._validate_receipts(entries)
        return _LedgerState(
            tuple(entries),
            frozenset(reserved_ids),
            open_bounds,
            reservation_known,
            latest_known,
            reservation_bounds,
            all_reservation_known,
            reconciliations,
        )

    def _validate_reservation_unlocked(
        self,
        state: _LedgerState,
        attempt_id: str,
        *,
        known_usage: Decimal,
        phase_bound: Decimal,
        rates_sha256: str,
    ) -> Decimal:
        if type(attempt_id) is not str or not attempt_id:
            raise ValueError("attempt_id must be a nonempty exact string")
        known = _require_decimal(known_usage, "known_usage")
        bound = _require_decimal(phase_bound, "phase_bound", positive=True)
        _lower_sha(rates_sha256, "rates_sha256")
        if attempt_id in state.reserved_ids:
            raise ValueError("attempt already has a reservation")
        if known < state.latest_known_usage:
            raise ValueError("known metered usage must not decrease; monotonic evidence required")
        pending = sum(state.open_bounds.values(), Decimal("0"))
        if known + pending + bound > self.internal_limit_usd:
            raise ValueError("launch would exceed the internal USD 27.00 limit")
        return pending

    def preview_reservation(
        self,
        attempt_id: str,
        *,
        known_usage: Decimal,
        phase_bound: Decimal,
        rates_sha256: str,
    ) -> Decimal:
        """Validate admission without appending; return currently open worst-case cost."""

        with private_lock(self.lock_path):
            state = self._state_unlocked()
            return self._validate_reservation_unlocked(
                state,
                attempt_id,
                known_usage=known_usage,
                phase_bound=phase_bound,
                rates_sha256=rates_sha256,
            )

    def require_pristine(self) -> None:
        """Reject a ledger that has ever admitted an attempt, open or reconciled."""

        with private_lock(self.lock_path):
            state = self._state_unlocked()
            if state.reserved_ids:
                raise ValueError("the one-shot pilot ledger contains a prior reservation")

    def attempt_cost(self, attempt_id: str) -> AttemptCost | None:
        """Read one reservation and any durable reconciliation under the ledger lock."""

        if type(attempt_id) is not str or not attempt_id:
            raise ValueError("attempt_id must be a nonempty exact string")
        with private_lock(self.lock_path):
            state = self._state_unlocked()
            if attempt_id not in state.reserved_ids:
                return None
            reconciliation = state.reconciliations.get(attempt_id)
            return AttemptCost(
                known_usage_before=state.all_reservation_known[attempt_id],
                phase_bound=state.reservation_bounds[attempt_id],
                reconciled_cost=None if reconciliation is None else reconciliation[0],
                known_usage_after=None if reconciliation is None else reconciliation[1],
            )

    def _append_unlocked(self, state: _LedgerState, body: dict[str, Any]) -> None:
        previous = state.entries[-1]["entry_sha256"] if state.entries else _ZERO_HASH
        unhashed = body | {"sequence": len(state.entries), "previous_sha256": previous}
        entry = unhashed | {
            "entry_sha256": hashlib.sha256(_canonical(unhashed).encode("ascii")).hexdigest()
        }
        existing = b"" if not state.entries else read_private_bytes(self.path)
        content = existing + _canonical(entry).encode("ascii") + b"\n"
        if state.entries:
            write_atomic_private_bytes(self.path, content)
        else:
            write_exclusive_private_bytes(self.path, content)
        self._publish_receipt(len(state.entries), cast(str, entry["entry_sha256"]))

    def reserve(
        self,
        attempt_id: str,
        *,
        known_usage: Decimal,
        phase_bound: Decimal,
        rates_sha256: str,
    ) -> None:
        known = _require_decimal(known_usage, "known_usage")
        bound = _require_decimal(phase_bound, "phase_bound", positive=True)
        rate_hash = _lower_sha(rates_sha256, "rates_sha256")
        with private_lock(self.lock_path):
            state = self._state_unlocked()
            pending = self._validate_reservation_unlocked(
                state,
                attempt_id,
                known_usage=known,
                phase_bound=bound,
                rates_sha256=rate_hash,
            )
            self._append_unlocked(
                state,
                {
                    "kind": "reserve",
                    "attempt_id": attempt_id,
                    "at": datetime.now(UTC).isoformat(),
                    "known_usage_usd": str(known),
                    "pending_before_usd": str(pending),
                    "phase_bound_usd": str(bound),
                    "rates_sha256": rate_hash,
                },
            )

    def reconcile(
        self,
        attempt_id: str,
        *,
        reconciled_cost: Decimal,
        known_usage_after: Decimal,
    ) -> None:
        if type(attempt_id) is not str or not attempt_id:
            raise ValueError("attempt_id must be a nonempty exact string")
        cost = _require_decimal(reconciled_cost, "reconciled_cost")
        after = _require_decimal(known_usage_after, "known_usage_after")
        with private_lock(self.lock_path):
            state = self._state_unlocked()
            if attempt_id not in state.open_bounds:
                prior = state.reconciliations.get(attempt_id)
                if prior == (cost, after):
                    return
                if prior is not None:
                    raise ValueError("attempt already has a different reconciliation")
                raise ValueError("attempt must have exactly one open reservation")
            before = state.reservation_known[attempt_id]
            if after < before:
                raise ValueError(
                    "PENDING: billing data has not caught up; another launch is forbidden"
                )
            if after - before != cost:
                raise ValueError("reconciled cost must equal the fresh metered usage delta")
            self._append_unlocked(
                state,
                {
                    "kind": "reconcile",
                    "attempt_id": attempt_id,
                    "at": datetime.now(UTC).isoformat(),
                    "reconciled_cost_usd": str(cost),
                    "known_usage_after_usd": str(after),
                },
            )

    def verify_hash_chain(self) -> None:
        with private_lock(self.lock_path):
            self._state_unlocked()
