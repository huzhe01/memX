from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Never, cast

from ratemem.pilot.private_io import (
    canonical_json_bytes,
    file_sha256,
    read_private_bytes,
    read_private_json,
)

_PROFILE = "ratemem-pilot"
_ENVIRONMENT = "main"
_BUDGET = "28.00"
_CONFIRMATION = (
    "I confirm the Modal dashboard Workspace usage budget is USD 28.00 before credits."
)
_RATE_KEYS = {
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
}
_ALLOWED_MODAL_ARGUMENTS = {
    ("profile", "list"),
    ("billing", "summary", "--for", "this month"),
    ("billing", "rates"),
}


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Modal JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Never:
    raise ValueError(f"non-finite Modal JSON constant: {value}")


def _decimal_text(value: object, name: str, *, positive: bool = False) -> Decimal:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a decimal string") from error
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return parsed


def _validate_rates(value: object) -> dict[str, str]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != _RATE_KEYS:
        raise ValueError("Modal rates have missing or unknown fields")
    rates = cast(dict[object, object], value)
    normalized: dict[str, str] = {}
    for key in sorted(_RATE_KEYS):
        raw = rates[key]
        _decimal_text(raw, f"rate {key}", positive=True)
        normalized[key] = cast(str, raw)
    return normalized


def _validate_profiles(value: object) -> tuple[str, str]:
    if type(value) is not list:
        raise ValueError("Modal profile response must be an exact list")
    active: list[tuple[str, str]] = []
    for entry in cast(list[object], value):
        if type(entry) is not dict or set(cast(dict[object, object], entry)) != {
            "name",
            "workspace",
            "active",
        }:
            raise ValueError("Modal profile response has unknown fields")
        payload = cast(dict[str, object], entry)
        if type(payload["name"]) is not str or type(payload["workspace"]) is not str:
            raise TypeError("Modal profile names and workspace IDs must be exact strings")
        if type(payload["active"]) is not bool:
            raise TypeError("Modal profile active must be an exact bool")
        if payload["active"]:
            active.append((payload["name"], payload["workspace"]))
    if len(active) != 1 or active[0][0] != _PROFILE or not active[0][1]:
        raise ValueError("ratemem-pilot must be the sole explicitly selected profile")
    return active[0]


def _validate_billing(value: object) -> str:
    expected = {"metered_cost", "billed_cost", "adjustments", "metered_cost_breakdown"}
    if type(value) is not dict or set(cast(dict[object, object], value)) != expected:
        raise ValueError("Modal billing response has missing or unknown fields")
    payload = cast(dict[str, object], value)
    metered = payload["metered_cost"]
    metered_amount = _decimal_text(metered, "metered_cost")
    billed_amount = _decimal_text(payload["billed_cost"], "billed_cost")
    adjustments = payload["adjustments"]
    breakdown = payload["metered_cost_breakdown"]
    if type(adjustments) is not dict or set(cast(dict[object, object], adjustments)) != {"credits"}:
        raise ValueError("Modal billing adjustments have unknown fields")
    credits = cast(dict[str, object], adjustments)["credits"]
    if type(credits) is not str:
        raise TypeError("Modal credits must be an exact decimal string")
    try:
        parsed_credit = Decimal(credits)
    except InvalidOperation as error:
        raise ValueError("Modal credits must be a decimal string") from error
    if not parsed_credit.is_finite() or parsed_credit > 0:
        raise ValueError("Modal credits must be finite and nonpositive")
    if type(breakdown) is not dict or set(cast(dict[object, object], breakdown)) != {"compute"}:
        raise ValueError("Modal metered breakdown has unknown fields")
    compute = _decimal_text(cast(dict[str, object], breakdown)["compute"], "metered compute")
    if compute != metered_amount:
        raise ValueError("Modal compute breakdown must equal metered cost")
    if billed_amount != metered_amount + parsed_credit:
        raise ValueError("Modal billed cost must equal metered cost plus credits")
    return cast(str, metered)


@dataclass(frozen=True, slots=True)
class OperatorBudgetAttestation:
    profile: str
    workspace: str
    environment: str
    workspace_budget_usd: str
    captured_at: datetime
    dashboard_evidence_path: Path
    dashboard_evidence_sha256: str
    confirmation_statement: str

    @classmethod
    def from_private_file(cls, path: Path) -> OperatorBudgetAttestation:
        payload = read_private_json(path)
        if read_private_bytes(path) != canonical_json_bytes(payload):
            raise ValueError("operator-attested dashboard evidence must be canonical JSON")
        expected = {
            "kind",
            "profile",
            "workspace",
            "environment",
            "workspace_budget_usd",
            "captured_at",
            "dashboard_evidence_path",
            "dashboard_evidence_sha256",
            "confirmation_statement",
        }
        if set(payload) != expected or payload["kind"] != "operator-dashboard-budget-v1":
            raise ValueError("operator-attested dashboard evidence schema is invalid")
        if any(type(payload[key]) is not str for key in expected):
            raise TypeError("operator-attested dashboard evidence fields must be exact strings")
        try:
            captured = datetime.fromisoformat(cast(str, payload["captured_at"]))
        except ValueError as error:
            raise ValueError("operator-attested dashboard captured_at is invalid") from error
        evidence_hash = cast(str, payload["dashboard_evidence_sha256"])
        _lowercase_sha(evidence_hash, "dashboard evidence")
        dashboard_path = Path(cast(str, payload["dashboard_evidence_path"]))
        if not dashboard_path.is_absolute():
            raise ValueError("operator-attested dashboard evidence path must be absolute")
        return cls(
            profile=cast(str, payload["profile"]),
            workspace=cast(str, payload["workspace"]),
            environment=cast(str, payload["environment"]),
            workspace_budget_usd=cast(str, payload["workspace_budget_usd"]),
            captured_at=captured,
            dashboard_evidence_path=dashboard_path,
            dashboard_evidence_sha256=evidence_hash,
            confirmation_statement=cast(str, payload["confirmation_statement"]),
        )

    def verify(self, *, expected_workspace: str, max_age_seconds: int) -> None:
        if (
            self.profile != _PROFILE
            or self.workspace != expected_workspace
            or self.environment != _ENVIRONMENT
            or self.workspace_budget_usd != _BUDGET
            or self.confirmation_statement != _CONFIRMATION
        ):
            raise ValueError(
                "operator-attested dashboard evidence does not bind the exact budget identity"
            )
        if self.captured_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("operator-attested dashboard captured_at must be UTC")
        age = (datetime.now(UTC) - self.captured_at).total_seconds()
        if age < 0 or age > max_age_seconds:
            raise ValueError("operator-attested dashboard evidence is stale or future-dated")
        if file_sha256(self.dashboard_evidence_path) != self.dashboard_evidence_sha256:
            raise ValueError("operator-attested dashboard evidence file hash changed")


def _lowercase_sha(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} SHA-256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    profile: str
    workspace: str
    environment: str
    workspace_budget_usd: str
    known_metered_usage_usd: str
    verified_at: datetime
    evidence_path: Path
    evidence_sha256: str
    rates: dict[str, str]

    def __post_init__(self) -> None:
        if type(self.profile) is not str or type(self.workspace) is not str:
            raise TypeError("workspace identity fields must be exact strings")
        if not self.profile or not self.workspace:
            raise ValueError("workspace identity fields must be nonempty")
        if type(self.environment) is not str or type(self.workspace_budget_usd) is not str:
            raise TypeError("workspace environment and budget must be exact strings")
        _decimal_text(self.workspace_budget_usd, "workspace budget")
        _decimal_text(self.known_metered_usage_usd, "known metered usage")
        if (
            type(self.verified_at) is not datetime
            or self.verified_at.utcoffset() != UTC.utcoffset(None)
        ):
            raise ValueError("verified_at must be a timezone-aware UTC datetime")
        if type(self.evidence_path) is not type(Path()):
            raise TypeError("evidence_path must be an exact Path")
        if type(self.evidence_sha256) is not str or (
            self.evidence_sha256
            and (
                len(self.evidence_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.evidence_sha256)
            )
        ):
            raise ValueError("evidence SHA-256 must be empty or 64 lowercase hex characters")
        object.__setattr__(self, "rates", _validate_rates(self.rates))

    def with_evidence_hash(self) -> WorkspaceSnapshot:
        return replace(self, evidence_sha256=file_sha256(self.evidence_path))

    def to_json(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "workspace": self.workspace,
            "environment": self.environment,
            "workspace_budget_usd": self.workspace_budget_usd,
            "known_metered_usage_usd": self.known_metered_usage_usd,
            "verified_at": self.verified_at.isoformat(),
            "evidence_path": str(self.evidence_path),
            "evidence_sha256": self.evidence_sha256,
            "rates": dict(self.rates),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> WorkspaceSnapshot:
        if type(payload) is not dict:
            raise TypeError("workspace attestation must be an exact object")
        expected = {
            "profile",
            "workspace",
            "environment",
            "workspace_budget_usd",
            "known_metered_usage_usd",
            "verified_at",
            "evidence_path",
            "evidence_sha256",
            "rates",
        }
        if set(payload) != expected:
            raise ValueError("workspace attestation has missing or unexpected keys")
        string_fields = expected - {"rates"}
        if any(type(payload[key]) is not str for key in string_fields):
            raise TypeError("workspace attestation scalar fields must be exact strings")
        try:
            verified_at = datetime.fromisoformat(cast(str, payload["verified_at"]))
        except ValueError as error:
            raise ValueError("workspace attestation verified_at is invalid") from error
        return cls(
            profile=cast(str, payload["profile"]),
            workspace=cast(str, payload["workspace"]),
            environment=cast(str, payload["environment"]),
            workspace_budget_usd=cast(str, payload["workspace_budget_usd"]),
            known_metered_usage_usd=cast(str, payload["known_metered_usage_usd"]),
            verified_at=verified_at,
            evidence_path=Path(cast(str, payload["evidence_path"])),
            evidence_sha256=cast(str, payload["evidence_sha256"]),
            rates=_validate_rates(payload["rates"]),
        )


def _modal_json(
    profile: str,
    arguments: list[str],
    *,
    config_path: Path | None = None,
) -> object:
    if type(profile) is not str or profile != _PROFILE:
        raise ValueError("Modal query must use the exact ratemem-pilot profile")
    if type(arguments) is not list or any(type(item) is not str for item in arguments):
        raise TypeError("Modal arguments must be an exact list of strings")
    if tuple(arguments) not in _ALLOWED_MODAL_ARGUMENTS:
        raise ValueError("Modal command is not an allowed non-secret billing query")
    selected_config = Path.home() / ".modal.toml" if config_path is None else config_path
    _validate_modal_config(selected_config)
    environment = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR")
        if key in os.environ
    }
    environment["MODAL_CONFIG_PATH"] = str(selected_config)
    environment["MODAL_PROFILE"] = profile
    try:
        completed = subprocess.run(
            ["modal", *arguments, "--json"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    except OSError as error:
        raise RuntimeError("Modal billing command is unavailable") from error
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("Modal billing command failed or is not authorized")
    try:
        return json.loads(
            completed.stdout,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError("Modal command did not return strict JSON") from error


def _validate_modal_config(path: Path) -> None:
    if type(path) is not type(Path()):
        raise TypeError("Modal config path must be an exact Path")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PermissionError("Modal config must be an owner mode-0600 regular file") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise PermissionError("Modal config must be an owner mode-0600 regular non-symlink file")


def capture_workspace_snapshot(
    *,
    evidence_path: Path,
    confirmed_budget: str,
    config_path: Path | None = None,
) -> WorkspaceSnapshot:
    if type(confirmed_budget) is not str or confirmed_budget != _BUDGET:
        raise ValueError("workspace usage budget must be exactly USD 28.00")
    selected_config = Path.home() / ".modal.toml" if config_path is None else config_path
    _profile, workspace = _validate_profiles(
        _modal_json(_PROFILE, ["profile", "list"], config_path=selected_config)
    )
    OperatorBudgetAttestation.from_private_file(evidence_path).verify(
        expected_workspace=workspace,
        max_age_seconds=900,
    )
    billing = _validate_billing(
        _modal_json(
            _PROFILE,
            ["billing", "summary", "--for", "this month"],
            config_path=selected_config,
        )
    )
    rates = _validate_rates(
        _modal_json(_PROFILE, ["billing", "rates"], config_path=selected_config)
    )
    return WorkspaceSnapshot(
        profile=_PROFILE,
        workspace=workspace,
        environment=_ENVIRONMENT,
        workspace_budget_usd=confirmed_budget,
        known_metered_usage_usd=billing,
        verified_at=datetime.now(UTC),
        evidence_path=evidence_path,
        evidence_sha256="",
        rates=rates,
    ).with_evidence_hash()


def verify_workspace_snapshot(
    snapshot: WorkspaceSnapshot,
    *,
    expected_workspace: str,
    max_age_seconds: int,
) -> WorkspaceSnapshot:
    if type(snapshot) is not WorkspaceSnapshot:
        raise TypeError("snapshot must be an exact WorkspaceSnapshot")
    if type(expected_workspace) is not str or not expected_workspace:
        raise ValueError("expected workspace must be a nonempty exact string")
    if type(max_age_seconds) is not int or max_age_seconds <= 0:
        raise ValueError("max age must be a positive exact int")
    if snapshot.profile != _PROFILE:
        raise ValueError("Modal profile mismatch")
    if snapshot.workspace != expected_workspace:
        raise ValueError("Modal workspace mismatch")
    if snapshot.environment != _ENVIRONMENT:
        raise ValueError("Modal environment mismatch")
    if snapshot.workspace_budget_usd != _BUDGET:
        raise ValueError("Modal workspace budget mismatch")
    _decimal_text(snapshot.known_metered_usage_usd, "known metered usage")
    age = (datetime.now(UTC) - snapshot.verified_at).total_seconds()
    if age < 0:
        raise ValueError("workspace attestation is from the future")
    if age > max_age_seconds:
        raise ValueError("workspace attestation is stale")
    if snapshot.with_evidence_hash().evidence_sha256 != snapshot.evidence_sha256:
        raise ValueError("workspace-budget evidence hash changed")
    OperatorBudgetAttestation.from_private_file(snapshot.evidence_path).verify(
        expected_workspace=expected_workspace,
        max_age_seconds=max_age_seconds,
    )
    return snapshot


def verify_fresh_attestation_file(path: Path) -> WorkspaceSnapshot:
    snapshot = WorkspaceSnapshot.from_json(read_private_json(path))
    _profile, selected_workspace = _validate_profiles(
        _modal_json(_PROFILE, ["profile", "list"])
    )
    verified = verify_workspace_snapshot(
        snapshot,
        expected_workspace=selected_workspace,
        max_age_seconds=900,
    )
    metered = _validate_billing(
        _modal_json(_PROFILE, ["billing", "summary", "--for", "this month"])
    )
    rates = _validate_rates(_modal_json(_PROFILE, ["billing", "rates"]))
    if Decimal(metered) < Decimal(verified.known_metered_usage_usd):
        raise ValueError("fresh Modal metered usage must be monotonic")
    return replace(verified, known_metered_usage_usd=metered, rates=rates)
