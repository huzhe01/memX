"""Strict process-isolated JSONL bridge for pinned external baselines."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import select
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, TypeAdapter

from ratemem.baselines.ledger import decode_state, ledger_from_export
from ratemem.baselines.protocol import (
    CausalEventView,
    EventReceipt,
    ExactByteLedger,
    FrozenComparisonContract,
    MethodSnapshot,
    ProbeResult,
    validate_operational_event_order,
)
from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.traces import LifecycleEvent, ProbeEvent

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_ALLOWED_ENVIRONMENT = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    }
)
_OPERATIONS = (
    "initialize",
    "event",
    "snapshot",
    "probe",
    "export_state",
    "import_state",
    "close",
)
_OUTCOMES = frozenset(
    {
        "created",
        "updated",
        "read",
        "deleted",
        "rejected",
        "evicted",
        "stale_handle",
    }
)
Operation = Literal[
    "initialize",
    "event",
    "snapshot",
    "probe",
    "export_state",
    "import_state",
    "close",
]


class ExternalProtocolError(RuntimeError):
    """Raised when an external worker violates framing, identity, or state rules."""


class ExternalWorkerManifest(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    method_id: str = Field(min_length=1, max_length=128)
    role: Literal["causal", "latency_control"] = "causal"
    command: tuple[str, ...]
    checkout: Path
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment: dict[str, str]
    timeout_seconds: PositiveFloat = 60.0
    termination_grace_seconds: PositiveFloat = 1.0
    maximum_line_bytes: PositiveInt = 16 * 1024 * 1024

    def model_post_init(self, __context: Any) -> None:
        if not self.command or any(not value or "\x00" in value for value in self.command):
            raise ValueError("external worker command must be non-empty argv")
        if not set(self.environment) <= _ALLOWED_ENVIRONMENT:
            raise ValueError("external worker environment contains a non-allowlisted key")
        if any("\x00" in key or "\x00" in value for key, value in self.environment.items()):
            raise ValueError("external worker environment contains NUL")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds > 3600.0:
            raise ValueError("external worker timeout must be finite and at most one hour")
        if self.maximum_line_bytes > 64 * 1024 * 1024:
            raise ValueError("external worker line cap exceeds 64 MiB")


class ExternalRequestMessage(BaseModel):
    model_config = _MODEL_CONFIG

    protocol_version: Literal["1.0"]
    request_id: int = Field(ge=0)
    method_id: str
    trace_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: Operation
    payload: dict[str, Any]


class ExternalResponseMessage(BaseModel):
    model_config = _MODEL_CONFIG

    protocol_version: Literal["1.0"]
    request_id: int = Field(ge=0)
    method_id: str
    trace_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: Operation
    status: Literal["ok", "error"]
    payload: dict[str, Any]


def external_message_schema() -> dict[str, object]:
    return cast(
        dict[str, object],
        TypeAdapter(ExternalRequestMessage | ExternalResponseMessage).json_schema(),
    )


def canonical_line(message: Mapping[str, object] | BaseModel) -> bytes:
    payload = message.model_dump(mode="json") if isinstance(message, BaseModel) else dict(message)
    return canonical_json_bytes(payload) + b"\n"


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ExternalProtocolError(
            f"{label} fields differ missing={sorted(expected - set(payload))} "
            f"extra={sorted(set(payload) - expected)}"
        )


def _decode_state_export(payload: Mapping[str, Any]) -> bytes:
    _require_exact_keys(payload, {"state_cbor_base64", "state_bytes"}, "state export")
    encoded = payload.get("state_cbor_base64")
    declared = payload.get("state_bytes")
    if type(encoded) is not str or type(declared) is not int or declared < 0:
        raise ExternalProtocolError("state export types are invalid")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ExternalProtocolError("invalid state export base64") from error
    if declared != len(decoded):
        raise ExternalProtocolError("worker state byte declaration disagrees with export")
    try:
        decode_state(decoded)
    except ValueError as error:
        raise ExternalProtocolError("worker exported invalid canonical state") from error
    return decoded


def state_export_payload(payload: bytes) -> dict[str, object]:
    """Build the canonical wire representation of a host-validated state."""

    decode_state(payload)
    return {
        "state_cbor_base64": base64.b64encode(payload).decode("ascii"),
        "state_bytes": len(payload),
    }


@dataclass(frozen=True, slots=True)
class ExternalLaunchView:
    argv: tuple[str, ...]
    checkout: Path
    environment: dict[str, str]
    shell: Literal[False] = False


@dataclass(frozen=True, slots=True)
class _WorkerSnapshot:
    worker_token: str
    state: bytes


class ExternalJsonlAdapter:
    """Host-owned BaselineAdapter facade over one strict persistent worker."""

    def __init__(self, manifest: ExternalWorkerManifest) -> None:
        if type(manifest) is not ExternalWorkerManifest:
            raise TypeError("manifest must be an exact ExternalWorkerManifest")
        self.manifest = manifest
        self.method_id = manifest.method_id
        self.role = manifest.role
        self.shared_trained_bytes = 0
        self.external_support_bytes = 0
        self._contract: FrozenComparisonContract | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: BinaryIO | None = None
        self._request_id = 0
        self._stdout_buffer = bytearray()
        self._state: bytes | None = None
        self._last_event_index: int | None = None
        self._snapshots: dict[str, _WorkerSnapshot] = {}
        self._poisoned = False
        self._closed = False

    def inspect_launch(self) -> ExternalLaunchView:
        return ExternalLaunchView(
            argv=self.manifest.command,
            checkout=self.manifest.checkout,
            environment=dict(self.manifest.environment),
        )

    def _validate_checkout(self) -> Path:
        checkout = self.manifest.checkout
        if checkout.is_symlink() or not checkout.is_dir():
            raise ExternalProtocolError("external checkout must be a real directory")
        return checkout.resolve(strict=True)

    def _launch(self) -> None:
        if self._process is not None:
            raise RuntimeError("external worker already launched")
        checkout = self._validate_checkout()
        stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            process = subprocess.Popen(
                list(self.manifest.command),
                shell=False,
                cwd=checkout,
                env=dict(self.manifest.environment),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                close_fds=True,
            )
        except OSError as error:
            stderr.close()
            raise ExternalProtocolError("external worker failed to launch") from error
        if process.stdin is None or process.stdout is None:
            process.kill()
            stderr.close()
            raise ExternalProtocolError("external worker pipes are unavailable")
        self._process = process
        self._stderr = stderr

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.manifest.termination_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.manifest.termination_grace_seconds)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        self._process = None

    def _fail(self, message: str, error: BaseException | None = None) -> ExternalProtocolError:
        self._poisoned = True
        self._terminate()
        if error is None:
            return ExternalProtocolError(message)
        return ExternalProtocolError(message).with_traceback(error.__traceback__)

    def _read_line(self) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise self._fail("external worker is not running")
        descriptor = process.stdout.fileno()
        deadline = time.monotonic() + self.manifest.timeout_seconds
        while b"\n" not in self._stdout_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise self._fail("external worker response deadline exceeded")
            ready, _writable, _errors = select.select([descriptor], [], [], remaining)
            if not ready:
                raise self._fail("external worker response deadline exceeded")
            chunk = os.read(descriptor, min(65536, self.manifest.maximum_line_bytes + 1))
            if not chunk:
                raise self._fail("external worker closed stdout before a response")
            self._stdout_buffer.extend(chunk)
            if len(self._stdout_buffer) > self.manifest.maximum_line_bytes:
                raise self._fail("external worker response exceeds maximum line bytes")
        newline = self._stdout_buffer.index(10)
        line = bytes(self._stdout_buffer[: newline + 1])
        del self._stdout_buffer[: newline + 1]
        if self._stdout_buffer:
            raise self._fail("external worker emitted additional stdout bytes")
        ready, _writable, _errors = select.select([descriptor], [], [], 0.0)
        if ready:
            extra = os.read(descriptor, self.manifest.maximum_line_bytes + 1)
            if extra:
                raise self._fail("external worker emitted additional stdout bytes")
        return line

    def _request(self, operation: Operation, payload: dict[str, Any]) -> dict[str, Any]:
        if self._poisoned or self._closed:
            raise ExternalProtocolError("external worker is poisoned or closed")
        process = self._process
        contract = self._contract
        if process is None or process.stdin is None or contract is None:
            raise ExternalProtocolError("external worker is not initialized")
        request = ExternalRequestMessage(
            protocol_version="1.0",
            request_id=self._request_id,
            method_id=self.method_id,
            trace_id=contract.trace_id,
            operation=operation,
            payload=payload,
        )
        wire = canonical_line(request)
        try:
            process.stdin.write(wire)
            process.stdin.flush()
            raw = self._read_line()
            decoded: Any = json.loads(raw)
            response = ExternalResponseMessage.model_validate(decoded)
        except (
            BrokenPipeError,
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            raise self._fail("external worker returned an invalid response", error) from error
        if canonical_line(response) != raw:
            raise self._fail("external worker response is not canonical JSONL")
        expected_identity = (
            request.protocol_version,
            request.request_id,
            request.method_id,
            request.trace_id,
            request.operation,
        )
        observed_identity = (
            response.protocol_version,
            response.request_id,
            response.method_id,
            response.trace_id,
            response.operation,
        )
        if observed_identity != expected_identity:
            raise self._fail("external worker response identity differs from the request")
        self._request_id += 1
        if response.status == "error":
            try:
                _require_exact_keys(response.payload, {"error_type", "message"}, "error")
                error_type = response.payload["error_type"]
                message = response.payload["message"]
                if type(error_type) is not str or type(message) is not str:
                    raise ExternalProtocolError("external worker error payload types are invalid")
            except ExternalProtocolError as error:
                raise self._fail(str(error), error) from error
            raise self._fail(f"external worker failed: {error_type}: {message}")
        return response.payload

    def initialize(self, contract: FrozenComparisonContract) -> None:
        if self._contract is not None or self._closed:
            raise RuntimeError("external adapter cannot be initialized twice or after close")
        self._contract = contract
        self._launch()
        try:
            payload = self._request("initialize", {"contract": contract.model_dump(mode="json")})
            _require_exact_keys(
                payload,
                {
                    "state_cbor_base64",
                    "state_bytes",
                    "shared_trained_bytes",
                    "external_support_bytes",
                },
                "initialize",
            )
            shared = payload["shared_trained_bytes"]
            support = payload["external_support_bytes"]
            if type(shared) is not int or shared < 0 or type(support) is not int or support < 0:
                raise ExternalProtocolError("external byte disclosures are invalid")
            state = _decode_state_export(
                {
                    "state_cbor_base64": payload["state_cbor_base64"],
                    "state_bytes": payload["state_bytes"],
                }
            )
            self.shared_trained_bytes = shared
            self.external_support_bytes = support
            self._state = state
            if len(state) > contract.byte_budget:
                raise ExternalProtocolError("external empty state exceeds the byte budget")
        except BaseException:
            self._terminate()
            self._contract = None
            self._poisoned = True
            raise

    def _require_active(self) -> tuple[FrozenComparisonContract, bytes]:
        if self._contract is None or self._state is None or self._poisoned or self._closed:
            raise RuntimeError("external adapter is not active")
        return self._contract, self._state

    def export_online_state(self) -> bytes:
        _contract, state = self._require_active()
        payload = self._request("export_state", {})
        exported = _decode_state_export(payload)
        if exported != state:
            raise self._fail("external worker mutated state outside an event")
        return state

    def import_online_state(self, payload: bytes) -> None:
        self._require_active()
        try:
            decode_state(payload)
        except ValueError as error:
            raise ValueError("external imported state is invalid") from error
        response = self._request("import_state", state_export_payload(payload))
        restored = _decode_state_export(response)
        if restored != payload:
            raise self._fail("external worker changed imported canonical state")
        self._state = payload
        components = decode_state(payload)
        controller = components["controller_state"]
        inferred: int | None = None
        if len(controller) == 1 and isinstance(controller[0], dict):
            value = controller[0].get("last_event_index")
            if value is None or (type(value) is int and value >= 0):
                inferred = value
        self._last_event_index = inferred

    def state_ledger(self) -> ExactByteLedger:
        _contract, state = self._require_active()
        return ledger_from_export(
            state,
            self.shared_trained_bytes,
            self.external_support_bytes,
        )

    def apply_event(self, event: LifecycleEvent, view: CausalEventView) -> EventReceipt:
        contract, before_state = self._require_active()
        if isinstance(event, ProbeEvent):
            raise TypeError("probe events must use score_probe")
        if event.event_index != view.current_index or view.at(event.event_index) != event:
            raise ValueError("event and causal view are not aligned")
        validate_operational_event_order(self._last_event_index, event, view)
        before = ledger_from_export(
            before_state,
            self.shared_trained_bytes,
            self.external_support_bytes,
        )
        response = self._request(
            "event",
            {
                "event": event.model_dump(mode="json"),
                "visible_history": [row.model_dump(mode="json") for row in view.history()],
            },
        )
        expected = {
            "state_cbor_base64",
            "state_bytes",
            "outcome",
            "affected_handles",
            "evicted_handles",
            "decoded_code_sha256",
            "generated_sample_sha256",
        }
        _require_exact_keys(response, expected, "event")
        state = _decode_state_export(
            {
                "state_cbor_base64": response["state_cbor_base64"],
                "state_bytes": response["state_bytes"],
            }
        )
        outcome = response["outcome"]
        if type(outcome) is not str or outcome not in _OUTCOMES:
            raise self._fail("external event outcome is invalid")
        affected = self._validate_handles(response["affected_handles"], "affected handles")
        evicted = self._validate_handles(response["evicted_handles"], "evicted handles")
        decoded_sha = self._optional_sha(response["decoded_code_sha256"], "decoded code")
        generated_sha = self._optional_sha(
            response["generated_sample_sha256"],
            "generated sample",
        )
        self._state = state
        self._last_event_index = event.event_index
        after = self.state_ledger()
        if after.online_state_bytes > contract.byte_budget:
            raise self._fail("external worker exceeded the exact online byte budget")
        input_sha = hashlib.sha256(
            canonical_json_bytes(event.model_dump(mode="json"))
        ).hexdigest()
        return EventReceipt(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=event.event_index,
            event_kind=event.kind,
            input_commitment_sha256=input_sha,
            method_state_sha256_before=before.online_state_sha256,
            method_state_sha256_after=after.online_state_sha256,
            candidate_stream_sha256=contract.candidate_stream_sha256,
            outcome=cast(Any, outcome),
            affected_handles=affected,
            evicted_handles=evicted,
            decoded_code_sha256=decoded_sha,
            generated_sample_sha256=generated_sha,
            ledger=after,
        )

    @staticmethod
    def _validate_handles(value: object, label: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(type(row) is not str or not row for row in value):
            raise ExternalProtocolError(f"external {label} are invalid")
        handles = tuple(cast(list[str], value))
        if handles != tuple(sorted(set(handles))):
            raise ExternalProtocolError(f"external {label} must be sorted and unique")
        return handles

    @staticmethod
    def _optional_sha(value: object, label: str) -> str | None:
        if value is None:
            return None
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ExternalProtocolError(f"external {label} SHA-256 is invalid")
        return value

    def copy_snapshot(self) -> MethodSnapshot:
        contract, state = self._require_active()
        if self._last_event_index is None:
            raise RuntimeError("cannot snapshot before the first event")
        response = self._request("snapshot", {})
        _require_exact_keys(
            response,
            {"worker_snapshot_token", "state_cbor_base64", "state_bytes"},
            "snapshot",
        )
        worker_token = response["worker_snapshot_token"]
        if type(worker_token) is not str or not worker_token:
            raise self._fail("external snapshot token is invalid")
        snapshot_state = _decode_state_export(
            {
                "state_cbor_base64": response["state_cbor_base64"],
                "state_bytes": response["state_bytes"],
            }
        )
        if snapshot_state != state:
            raise self._fail("external snapshot differs from current state")
        state_sha = hashlib.sha256(state).hexdigest()
        host_token = hashlib.sha256(
            b"external-snapshot-v1\0"
            + self.method_id.encode("utf-8")
            + b"\0"
            + worker_token.encode("utf-8")
            + b"\0"
            + state
        ).hexdigest()
        self._snapshots[host_token] = _WorkerSnapshot(worker_token, state)
        return MethodSnapshot(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            event_index=self._last_event_index,
            state_sha256=state_sha,
            online_state_bytes=len(state),
            opaque_snapshot_token=host_token,
        )

    def score_probe(self, snapshot: MethodSnapshot, probe: ProbeEvent) -> ProbeResult:
        contract, current_state = self._require_active()
        if snapshot.method_id != self.method_id or snapshot.trace_id != contract.trace_id:
            raise ValueError("probe snapshot belongs to a different method or trace")
        stored = self._snapshots.get(snapshot.opaque_snapshot_token)
        if stored is None or hashlib.sha256(stored.state).hexdigest() != snapshot.state_sha256:
            raise ValueError("probe snapshot token is unknown or corrupted")
        response = self._request(
            "probe",
            {
                "worker_snapshot_token": stored.worker_token,
                "probe": probe.model_dump(mode="json"),
            },
        )
        _require_exact_keys(
            response,
            {"generated_sample_sha256", "state_cbor_base64", "state_bytes"},
            "probe",
        )
        generated_sha = self._optional_sha(
            response["generated_sample_sha256"],
            "generated sample",
        )
        if generated_sha is None:
            raise self._fail("external probe omitted generated sample SHA-256")
        after_state = _decode_state_export(
            {
                "state_cbor_base64": response["state_cbor_base64"],
                "state_bytes": response["state_bytes"],
            }
        )
        if after_state != current_state:
            raise self._fail("external probe mutated online state")
        input_sha = hashlib.sha256(
            canonical_json_bytes(probe.model_dump(mode="json"))
        ).hexdigest()
        return ProbeResult(
            method_id=self.method_id,
            trace_id=contract.trace_id,
            probe_event_index=probe.event_index,
            snapshot_state_sha256=snapshot.state_sha256,
            input_commitment_sha256=input_sha,
            generated_sample_sha256=generated_sha,
            update_usage=False,
        )

    def stderr_sha256(self) -> str:
        stderr = self._stderr
        if stderr is None:
            return hashlib.sha256(b"").hexdigest()
        position = stderr.tell()
        stderr.seek(0)
        payload = stderr.read()
        stderr.seek(position)
        return hashlib.sha256(payload).hexdigest()

    def close(self) -> None:
        if self._closed:
            return
        if self._process is not None and not self._poisoned and self._contract is not None:
            try:
                response = self._request("close", {})
                _require_exact_keys(response, set(), "close")
            except ExternalProtocolError:
                self._poisoned = True
                raise
            finally:
                self._terminate()
        else:
            self._terminate()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        self._snapshots.clear()
        self._state = None
        self._contract = None
        self._closed = True


__all__ = [
    "ExternalJsonlAdapter",
    "ExternalLaunchView",
    "ExternalProtocolError",
    "ExternalRequestMessage",
    "ExternalResponseMessage",
    "ExternalWorkerManifest",
    "canonical_line",
    "external_message_schema",
    "state_export_payload",
]
