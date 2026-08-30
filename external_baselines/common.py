"""Protocol server shared by thin, source-pinned external baseline runners."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from ratemem.baselines.external_jsonl import (
    ExternalRequestMessage,
    canonical_line,
    state_export_payload,
)


@dataclass(frozen=True, slots=True)
class WorkerEventResult:
    state: bytes
    outcome: str
    affected_handles: tuple[str, ...]
    evicted_handles: tuple[str, ...]
    decoded_code_sha256: str | None
    generated_sample_sha256: str | None


class ExternalBackend(Protocol):
    shared_trained_bytes: int
    external_support_bytes: int

    def initialize(self, contract: Mapping[str, object]) -> bytes: ...

    def apply_event(
        self,
        event: Mapping[str, object],
        visible_history: Sequence[Mapping[str, object]],
    ) -> WorkerEventResult: ...

    def export_state(self) -> bytes: ...

    def import_state(self, state: bytes) -> None: ...

    def copy_snapshot(self) -> tuple[str, bytes]: ...

    def score_probe(
        self,
        worker_snapshot_token: str,
        probe: Mapping[str, object],
    ) -> tuple[str, bytes]: ...

    def close(self) -> None: ...


def _load_factory(specification: str, checkout: Path) -> ExternalBackend:
    if specification.count(":") != 1:
        raise ValueError("backend factory must use module:callable syntax")
    module_name, attribute = specification.split(":")
    if not module_name or not attribute:
        raise ValueError("backend factory module and callable are required")
    module = importlib.import_module(module_name)
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError("backend factory module has no auditable source file")
    source_path = Path(source).resolve(strict=True)
    if not source_path.is_relative_to(checkout):
        raise RuntimeError("backend factory must be loaded from the pinned checkout")
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError("backend factory attribute is not callable")
    backend = factory()
    return cast(ExternalBackend, backend)


def _response(
    request: ExternalRequestMessage,
    payload: dict[str, object],
    *,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "method_id": request.method_id,
        "operation": request.operation,
        "payload": payload,
        "protocol_version": request.protocol_version,
        "request_id": request.request_id,
        "status": status,
        "trace_id": request.trace_id,
    }


def _dispatch(
    backend: ExternalBackend,
    request: ExternalRequestMessage,
) -> dict[str, object]:
    payload = request.payload
    operation = request.operation
    if operation == "initialize":
        if set(payload) != {"contract"} or not isinstance(payload["contract"], dict):
            raise ValueError("initialize payload is invalid")
        state = backend.initialize(payload["contract"])
        return {
            **state_export_payload(state),
            "shared_trained_bytes": backend.shared_trained_bytes,
            "external_support_bytes": backend.external_support_bytes,
        }
    if operation == "event":
        if set(payload) != {"event", "visible_history"}:
            raise ValueError("event payload is invalid")
        event = payload["event"]
        history = payload["visible_history"]
        if not isinstance(event, dict) or not isinstance(history, list):
            raise ValueError("event payload types are invalid")
        if any(not isinstance(row, dict) for row in history):
            raise ValueError("event visible history is invalid")
        result = backend.apply_event(event, history)
        if type(result) is not WorkerEventResult:
            raise TypeError("external backend must return an exact WorkerEventResult")
        return {
            **state_export_payload(result.state),
            "outcome": result.outcome,
            "affected_handles": list(result.affected_handles),
            "evicted_handles": list(result.evicted_handles),
            "decoded_code_sha256": result.decoded_code_sha256,
            "generated_sample_sha256": result.generated_sample_sha256,
        }
    if operation == "export_state":
        if payload:
            raise ValueError("export_state payload must be empty")
        return state_export_payload(backend.export_state())
    if operation == "import_state":
        if set(payload) != {"state_cbor_base64", "state_bytes"}:
            raise ValueError("import_state payload is invalid")
        import base64

        state = base64.b64decode(str(payload["state_cbor_base64"]), validate=True)
        if payload["state_bytes"] != len(state):
            raise ValueError("import_state byte count differs")
        backend.import_state(state)
        return state_export_payload(backend.export_state())
    if operation == "snapshot":
        if payload:
            raise ValueError("snapshot payload must be empty")
        token, state = backend.copy_snapshot()
        return {**state_export_payload(state), "worker_snapshot_token": token}
    if operation == "probe":
        if set(payload) != {"worker_snapshot_token", "probe"}:
            raise ValueError("probe payload is invalid")
        token = payload["worker_snapshot_token"]
        probe = payload["probe"]
        if type(token) is not str or not isinstance(probe, dict):
            raise ValueError("probe payload types are invalid")
        generated_sha, state = backend.score_probe(token, probe)
        return {
            **state_export_payload(state),
            "generated_sample_sha256": generated_sha,
        }
    if operation == "close":
        if payload:
            raise ValueError("close payload must be empty")
        backend.close()
        return {}
    raise ValueError(f"unsupported operation: {operation}")


def serve_external(method_id: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-factory", required=True)
    arguments = parser.parse_args()
    checkout = Path.cwd().resolve(strict=True)
    backend = _load_factory(arguments.backend_factory, checkout)
    for raw in sys.stdin.buffer:
        request: ExternalRequestMessage | None = None
        try:
            decoded: Any = json.loads(raw)
            request = ExternalRequestMessage.model_validate(decoded)
            if canonical_line(request) != raw:
                raise ValueError("request is not canonical JSONL")
            if request.method_id != method_id:
                raise ValueError("request method differs from this runner")
            payload = _dispatch(backend, request)
            response = _response(request, payload)
        except Exception as error:
            if request is None:
                print(f"unframed request failure: {type(error).__name__}: {error}", file=sys.stderr)
                return 2
            response = _response(
                request,
                {"error_type": type(error).__name__, "message": str(error)},
                status="error",
            )
        sys.stdout.buffer.write(canonical_line(response))
        sys.stdout.buffer.flush()
        if request is not None and request.operation == "close":
            return 0
    return 2


__all__ = ["ExternalBackend", "WorkerEventResult", "serve_external"]
