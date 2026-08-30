from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from typing import Any

from ratemem.baselines.ledger import decode_state, empty_components, export_state
from ratemem.evaluation.canonical import canonical_json_bytes


def _wire(message: dict[str, object]) -> bytes:
    return canonical_json_bytes(message) + b"\n"


def _export(payload: bytes) -> dict[str, object]:
    return {
        "state_cbor_base64": base64.b64encode(payload).decode("ascii"),
        "state_bytes": len(payload),
    }


def _state(handles: set[str], last_event_index: int | None) -> bytes:
    components = empty_components()
    for handle in sorted(handles):
        components["handles"].append(handle)
        components["base_codes"].append(
            {"handle": handle, "data": hashlib.sha256(handle.encode()).digest()}
        )
    components["controller_state"].append(
        {"worker": "fixture", "last_event_index": last_event_index}
    )
    return export_state(components)


def _restore(payload: bytes) -> tuple[set[str], int | None]:
    components = decode_state(payload)
    handles = {str(handle) for handle in components["handles"]}
    controller = components["controller_state"]
    if len(controller) != 1 or not isinstance(controller[0], dict):
        raise ValueError("invalid fixture controller")
    event_index = controller[0].get("last_event_index")
    if event_index is not None and type(event_index) is not int:
        raise ValueError("invalid fixture event index")
    return handles, event_index


def _response(
    request: dict[str, Any],
    payload: dict[str, object],
    *,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "method_id": request["method_id"],
        "operation": request["operation"],
        "payload": payload,
        "protocol_version": request["protocol_version"],
        "request_id": request["request_id"],
        "status": status,
        "trace_id": request["trace_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("valid", "stdout_log", "extra_field", "wrong_index", "invalid_base64", "hang"),
        default="valid",
    )
    mode = parser.parse_args().mode
    handles: set[str] = set()
    state = _state(handles, None)
    snapshots: dict[str, bytes] = {}
    for raw in sys.stdin.buffer:
        request = json.loads(raw)
        if mode == "hang":
            time.sleep(30)
            continue
        operation = request["operation"]
        payload: dict[str, object]
        if operation == "initialize":
            handles.clear()
            state = _state(handles, None)
            payload = {
                **_export(state),
                "shared_trained_bytes": 123,
                "external_support_bytes": 0,
            }
            if mode == "invalid_base64":
                payload["state_cbor_base64"] = "not-base64!"
        elif operation == "event":
            event = request["payload"]["event"]
            visible = request["payload"]["visible_history"]
            if visible[-1] != event:
                raise RuntimeError("worker received a non-causal history")
            handle = event["handle"]
            kind = event["kind"]
            affected: list[str] = []
            if kind == "create" and handle not in handles:
                handles.add(handle)
                outcome = "created"
                affected = [handle]
            elif kind == "delete" and handle in handles:
                handles.remove(handle)
                outcome = "deleted"
                affected = [handle]
            elif kind == "update" and handle in handles:
                outcome = "updated"
                affected = [handle]
            elif kind == "read" and handle in handles:
                outcome = "read"
                affected = [handle]
            else:
                outcome = "stale_handle"
            state = _state(handles, event["event_index"])
            code_sha = (
                hashlib.sha256(handle.encode()).hexdigest() if handle in handles else None
            )
            generated_sha = None
            if kind == "read" and handle in handles:
                generated_sha = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "handle": handle,
                            "prompt_id": event["prompt_id"],
                            "seed": event["generation_seed"],
                        }
                    )
                ).hexdigest()
            payload = {
                **_export(state),
                "outcome": outcome,
                "affected_handles": affected,
                "evicted_handles": [],
                "decoded_code_sha256": code_sha,
                "generated_sample_sha256": generated_sha,
            }
        elif operation == "export_state":
            payload = _export(state)
        elif operation == "import_state":
            encoded = request["payload"]["state_cbor_base64"]
            state = base64.b64decode(encoded, validate=True)
            handles, _last_event = _restore(state)
            payload = _export(state)
        elif operation == "snapshot":
            token = hashlib.sha256(b"fixture-snapshot\0" + state).hexdigest()
            snapshots[token] = state
            payload = {**_export(state), "worker_snapshot_token": token}
        elif operation == "probe":
            token = request["payload"]["worker_snapshot_token"]
            probe = request["payload"]["probe"]
            snapshot_state = snapshots[token]
            snapshot_handles, _snapshot_event = _restore(snapshot_state)
            if probe["handle"] not in snapshot_handles:
                raise RuntimeError("probe handle is absent")
            generated = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "handle": probe["handle"],
                        "prompt_id": probe["prompt_id"],
                        "seed": probe["generation_seed"],
                        "snapshot": hashlib.sha256(snapshot_state).hexdigest(),
                    }
                )
            ).hexdigest()
            payload = {**_export(state), "generated_sample_sha256": generated}
        elif operation == "close":
            payload = {}
        else:
            raise RuntimeError("unknown fixture operation")
        response = _response(request, payload)
        if mode == "wrong_index":
            response["request_id"] = int(response["request_id"]) + 1
        if mode == "extra_field":
            response["unexpected"] = True
        if mode == "stdout_log":
            sys.stdout.buffer.write(_wire({"log": "invalid protocol stdout"}))
        sys.stdout.buffer.write(_wire(response))
        sys.stdout.buffer.flush()
        if operation == "close":
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
