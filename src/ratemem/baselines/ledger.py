"""Canonical state export and host-owned exact byte accounting."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cbor2
import numpy as np
from numpy.typing import NDArray

from ratemem.baselines.protocol import ExactByteLedger

ONLINE_COMPONENT_NAMES = (
    "base_codes",
    "packet_payloads",
    "packet_hashes",
    "incidences_gains",
    "feature_cache",
    "optional_tokens",
    "handles",
    "usage_age",
    "reference_counts",
    "controller_state",
    "allocator_state",
    "checksums",
    "alignment",
)


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    array: NDArray[np.generic]


def tensor_record(record: TensorRecord) -> dict[str, object]:
    """Encode one tensor without pickle, strides, or native-endian ambiguity."""

    if type(record) is not TensorRecord:
        raise TypeError("record must be an exact TensorRecord")
    if not record.name:
        raise ValueError("tensor name must be non-empty")
    array = np.ascontiguousarray(record.array)
    dtype = array.dtype.newbyteorder("<")
    little = array.astype(dtype, copy=False)
    return {
        "name": record.name,
        "dtype": dtype.str,
        "shape": list(array.shape),
        "data": little.tobytes(order="C"),
    }


def component_blob(name: str, records: Sequence[object]) -> bytes:
    if name not in ONLINE_COMPONENT_NAMES:
        raise ValueError(f"unknown online-state component: {name}")
    normalized = [tensor_record(row) if type(row) is TensorRecord else row for row in records]
    return cbor2.dumps({"component": name, "records": normalized}, canonical=True)


def export_state(components: Mapping[str, Sequence[object]]) -> bytes:
    """Export every online component into one deterministic canonical-CBOR envelope."""

    supplied = set(components)
    required = set(ONLINE_COMPONENT_NAMES)
    if supplied != required:
        missing = sorted(required - supplied)
        extra = sorted(supplied - required)
        raise ValueError(f"state components mismatch missing={missing} extra={extra}")
    framed = {
        name: component_blob(name, components[name])
        for name in ONLINE_COMPONENT_NAMES
    }
    return cbor2.dumps(
        {"format": "ratemem-baseline-cbor-v1", "components": framed},
        canonical=True,
    )


def decode_state(payload: bytes) -> dict[str, tuple[object, ...]]:
    """Validate and decode canonical state without accepting alternate encodings."""

    if type(payload) is not bytes or not payload:
        raise ValueError("online state must be non-empty bytes")
    try:
        decoded = cbor2.loads(payload)
    except Exception as error:
        raise ValueError("online state is not valid CBOR") from error
    if type(decoded) is not dict or decoded.get("format") != "ratemem-baseline-cbor-v1":
        raise ValueError("online state serializer identity mismatch")
    framed = decoded.get("components")
    if type(framed) is not dict or set(framed) != set(ONLINE_COMPONENT_NAMES):
        raise ValueError("online state component set mismatch")
    if cbor2.dumps(decoded, canonical=True) != payload:
        raise ValueError("online state is not canonical CBOR")
    result: dict[str, tuple[object, ...]] = {}
    for name in ONLINE_COMPONENT_NAMES:
        blob = framed[name]
        if type(blob) is not bytes:
            raise ValueError("online state component frame must be bytes")
        component = cbor2.loads(blob)
        if (
            type(component) is not dict
            or component.get("component") != name
            or type(component.get("records")) is not list
            or cbor2.dumps(component, canonical=True) != blob
        ):
            raise ValueError(f"online state component frame is invalid: {name}")
        result[name] = tuple(component["records"])
    return result


def ledger_from_export(
    payload: bytes,
    shared_trained_bytes: int,
    external_support_bytes: int,
) -> ExactByteLedger:
    """Recompute accounting from bytes; comparator-supplied totals are never trusted."""

    if type(shared_trained_bytes) is not int or shared_trained_bytes < 0:
        raise ValueError("shared trained bytes must be a nonnegative integer")
    if type(external_support_bytes) is not int or external_support_bytes < 0:
        raise ValueError("external support bytes must be a nonnegative integer")
    decoded = cbor2.loads(payload)
    decode_state(payload)
    framed = decoded["components"]
    component_bytes = {
        name: len(framed[name])
        for name in ONLINE_COMPONENT_NAMES
    }
    envelope_bytes = len(payload) - sum(component_bytes.values())
    if envelope_bytes < 0:
        raise ValueError("online state framing length is invalid")
    component_bytes["checksums"] += envelope_bytes
    return ExactByteLedger(
        serializer_id="ratemem-baseline-cbor-v1",
        online_state_bytes=len(payload),
        online_state_sha256=hashlib.sha256(payload).hexdigest(),
        component_bytes=component_bytes,
        shared_trained_bytes=shared_trained_bytes,
        external_support_bytes=external_support_bytes,
    )


def empty_components() -> dict[str, list[object]]:
    """Return a fresh complete component map for adapter implementations."""

    return {name: [] for name in ONLINE_COMPONENT_NAMES}


__all__ = [
    "ONLINE_COMPONENT_NAMES",
    "TensorRecord",
    "component_blob",
    "decode_state",
    "empty_components",
    "export_state",
    "ledger_from_export",
    "tensor_record",
]
