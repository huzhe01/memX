from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ratemem.state.model import Packet
from ratemem.state.serialization import packet_from_payload

_PACKET_HEADER = struct.Struct("<II")
FloatArray: TypeAlias = NDArray[np.float32]


def _encode_residual(group: int, start: int, values: FloatArray) -> bytes:
    body = np.asarray(values, dtype="<f2").tobytes(order="C")
    return _PACKET_HEADER.pack(group, start) + body


def _decode_residual(payload: bytes) -> tuple[int, int, FloatArray]:
    if len(payload) < _PACKET_HEADER.size:
        raise ValueError("truncated residual packet")
    group, start = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
    values = np.frombuffer(payload[_PACKET_HEADER.size :], dtype="<f2").astype(
        np.float32
    )
    return group, start, values


@dataclass(frozen=True, slots=True)
class EncodedPacket:
    group: int
    packet: Packet


@dataclass(frozen=True, slots=True)
class EncodedCode:
    handle: str
    shape: tuple[int, ...]
    base_payload: bytes
    packets: tuple[EncodedPacket, ...]

    def decode(self, packet_count: int) -> FloatArray:
        if not 0 <= packet_count <= len(self.packets):
            raise ValueError("packet_count is outside the progressive stream")
        with io.BytesIO(self.base_payload) as stream:
            base = cast(
                FloatArray, np.load(stream, allow_pickle=False).astype(np.float32)
            )
        output = base.reshape(-1).copy()
        for encoded in self.packets[:packet_count]:
            group, start, values = _decode_residual(encoded.packet.payload)
            if group != encoded.group:
                raise ValueError("packet group mismatch")
            output[start : start + len(values)] += values
        return cast(FloatArray, output.reshape(self.shape))


class ProgressiveCodec:
    def __init__(self, group_size: int) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.group_size = group_size

    def encode(self, handle: str, code: FloatArray) -> EncodedCode:
        flat = cast(FloatArray, np.asarray(code, dtype=np.float32).reshape(-1))
        if flat.size == 0 or not np.all(np.isfinite(flat)):
            raise ValueError("code must be finite and nonempty")
        scale = max(
            float(np.max(np.abs(flat))) / 127.0,
            float(np.finfo(np.float32).eps),
        )
        base = (
            np.round(flat / scale).clip(-127, 127).astype(np.int8).astype(np.float32)
            * scale
        )
        base_stream = io.BytesIO()
        np.save(base_stream, base.reshape(code.shape).astype(np.float16), allow_pickle=False)
        decoded_base = base.astype(np.float16).astype(np.float32)
        residual = flat - decoded_base
        packets: list[EncodedPacket] = []
        for group, start in enumerate(range(0, len(flat), self.group_size)):
            payload = _encode_residual(group, start, residual[start : start + self.group_size])
            packets.append(EncodedPacket(group, packet_from_payload(payload)))
        return EncodedCode(handle, tuple(code.shape), base_stream.getvalue(), tuple(packets))
