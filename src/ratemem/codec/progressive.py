from __future__ import annotations

import hashlib
import io
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ratemem.state.model import Packet
from ratemem.state.serialization import packet_from_payload

_PACKET_HEADER = struct.Struct("<II")
_FLOAT16_MAX = float(np.finfo(np.float16).max)
FloatArray: TypeAlias = NDArray[np.float32]
NpyHeader: TypeAlias = tuple[tuple[int, ...], bool, np.dtype[Any]]
_READ_MAGIC = cast(Callable[[io.BytesIO], tuple[int, int]], np.lib.format.read_magic)
_READ_HEADER_1 = cast(Callable[[io.BytesIO], NpyHeader], np.lib.format.read_array_header_1_0)
_READ_HEADER_2 = cast(Callable[[io.BytesIO], NpyHeader], np.lib.format.read_array_header_2_0)


def _encode_residual(group: int, start: int, values: FloatArray) -> bytes:
    half_values = np.asarray(values, dtype="<f2")
    if not np.all(np.isfinite(half_values)):
        raise ValueError("residual must remain finite in float16")
    body = half_values.tobytes(order="C")
    return _PACKET_HEADER.pack(group, start) + body


def _decode_residual(payload: bytes) -> tuple[int, int, FloatArray]:
    if len(payload) < _PACKET_HEADER.size:
        raise ValueError("truncated residual packet")
    group, start = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
    values = np.frombuffer(payload[_PACKET_HEADER.size :], dtype="<f2").astype(np.float32)
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
    group_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "base_payload", bytes(self.base_payload))
        object.__setattr__(self, "packets", tuple(self.packets))
        if self.group_size < 1:
            raise ValueError("group_size must be positive")

    def _element_count(self) -> int:
        if any(type(dimension) is not int or dimension < 0 for dimension in self.shape):
            raise ValueError("declared shape must contain nonnegative integers")
        element_count = math.prod(self.shape)
        if element_count < 1:
            raise ValueError("declared shape must contain data")
        return element_count

    def _decode_base_payload(self) -> FloatArray:
        stream = io.BytesIO(self.base_payload)
        try:
            version = _READ_MAGIC(stream)
            if version == (1, 0):
                stored_shape, fortran_order, dtype = _READ_HEADER_1(stream)
            elif version == (2, 0):
                stored_shape, fortran_order, dtype = _READ_HEADER_2(stream)
            else:
                raise ValueError("unsupported NPY base-payload version")
        except Exception as error:
            raise ValueError("malformed base payload") from error

        if dtype != np.dtype("<f2") or dtype.hasobject:
            raise ValueError("base payload must contain little-endian float16")
        if fortran_order:
            raise ValueError("base payload must use C order")
        if tuple(stored_shape) != self.shape:
            raise ValueError("stored base shape does not match the declared shape")
        element_count = self._element_count()
        data_offset = stream.tell()
        expected_data_bytes = element_count * dtype.itemsize
        if len(self.base_payload) - data_offset != expected_data_bytes:
            raise ValueError("base payload data length is not canonical")
        half_values = np.frombuffer(
            self.base_payload,
            dtype="<f2",
            count=element_count,
            offset=data_offset,
        )
        if not np.all(np.isfinite(half_values)):
            raise ValueError("base payload values must be finite")
        return cast(
            FloatArray,
            half_values.reshape(self.shape, order="C").astype(np.float32),
        )

    def _validated_residuals(self, packet_count: int) -> tuple[FloatArray, ...]:
        element_count = self._element_count()
        expected_packet_count = (element_count + self.group_size - 1) // self.group_size
        if len(self.packets) != expected_packet_count:
            raise ValueError("packet count does not match the declared shape")
        selected_packets = self.packets[:packet_count]
        packet_ids = [encoded.packet.packet_id for encoded in selected_packets]
        if len(set(packet_ids)) != len(packet_ids):
            raise ValueError("progressive stream contains a repeated packet")

        half_residuals: list[NDArray[np.float16]] = []
        for position, encoded in enumerate(selected_packets):
            if encoded.group != position:
                raise ValueError("packet tuple group is not canonical")
            payload = encoded.packet.payload
            if hashlib.sha256(payload).hexdigest() != encoded.packet.packet_id:
                raise ValueError("packet hash mismatch")
            expected_start = position * self.group_size
            expected_elements = min(self.group_size, element_count - expected_start)
            expected_payload_size = _PACKET_HEADER.size + expected_elements * 2
            if len(payload) != expected_payload_size:
                raise ValueError("residual packet body has a noncanonical size")
            group, start = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
            if group != encoded.group:
                raise ValueError("packet header group mismatch")
            if start != expected_start:
                raise ValueError("packet start is not canonical")
            half_values = np.frombuffer(payload[_PACKET_HEADER.size :], dtype="<f2")
            if not np.all(np.isfinite(half_values)):
                raise ValueError("residual packet values must be finite")
            half_residuals.append(half_values)
        return tuple(values.astype(np.float32) for values in half_residuals)

    def decode(self, packet_count: int) -> FloatArray:
        if not 0 <= packet_count <= len(self.packets):
            raise ValueError("packet_count is outside the progressive stream")
        residuals = self._validated_residuals(packet_count)
        base = self._decode_base_payload()
        output = base.reshape(-1).copy()
        for position, values in enumerate(residuals[:packet_count]):
            start = position * self.group_size
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
        if float(np.max(np.abs(flat))) > _FLOAT16_MAX:
            raise ValueError("code exceeds the supported float16 range")
        scale = max(
            float(np.max(np.abs(flat))) / 127.0,
            float(np.finfo(np.float32).eps),
        )
        base = np.round(flat / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale
        base_half = base.astype(np.float16)
        if not np.all(np.isfinite(base_half)):
            raise ValueError("base must remain finite in float16")
        base_stream = io.BytesIO()
        np.save(base_stream, base_half.reshape(code.shape), allow_pickle=False)
        decoded_base = base_half.astype(np.float32)
        if not np.all(np.isfinite(decoded_base)):
            raise ValueError("decoded base must remain finite")
        residual = flat - decoded_base
        packets: list[EncodedPacket] = []
        for group, start in enumerate(range(0, len(flat), self.group_size)):
            payload = _encode_residual(group, start, residual[start : start + self.group_size])
            packets.append(EncodedPacket(group, packet_from_payload(payload)))
        return EncodedCode(
            handle,
            tuple(code.shape),
            base_stream.getvalue(),
            tuple(packets),
            self.group_size,
        )
