"""Deterministic grouped symmetric base quantization with a fixed wire format."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
UIntArray: TypeAlias = NDArray[np.uint8]

_MAGIC = b"RTBASE01"
_HEADER = struct.Struct("<8sBHI")
_SUPPORTED_BITS = frozenset({2, 4, 8})


def _pack(values: UIntArray, bits: int) -> bytes:
    output = bytearray(math.ceil(int(values.size) * bits / 8))
    accumulator = 0
    occupied = 0
    cursor = 0
    for value in values.tolist():
        accumulator |= int(value) << occupied
        occupied += bits
        while occupied >= 8:
            output[cursor] = accumulator & 0xFF
            cursor += 1
            accumulator >>= 8
            occupied -= 8
    if occupied:
        output[cursor] = accumulator & 0xFF
    return bytes(output)


def _unpack(payload: bytes, count: int, bits: int) -> UIntArray:
    expected_bytes = math.ceil(count * bits / 8)
    if len(payload) != expected_bytes:
        raise ValueError("packed base payload has a noncanonical byte length")
    mask = (1 << bits) - 1
    output = np.empty(count, dtype=np.uint8)
    accumulator = 0
    occupied = 0
    cursor = 0
    for index in range(count):
        while occupied < bits:
            accumulator |= payload[cursor] << occupied
            cursor += 1
            occupied += 8
        output[index] = accumulator & mask
        accumulator >>= bits
        occupied -= bits
    if accumulator:
        raise ValueError("noncanonical trailing base bits")
    return output


@dataclass(frozen=True, slots=True)
class QuantizedBase:
    payload: bytes

    def decode(self) -> FloatArray:
        return decode_base_payload(self.payload)

    def scales(self) -> FloatArray:
        return decode_base_scales(self.payload)


class BlockwiseBaseQuantizer:
    def __init__(self, group_size: int, bits: int) -> None:
        if type(group_size) is not int or group_size < 1 or group_size > 0xFFFF:
            raise ValueError("group_size must fit a positive uint16")
        if type(bits) is not int or bits not in _SUPPORTED_BITS:
            raise ValueError("bits must be 2, 4, or 8")
        self.group_size = group_size
        self.bits = bits

    def encode(self, code: NDArray[np.generic]) -> QuantizedBase:
        flat = cast(FloatArray, np.asarray(code, dtype=np.float32).reshape(-1))
        if flat.size == 0 or flat.size > 0xFFFFFFFF or flat.size % self.group_size:
            raise ValueError("code width must be nonempty and divisible by group_size")
        if not np.isfinite(flat).all():
            raise ValueError("code must be finite")
        groups = flat.reshape(-1, self.group_size)
        qmax = (1 << (self.bits - 1)) - 1
        scales = np.maximum(
            np.max(np.abs(groups), axis=1) / qmax,
            np.finfo(np.float16).tiny,
        )
        scales_f16 = scales.astype("<f2")
        if not np.isfinite(scales_f16).all():
            raise ValueError("base scale cannot be represented as float16")
        restored_scales = scales_f16.astype(np.float32)
        signed = (
            np.rint(groups / restored_scales[:, None])
            .clip(-qmax, qmax)
            .astype(np.int16)
        )
        unsigned = cast(UIntArray, (signed + qmax).astype(np.uint8).reshape(-1))
        header = _HEADER.pack(_MAGIC, self.bits, self.group_size, int(flat.size))
        return QuantizedBase(
            header + scales_f16.tobytes(order="C") + _pack(unsigned, self.bits)
        )


def _decode_metadata(payload: bytes) -> tuple[int, int, int, int]:
    if type(payload) is not bytes or len(payload) < _HEADER.size:
        raise ValueError("truncated base header")
    magic, bits, group_size, count = _HEADER.unpack_from(payload)
    if (
        magic != _MAGIC
        or bits not in _SUPPORTED_BITS
        or not group_size
        or not count
        or count % group_size
    ):
        raise ValueError("unsupported base payload")
    group_count = count // group_size
    scales_end = _HEADER.size + group_count * 2
    expected = scales_end + math.ceil(count * bits / 8)
    if len(payload) != expected:
        raise ValueError("base payload has a noncanonical byte length")
    return bits, group_size, count, scales_end


def decode_base_payload(payload: bytes) -> FloatArray:
    bits, group_size, count, scales_end = _decode_metadata(payload)
    scales = np.frombuffer(payload[_HEADER.size:scales_end], dtype="<f2").astype(
        np.float32
    )
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("base scales must be finite and positive")
    unsigned = _unpack(payload[scales_end:], count, bits).astype(np.int16)
    qmax = (1 << (bits - 1)) - 1
    if np.any(unsigned > 2 * qmax):
        raise ValueError("base payload contains an unused integer code")
    signed = unsigned - qmax
    decoded = signed.reshape(-1, group_size).astype(np.float32) * scales[:, None]
    return cast(FloatArray, decoded.reshape(-1))


def decode_base_scales(payload: bytes) -> FloatArray:
    decode_base_payload(payload)
    _bits, _group_size, _count, scales_end = _decode_metadata(payload)
    return (
        np.frombuffer(payload[_HEADER.size:scales_end], dtype="<f2")
        .astype(np.float32)
        .copy()
    )


__all__ = [
    "BlockwiseBaseQuantizer",
    "FloatArray",
    "QuantizedBase",
    "decode_base_payload",
    "decode_base_scales",
]
