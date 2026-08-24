import io
import struct
from dataclasses import replace

import numpy as np
import pytest

from ratemem.codec.progressive import EncodedCode, ProgressiveCodec
from ratemem.state.serialization import packet_from_payload

_PACKET_HEADER = struct.Struct("<II")


def _with_packet_payload(encoded: EncodedCode, packet_index: int, payload: bytes) -> EncodedCode:
    packets = list(encoded.packets)
    packets[packet_index] = replace(packets[packet_index], packet=packet_from_payload(payload))
    return replace(encoded, packets=tuple(packets))


def _npy_payload(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=allow_pickle)
    return stream.getvalue()


def test_codec_bytes_match_the_planned_float32_quantizer() -> None:
    code = np.array([61370.43359375, 14255.3369140625], dtype=np.float32)
    flat = code.reshape(-1)
    scale = max(
        float(np.max(np.abs(flat))) / 127.0,
        float(np.finfo(np.float32).eps),
    )
    base = np.round(flat / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale
    expected_base_payload = _npy_payload(base.reshape(code.shape).astype(np.float16))
    decoded_base = base.astype(np.float16).astype(np.float32)
    residual = flat - decoded_base
    expected_packet_payloads = [
        _PACKET_HEADER.pack(group, start)
        + np.asarray(residual[start : start + 1], dtype="<f2").tobytes(order="C")
        for group, start in enumerate(range(len(flat)))
    ]

    encoded = ProgressiveCodec(group_size=1).encode("float32-reference", code)

    assert encoded.base_payload == expected_base_payload
    assert [item.packet.payload for item in encoded.packets] == expected_packet_payloads
    assert [item.packet.packet_id for item in encoded.packets] == [
        packet_from_payload(payload).packet_id for payload in expected_packet_payloads
    ]


def test_packets_monotonically_reduce_code_error() -> None:
    code = np.array([0.1, -1.7, 0.3, 2.2, -0.8, 0.4, 1.1, -0.2], dtype=np.float32)
    encoded = ProgressiveCodec(group_size=2).encode("a", code)
    errors = []
    for count in range(len(encoded.packets) + 1):
        decoded = encoded.decode(packet_count=count)
        errors.append(float(np.mean((decoded - code) ** 2)))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


def test_packet_payloads_are_deterministic() -> None:
    code = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    codec = ProgressiveCodec(group_size=3)
    first = codec.encode("a", code)
    second = codec.encode("a", code)
    assert first.base_payload == second.base_payload
    assert [item.packet.packet_id for item in first.packets] == [
        item.packet.packet_id for item in second.packets
    ]


def test_truncated_packet_is_rejected() -> None:
    import pytest

    from ratemem.codec.progressive import _decode_residual

    with pytest.raises(ValueError, match="truncated residual packet"):
        _decode_residual(b"bad")


def test_float16_boundaries_are_finite_and_monotone() -> None:
    code = np.array([65504.0, -65504.0], dtype=np.float32)
    encoded = ProgressiveCodec(group_size=1).encode("boundary", code)
    errors = []
    for count in range(len(encoded.packets) + 1):
        decoded = encoded.decode(packet_count=count)
        assert np.all(np.isfinite(decoded))
        errors.append(float(np.mean((decoded - code) ** 2)))
    assert errors == sorted(errors, reverse=True)


def test_values_immediately_outside_float16_range_are_rejected() -> None:
    import pytest

    above = np.nextafter(np.float32(65504.0), np.float32(np.inf))
    for value in (above, -above):
        with pytest.raises(ValueError, match="float16 range"):
            ProgressiveCodec(group_size=1).encode(
                "outside-boundary", np.array([value], dtype=np.float32)
            )


def test_large_float32_values_are_rejected() -> None:
    import pytest

    maximum = np.finfo(np.float32).max
    for value in (np.float32(1e10), np.float32(-1e10), maximum, -maximum):
        with pytest.raises(ValueError, match="float16 range"):
            ProgressiveCodec(group_size=1).encode("large", np.array([value], dtype=np.float32))


def test_encoded_code_defensively_owns_stream_metadata() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("source", np.arange(7, dtype=np.float32))
    shape = list(encoded.shape)
    base_payload = bytearray(encoded.base_payload)
    packets = list(encoded.packets)
    owned = EncodedCode(  # type: ignore[arg-type]
        handle="owned",
        shape=shape,
        base_payload=base_payload,
        packets=packets,
        group_size=3,
    )
    shape.clear()
    base_payload.clear()
    packets.clear()
    assert owned.shape == (7,)
    assert owned.base_payload
    assert len(owned.packets) == 3


def test_forged_packet_id_is_rejected_when_selected() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("forged", np.arange(7, dtype=np.float32))
    packets = list(encoded.packets)
    packets[0] = replace(packets[0], packet=replace(packets[0].packet, packet_id="0" * 64))
    with pytest.raises(ValueError):
        replace(encoded, packets=tuple(packets)).decode(1)


def test_repeated_packet_is_rejected_when_selected() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("repeated", np.arange(7, dtype=np.float32))
    packets = list(encoded.packets)
    packets[1] = packets[0]
    with pytest.raises(ValueError):
        replace(encoded, packets=tuple(packets)).decode(2)


def test_packet_tuple_order_is_canonical() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("order", np.arange(7, dtype=np.float32))
    packets = list(encoded.packets)
    packets[0], packets[1] = packets[1], packets[0]
    with pytest.raises(ValueError):
        replace(encoded, packets=tuple(packets)).decode(1)


def test_repeated_packet_group_is_rejected_when_selected() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("group", np.arange(7, dtype=np.float32))
    packets = list(encoded.packets)
    packets[1] = replace(packets[1], group=packets[0].group)
    with pytest.raises(ValueError):
        replace(encoded, packets=tuple(packets)).decode(2)


def test_packet_header_group_must_match_tuple_group() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("header-group", np.arange(7, dtype=np.float32))
    payload = encoded.packets[0].packet.payload
    _, start = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
    tampered = _PACKET_HEADER.pack(99, start) + payload[_PACKET_HEADER.size :]
    with pytest.raises(ValueError):
        _with_packet_payload(encoded, 0, tampered).decode(1)


@pytest.mark.parametrize("start", [1, 4, 0xFFFFFFFF, 7])
def test_packet_start_must_be_canonical_and_in_bounds(start: int) -> None:
    encoded = ProgressiveCodec(group_size=3).encode("start", np.arange(7, dtype=np.float32))
    payload = encoded.packets[1].packet.payload
    group, _ = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
    tampered = _PACKET_HEADER.pack(group, start) + payload[_PACKET_HEADER.size :]
    with pytest.raises(ValueError):
        _with_packet_payload(encoded, 1, tampered).decode(2)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"\x00",
        np.zeros(2, dtype="<f2").tobytes(),
        np.zeros(4, dtype="<f2").tobytes(),
    ],
)
def test_packet_body_size_must_match_canonical_segment(body: bytes) -> None:
    encoded = ProgressiveCodec(group_size=3).encode("body-size", np.arange(7, dtype=np.float32))
    payload = encoded.packets[0].packet.payload
    tampered = payload[: _PACKET_HEADER.size] + body
    with pytest.raises(ValueError):
        _with_packet_payload(encoded, 0, tampered).decode(1)


def test_last_packet_cannot_extend_past_declared_shape() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("end", np.arange(7, dtype=np.float32))
    payload = encoded.packets[-1].packet.payload
    oversized = payload + np.zeros(1, dtype="<f2").tobytes()
    with pytest.raises(ValueError):
        _with_packet_payload(encoded, len(encoded.packets) - 1, oversized).decode(
            len(encoded.packets)
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float16_packet_body_is_rejected(value: float) -> None:
    encoded = ProgressiveCodec(group_size=3).encode(
        "nonfinite-body", np.arange(7, dtype=np.float32)
    )
    payload = encoded.packets[0].packet.payload
    body = np.array([value, 0.0, 0.0], dtype="<f2").tobytes()
    tampered = payload[: _PACKET_HEADER.size] + body
    with pytest.raises(ValueError):
        _with_packet_payload(encoded, 0, tampered).decode(1)


def test_malformed_unused_suffix_is_isolated_from_shorter_prefixes() -> None:
    encoded = ProgressiveCodec(group_size=3).encode(
        "suffix-isolation", np.arange(7, dtype=np.float32)
    )
    payload = encoded.packets[-1].packet.payload
    nonfinite_body = np.array([np.nan], dtype="<f2").tobytes()
    corrupted = _with_packet_payload(
        encoded,
        len(encoded.packets) - 1,
        payload[: _PACKET_HEADER.size] + nonfinite_body,
    )

    for packet_count in range(len(encoded.packets)):
        np.testing.assert_array_equal(corrupted.decode(packet_count), encoded.decode(packet_count))
    with pytest.raises(ValueError):
        corrupted.decode(len(encoded.packets))


@pytest.mark.parametrize("extra", [False, True])
def test_packet_count_must_cover_declared_shape_exactly(extra: bool) -> None:
    encoded = ProgressiveCodec(group_size=3).encode("packet-count", np.arange(7, dtype=np.float32))
    packets = encoded.packets + (encoded.packets[-1],) if extra else encoded.packets[:-1]
    with pytest.raises(ValueError):
        replace(encoded, packets=packets).decode(0)


def test_declared_shape_must_match_stored_base_shape_exactly() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("base-shape", np.arange(7, dtype=np.float32))
    with pytest.raises(ValueError):
        replace(encoded, shape=(1, 7)).decode(0)


def test_base_payload_rejects_trailing_bytes() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("base-trailing", np.arange(7, dtype=np.float32))
    with pytest.raises(ValueError):
        replace(encoded, base_payload=encoded.base_payload + b"trailing").decode(0)


def test_base_payload_requires_canonical_npy_header_bytes() -> None:
    encoded = ProgressiveCodec(group_size=3).encode(
        "base-canonical", np.arange(7, dtype=np.float32)
    )
    header_size = struct.unpack("<H", encoded.base_payload[8:10])[0]
    header_start = 10
    header_end = header_start + header_size
    header = encoded.base_payload[header_start:header_end]
    assert b", }" in header
    assert header.endswith(b" \n")

    widened_dict = header.replace(b", }", b",  }", 1)
    noncanonical_header = widened_dict[:-2] + widened_dict[-1:]
    assert len(noncanonical_header) == len(header)
    noncanonical = (
        encoded.base_payload[:header_start]
        + noncanonical_header
        + encoded.base_payload[header_end:]
    )
    assert len(noncanonical) == len(encoded.base_payload)

    with pytest.raises(ValueError, match="canonical"):
        replace(encoded, base_payload=noncanonical).decode(0)


def test_base_payload_requires_float16_dtype() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("base-dtype", np.arange(7, dtype=np.float32))
    wrong_dtype = _npy_payload(np.arange(7, dtype=np.float32))
    with pytest.raises(ValueError):
        replace(encoded, base_payload=wrong_dtype).decode(0)


def test_base_payload_requires_c_order() -> None:
    code = np.arange(6, dtype=np.float32).reshape(2, 3)
    encoded = ProgressiveCodec(group_size=3).encode("base-order", code)
    fortran = np.asfortranarray(code.astype(np.float16))
    assert fortran.flags.f_contiguous and not fortran.flags.c_contiguous
    with pytest.raises(ValueError):
        replace(encoded, base_payload=_npy_payload(fortran)).decode(0)


def test_object_base_payload_is_rejected_without_unpickling() -> None:
    encoded = ProgressiveCodec(group_size=3).encode("base-object", np.arange(7, dtype=np.float32))
    object_payload = _npy_payload(np.array(["unsafe"] * 7, dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        replace(encoded, base_payload=object_payload).decode(0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float16_base_payload_is_rejected(value: float) -> None:
    encoded = ProgressiveCodec(group_size=3).encode(
        "base-nonfinite", np.arange(7, dtype=np.float32)
    )
    values = np.zeros(7, dtype=np.float16)
    values[3] = value
    with pytest.raises(ValueError):
        replace(encoded, base_payload=_npy_payload(values)).decode(0)


def test_empty_base_payload_is_rejected() -> None:
    encoded = EncodedCode(
        handle="base-empty",
        shape=(0,),
        base_payload=_npy_payload(np.array([], dtype=np.float16)),
        packets=(),
        group_size=3,
    )
    with pytest.raises(ValueError):
        encoded.decode(0)


@pytest.mark.parametrize("kind", ["short-header", "short-data", "malformed"])
def test_malformed_or_truncated_base_payload_is_normalized_to_value_error(
    kind: str,
) -> None:
    encoded = ProgressiveCodec(group_size=3).encode(
        "base-malformed", np.arange(7, dtype=np.float32)
    )
    if kind == "short-header":
        payload = encoded.base_payload[:5]
    elif kind == "short-data":
        payload = encoded.base_payload[:-1]
    else:
        payload = b"not a numpy payload"
    with pytest.raises(ValueError):
        replace(encoded, base_payload=payload).decode(0)
