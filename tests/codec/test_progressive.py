import numpy as np

from ratemem.codec.progressive import ProgressiveCodec


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
