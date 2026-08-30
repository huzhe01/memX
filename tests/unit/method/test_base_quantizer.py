from __future__ import annotations

import numpy as np
import pytest

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer, decode_base_payload


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_blockwise_payload_is_deterministic_and_finite(bits: int) -> None:
    code = np.linspace(-1.25, 1.5, 48, dtype=np.float32)
    codec = BlockwiseBaseQuantizer(group_size=16, bits=bits)
    first = codec.encode(code)
    second = codec.encode(code.copy())
    assert first.payload == second.payload
    decoded = decode_base_payload(first.payload)
    assert decoded.shape == (48,)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()


def test_more_base_bits_do_not_increase_error() -> None:
    code = np.array([0.91, -0.52, 0.33, -1.0] * 12, dtype=np.float32)
    errors = [
        float(np.mean((BlockwiseBaseQuantizer(16, bits).encode(code).decode() - code) ** 2))
        for bits in (2, 4, 8)
    ]
    assert errors[2] <= errors[1] <= errors[0]


def test_nonfinite_or_wrong_width_input_is_rejected() -> None:
    codec = BlockwiseBaseQuantizer(group_size=16, bits=4)
    with pytest.raises(ValueError, match="divisible"):
        codec.encode(np.zeros(17, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        codec.encode(np.array([float("nan")] * 16, dtype=np.float32))
