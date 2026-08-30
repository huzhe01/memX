from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer, decode_base_payload


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_payload_length_matches_header_scales_and_packed_integers(bits: int) -> None:
    count, group_size = 480, 16
    payload = BlockwiseBaseQuantizer(group_size, bits).encode(
        np.linspace(-1.0, 1.0, count, dtype=np.float32)
    ).payload
    assert len(payload) == (
        struct.calcsize("<8sBHI")
        + (count // group_size) * 2
        + math.ceil(count * bits / 8)
    )


def test_payload_rejects_truncation_and_noncanonical_suffix() -> None:
    payload = BlockwiseBaseQuantizer(16, 4).encode(
        np.ones(32, dtype=np.float32)
    ).payload
    with pytest.raises(ValueError):
        decode_base_payload(payload[:-1])
    with pytest.raises(ValueError):
        decode_base_payload(payload + b"\x00")


def test_public_scale_decode_matches_the_thirty_production_groups() -> None:
    encoded = BlockwiseBaseQuantizer(16, 4).encode(
        np.linspace(-1.0, 1.0, 480, dtype=np.float32)
    )
    assert encoded.scales().shape == (30,)
    assert encoded.scales().dtype == np.float32
