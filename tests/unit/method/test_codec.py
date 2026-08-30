from __future__ import annotations

import numpy as np
import pytest
import torch

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import (
    PacketCandidateKey,
    RateMemHardCodec,
    dequantize_gain,
    quantize_gain,
    select_packet_topk,
)
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary


def _dictionary() -> GroupRVQDictionary:
    torch.manual_seed(71)
    return GroupRVQDictionary(group_count=2, group_size=4, stages=2, entries=5)


def test_topk_ties_use_group_stage_entry_order_after_quantized_gain() -> None:
    rows = (
        PacketCandidateKey(group=1, stage=0, entry=2, gain_q=8),
        PacketCandidateKey(group=0, stage=1, entry=1, gain_q=-8),
        PacketCandidateKey(group=0, stage=0, entry=2, gain_q=8),
        PacketCandidateKey(group=0, stage=2, entry=1, gain_q=7),
    )
    selected = select_packet_topk(rows, maximum_packets=3)
    assert [(row.group, row.stage, row.entry, row.gain_q) for row in selected] == [
        (0, 0, 2, 8),
        (0, 1, 1, -8),
        (1, 0, 2, 8),
    ]


def test_gain_quantization_is_int16_saturating_and_finite() -> None:
    assert quantize_gain(1.0, 1 / 256) == 256
    assert quantize_gain(1e9, 1 / 256) == 32767
    assert quantize_gain(-1e9, 1 / 256) == -32768
    assert dequantize_gain(-8, 1 / 256) == -0.03125
    with pytest.raises(ValueError, match="finite"):
        quantize_gain(float("nan"), 1 / 256)


def test_hard_codec_decodes_only_ranked_quantized_topk() -> None:
    frozen = freeze_dictionary(_dictionary())
    codec = RateMemHardCodec(
        BlockwiseBaseQuantizer(4, 4),
        frozen,
        gain_step=1 / 256,
        maximum_packets=2,
    )
    code = np.array(
        [1.0, 0.1, 0.0, 0.0, 0.0, 1.2, 0.8, 0.0],
        dtype=np.float32,
    )
    encoded = codec.encode("a", code)
    assert len(encoded.all_candidates) == 4
    assert len(encoded.incidences) == 2
    decoded = codec.decode(encoded.base_payload, encoded.incidences)
    assert decoded.shape == code.shape
    assert np.isfinite(decoded).all()
    assert tuple(row.key for row in encoded.incidences) == select_packet_topk(
        tuple(row.key for row in encoded.all_candidates), maximum_packets=2
    )
