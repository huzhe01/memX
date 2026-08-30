from __future__ import annotations

import numpy as np
import torch

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import RateMemDifferentiableCodec, RateMemHardCodec
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary


def test_production_ste_matches_actual_hard_60_candidate_top8() -> None:
    torch.manual_seed(20260824)
    dictionary = GroupRVQDictionary(30, 16, 2, 64)
    differentiable = RateMemDifferentiableCodec(
        dictionary,
        group_size=16,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=8,
    )
    code = torch.randn(2, 480, requires_grad=True)
    ste = differentiable(code, temperature=0.25, mode="ste")
    hard = RateMemHardCodec(
        BlockwiseBaseQuantizer(16, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=8,
    )
    for batch_index in range(2):
        encoded = hard.encode(
            f"h{batch_index}",
            code[batch_index].detach().numpy().astype(np.float32),
        )
        actual = torch.from_numpy(hard.decode(encoded.base_payload, encoded.incidences))
        assert len(encoded.all_candidates) == 60
        assert len(encoded.incidences) == 8
        assert ste.selected_keys[batch_index] == tuple(
            row.key for row in encoded.incidences
        )
        assert int(ste.selected_mask[batch_index].detach().sum().item()) == 8
        for row in encoded.incidences:
            assert int(
                ste.quantized_gains[batch_index, row.group, row.stage].item()
            ) == row.key.gain_q
        torch.testing.assert_close(
            ste.reconstruction[batch_index].detach().cpu(),
            actual,
            rtol=0.0,
            atol=1e-6,
        )
    ste.reconstruction.square().mean().backward()
    assert code.grad is not None
    assert dictionary.codebooks.grad is not None


def test_zero_gain_ties_match_deployed_lexicographic_top8() -> None:
    torch.manual_seed(20260824)
    dictionary = GroupRVQDictionary(30, 16, 2, 64)
    differentiable = RateMemDifferentiableCodec(
        dictionary,
        group_size=16,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=8,
    )
    code = torch.zeros(1, 480, requires_grad=True)
    ste = differentiable(code, temperature=0.25, mode="ste")
    hard = RateMemHardCodec(
        BlockwiseBaseQuantizer(16, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=8,
    ).encode("zero-tie", np.zeros(480, dtype=np.float32))
    assert ste.selected_keys[0] == tuple(row.key for row in hard.incidences)
    assert [(row.group, row.stage) for row in ste.selected_keys[0]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
