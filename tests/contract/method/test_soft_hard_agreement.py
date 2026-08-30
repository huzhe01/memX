from __future__ import annotations

import pytest
import torch

from ratemem.method.codec import (
    RateMemDifferentiableCodec,
    SoftHardAgreement,
    enforce_agreement,
    measure_soft_hard_agreement,
)
from ratemem.method.dictionary import GroupRVQDictionary


def test_agreement_report_includes_deployed_topk_membership() -> None:
    torch.manual_seed(99)
    codec = RateMemDifferentiableCodec(
        GroupRVQDictionary(2, 4, 2, 8),
        group_size=4,
        base_bits=4,
        gain_step=1 / 256,
        maximum_packets=2,
    )
    code = torch.tensor([[1.5, 0.0, 0.0, 0.0, 0.0, 1.3, 1.2, 0.0]])
    soft = codec(code, temperature=0.0001, mode="soft")
    hard = codec(code, temperature=1.0, mode="ste")
    report = measure_soft_hard_agreement(soft, hard)
    assert report.topk_disagreement == 0.0
    assert report.mean_code_error >= 0.0
    assert report.assignment_disagreement >= 0.0
    enforce_agreement(
        report,
        maximum_mean_code_error=0.02,
        maximum_assignment_disagreement=0.05,
        maximum_topk_disagreement=0.0,
    )


def test_release_gate_rejects_any_topk_disagreement() -> None:
    report = SoftHardAgreement(
        mean_code_error=0.0,
        maximum_code_error=0.0,
        assignment_disagreement=0.0,
        topk_disagreement=0.01,
    )
    with pytest.raises(RuntimeError, match="soft-hard agreement"):
        enforce_agreement(
            report,
            maximum_mean_code_error=0.02,
            maximum_assignment_disagreement=0.05,
            maximum_topk_disagreement=0.0,
        )
