from __future__ import annotations

import torch
from torch import nn

from ratemem.method.dictionary import GroupRVQDictionary
from ratemem.method.model import RateMemTrainableMethod
from ratemem.method.utility import NonnegativeUtilityCalibrator
from ratemem.support.amortizer import SupportAmortizer


def _method() -> RateMemTrainableMethod:
    amortizer = SupportAmortizer(
        support_dim=4,
        description_dim=6,
        hidden_dim=8,
        projection_count=2,
        atom_count=2,
        layers=1,
        heads=2,
    )
    dictionary = GroupRVQDictionary(2, 2, 1, 4)
    utility = NonnegativeUtilityCalibrator(3, 4, 8, 2)
    atoms = (
        nn.Parameter(torch.zeros(2, 2)),
        nn.Parameter(torch.ones(2, 2)),
    )
    return RateMemTrainableMethod(atoms, amortizer, dictionary, utility)


def test_trainable_method_has_only_four_approved_namespaces() -> None:
    method = _method()

    assert {name.split(".", 1)[0] for name in method.state_dict()} == {
        "adapter_bank",
        "amortizer",
        "dictionary",
        "utility",
    }
    assert method.trainable_parameter_count() == sum(
        parameter.numel() for parameter in method.parameters()
    )


def test_dictionary_revision_changes_with_dictionary_only() -> None:
    method = _method()
    before = method.frozen_dictionary_revision()
    with torch.no_grad():
        method.amortizer.head.bias.add_(1.0)
    assert method.frozen_dictionary_revision() == before
    with torch.no_grad():
        method.dictionary.codebooks[0, 0, 0, 0].add_(0.5)
    assert method.frozen_dictionary_revision() != before
