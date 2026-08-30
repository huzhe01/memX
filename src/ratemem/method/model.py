"""One checkpointable ownership boundary for every learned RateMem tensor."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary
from ratemem.method.utility import NonnegativeUtilityCalibrator
from ratemem.support.amortizer import SupportAmortizer


class AdapterParameterBank(nn.Module):
    """Register transformer-owned atom parameters without registering the backbone."""

    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        super().__init__()
        values = tuple(parameters)
        if not values or any(type(parameter) is not nn.Parameter for parameter in values):
            raise TypeError("adapter bank requires exact nn.Parameter values")
        if len({id(parameter) for parameter in values}) != len(values):
            raise ValueError("adapter bank parameter aliases are forbidden")
        self.atoms = nn.ParameterList(values)


class RateMemTrainableMethod(nn.Module):
    """The sole optimizer/checkpoint owner; frozen encoders remain outside this module."""

    def __init__(
        self,
        adapter_parameters: Iterable[nn.Parameter],
        amortizer: SupportAmortizer,
        dictionary: GroupRVQDictionary,
        utility: NonnegativeUtilityCalibrator,
    ) -> None:
        super().__init__()
        if type(amortizer) is not SupportAmortizer:
            raise TypeError("amortizer must be an exact SupportAmortizer")
        if type(dictionary) is not GroupRVQDictionary:
            raise TypeError("dictionary must be an exact GroupRVQDictionary")
        if type(utility) is not NonnegativeUtilityCalibrator:
            raise TypeError("utility must be an exact NonnegativeUtilityCalibrator")
        self.adapter_bank = AdapterParameterBank(adapter_parameters)
        self.amortizer = amortizer
        self.dictionary = dictionary
        self.utility = utility

    def frozen_dictionary_revision(self) -> str:
        return freeze_dictionary(self.dictionary).revision_sha256

    def trainable_parameter_count(self) -> int:
        parameters = tuple(self.parameters())
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("RateMem trainable parameter ownership contains aliases")
        if any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError("RateMem trainable method contains a frozen parameter")
        return sum(parameter.numel() for parameter in parameters)


__all__ = ["AdapterParameterBank", "RateMemTrainableMethod"]
