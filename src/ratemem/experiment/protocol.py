from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    features: Tensor
    targets: Tensor

    def __post_init__(self) -> None:
        if type(self.features) is not Tensor or type(self.targets) is not Tensor:
            raise TypeError("episode features and targets must be exact Tensors")
        if self.features.ndim != 2 or self.features.shape[1] != 6:
            raise ValueError("fixture features must have shape [batch, 6]")
        if self.targets.shape != (self.features.shape[0], 3):
            raise ValueError("fixture targets must have shape [batch, 3]")
        if self.features.dtype != self.targets.dtype:
            raise TypeError("fixture features and targets must share one dtype")
        if self.features.device != self.targets.device:
            raise ValueError("fixture features and targets must share one device")
        if self.features.shape[0] < 1:
            raise ValueError("episode batch must not be empty")


@dataclass(frozen=True, slots=True)
class StepMetrics:
    loss: float

    def __post_init__(self) -> None:
        if type(self.loss) is not float or not math.isfinite(self.loss) or self.loss < 0:
            raise ValueError("step loss must be a finite nonnegative exact float")


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    squared_error_sum: float
    example_count: int

    def __post_init__(self) -> None:
        if (
            type(self.squared_error_sum) is not float
            or not math.isfinite(self.squared_error_sum)
            or self.squared_error_sum < 0
        ):
            raise ValueError("evaluation squared-error sum must be finite and nonnegative")
        if type(self.example_count) is not int or self.example_count < 1:
            raise ValueError("evaluation example count must be a positive exact int")


class Experiment(Protocol):
    def train_step(self, batch: EpisodeBatch) -> StepMetrics: ...

    def evaluate(self, batch: EpisodeBatch) -> EvaluationMetrics: ...

    def model_state_dict(self) -> dict[str, Tensor]: ...

    def optimizer_state_dict(self) -> dict[str, object]: ...

    def load_state_dicts(
        self,
        model_state: Mapping[str, Tensor],
        optimizer_state: Mapping[str, object],
    ) -> None: ...
