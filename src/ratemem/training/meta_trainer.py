"""Bounded sequential meta-training over visible lifecycle segments."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from ratemem.method.codec import (
    DifferentiableEncoding,
    RateMemDifferentiableCodec,
)
from ratemem.method.utility import CausalFeatureBatch, NonnegativeUtilityCalibrator
from ratemem.training.functional_state import FunctionalMemoryState
from ratemem.training.losses import (
    LossWeights,
    combine_losses,
    dictionary_balance_loss,
    dictionary_commitment_loss,
    expected_rate_loss,
    nonnegative_calibration_loss,
    reconstruction_loss,
    reuse_affinity_loss,
)
from ratemem.training.segments import TrainingSegment


class PreparedSegmentResolver(Protocol):
    def target_code(self, trace_id: str, event_index: int) -> Tensor: ...

    def one_timestep_flow_loss(
        self,
        trace_id: str,
        event_index: int,
        adapter_code: Tensor,
    ) -> Tensor: ...

    def utility_supervision(
        self,
        trace_id: str,
        event_index: int,
    ) -> tuple[CausalFeatureBatch, Tensor, Tensor]: ...


@dataclass(frozen=True, slots=True)
class MetaStepReceipt:
    trace_id: str
    segment_index: int
    transformer_passes: int
    event_count: int
    total_loss: float
    detached_state: FunctionalMemoryState


class SequentialMetaTrainer:
    def __init__(
        self,
        codec: RateMemDifferentiableCodec,
        utility: NonnegativeUtilityCalibrator,
        optimizer: torch.optim.Optimizer,
        resolver: PreparedSegmentResolver,
        weights: LossWeights,
        maximum_transformer_passes: int = 2,
        gradient_synchronizer: Callable[[tuple[nn.Parameter, ...]], None] | None = None,
    ) -> None:
        if type(codec) is not RateMemDifferentiableCodec:
            raise TypeError("codec must be an exact RateMemDifferentiableCodec")
        if type(utility) is not NonnegativeUtilityCalibrator:
            raise TypeError("utility must be an exact NonnegativeUtilityCalibrator")
        if type(maximum_transformer_passes) is not int or not (
            1 <= maximum_transformer_passes <= 2
        ):
            raise ValueError("transformer pass cap must be one or two")
        self.codec = codec
        self.utility = utility
        self.optimizer = optimizer
        self.resolver = resolver
        self.weights = weights
        self.maximum_transformer_passes = maximum_transformer_passes
        if gradient_synchronizer is not None and not callable(gradient_synchronizer):
            raise TypeError("gradient_synchronizer must be callable or None")
        parameters: list[nn.Parameter] = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError("optimizer parameters must be nn.Parameter values")
                parameters.append(parameter)
        if not parameters or len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("optimizer parameters must be nonempty and unique")
        self.parameters = tuple(parameters)
        self.gradient_synchronizer = gradient_synchronizer

    def train_segment(
        self,
        segment: TrainingSegment,
        state: FunctionalMemoryState,
        *,
        temperature: float,
        candidate_cost_bytes: Tensor,
        budget_bytes: int,
    ) -> MetaStepReceipt:
        if type(segment) is not TrainingSegment or type(state) is not FunctionalMemoryState:
            raise TypeError("trainer requires exact segment and functional state values")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("training temperature must be finite and positive")
        self.optimizer.zero_grad(set_to_none=True)
        terms: dict[str, Tensor] = {}
        transformer_passes = 0
        current = state
        encodings: list[DifferentiableEncoding] = []
        targets: list[Tensor] = []
        for event in segment.events:
            if event.kind in {"create", "update"}:
                target = self.resolver.target_code(
                    segment.trace_id,
                    event.event_index,
                ).float()
                if target.ndim != 2:
                    raise ValueError("resolver target code must be rank two")
                encoded = self.codec(
                    target,
                    temperature=temperature,
                    mode="ste",
                )
                current = current.upsert(
                    event.handle,
                    encoded.reconstruction,
                    event.event_index,
                )
                encodings.append(encoded)
                targets.append(target)
            elif event.kind == "delete":
                current = current.delete(event.handle)
            if event.has_training_query:
                if event.handle not in current.codes:
                    continue
                if transformer_passes >= self.maximum_transformer_passes:
                    raise RuntimeError(
                        "segment exceeded the locked transformer-pass cap"
                    )
                transformer_passes += 1
                flow = self.resolver.one_timestep_flow_loss(
                    segment.trace_id,
                    event.event_index,
                    current.codes[event.handle],
                )
                if flow.ndim != 0 or not bool(torch.isfinite(flow).item()):
                    raise ValueError("one-timestep flow loss must be a finite scalar")
                terms["flow"] = terms.get("flow", flow * 0.0) + flow
        if not encodings:
            zero = next(self.codec.parameters()).sum() * 0.0
            terms.update(
                {
                    name: terms.get(name, zero)
                    for name in self.weights.__dataclass_fields__
                }
            )
        else:
            target_batch = torch.cat(targets, dim=0)
            reconstruction_batch = torch.cat(
                [row.reconstruction for row in encodings],
                dim=0,
            )
            probabilities = torch.cat(
                [row.assignment_probabilities for row in encodings],
                dim=0,
            )
            base_batch = torch.cat(
                [row.base_reconstruction for row in encodings],
                dim=0,
            )
            residuals = target_batch - base_batch
            packet_reconstruction = reconstruction_batch - base_batch
            utility_features, observed, mask = self.resolver.utility_supervision(
                segment.trace_id,
                segment.events[-1].event_index,
            )
            predicted = self.utility(utility_features).value
            terms.update(
                {
                    "reconstruction": reconstruction_loss(
                        target_batch,
                        reconstruction_batch,
                    ),
                    "rate": expected_rate_loss(
                        torch.cat(
                            [row.selected_mask for row in encodings],
                            dim=0,
                        ),
                        candidate_cost_bytes,
                        budget_bytes,
                    ),
                    "reuse_affinity": reuse_affinity_loss(
                        probabilities,
                        residuals,
                        0.8,
                        0.1,
                    ),
                    "dictionary_balance": dictionary_balance_loss(probabilities),
                    "dictionary_commitment": dictionary_commitment_loss(
                        residuals,
                        packet_reconstruction,
                    ),
                    "utility_calibration": nonnegative_calibration_loss(
                        predicted,
                        observed,
                        mask,
                    ),
                }
            )
            terms["flow"] = terms.get(
                "flow",
                reconstruction_batch.sum() * 0.0,
            )
        total = combine_losses(terms, self.weights)
        resolver_backward = getattr(self.resolver, "backward", None)
        if callable(resolver_backward):
            resolver_backward(total)
        else:
            torch.autograd.backward(total)
        if self.gradient_synchronizer is not None:
            self.gradient_synchronizer(self.parameters)
        self.optimizer.step()
        self.codec.dictionary.normalize_codebooks_()
        detached = current.detach_boundary()
        return MetaStepReceipt(
            trace_id=segment.trace_id,
            segment_index=segment.segment_index,
            transformer_passes=transformer_passes,
            event_count=len(segment.events),
            total_loss=float(total.detach()),
            detached_state=detached,
        )
