"""Locked RateMem training objective terms."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F


def reconstruction_loss(target: Tensor, reconstructed: Tensor) -> Tensor:
    if target.shape != reconstructed.shape or target.numel() == 0:
        raise ValueError("reconstruction tensors must be nonempty and aligned")
    return F.mse_loss(reconstructed.float(), target.float())


def expected_rate_loss(
    selected_mask: Tensor,
    candidate_cost_bytes: Tensor,
    budget_bytes: int,
) -> Tensor:
    if (
        type(budget_bytes) is not int
        or budget_bytes <= 0
        or selected_mask.ndim != 3
        or candidate_cost_bytes.shape != selected_mask.shape[1:]
    ):
        raise ValueError("rate tensors or byte budget have the wrong shape")
    if torch.any(candidate_cost_bytes < 0) or not bool(
        torch.isfinite(candidate_cost_bytes).all().item()
    ):
        raise ValueError("candidate byte costs must be finite and nonnegative")
    expected = (
        selected_mask.float()
        * candidate_cost_bytes.to(selected_mask).unsqueeze(0)
    ).sum()
    return expected / (selected_mask.shape[0] * budget_bytes)


def reuse_affinity_loss(
    probabilities: Tensor,
    residuals: Tensor,
    similarity_center: float,
    similarity_width: float,
) -> Tensor:
    if residuals.shape[0] != probabilities.shape[0]:
        raise ValueError("reuse probability and residual batches differ")
    if not math.isfinite(similarity_center) or not math.isfinite(similarity_width):
        raise ValueError("similarity parameters must be finite")
    if similarity_width <= 0.0:
        raise ValueError("similarity width must be positive")
    if residuals.shape[0] < 2:
        return probabilities.sum() * 0.0
    normalized = F.normalize(
        residuals.float().reshape(residuals.shape[0], -1),
        dim=-1,
        eps=1e-8,
    )
    target = torch.sigmoid(
        (normalized @ normalized.T - similarity_center) / similarity_width
    )
    flat = probabilities.float().reshape(
        probabilities.shape[0],
        -1,
        probabilities.shape[-1],
    )
    match = (
        torch.einsum("bke,cke->bck", flat, flat)
        .mean(dim=-1)
        .clamp(1e-6, 1 - 1e-6)
    )
    mask = ~torch.eye(match.shape[0], dtype=torch.bool, device=match.device)
    return F.binary_cross_entropy(match[mask], target[mask])


def dictionary_balance_loss(probabilities: Tensor) -> Tensor:
    if probabilities.ndim < 2 or probabilities.shape[-1] < 1:
        raise ValueError("dictionary probabilities have invalid shape")
    usage = probabilities.float().mean(
        dim=tuple(range(probabilities.ndim - 1))
    ).clamp_min(1e-8)
    uniform = torch.full_like(usage, 1.0 / usage.numel())
    return torch.sum(usage * (usage.log() - uniform.log()))


def dictionary_commitment_loss(
    residual: Tensor,
    packet_reconstruction: Tensor,
) -> Tensor:
    if residual.shape != packet_reconstruction.shape or residual.numel() == 0:
        raise ValueError("commitment tensors must be nonempty and aligned")
    encoder_term = F.mse_loss(
        residual.float(),
        packet_reconstruction.detach().float(),
    )
    dictionary_term = F.mse_loss(
        residual.detach().float(),
        packet_reconstruction.float(),
    )
    return encoder_term + dictionary_term


def nonnegative_calibration_loss(
    predicted: Tensor,
    observed: Tensor,
    mask: Tensor,
) -> Tensor:
    if predicted.shape != observed.shape or mask.shape != predicted.shape:
        raise ValueError("utility calibration tensors must be aligned")
    if torch.any(predicted < 0) or torch.any(observed < 0):
        raise ValueError("utility calibration values must be nonnegative")
    selected = mask.to(torch.bool)
    if not bool(selected.any().item()):
        raise ValueError("utility calibration mask must select at least one value")
    return F.mse_loss(
        predicted[selected].float(),
        observed[selected].float(),
    )


@dataclass(frozen=True, slots=True)
class LossWeights:
    flow: float
    reconstruction: float
    rate: float
    reuse_affinity: float
    dictionary_balance: float
    dictionary_commitment: float
    utility_calibration: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(getattr(self, row.name))
            or getattr(self, row.name) < 0.0
            for row in fields(self)
        ):
            raise ValueError("loss weights must be finite and nonnegative")


def combine_losses(terms: dict[str, Tensor], weights: LossWeights) -> Tensor:
    names = {row.name for row in fields(weights)}
    if set(terms) != names:
        raise ValueError("loss terms do not match the locked method objective")
    ordered = [
        terms[name] * cast(float, getattr(weights, name))
        for name in sorted(terms)
    ]
    total = ordered[0]
    for term in ordered[1:]:
        total = total + term
    if total.ndim != 0 or not bool(torch.isfinite(total).item()):
        raise RuntimeError("combined training loss is not one finite scalar")
    return total
