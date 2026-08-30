"""Causal nonnegative utility calibration for RateMem packet allocation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.nn import functional as F

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.method.proposal import ImmutableBundleProposal


@dataclass(frozen=True, slots=True)
class CausalRequestHistory:
    decay: float
    reads: Mapping[str, Sequence[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.decay) or not 0.0 < self.decay <= 1.0:
            raise ValueError("request decay must be finite and in (0, 1]")
        normalized: dict[str, tuple[int, ...]] = {}
        for handle, indices in self.reads.items():
            if type(handle) is not str or not handle:
                raise ValueError("request history handles must be nonempty exact strings")
            canonical = tuple(indices)
            if any(type(index) is not int or index < 0 for index in canonical):
                raise ValueError("request history indices must be nonnegative exact integers")
            if tuple(sorted(canonical)) != canonical or len(set(canonical)) != len(canonical):
                raise ValueError("request history indices must be unique and sorted")
            normalized[handle] = canonical
        object.__setattr__(self, "reads", MappingProxyType(dict(sorted(normalized.items()))))

    def observe_read(
        self,
        handle: str,
        event_index: int,
        operational: bool,
    ) -> CausalRequestHistory:
        if type(handle) is not str or not handle:
            raise ValueError("read handle must be a nonempty exact string")
        if type(event_index) is not int or event_index < 0:
            raise ValueError("read event_index must be a nonnegative exact integer")
        if type(operational) is not bool:
            raise TypeError("operational must be an exact bool")
        if not operational:
            return self
        rows = dict(self.reads)
        if rows.get(handle) and event_index <= rows[handle][-1]:
            raise ValueError("operational reads must be observed in increasing event order")
        rows[handle] = (*rows.get(handle, ()), event_index)
        return CausalRequestHistory(self.decay, rows)

    def weight(self, handle: str, allocation_event_index: int) -> float:
        if type(allocation_event_index) is not int or allocation_event_index < 0:
            raise ValueError("allocation_event_index must be a nonnegative exact integer")
        events = self.reads.get(handle, ())
        if any(index >= allocation_event_index for index in events):
            raise ValueError("future or current-event read reached allocation history")
        return float(
            sum(
                self.decay ** (allocation_event_index - 1 - index)
                for index in events
            )
        )


@dataclass(frozen=True, slots=True)
class CausalFeatureBatch:
    concept: Tensor
    incidence: Tensor
    incidence_mask: Tensor
    maximum_source_event_index: Tensor
    allocation_event_index: Tensor


@dataclass(frozen=True, slots=True)
class UtilityPrediction:
    beta: Tensor
    value: Tensor


class NonnegativeUtilityCalibrator(nn.Module):
    def __init__(
        self,
        concept_features: int,
        incidence_features: int,
        hidden: int,
        groups: int,
    ) -> None:
        super().__init__()
        if any(
            type(value) is not int or value < 1
            for value in (concept_features, incidence_features, hidden, groups)
        ):
            raise ValueError("utility dimensions must be positive exact integers")
        self.groups = groups
        self.concept_net = nn.Sequential(
            nn.Linear(concept_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, groups),
        )
        self.incidence_net = nn.Sequential(
            nn.Linear(incidence_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, groups),
        )

    def forward(self, batch: CausalFeatureBatch) -> UtilityPrediction:
        if type(batch) is not CausalFeatureBatch:
            raise TypeError("utility input must be an exact CausalFeatureBatch")
        if batch.concept.ndim != 2 or batch.incidence.ndim != 3:
            raise ValueError("utility concept/incidence features have invalid rank")
        if batch.incidence.shape[0] != batch.concept.shape[0]:
            raise ValueError("utility feature batch dimensions differ")
        if batch.incidence_mask.shape != batch.incidence.shape[:2]:
            raise ValueError("incidence mask shape does not match feature rows")
        if batch.incidence_mask.dtype is not torch.bool:
            raise TypeError("incidence mask must be boolean")
        if batch.maximum_source_event_index.shape != batch.allocation_event_index.shape:
            raise ValueError("utility event-index shapes differ")
        if batch.maximum_source_event_index.shape != (batch.concept.shape[0],):
            raise ValueError("utility event indices must contain one value per batch row")
        if torch.any(batch.maximum_source_event_index > batch.allocation_event_index):
            raise ValueError("future feature reached the utility calibrator")
        beta = F.softplus(self.concept_net(batch.concept.float()))
        raw = F.softplus(self.incidence_net(batch.incidence.float()))
        value = raw * batch.incidence_mask.unsqueeze(-1).to(raw.dtype)
        return UtilityPrediction(beta=beta, value=value)


class CalibrationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    split: Literal["calibration"]
    method_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bin_edges: list[float]
    bin_count: list[int]
    predicted_mean: list[float]
    observed_mean: list[float]
    expected_calibration_error: float = Field(ge=0.0, allow_inf_nan=False)
    maximum_allowed_ece: float = Field(ge=0.0, allow_inf_nan=False)


def calibration_receipt(
    predicted: NDArray[np.float64] | Sequence[float],
    observed: NDArray[np.float64] | Sequence[float],
    *,
    bins: int,
    method_lock_sha256: str,
    feature_manifest_sha256: str,
    label_artifact_sha256: str,
    maximum_allowed_ece: float,
) -> CalibrationReceipt:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=np.float64)
    if (
        predicted_array.shape != observed_array.shape
        or predicted_array.ndim != 1
        or predicted_array.size == 0
    ):
        raise ValueError("calibration arrays must be nonempty aligned vectors")
    if type(bins) is not int or bins < 1:
        raise ValueError("calibration bins must be a positive exact integer")
    if (
        np.any(predicted_array < 0)
        or np.any(observed_array < 0)
        or not np.isfinite(predicted_array).all()
        or not np.isfinite(observed_array).all()
    ):
        raise ValueError("calibration gains must be finite and nonnegative")
    edges = np.linspace(0.0, max(1.0, float(predicted_array.max())), bins + 1)
    assignments = np.minimum(np.digitize(predicted_array, edges[1:-1]), bins - 1)
    counts: list[int] = []
    predicted_means: list[float] = []
    observed_means: list[float] = []
    weighted_error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        predicted_mean = float(predicted_array[mask].mean()) if count else 0.0
        observed_mean = float(observed_array[mask].mean()) if count else 0.0
        counts.append(count)
        predicted_means.append(predicted_mean)
        observed_means.append(observed_mean)
        weighted_error += count * abs(predicted_mean - observed_mean)
    return CalibrationReceipt(
        split="calibration",
        method_lock_sha256=method_lock_sha256,
        feature_manifest_sha256=feature_manifest_sha256,
        label_artifact_sha256=label_artifact_sha256,
        bin_edges=[float(value) for value in edges],
        bin_count=counts,
        predicted_mean=predicted_means,
        observed_mean=observed_means,
        expected_calibration_error=weighted_error / predicted_array.size,
        maximum_allowed_ece=maximum_allowed_ece,
    )


def enforce_calibration(receipt: CalibrationReceipt) -> None:
    if type(receipt) is not CalibrationReceipt:
        raise TypeError("calibration receipt must be an exact CalibrationReceipt")
    if receipt.expected_calibration_error > receipt.maximum_allowed_ece:
        raise RuntimeError("utility calibration ECE exceeds the locked threshold")


@dataclass(frozen=True, slots=True)
class UtilityAudit:
    allocation_event_index: int
    maximum_feature_event_index: int
    cold_start_handles: tuple[str, ...]
    request_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_weights",
            MappingProxyType(dict(sorted(self.request_weights.items()))),
        )


def _gain_vector(value: Sequence[float], groups: int) -> tuple[float, ...]:
    vector = tuple(float(item) for item in value)
    if len(vector) != groups:
        raise ValueError("incidence prediction width differs from concept beta width")
    if any(not math.isfinite(item) or item < 0.0 for item in vector):
        raise ValueError("incidence predictions must be finite and nonnegative")
    return vector


def build_coverage_oracle(
    cohort: Sequence[str],
    bundles: Sequence[ImmutableBundleProposal],
    history: CausalRequestHistory,
    allocation_event_index: int,
    incidence_predictions: Mapping[tuple[str, str], Sequence[float]],
    concept_betas: Mapping[str, Sequence[float]],
    calibration: CalibrationReceipt,
    *,
    cold_start_handles: Sequence[str] = (),
    maximum_feature_event_index: int,
) -> tuple[CoverageOracle, UtilityAudit]:
    """Build the exact oracle from features available no later than this event."""

    enforce_calibration(calibration)
    canonical_cohort = tuple(sorted(cohort))
    if not canonical_cohort or len(set(canonical_cohort)) != len(canonical_cohort):
        raise ValueError("oracle cohort must be nonempty and unique")
    if tuple(cohort) != canonical_cohort:
        raise ValueError("oracle cohort must use canonical order")
    if type(allocation_event_index) is not int or allocation_event_index < 0:
        raise ValueError("allocation event index must be nonnegative")
    if (
        type(maximum_feature_event_index) is not int
        or maximum_feature_event_index < 0
        or maximum_feature_event_index > allocation_event_index
    ):
        raise ValueError("future feature reached oracle construction")
    if set(concept_betas) != set(canonical_cohort):
        raise ValueError("concept beta rows must exactly match the admitted cohort")
    beta_rows = {
        handle: tuple(float(value) for value in concept_betas[handle])
        for handle in canonical_cohort
    }
    if any(
        not row or any(not math.isfinite(value) or value < 0.0 for value in row)
        for row in beta_rows.values()
    ):
        raise ValueError("concept beta rows must be finite nonnegative vectors")
    group_widths = {handle: len(row) for handle, row in beta_rows.items()}

    cold = tuple(sorted(cold_start_handles))
    if len(cold) > 1 or len(set(cold)) != len(cold) or not set(cold) <= set(canonical_cohort):
        raise ValueError("cold start must name at most the current admitted concept")
    request_weights = {
        handle: history.weight(handle, allocation_event_index)
        + (1.0 if handle in cold else 0.0)
        for handle in canonical_cohort
    }

    observed_incidence_keys = {
        (edge.handle, edge.packet_id)
        for bundle in bundles
        for edge in bundle.incidences
        if edge.handle in set(canonical_cohort)
    }
    if set(incidence_predictions) != observed_incidence_keys:
        raise ValueError("incidence predictions must exactly match proposal incidences")
    packet_bundles: dict[str, PacketBundle] = {}
    for proposal in bundles:
        if proposal.packet.packet_id in packet_bundles:
            raise ValueError("proposal repeats a packet bundle")
        gains = {
            handle: tuple(0.0 for _ in range(group_widths[handle]))
            for handle in canonical_cohort
        }
        for edge in proposal.incidences:
            if edge.handle not in gains:
                continue
            gains[edge.handle] = _gain_vector(
                incidence_predictions[(edge.handle, edge.packet_id)],
                group_widths[edge.handle],
            )
        packet_bundles[proposal.packet.packet_id] = PacketBundle(
            packet_id=proposal.packet.packet_id,
            cost_bytes=proposal.measured_cost_bytes(),
            gains=gains,
        )
    oracle = CoverageOracle(
        bundles=packet_bundles,
        request_weights=request_weights,
        group_weights=beta_rows,
    )
    audit = UtilityAudit(
        allocation_event_index=allocation_event_index,
        maximum_feature_event_index=maximum_feature_event_index,
        cold_start_handles=cold,
        request_weights=cast(Mapping[str, float], request_weights),
    )
    return oracle, audit
