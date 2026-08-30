"""Calibration-only sample-size planning for locked deployment episodes."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import canonical_json_bytes
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class CalibrationLeakageError(ValueError):
    """Raised when sample-size planning consumes a non-calibration split."""


class PairedPilotEffect(BaseModel):
    model_config = _MODEL_CONFIG

    inference_unit_id: str
    metric_id: str
    paired_effect: float
    source_artifact_sha256: Sha256

    @field_validator("inference_unit_id", "metric_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value or value != value.strip() or not value.replace("_", "").isalnum():
            raise ValueError("calibration identifiers must be canonical alphanumeric text")
        return value

    @field_validator("paired_effect")
    @classmethod
    def validate_effect(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("paired effect must be finite")
        return value


class CalibrationRecord(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    record_sha256: Sha256
    dataset_lock_id: Sha256
    evaluator_revision: GitCommit
    pool_sha256: Sha256
    split: Literal["calibration"]
    rows: tuple[PairedPilotEffect, ...]

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("record_sha256")
        return canonical_json_bytes(payload)

    @model_validator(mode="after")
    def validate_rows(self) -> CalibrationRecord:
        if len(self.rows) < 4:
            raise ValueError("calibration record requires at least four paired rows")
        if len({row.source_artifact_sha256 for row in self.rows}) != len(self.rows):
            raise ValueError("calibration source artifact hashes must be unique")
        if len({row.metric_id for row in self.rows}) != 1:
            raise ValueError("one calibration record must contain exactly one metric")
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_lock_id: str,
        evaluator_revision: str,
        pool_sha256: str,
        split: Literal["calibration"],
        rows: Sequence[PairedPilotEffect],
    ) -> CalibrationRecord:
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.inference_unit_id,
                    row.source_artifact_sha256,
                ),
            )
        )
        provisional = cls(
            schema_version="1.0",
            record_sha256="0" * 64,
            dataset_lock_id=dataset_lock_id,
            evaluator_revision=evaluator_revision,
            pool_sha256=pool_sha256,
            split=split,
            rows=ordered,
        )
        return provisional.model_copy(
            update={
                "record_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()
            }
        )


class PowerSearchPoint(BaseModel):
    model_config = _MODEL_CONFIG

    units: PositiveInt
    ci_half_width: PositiveFloat
    simulated_power: float = Field(ge=0.0, le=1.0)


class RequiredUnits(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    calibration_record_sha256: Sha256
    calibration_pool_sha256: Sha256
    maximum_half_width: PositiveFloat
    minimum_effect: PositiveFloat
    alpha: float = Field(gt=0.0, lt=1.0)
    target_power: float = Field(gt=0.0, lt=1.0)
    minimum_units: PositiveInt
    simulation_seed: int
    monte_carlo_draws: PositiveInt
    ci_required_units: PositiveInt
    power_required_units: PositiveInt
    required_units: PositiveInt
    search_curve: tuple[PowerSearchPoint, ...]
    record_sha256: Sha256

    @property
    def semantic_bytes(self) -> bytes:
        payload = self.model_dump(mode="json")
        payload.pop("record_sha256")
        return canonical_json_bytes(payload)


def _unit_effects(record: CalibrationRecord) -> NDArray[np.float64]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in record.rows:
        grouped[row.inference_unit_id].append(row.paired_effect)
    effects = np.asarray(
        [float(np.mean(grouped[key])) for key in sorted(grouped)],
        dtype=np.float64,
    )
    if effects.size < 4 or float(np.std(effects, ddof=1)) <= 0.0:
        raise ValueError("calibration requires at least four variable inference units")
    return effects


def _simulate_point(
    centered_effects: NDArray[np.float64],
    *,
    units: int,
    minimum_effect: float,
    alpha: float,
    draws: int,
    seed: int,
) -> PowerSearchPoint:
    generator = np.random.default_rng(np.random.SeedSequence([seed, units]))
    indices = generator.integers(
        0,
        centered_effects.size,
        size=(draws, units),
        endpoint=False,
    )
    sampled = centered_effects[indices]
    means = sampled.mean(axis=1)
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    half_width = float((upper - lower) / 2.0)
    shifted = sampled + minimum_effect
    shifted_means = shifted.mean(axis=1)
    standard_errors = shifted.std(axis=1, ddof=1) / math.sqrt(units)
    valid = standard_errors > 0.0
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rejections = np.zeros(draws, dtype=bool)
    rejections[valid] = np.abs(shifted_means[valid] / standard_errors[valid]) >= critical
    return PowerSearchPoint(
        units=units,
        ci_half_width=max(half_width, float(np.finfo(np.float64).eps)),
        simulated_power=float(np.mean(rejections)),
    )


def plan_required_units(
    record: CalibrationRecord,
    maximum_half_width: float,
    minimum_effect: float,
    alpha: float,
    power: float,
    minimum_units: int,
    simulation_seed: int,
    *,
    monte_carlo_draws: int = 2048,
) -> RequiredUnits:
    """Find separate CI-width and power requirements using cluster resampling."""

    if type(record) is not CalibrationRecord or record.split != "calibration":
        raise CalibrationLeakageError("power planning accepts only the calibration split")
    if hashlib.sha256(record.semantic_bytes).hexdigest() != record.record_sha256:
        raise ValueError("calibration record content hash changed")
    for name, value in (
        ("maximum_half_width", maximum_half_width),
        ("minimum_effect", minimum_effect),
    ):
        if type(value) is not float or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive exact float")
    for name, value in (("alpha", alpha), ("power", power)):
        if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be a probability exact float")
    if type(minimum_units) is not int or minimum_units < 2:
        raise ValueError("minimum_units must be an exact int of at least two")
    if type(simulation_seed) is not int or not 0 <= simulation_seed < 2**63:
        raise ValueError("simulation_seed must be a nonnegative signed 64-bit exact int")
    if type(monte_carlo_draws) is not int or monte_carlo_draws < 128:
        raise ValueError("monte_carlo_draws must be an exact int of at least 128")

    effects = _unit_effects(record)
    centered = effects - effects.mean()
    curve: list[PowerSearchPoint] = []
    ci_required: int | None = None
    power_required: int | None = None
    for units in range(minimum_units, 4097):
        point = _simulate_point(
            centered,
            units=units,
            minimum_effect=minimum_effect,
            alpha=alpha,
            draws=monte_carlo_draws,
            seed=simulation_seed,
        )
        curve.append(point)
        if ci_required is None and point.ci_half_width <= maximum_half_width:
            ci_required = units
        if power_required is None and point.simulated_power >= power:
            power_required = units
        if ci_required is not None and power_required is not None:
            break
    if ci_required is None or power_required is None:
        raise RuntimeError("power planning did not converge within 4096 inference units")
    required = max(ci_required, power_required)
    provisional = RequiredUnits(
        schema_version="1.0",
        calibration_record_sha256=record.record_sha256,
        calibration_pool_sha256=record.pool_sha256,
        maximum_half_width=maximum_half_width,
        minimum_effect=minimum_effect,
        alpha=alpha,
        target_power=power,
        minimum_units=minimum_units,
        simulation_seed=simulation_seed,
        monte_carlo_draws=monte_carlo_draws,
        ci_required_units=ci_required,
        power_required_units=power_required,
        required_units=required,
        search_curve=tuple(curve),
        record_sha256="0" * 64,
    )
    return provisional.model_copy(
        update={"record_sha256": hashlib.sha256(provisional.semantic_bytes).hexdigest()}
    )


__all__ = [
    "CalibrationLeakageError",
    "CalibrationRecord",
    "PairedPilotEffect",
    "PowerSearchPoint",
    "RequiredUnits",
    "plan_required_units",
]
