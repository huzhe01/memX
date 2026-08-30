from __future__ import annotations

from pathlib import Path

import pytest

from ratemem.evaluation.statistics import (
    CalibrationLeakageError,
    CalibrationRecord,
    PairedPilotEffect,
    plan_required_units,
)


def _calibration_record() -> CalibrationRecord:
    effects = (-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06) * 3
    rows = tuple(
        PairedPilotEffect(
            inference_unit_id=f"calibration_unit_{index:03d}",
            metric_id="identity_similarity_delta",
            paired_effect=effect,
            source_artifact_sha256=f"{index + 1:064x}",
        )
        for index, effect in enumerate(effects)
    )
    return CalibrationRecord.create(
        dataset_lock_id="1" * 64,
        evaluator_revision="2" * 40,
        pool_sha256="3" * 64,
        split="calibration",
        rows=rows,
    )


def test_required_units_uses_larger_ci_or_power_requirement() -> None:
    record = _calibration_record()
    result = plan_required_units(
        record,
        maximum_half_width=0.02,
        minimum_effect=0.03,
        alpha=0.05,
        power=0.80,
        minimum_units=12,
        simulation_seed=314159,
        monte_carlo_draws=512,
    )

    assert result.required_units == max(
        result.ci_required_units,
        result.power_required_units,
    )
    assert result.calibration_pool_sha256 == record.pool_sha256
    assert result.calibration_record_sha256 == record.record_sha256
    assert result.search_curve


def test_power_record_rejects_final_test_concepts() -> None:
    leaked = _calibration_record().model_copy(update={"split": "final_test"})
    with pytest.raises(CalibrationLeakageError, match="calibration split"):
        plan_required_units(
            leaked,
            maximum_half_width=0.02,
            minimum_effect=0.03,
            alpha=0.05,
            power=0.80,
            minimum_units=12,
            simulation_seed=314159,
            monte_carlo_draws=256,
        )


def test_calibration_schema_command_is_registered(tmp_path: Path) -> None:
    from ratemem.evaluation.cli import app

    assert app.registered_groups
