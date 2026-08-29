from __future__ import annotations

from decimal import Decimal

import pytest

from ratemem.pilot.probes import ALLOWED_PROBES, held_in_step_cap, percentile


def test_probe_set_is_closed_and_has_no_scientific_endpoint() -> None:
    assert ALLOWED_PROBES == (
        "checkpoint_compatibility",
        "dynamic_numerics",
        "gradient_flow",
        "frozen_backbone",
        "peak_memory",
        "one_step_inference",
        "one_timestep_backward",
        "step_timing",
        "held_in_loss",
    )
    forbidden = {
        "identity",
        "kid",
        "fid",
        "memory_policy",
        "lifecycle",
        "composition",
        "augmentation",
        "validation",
        "scientific",
    }
    assert not any(token in probe for token in forbidden for probe in ALLOWED_PROBES)


def test_percentiles_use_nearest_rank_without_interpolation() -> None:
    values = [float(value) for value in range(1, 21)]
    assert percentile(values, 0.50) == 10.0
    assert percentile(values, 0.95) == 19.0
    assert percentile(list(reversed(values)), 1.0) == 20.0


@pytest.mark.parametrize(
    ("values", "quantile", "error"),
    [
        ([], 0.5, ValueError),
        ([1.0], 0.0, ValueError),
        ([1.0], 1.01, ValueError),
        ([1.0, float("nan")], 0.5, ValueError),
        ([1.0, float("inf")], 0.5, ValueError),
        ([1], 0.5, TypeError),
        ([1.0], True, TypeError),
    ],
)
def test_percentile_rejects_empty_nonfinite_and_nonexact_inputs(
    values: list[float], quantile: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        percentile(values, quantile)


def test_step_cap_is_minimum_of_dollars_and_timeout() -> None:
    cost_limited = held_in_step_cap(
        p95_step_seconds=Decimal("2.0"),
        remaining_compute_usd=Decimal("4.00"),
        requested_resource_usd_per_second=Decimal("0.001"),
        remaining_timeout_seconds=3000,
        shutdown_reserve_seconds=120,
    )
    timeout_limited = held_in_step_cap(
        p95_step_seconds=Decimal("2.0"),
        remaining_compute_usd=Decimal("100.00"),
        requested_resource_usd_per_second=Decimal("0.001"),
        remaining_timeout_seconds=3000,
        shutdown_reserve_seconds=120,
    )
    assert cost_limited == 1440
    assert timeout_limited == 1440


def test_step_cap_allows_zero_remaining_budget_or_usable_timeout() -> None:
    common = {
        "p95_step_seconds": Decimal("2.0"),
        "requested_resource_usd_per_second": Decimal("0.001"),
        "shutdown_reserve_seconds": 120,
    }
    assert (
        held_in_step_cap(
            **common,
            remaining_compute_usd=Decimal("0"),
            remaining_timeout_seconds=3000,
        )
        == 0
    )
    assert (
        held_in_step_cap(
            **common,
            remaining_compute_usd=Decimal("4"),
            remaining_timeout_seconds=120,
        )
        == 0
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("p95_step_seconds", Decimal("0"), ValueError),
        ("p95_step_seconds", Decimal("NaN"), ValueError),
        ("p95_step_seconds", Decimal("Infinity"), ValueError),
        ("remaining_compute_usd", Decimal("-0.01"), ValueError),
        ("remaining_compute_usd", Decimal("NaN"), ValueError),
        ("requested_resource_usd_per_second", Decimal("0"), ValueError),
        ("requested_resource_usd_per_second", Decimal("-1"), ValueError),
        ("remaining_timeout_seconds", -1, ValueError),
        ("shutdown_reserve_seconds", -1, ValueError),
        ("p95_step_seconds", 2.0, TypeError),
        ("remaining_compute_usd", "4.00", TypeError),
        ("remaining_timeout_seconds", True, TypeError),
    ],
)
def test_step_cap_rejects_non_decimal_nonfinite_and_negative_inputs(
    field: str, value: object, error: type[Exception]
) -> None:
    arguments: dict[str, object] = {
        "p95_step_seconds": Decimal("2.0"),
        "remaining_compute_usd": Decimal("4.00"),
        "requested_resource_usd_per_second": Decimal("0.001"),
        "remaining_timeout_seconds": 3000,
        "shutdown_reserve_seconds": 120,
    }
    arguments[field] = value
    with pytest.raises(error):
        held_in_step_cap(**arguments)  # type: ignore[arg-type]
