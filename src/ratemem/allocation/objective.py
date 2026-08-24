from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Real
from types import MappingProxyType

ExactCoverage = dict[tuple[str, int], Fraction]


def _nonempty_id(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must be nonempty")
    return value


def _nonnegative_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be finite and nonnegative") from error
    if normalized < 0.0 or not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite and nonnegative")
    return normalized


def _nonempty_vector(value: object, label: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a numeric sequence")
    if not value:
        raise ValueError(f"{label} must be nonempty")
    return tuple(
        _nonnegative_real(scalar, f"{label} scalar") for scalar in value
    )


@dataclass(frozen=True, slots=True)
class PacketBundle:
    packet_id: str
    cost_bytes: int
    gains: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        _nonempty_id(self.packet_id, "packet id")
        if type(self.cost_bytes) is not int:
            raise TypeError("packet cost must be an integer byte count")
        if self.cost_bytes <= 0:
            raise ValueError("packet cost must be positive")

        raw_gains: object = self.gains
        if not isinstance(raw_gains, Mapping):
            raise TypeError("gains must be a mapping")
        if not raw_gains:
            raise ValueError("packet gains must contain at least one incidence")
        normalized_gains: dict[str, tuple[float, ...]] = {}
        for raw_handle, raw_values in raw_gains.items():
            handle = _nonempty_id(raw_handle, "concept id")
            normalized_gains[handle] = _nonempty_vector(raw_values, "gain vector")
        object.__setattr__(
            self,
            "gains",
            MappingProxyType(dict(sorted(normalized_gains.items()))),
        )


@dataclass(frozen=True, slots=True)
class CoverageOracle:
    bundles: Mapping[str, PacketBundle]
    request_weights: Mapping[str, float]
    group_weights: Mapping[str, tuple[float, ...]]
    _exact_gains: Mapping[str, Mapping[str, tuple[Fraction, ...]]] = field(
        init=False, repr=False, compare=False
    )
    _exact_coefficients: Mapping[str, tuple[Fraction, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        raw_bundles: object = self.bundles
        raw_request_weights: object = self.request_weights
        raw_group_weights: object = self.group_weights
        if not isinstance(raw_bundles, Mapping):
            raise TypeError("bundles must be a mapping")
        if not isinstance(raw_request_weights, Mapping):
            raise TypeError("request_weights must be a mapping")
        if not isinstance(raw_group_weights, Mapping):
            raise TypeError("group_weights must be a mapping")

        normalized_bundles: dict[str, PacketBundle] = {}
        for raw_key, raw_bundle in raw_bundles.items():
            key = _nonempty_id(raw_key, "bundle id")
            if not isinstance(raw_bundle, PacketBundle):
                raise TypeError("bundle values must be PacketBundle instances")
            if key != raw_bundle.packet_id:
                raise ValueError("bundle map key must equal packet_id")
            normalized_bundles[key] = raw_bundle

        normalized_request_weights: dict[str, float] = {}
        for raw_handle, raw_weight in raw_request_weights.items():
            handle = _nonempty_id(raw_handle, "concept id")
            normalized_request_weights[handle] = _nonnegative_real(
                raw_weight, "request weight"
            )

        normalized_group_weights: dict[str, tuple[float, ...]] = {}
        for raw_handle, raw_weights in raw_group_weights.items():
            handle = _nonempty_id(raw_handle, "concept id")
            normalized_group_weights[handle] = _nonempty_vector(
                raw_weights, "group weight vector"
            )

        if set(normalized_request_weights) != set(normalized_group_weights):
            raise ValueError("request and group weights must name the same concepts")
        for bundle in normalized_bundles.values():
            for handle, gains in bundle.gains.items():
                if handle not in normalized_group_weights:
                    raise ValueError(f"packet gain names unknown concept: {handle}")
                if len(gains) > len(normalized_group_weights[handle]):
                    raise ValueError(f"packet gain exceeds group width: {handle}")

        exact_gains: dict[str, Mapping[str, tuple[Fraction, ...]]] = {}
        for packet_id, bundle in normalized_bundles.items():
            exact_gains[packet_id] = MappingProxyType(
                {
                    handle: tuple(Fraction.from_float(gain) for gain in gains)
                    for handle, gains in bundle.gains.items()
                }
            )

        exact_coefficients: dict[str, tuple[Fraction, ...]] = {}
        for handle, weight in normalized_request_weights.items():
            exact_weight = Fraction.from_float(weight)
            handle_coefficients = []
            for beta in normalized_group_weights[handle]:
                coefficient = exact_weight * Fraction.from_float(beta)
                try:
                    reporting_coefficient = float(coefficient)
                except OverflowError as error:
                    raise ValueError(
                        "oracle coefficient must be representable as a finite float for reporting"
                    ) from error
                if not math.isfinite(reporting_coefficient):
                    raise ValueError(
                        "oracle coefficient must be representable as a finite float for reporting"
                    )
                handle_coefficients.append(coefficient)
            exact_coefficients[handle] = tuple(handle_coefficients)

        maximum_objective = sum(
            (
                coefficient
                for coefficients in exact_coefficients.values()
                for coefficient in coefficients
            ),
            start=Fraction(),
        )
        try:
            reporting_maximum = float(maximum_objective)
        except OverflowError as error:
            raise ValueError(
                "maximum objective mass must be representable as a finite float for reporting"
            ) from error
        if not math.isfinite(reporting_maximum):
            raise ValueError(
                "maximum objective mass must be representable as a finite float for reporting"
            )

        object.__setattr__(
            self,
            "bundles",
            MappingProxyType(dict(sorted(normalized_bundles.items()))),
        )
        object.__setattr__(
            self,
            "request_weights",
            MappingProxyType(dict(sorted(normalized_request_weights.items()))),
        )
        object.__setattr__(
            self,
            "group_weights",
            MappingProxyType(dict(sorted(normalized_group_weights.items()))),
        )
        object.__setattr__(
            self,
            "_exact_gains",
            MappingProxyType(dict(sorted(exact_gains.items()))),
        )
        object.__setattr__(
            self,
            "_exact_coefficients",
            MappingProxyType(dict(sorted(exact_coefficients.items()))),
        )

    def _selected_ids(self, selected: frozenset[str]) -> tuple[str, ...]:
        selected_ids = tuple(sorted(selected))
        for packet_id in selected_ids:
            self.bundles[packet_id]
        return selected_ids

    def _empty_exact_coverage(self) -> ExactCoverage:
        return {
            (handle, group): Fraction()
            for handle, coefficients in self._exact_coefficients.items()
            for group in range(len(coefficients))
        }

    def _add_exact_gains(self, coverage: ExactCoverage, item: str) -> None:
        for handle, gains in self._exact_gains[item].items():
            for group, gain in enumerate(gains):
                key = (handle, group)
                coverage[key] = min(Fraction(1), coverage[key] + gain)

    def _exact_coverage(self, selected_ids: tuple[str, ...]) -> ExactCoverage:
        coverage = self._empty_exact_coverage()
        for item in selected_ids:
            self._add_exact_gains(coverage, item)
        return coverage

    def _exact_value_from_coverage(self, coverage: Mapping[tuple[str, int], Fraction]) -> Fraction:
        return sum(
            (
                coefficient * coverage[(handle, group)]
                for handle, coefficients in self._exact_coefficients.items()
                for group, coefficient in enumerate(coefficients)
            ),
            start=Fraction(),
        )

    def _exact_marginal_from_coverage(
        self, coverage: Mapping[tuple[str, int], Fraction], item: str
    ) -> Fraction:
        terms = []
        bundle_gains = self._exact_gains[item]
        for handle, coefficients in self._exact_coefficients.items():
            item_gains = bundle_gains.get(handle, ())
            for group, coefficient in enumerate(coefficients):
                item_gain = item_gains[group] if group < len(item_gains) else Fraction()
                remaining = Fraction(1) - coverage[(handle, group)]
                terms.append(coefficient * min(remaining, item_gain))
        return sum(terms, start=Fraction())

    def exact_value(self, selected: frozenset[str]) -> Fraction:
        """Return certified utility over the exact binary-rational normalized inputs."""
        selected_ids = self._selected_ids(selected)
        return self._exact_value_from_coverage(self._exact_coverage(selected_ids))

    def exact_marginal(self, selected: frozenset[str], item: str) -> Fraction:
        """Return a direct exact marginal without subtracting rounded reporting values."""
        selected_ids = self._selected_ids(selected)
        self.bundles[item]
        if item in selected:
            return Fraction()
        return self._exact_marginal_from_coverage(self._exact_coverage(selected_ids), item)

    def value(self, selected: frozenset[str]) -> float:
        """Return a rounded float report; certification uses exact_value instead."""
        return float(self.exact_value(selected))

    def marginal(self, selected: frozenset[str], item: str) -> float:
        """Return a rounded float report; certification uses exact_marginal instead."""
        return float(self.exact_marginal(selected, item))

    def cost(self, selected: frozenset[str]) -> int:
        return sum(
            self.bundles[item].cost_bytes for item in self._selected_ids(selected)
        )
