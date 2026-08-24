from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType


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

        coefficients = []
        for handle, weight in normalized_request_weights.items():
            for beta in normalized_group_weights[handle]:
                coefficient = weight * beta
                if not math.isfinite(coefficient):
                    raise ValueError("oracle coefficient must be finite")
                coefficients.append(coefficient)
        try:
            maximum_objective = math.fsum(coefficients)
        except OverflowError as error:
            raise ValueError("maximum objective mass must be finite") from error
        if not math.isfinite(maximum_objective):
            raise ValueError("maximum objective mass must be finite")

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

    def _selected_ids(self, selected: frozenset[str]) -> tuple[str, ...]:
        selected_ids = tuple(sorted(selected))
        for packet_id in selected_ids:
            self.bundles[packet_id]
        return selected_ids

    def _coverage(
        self, selected_ids: tuple[str, ...], handle: str, group: int
    ) -> float:
        gains = []
        for packet_id in selected_ids:
            vector = self.bundles[packet_id].gains.get(handle, ())
            if group < len(vector):
                gain = vector[group]
                if gain >= 1.0:
                    return 1.0
                gains.append(gain)
        return min(1.0, math.fsum(gains))

    def value(self, selected: frozenset[str]) -> float:
        selected_ids = self._selected_ids(selected)
        terms = []
        for handle, weight in self.request_weights.items():
            for group, beta in enumerate(self.group_weights[handle]):
                terms.append(
                    weight * beta * self._coverage(selected_ids, handle, group)
                )
        return math.fsum(terms)

    def marginal(self, selected: frozenset[str], item: str) -> float:
        selected_ids = self._selected_ids(selected)
        bundle = self.bundles[item]
        if item in selected:
            return 0.0

        terms = []
        for handle, weight in self.request_weights.items():
            item_gains = bundle.gains.get(handle, ())
            for group, beta in enumerate(self.group_weights[handle]):
                item_gain = item_gains[group] if group < len(item_gains) else 0.0
                remaining = 1.0 - self._coverage(selected_ids, handle, group)
                terms.append(weight * beta * min(remaining, item_gain))
        return math.fsum(terms)

    def cost(self, selected: frozenset[str]) -> int:
        return sum(
            self.bundles[item].cost_bytes for item in self._selected_ids(selected)
        )
