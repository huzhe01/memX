from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class PacketBundle:
    packet_id: str
    cost_bytes: int
    gains: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gains",
            MappingProxyType(
                {handle: tuple(values) for handle, values in self.gains.items()}
            ),
        )
        if self.cost_bytes <= 0:
            raise ValueError("packet cost must be positive")
        if any(
            value < 0.0 or not math.isfinite(value)
            for rows in self.gains.values()
            for value in rows
        ):
            raise ValueError("certified packet gains must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class CoverageOracle:
    bundles: Mapping[str, PacketBundle]
    request_weights: Mapping[str, float]
    group_weights: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundles", MappingProxyType(dict(self.bundles)))
        object.__setattr__(
            self, "request_weights", MappingProxyType(dict(self.request_weights))
        )
        object.__setattr__(
            self,
            "group_weights",
            MappingProxyType(
                {handle: tuple(values) for handle, values in self.group_weights.items()}
            ),
        )
        scalars = list(self.request_weights.values()) + [
            value for rows in self.group_weights.values() for value in rows
        ]
        if any(value < 0.0 or not math.isfinite(value) for value in scalars):
            raise ValueError("oracle weights must be finite and nonnegative")
        if set(self.request_weights) != set(self.group_weights):
            raise ValueError("request and group weights must name the same concepts")
        if any(key != bundle.packet_id for key, bundle in self.bundles.items()):
            raise ValueError("bundle map key must equal packet_id")
        for bundle in self.bundles.values():
            for handle, gains in bundle.gains.items():
                if handle not in self.group_weights:
                    raise ValueError(f"packet gain names unknown concept: {handle}")
                if len(gains) > len(self.group_weights[handle]):
                    raise ValueError(f"packet gain exceeds group width: {handle}")

    def value(self, selected: frozenset[str]) -> float:
        total = 0.0
        for handle, weight in self.request_weights.items():
            for group, beta in enumerate(self.group_weights[handle]):
                coverage = sum(
                    self.bundles[item].gains.get(handle, ())[group]
                    if group < len(self.bundles[item].gains.get(handle, ()))
                    else 0.0
                    for item in selected
                )
                total += weight * beta * min(1.0, coverage)
        return total

    def marginal(self, selected: frozenset[str], item: str) -> float:
        return self.value(selected | {item}) - self.value(selected)

    def cost(self, selected: frozenset[str]) -> int:
        return sum(self.bundles[item].cost_bytes for item in selected)
