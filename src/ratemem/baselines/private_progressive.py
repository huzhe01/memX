"""Private progressive-code rate allocation controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class RateChoice:
    handle: str
    prefix_length: int
    serialized_bytes: int
    value: Decimal

    def __post_init__(self) -> None:
        if not self.handle:
            raise ValueError("rate choice handle must be non-empty")
        if type(self.prefix_length) is not int or self.prefix_length < 0:
            raise ValueError("rate choice prefix must be nonnegative")
        if type(self.serialized_bytes) is not int or self.serialized_bytes < 0:
            raise ValueError("rate choice bytes must be nonnegative")
        if type(self.value) is not Decimal or not self.value.is_finite() or self.value < 0:
            raise ValueError("rate choice value must be a finite nonnegative Decimal")


@dataclass(frozen=True, slots=True)
class RateAllocation:
    prefix_by_handle: dict[str, int]
    total_bytes: int
    total_value: Decimal


def _validate_options(options: dict[str, Sequence[RateChoice]]) -> None:
    if not options:
        raise ValueError("rate options must contain at least one handle")
    for handle, choices in options.items():
        if not handle or not choices:
            raise ValueError("every handle requires at least one rate choice")
        if any(choice.handle != handle for choice in choices):
            raise ValueError("rate option map key differs from its choice handle")
        prefixes = tuple(choice.prefix_length for choice in choices)
        if prefixes != tuple(range(len(prefixes))):
            raise ValueError("rate choices must enumerate every legal prefix from zero")
        if any(
            right.serialized_bytes < left.serialized_bytes
            for left, right in zip(choices, choices[1:], strict=False)
        ):
            raise ValueError("longer progressive prefixes cannot use fewer bytes")


def exact_separable_allocation(
    options: dict[str, Sequence[RateChoice]],
    budget: int,
) -> RateAllocation:
    """Solve the multiple-choice prefix knapsack exactly with sparse dominance pruning."""

    _validate_options(options)
    if type(budget) is not int or budget < 0:
        raise ValueError("rate budget must be a nonnegative integer")
    frontier: dict[int, tuple[Decimal, tuple[tuple[str, int], ...]]] = {
        0: (Decimal(0), ())
    }
    for handle in sorted(options):
        expanded: dict[int, tuple[Decimal, tuple[tuple[str, int], ...]]] = {}
        for used, (value, choices) in frontier.items():
            for choice in options[handle]:
                candidate_bytes = used + choice.serialized_bytes
                if candidate_bytes > budget:
                    continue
                candidate = (
                    value + choice.value,
                    choices + ((handle, choice.prefix_length),),
                )
                incumbent = expanded.get(candidate_bytes)
                if incumbent is None or candidate[0] > incumbent[0] or (
                    candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
                ):
                    expanded[candidate_bytes] = candidate
        if not expanded:
            raise ValueError("no feasible separable rate allocation")
        best_value = Decimal("-Infinity")
        frontier = {}
        for used in sorted(expanded):
            value, choices = expanded[used]
            if value > best_value:
                frontier[used] = (value, choices)
                best_value = value
    used, (value, choices) = min(
        frontier.items(),
        key=lambda row: (-row[1][0], row[0], row[1][1]),
    )
    return RateAllocation(dict(choices), used, value)


__all__ = ["RateAllocation", "RateChoice", "exact_separable_allocation"]
