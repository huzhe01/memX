import math
import random
from collections.abc import Callable
from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.allocation.oracle import exhaustive_optimum
from ratemem.allocation.snapshot import allocate_snapshot

Allocator = Callable[[CoverageOracle, int], frozenset[str]]


@given(
    costs=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=9),
    values=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=9),
)
def test_allocator_meets_certified_factor(costs: list[int], values: list[float]) -> None:
    size = min(len(costs), len(values))
    bundles = {
        f"p{index}": PacketBundle(f"p{index}", costs[index], {"a": (values[index],)})
        for index in range(size)
    }
    oracle = CoverageOracle(bundles, {"a": 1.0}, {"a": (1.0,)})
    budget = max(1, sum(costs[:size]) // 2)
    chosen = allocate_snapshot(oracle, budget)
    optimum = exhaustive_optimum(oracle, budget)
    assert oracle.cost(chosen) <= budget
    assert oracle.value(chosen) + 1e-9 >= (1.0 - 1.0 / math.e) * oracle.value(optimum)
    assert allocate_snapshot(oracle, budget) == chosen


def test_allocator_factor_on_exhaustive_scalar_grid() -> None:
    for costs in product((1, 2), repeat=4):
        for values in product((0.0, 0.5, 1.0), repeat=4):
            bundles = {
                f"p{index}": PacketBundle(
                    f"p{index}", costs[index], {"a": (values[index],)}
                )
                for index in range(4)
            }
            oracle = CoverageOracle(bundles, {"a": 1.0}, {"a": (1.0,)})
            for budget in range(9):
                chosen = allocate_snapshot(oracle, budget)
                optimum = exhaustive_optimum(oracle, budget)
                assert oracle.cost(chosen) <= budget
                assert oracle.value(chosen) + 1e-9 >= (
                    (1.0 - 1.0 / math.e) * oracle.value(optimum)
                )


def test_allocator_factor_on_seeded_multiconcept_instances() -> None:
    rng = random.Random(20260824)
    concepts = ("a", "b", "c")
    for _ in range(40):
        bundles = {
            f"p{index}": PacketBundle(
                f"p{index}",
                rng.randint(1, 10),
                {handle: tuple(rng.random() for _ in range(3)) for handle in concepts},
            )
            for index in range(rng.randint(1, 9))
        }
        oracle = CoverageOracle(
            bundles,
            {handle: rng.random() for handle in concepts},
            {handle: tuple(rng.random() for _ in range(3)) for handle in concepts},
        )
        budget = rng.randint(1, sum(bundle.cost_bytes for bundle in bundles.values()))
        chosen = allocate_snapshot(oracle, budget)
        optimum = exhaustive_optimum(oracle, budget)
        assert oracle.cost(chosen) <= budget
        assert oracle.value(chosen) + 1e-9 >= (
            (1.0 - 1.0 / math.e) * oracle.value(optimum)
        )


def _single_group_oracle(packet_ids: tuple[str, ...]) -> CoverageOracle:
    return CoverageOracle(
        bundles={
            packet_id: PacketBundle(packet_id, cost_bytes=1, gains={"a": (1.0,)})
            for packet_id in packet_ids
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )


@pytest.mark.parametrize("allocator", [allocate_snapshot, exhaustive_optimum])
@pytest.mark.parametrize("budget", [True, 1.0, "1"], ids=["bool", "float", "string"])
def test_budget_requires_an_exact_integer(allocator: Allocator, budget: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        allocator(_single_group_oracle(("p",)), budget)  # type: ignore[arg-type]


@pytest.mark.parametrize("allocator", [allocate_snapshot, exhaustive_optimum])
def test_budget_must_be_nonnegative(allocator: Allocator) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        allocator(_single_group_oracle(("p",)), -1)


def test_exhaustive_optimum_rejects_more_than_24_bundles() -> None:
    oracle = _single_group_oracle(tuple(f"p{index:02d}" for index in range(25)))
    with pytest.raises(ValueError, match="at most 24"):
        exhaustive_optimum(oracle, budget_bytes=1)


def test_allocators_use_deterministic_packet_id_tie_breaking() -> None:
    oracle = _single_group_oracle(("z", "a"))
    expected = frozenset({"z"})
    assert exhaustive_optimum(oracle, budget_bytes=1) == expected
    assert allocate_snapshot(oracle, budget_bytes=1) == expected


@pytest.mark.parametrize("allocator", [allocate_snapshot, exhaustive_optimum])
def test_zero_budget_returns_the_empty_feasible_allocation(allocator: Allocator) -> None:
    oracle = _single_group_oracle(("p",))
    chosen = allocator(oracle, 0)
    assert chosen == frozenset()
    assert oracle.cost(chosen) == 0
