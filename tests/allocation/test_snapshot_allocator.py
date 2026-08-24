import random
from collections.abc import Callable
from fractions import Fraction
from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.allocation.oracle import exhaustive_optimum
from ratemem.allocation.snapshot import allocate_density_greedy_heuristic, allocate_snapshot

Allocator = Callable[[CoverageOracle, int], frozenset[str]]
CERTIFIED_FACTOR_LOWER_BOUND = Fraction(6_321_205_588_285_576, 10**16)


def _assert_certified_ratio(
    oracle: CoverageOracle, chosen: frozenset[str], optimum: frozenset[str]
) -> None:
    assert (
        oracle.exact_value(chosen) * CERTIFIED_FACTOR_LOWER_BOUND.denominator
        >= oracle.exact_value(optimum) * CERTIFIED_FACTOR_LOWER_BOUND.numerator
    )


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
    _assert_certified_ratio(oracle, chosen, optimum)
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
                _assert_certified_ratio(oracle, chosen, optimum)


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
        _assert_certified_ratio(oracle, chosen, optimum)


def test_allocator_certification_uses_exact_underflowed_coefficients() -> None:
    groups = tuple(range(5))
    useful_ids = tuple(chr(ord("a") + group) for group in groups)
    bundles = {
        packet_id: PacketBundle(
            packet_id,
            cost_bytes=1,
            gains={"concept": tuple(0.5 if index == group else 0.0 for index in groups)},
        )
        for group, packet_id in enumerate(useful_ids)
    }
    bundles.update(
        {
            packet_id: PacketBundle(packet_id, cost_bytes=1, gains={"concept": (0.0,) * 5})
            for packet_id in ("y", "z")
        }
    )
    oracle = CoverageOracle(
        bundles,
        request_weights={"concept": 5e-324},
        group_weights={"concept": (1.0,) * 5},
    )

    chosen = allocate_snapshot(oracle, budget_bytes=5)
    optimum = exhaustive_optimum(oracle, budget_bytes=5)

    assert chosen == frozenset(useful_ids)
    _assert_certified_ratio(oracle, chosen, optimum)


def test_density_ranking_preserves_subnormal_marginal_gains() -> None:
    positive_ids = tuple(f"p{index}" for index in range(6))
    bundles = {
        **{
            packet_id: PacketBundle(packet_id, cost_bytes=10, gains={"a": (5e-324,)})
            for packet_id in positive_ids
        },
        "z": PacketBundle("z", cost_bytes=30, gains={"a": (0.0,)}),
    }
    oracle = CoverageOracle(
        bundles,
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )

    chosen = allocate_snapshot(oracle, budget_bytes=60)
    optimum = exhaustive_optimum(oracle, budget_bytes=60)

    assert chosen == frozenset(positive_ids)
    _assert_certified_ratio(oracle, chosen, optimum)


def test_density_ranking_supports_integer_costs_beyond_float_range() -> None:
    huge_cost = 10**400
    oracle = CoverageOracle(
        bundles={
            packet_id: PacketBundle(packet_id, huge_cost, {"a": (0.5,)})
            for packet_id in ("z", "a")
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )

    chosen = allocate_snapshot(oracle, budget_bytes=huge_cost)

    assert chosen == frozenset({"z"})
    assert chosen == exhaustive_optimum(oracle, budget_bytes=huge_cost)


def _single_group_oracle(packet_ids: tuple[str, ...]) -> CoverageOracle:
    return CoverageOracle(
        bundles={
            packet_id: PacketBundle(packet_id, cost_bytes=1, gains={"a": (1.0,)})
            for packet_id in packet_ids
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )


@pytest.mark.parametrize(
    "allocator", [allocate_snapshot, allocate_density_greedy_heuristic, exhaustive_optimum]
)
@pytest.mark.parametrize("budget", [True, 1.0, "1"], ids=["bool", "float", "string"])
def test_budget_requires_an_exact_integer(allocator: Allocator, budget: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        allocator(_single_group_oracle(("p",)), budget)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "allocator", [allocate_snapshot, allocate_density_greedy_heuristic, exhaustive_optimum]
)
def test_budget_must_be_nonnegative(allocator: Allocator) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        allocator(_single_group_oracle(("p",)), -1)


def test_exhaustive_optimum_rejects_more_than_24_bundles() -> None:
    oracle = _single_group_oracle(tuple(f"p{index:02d}" for index in range(25)))
    with pytest.raises(ValueError, match="at most 24"):
        exhaustive_optimum(oracle, budget_bytes=1)


def test_exhaustive_optimum_enforces_default_state_budget_and_allows_override() -> None:
    oracle = _single_group_oracle(tuple(f"p{index:02d}" for index in range(19)))

    with pytest.raises(ValueError, match="max_states"):
        exhaustive_optimum(oracle, budget_bytes=1)

    assert exhaustive_optimum(
        oracle, budget_bytes=1, max_states=2**19
    ) == frozenset({"p18"})


def test_exhaustive_optimum_prefilters_individually_infeasible_bundles() -> None:
    bundles = {
        "feasible": PacketBundle("feasible", 1, {"a": (1.0,)}),
        **{
            f"infeasible-{index:02d}": PacketBundle(
                f"infeasible-{index:02d}", 2, {"a": (1.0,)}
            )
            for index in range(23)
        },
    }
    oracle = CoverageOracle(bundles, {"a": 1.0}, {"a": (1.0,)})

    assert exhaustive_optimum(oracle, budget_bytes=1) == frozenset({"feasible"})


@pytest.mark.parametrize("max_states", [True, 1.5, "16"], ids=["bool", "float", "string"])
def test_exhaustive_state_budget_requires_an_exact_integer(max_states: object) -> None:
    with pytest.raises(TypeError, match="max_states.*integer"):
        exhaustive_optimum(
            _single_group_oracle(("p",)),
            budget_bytes=1,
            max_states=max_states,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_states", [0, -1])
def test_exhaustive_state_budget_must_be_positive(max_states: int) -> None:
    with pytest.raises(ValueError, match="max_states.*positive"):
        exhaustive_optimum(
            _single_group_oracle(("p",)), budget_bytes=1, max_states=max_states
        )


def test_certified_allocator_cap_has_explicit_override_and_no_heuristic_fallback() -> None:
    oracle = _single_group_oracle(tuple(f"p{index:02d}" for index in range(25)))

    with pytest.raises(ValueError, match="max_bundles"):
        allocate_snapshot(oracle, budget_bytes=1)

    expected = frozenset({"p24"})
    assert allocate_snapshot(oracle, budget_bytes=1, max_bundles=25) == expected
    assert allocate_density_greedy_heuristic(oracle, budget_bytes=1) == expected


@pytest.mark.parametrize("max_bundles", [True, 1.5, "24"], ids=["bool", "float", "string"])
def test_certified_bundle_cap_requires_an_exact_integer(max_bundles: object) -> None:
    with pytest.raises(TypeError, match="max_bundles.*integer"):
        allocate_snapshot(
            _single_group_oracle(("p",)),
            budget_bytes=1,
            max_bundles=max_bundles,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_bundles", [0, -1])
def test_certified_bundle_cap_must_be_positive(max_bundles: int) -> None:
    with pytest.raises(ValueError, match="max_bundles.*positive"):
        allocate_snapshot(
            _single_group_oracle(("p",)), budget_bytes=1, max_bundles=max_bundles
        )


def test_allocators_use_deterministic_packet_id_tie_breaking() -> None:
    oracle = _single_group_oracle(("z", "a"))
    expected = frozenset({"z"})
    assert exhaustive_optimum(oracle, budget_bytes=1) == expected
    assert allocate_snapshot(oracle, budget_bytes=1) == expected


@pytest.mark.parametrize(
    "allocator", [allocate_snapshot, allocate_density_greedy_heuristic, exhaustive_optimum]
)
def test_zero_budget_returns_the_empty_feasible_allocation(allocator: Allocator) -> None:
    oracle = _single_group_oracle(("p",))
    chosen = allocator(oracle, 0)
    assert chosen == frozenset()
    assert oracle.cost(chosen) == 0
