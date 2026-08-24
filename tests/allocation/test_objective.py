from itertools import combinations

from ratemem.allocation.objective import CoverageOracle, PacketBundle


def _oracle() -> CoverageOracle:
    bundles = {
        "shared": PacketBundle("shared", cost_bytes=12, gains={"a": (0.7,), "b": (0.6,)}),
        "a-only": PacketBundle("a-only", cost_bytes=8, gains={"a": (0.5,)}),
        "b-only": PacketBundle("b-only", cost_bytes=8, gains={"b": (0.5,)}),
    }
    return CoverageOracle(
        bundles=bundles,
        request_weights={"a": 2.0, "b": 1.0},
        group_weights={"a": (1.0,), "b": (1.0,)},
    )


def test_coverage_is_normalized_monotone_and_submodular() -> None:
    oracle = _oracle()
    names = tuple(oracle.bundles)
    assert oracle.value(frozenset()) == 0.0
    subsets = [frozenset(c) for size in range(4) for c in combinations(names, size)]
    for left in subsets:
        for right in subsets:
            if left.issubset(right):
                assert oracle.value(left) <= oracle.value(right) + 1e-12
                for item in set(names) - set(right):
                    assert oracle.marginal(left, item) + 1e-12 >= oracle.marginal(right, item)


def test_one_payload_can_benefit_two_concepts() -> None:
    oracle = _oracle()
    assert oracle.value(frozenset({"shared"})) == 2.0


def test_group_weights_scale_only_their_declared_coverage_group() -> None:
    bundle = PacketBundle("p", cost_bytes=4, gains={"a": (0.5, 0.5)})
    oracle = CoverageOracle(
        bundles={"p": bundle},
        request_weights={"a": 2.0},
        group_weights={"a": (1.0, 3.0)},
    )
    assert oracle.value(frozenset({"p"})) == 4.0
