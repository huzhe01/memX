from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "cost_bytes",
    [True, 1.5, float("nan"), float("inf"), "4"],
    ids=["bool", "fractional", "nan", "infinity", "string"],
)
def test_packet_cost_requires_an_exact_integer(cost_bytes: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        PacketBundle("p", cost_bytes=cost_bytes, gains={"a": (0.5,)})  # type: ignore[arg-type]


@pytest.mark.parametrize("cost_bytes", [0, -1])
def test_packet_cost_requires_a_positive_integer(cost_bytes: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        PacketBundle("p", cost_bytes=cost_bytes, gains={"a": (0.5,)})


@pytest.mark.parametrize("packet_id", ["", 3], ids=["empty", "non-string"])
def test_packet_id_requires_a_nonempty_string(packet_id: object) -> None:
    error = ValueError if packet_id == "" else TypeError
    with pytest.raises(error, match="packet id"):
        PacketBundle(packet_id, cost_bytes=1, gains={"a": (0.5,)})  # type: ignore[arg-type]


def test_packet_gains_require_a_nonempty_mapping() -> None:
    with pytest.raises(TypeError, match="gains must be a mapping"):
        PacketBundle("p", cost_bytes=1, gains=[("a", (0.5,))])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one incidence"):
        PacketBundle("p", cost_bytes=1, gains={})


@pytest.mark.parametrize("concept_id", ["", 3], ids=["empty", "non-string"])
def test_packet_gain_concept_ids_require_nonempty_strings(concept_id: object) -> None:
    error = ValueError if concept_id == "" else TypeError
    with pytest.raises(error, match="concept id"):
        PacketBundle("p", cost_bytes=1, gains={concept_id: (0.5,)})  # type: ignore[dict-item]


@pytest.mark.parametrize("gains", [{"a": ()}, {"a": 0.5}, {"a": "0.5"}])
def test_packet_gain_vectors_require_nonempty_numeric_sequences(gains: object) -> None:
    error = ValueError if gains == {"a": ()} else TypeError
    with pytest.raises(error, match="gain vector"):
        PacketBundle("p", cost_bytes=1, gains=gains)  # type: ignore[arg-type]


@pytest.mark.parametrize("gain", [True, "0.5", 1 + 2j], ids=["bool", "string", "complex"])
def test_packet_gains_reject_non_real_scalars(gain: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        PacketBundle("p", cost_bytes=1, gains={"a": (gain,)})  # type: ignore[dict-item]


@pytest.mark.parametrize("field", ["bundles", "request_weights", "group_weights"])
def test_oracle_inputs_must_be_mappings(field: str) -> None:
    inputs: dict[str, object] = {
        "bundles": {},
        "request_weights": {},
        "group_weights": {},
    }
    inputs[field] = []
    with pytest.raises(TypeError, match=f"{field} must be a mapping"):
        CoverageOracle(**inputs)  # type: ignore[arg-type]


def test_oracle_rejects_non_bundle_values() -> None:
    with pytest.raises(TypeError, match="PacketBundle"):
        CoverageOracle(bundles={"p": object()}, request_weights={}, group_weights={})  # type: ignore[dict-item]


@pytest.mark.parametrize("bundle_key", ["", 3], ids=["empty", "non-string"])
def test_oracle_bundle_keys_require_nonempty_strings(bundle_key: object) -> None:
    bundle = PacketBundle("p", cost_bytes=1, gains={"a": (0.5,)})
    error = ValueError if bundle_key == "" else TypeError
    with pytest.raises(error, match="bundle id"):
        CoverageOracle(
            bundles={bundle_key: bundle},  # type: ignore[dict-item]
            request_weights={"a": 1.0},
            group_weights={"a": (1.0,)},
        )


def test_oracle_requires_bundle_key_to_match_packet_id() -> None:
    bundle = PacketBundle("p", cost_bytes=1, gains={"a": (0.5,)})
    with pytest.raises(ValueError, match="equal packet_id"):
        CoverageOracle(
            bundles={"other": bundle},
            request_weights={"a": 1.0},
            group_weights={"a": (1.0,)},
        )


@pytest.mark.parametrize("concept_id", ["", 3], ids=["empty", "non-string"])
def test_oracle_concept_ids_require_nonempty_strings(concept_id: object) -> None:
    error = ValueError if concept_id == "" else TypeError
    with pytest.raises(error, match="concept id"):
        CoverageOracle(
            bundles={},
            request_weights={concept_id: 1.0},  # type: ignore[dict-item]
            group_weights={concept_id: (1.0,)},  # type: ignore[dict-item]
        )


def test_oracle_requires_nonempty_group_vectors() -> None:
    with pytest.raises(ValueError, match="group weight vector"):
        CoverageOracle(
            bundles={},
            request_weights={"a": 1.0},
            group_weights={"a": ()},
        )


@pytest.mark.parametrize("weight", [True, "1", 1 + 2j], ids=["bool", "string", "complex"])
def test_oracle_request_weights_reject_non_real_scalars(weight: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        CoverageOracle(
            bundles={},
            request_weights={"a": weight},  # type: ignore[dict-item]
            group_weights={"a": (1.0,)},
        )


@pytest.mark.parametrize(
    "weights",
    [1.0, "1.0", (True,), ("1.0",), (1 + 2j,)],
    ids=["scalar", "string-container", "bool", "string", "complex"],
)
def test_oracle_group_weights_require_real_numeric_sequences(weights: object) -> None:
    with pytest.raises(TypeError, match="group weight"):
        CoverageOracle(
            bundles={},
            request_weights={"a": 1.0},
            group_weights={"a": weights},  # type: ignore[dict-item]
        )


def test_empty_oracle_is_valid() -> None:
    oracle = CoverageOracle(bundles={}, request_weights={}, group_weights={})
    assert tuple(oracle.bundles) == ()
    assert oracle.value(frozenset()) == 0.0
    assert oracle.cost(frozenset()) == 0


def test_inputs_are_normalized_owned_and_immune_to_source_mutation() -> None:
    gain_vector = [Fraction(1, 2)]
    source_gains = {"a": gain_vector}
    bundle = PacketBundle("p", cost_bytes=1, gains=source_gains)  # type: ignore[arg-type]
    gain_vector[0] = Fraction(1, 4)
    source_gains.clear()

    source_bundles = {"p": bundle}
    request_weights = {"a": 2}
    group_vector = [Fraction(3, 2)]
    group_weights = {"a": group_vector}
    oracle = CoverageOracle(
        bundles=source_bundles,
        request_weights=request_weights,
        group_weights=group_weights,  # type: ignore[arg-type]
    )
    source_bundles.clear()
    request_weights["a"] = 99
    group_vector[0] = Fraction(1, 3)
    group_weights.clear()

    assert bundle.gains == {"a": (0.5,)}
    assert oracle.request_weights == {"a": 2.0}
    assert oracle.group_weights == {"a": (1.5,)}
    assert oracle.value(frozenset({"p"})) == 1.5


def test_oracle_iteration_and_value_are_insertion_order_independent() -> None:
    gains = {"a": (0.1,), "b": (0.25,)}
    forward_bundles = {
        packet_id: PacketBundle(packet_id, 1, gains)
        for packet_id in ("p1", "p2", "p3")
    }
    reverse_bundles = {
        packet_id: PacketBundle(packet_id, 1, dict(reversed(tuple(gains.items()))))
        for packet_id in ("p3", "p2", "p1")
    }
    forward = CoverageOracle(
        bundles=forward_bundles,
        request_weights={"a": 0.1, "b": 0.2},
        group_weights={"a": (0.3,), "b": (0.4,)},
    )
    reverse = CoverageOracle(
        bundles=reverse_bundles,
        request_weights={"b": 0.2, "a": 0.1},
        group_weights={"b": (0.4,), "a": (0.3,)},
    )
    forward_selected = frozenset(("p1", "p2", "p3"))
    reverse_selected = frozenset(("p3", "p2", "p1"))

    assert tuple(forward.bundles) == ("p1", "p2", "p3")
    assert tuple(reverse.bundles) == ("p1", "p2", "p3")
    assert tuple(reverse.request_weights) == ("a", "b")
    assert forward.value(forward_selected).hex() == reverse.value(reverse_selected).hex()


@pytest.mark.parametrize("method", ["value", "marginal", "cost"])
def test_unknown_selected_packet_ids_raise_key_error(method: str) -> None:
    oracle = _oracle()
    with pytest.raises(KeyError, match="missing"):
        if method == "value":
            oracle.value(frozenset({"missing"}))
        elif method == "marginal":
            oracle.marginal(frozenset(), "missing")
        else:
            oracle.cost(frozenset({"missing"}))


def test_huge_gains_saturate_without_overflow() -> None:
    oracle = CoverageOracle(
        bundles={
            "p1": PacketBundle("p1", 1, {"a": (1e308,)}),
            "p2": PacketBundle("p2", 1, {"a": (1e308,)}),
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )
    assert oracle.value(frozenset({"p1", "p2"})) == 1.0


def test_marginal_preserves_small_gain_beside_huge_unrelated_mass() -> None:
    oracle = CoverageOracle(
        bundles={
            "mass": PacketBundle("mass", 1, {"huge": (1.0,)}),
            "small": PacketBundle("small", 1, {"small": (1.0,)}),
            "zero": PacketBundle("zero", 1, {"small": (0.0,)}),
        },
        request_weights={"huge": 1e308, "small": 1.0},
        group_weights={"huge": (1.0,), "small": (1.0,)},
    )
    selected = frozenset({"mass"})
    assert oracle.marginal(selected, "small") == 1.0
    assert oracle.marginal(selected, "zero") == 0.0


def test_oracle_rejects_nonfinite_weight_products() -> None:
    with pytest.raises(ValueError, match="coefficient"):
        CoverageOracle(
            bundles={},
            request_weights={"a": 1e308},
            group_weights={"a": (2.0,)},
        )


def test_oracle_rejects_nonfinite_total_maximum_objective_mass() -> None:
    with pytest.raises(ValueError, match="maximum objective"):
        CoverageOracle(
            bundles={},
            request_weights={"a": 1e308, "b": 1e308},
            group_weights={"a": (1.0,), "b": (1.0,)},
        )


def test_exact_objective_preserves_underflowed_multigroup_coefficients() -> None:
    smallest_subnormal = 5e-324
    oracle = CoverageOracle(
        bundles={"p": PacketBundle("p", 1, {"a": (0.5,) * 5})},
        request_weights={"a": smallest_subnormal},
        group_weights={"a": (1.0,) * 5},
    )
    selected = frozenset({"p"})
    exact_tiny = Fraction.from_float(smallest_subnormal)

    assert oracle.exact_value(selected) == 5 * exact_tiny / 2
    assert oracle.exact_marginal(frozenset(), "p") == 5 * exact_tiny / 2
    assert isinstance(oracle.exact_value(selected), Fraction)
    assert isinstance(oracle.exact_marginal(frozenset(), "p"), Fraction)


def test_exact_objective_is_submodular_when_reporting_rounding_is_not() -> None:
    above_half = float(np.nextafter(0.5, 1.0))
    oracle = CoverageOracle(
        bundles={
            "below-half": PacketBundle("below-half", 1, {"a": (1.0 - above_half,)}),
            "half": PacketBundle("half", 1, {"a": (0.5,)}),
            "small": PacketBundle("small", 1, {"a": (2**-54,)}),
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )
    left = frozenset({"below-half"})
    right = frozenset({"half", "small"})
    union = left | right
    intersection = left & right

    assert oracle.value(left) + oracle.value(right) < (
        oracle.value(union) + oracle.value(intersection)
    )
    assert oracle.exact_value(left) + oracle.exact_value(right) >= (
        oracle.exact_value(union) + oracle.exact_value(intersection)
    )
    assert oracle.exact_marginal(left, "small") == (
        oracle.exact_value(left | {"small"}) - oracle.exact_value(left)
    )


def test_exact_objective_keeps_unusual_packet_ids_distinct() -> None:
    packet_ids = ("nul\x00packet", "path/packet", "snowman-☃")
    oracle = CoverageOracle(
        bundles={
            packet_id: PacketBundle(packet_id, 1, {"a": (0.25,)})
            for packet_id in packet_ids
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0,)},
    )

    assert oracle.exact_value(frozenset(packet_ids)) == Fraction(3, 4)
    assert oracle.exact_marginal(frozenset({packet_ids[0]}), packet_ids[1]) == Fraction(
        1, 4
    )
