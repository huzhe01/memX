# RateMem Core Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-testable, byte-exact RateMem packet store, certified coverage objective, causal snapshot allocator, lifecycle replay engine, and artifact contract before integrating any diffusion model.

**Architecture:** A clean `src/ratemem` package replaces the historical TensorFlow code without importing it. Immutable base records, packet payloads, and incidence bundles serialize as a fixed header plus length-framed canonical-CBOR records, so the measured packet-bundle increment is exactly modular. Lifecycle operations return new states. A coverage oracle exposes the exact normalized monotone-submodular objective consumed by an exact small-instance oracle and a partial-enumeration density allocator.

**Tech Stack:** Python 3.11, uv, NumPy, Pydantic 2, cbor2, pytest, Hypothesis, Ruff, mypy

---

## File map

- `pyproject.toml`: package metadata, runtime/dev dependencies, CLI entry point, pytest/Ruff/mypy configuration.
- `uv.lock`: exact, cross-plan dependency lock used by local, Modal, evaluation, and paper jobs.
- `src/ratemem/__init__.py`: public package version.
- `src/ratemem/state/model.py`: immutable handles, base records, packets, incidences, and memory state.
- `src/ratemem/state/serialization.py`: fixed-header, length-framed canonical CBOR encoding, packet hashing, modular bundle costs, exact byte accounting, and round-trip decoding.
- `src/ratemem/state/store.py`: transactional create/update/read/delete/packet-redirection/packet-GC operations.
- `src/ratemem/codec/progressive.py`: deterministic base-code and enhancement-packet codec used by CPU tests.
- `src/ratemem/allocation/objective.py`: certified nonnegative coverage objective and packet-bundle costs.
- `src/ratemem/allocation/oracle.py`: exhaustive small-instance optimum.
- `src/ratemem/allocation/snapshot.py`: partial-enumeration marginal-density allocator.
- `src/ratemem/lifecycle/events.py`: typed lifecycle events and canonical trace records.
- `src/ratemem/lifecycle/replay.py`: deterministic event replay and read-only probes.
- `src/ratemem/artifacts/schema.py`: run-manifest and result schemas with credential-safe serialization.
- `src/ratemem/cli.py`: `ratemem smoke-core` integration command.
- `tests/`: one focused test module per component.

## Companion plans and execution order

This is the first implementation plan. Follow the interleaved six-plan order in
`2026-08-24-ratemem-master-execution.md`: freeze this interface; complete the free SANA engineering
gate and separately authorized pilot; seal scientific Tasks 1--7; implement and audit the matched
baselines before sealing scientific Task 8; complete the learned-method CPU gate; then authorize
scientific phases, replay/evaluate, and build the paper. No comparative model selection or
scientific run may start before its required locks pass. Only the paper template/scaffold and
artifact readers may be built before the result-bearing scientific release.

### Task 1: Create the clean Python package and locked development environment

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `uv.lock`
- Create: `src/ratemem/__init__.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Add a failing package import test**

```python
# tests/test_package.py
from ratemem import __version__


def test_package_version_is_explicit() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Create an isolated environment and confirm the test fails**

Run:

```bash
python3 -m pytest tests/test_package.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ratemem'`; on a bare host,
`No module named pytest` is also an acceptable pre-bootstrap failure.

- [ ] **Step 3: Add package metadata and tooling configuration**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[project]
name = "ratemem"
version = "0.1.0"
description = "Byte-bounded shared packet memory for image personalization"
requires-python = ">=3.11,<3.12"
dependencies = [
  "cbor2==5.7.0",
  "numpy==2.2.6",
  "pydantic==2.11.7",
  "PyYAML==6.0.2",
]

[dependency-groups]
dev = [
  "hypothesis>=6.120,<7",
  "mypy==1.17.1",
  "pytest==8.4.1",
  "ruff==0.12.11",
]

[project.scripts]
ratemem = "ratemem.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/ratemem"]

[tool.pytest.ini_options]
addopts = "-ra --strict-markers"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["ratemem"]

[[tool.mypy.overrides]]
module = ["cbor2"]
ignore_missing_imports = true
```

```gitignore
# .gitignore
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.py[cod]
/artifacts/
data/cache/
```

```text
# .python-version
3.11.13
```

```python
# src/ratemem/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Resolve the exact dependency lock and run the package test**

Run:

```bash
if ! command -v uv >/dev/null; then
  curl --fail --proto '=https' --tlsv1.2 --silent --show-error \
    https://astral.sh/uv/0.8.14/install.sh | sh
fi
test "$(uv --version)" = "uv 0.8.14"
uv python install 3.11.13
uv lock
uv sync --frozen --python 3.11.13
uv run --python 3.11.13 pytest tests/test_package.py -q
```

Expected: `uv 0.8.14`, then `1 passed`; `uv.lock` records exact source distributions and hashes.

- [ ] **Step 5: Commit the scaffold**

```bash
git add .gitignore .python-version pyproject.toml uv.lock src/ratemem/__init__.py tests/test_package.py
git commit -m "build: scaffold RateMem package"
```

### Task 2: Define immutable state and canonical byte accounting

**Files:**
- Create: `src/ratemem/state/__init__.py`
- Create: `src/ratemem/state/model.py`
- Create: `src/ratemem/state/serialization.py`
- Create: `tests/state/test_serialization.py`

This task is now a freeze-and-verify step. The exact checked-in state model, serializer, and test
file above are normative. Do not recreate them from an inline dataclass or CBOR skeleton: the
hardened boundary rejects noncanonical types, keys, records, and encodings that the former compact
examples accepted.

- [ ] **Step 1: Audit the exact checked-in serialization tests**

Use `tests/state/test_serialization.py` itself as the executable inventory. It must retain
coverage for exact byte accounting and bundle deltas; deterministic canonical round trips;
immutable mapping and payload ownership; packet-hash and dangling-reference rejection; exact
single-value canonical CBOR, section ordering, row arity/type, and fixed-width fields; duplicate
and mismatched embedded identities; and exact record/key/value types.

In particular, preserve adversarial cases for booleans, `int`/`str`/`tuple` subclasses,
spoofed values, and record subclasses. A small illustrative excerpt is not a substitute for this
checked-in suite.

- [ ] **Step 2: Verify the exact checked-in state model**

Review `src/ratemem/state/model.py` against `docs/contracts/core-interface.md`. Preserve all of
these boundaries:

- `BaseRecord.handle`, `Packet.packet_id`, and both `Incidence` identities are exact built-in
  nonempty strings.
- `BaseRecord.reads` and `BaseRecord.created_at` are exact built-in uint64 integers;
  `Incidence.gain_q` is an exact built-in int16 integer. Booleans, subclasses, and spoofed numeric
  objects are rejected.
- Base and packet keys are exact built-in nonempty strings. Incidence keys are exact built-in
  two-tuples whose two members are exact built-in nonempty strings.
- Mapping values are exact `BaseRecord`, `Packet`, or `Incidence` instances as appropriate;
  mapping keys equal their embedded identities, and duplicate embedded identities are rejected.
- Records own immutable payload-byte copies and `MemoryState` owns immutable mapping copies.

Raw `Packet` construction intentionally validates identity and ownership but not the content hash.

- [ ] **Step 3: Verify canonical serialization and validation**

Review the exact checked-in `src/ratemem/state/serialization.py` together with the model and tests.
`packet_from_payload` must hash the same owned bytes stored by the returned packet.
`encode_state` is the fixed header plus length-framed canonical-CBOR records in canonical section
order. `decode_state` accepts exactly one canonical value per frame, exact row schemas and fixed
integer widths, canonical ordering, no duplicates or trailing bytes, valid packet hashes, and no
dangling references; re-encoding the decoded state must reproduce the input exactly.
`bundle_cost_bytes` remains the measured positive additive packet-plus-incidence delta under the
fixed-cohort proof assumptions.

- [ ] **Step 4: Run the state serialization gate and commit**

Run:

```bash
uv run pytest tests/state/test_serialization.py -q
uv run ruff check src/ratemem/state tests/state/test_serialization.py
uv run mypy src/ratemem/state
```

Expected: every checked-in serialization/model test passes and both static checks exit 0.

```bash
git add src/ratemem/state tests/state/test_serialization.py
git commit -m "feat: add canonical memory state"
```

### Task 3: Implement atomic packet-store lifecycle operations

**Files:**
- Create: `src/ratemem/state/store.py`
- Create: `tests/state/test_store.py`

This task is now a freeze-and-verify step. The exact checked-in store and store tests are normative.
Do not replace them with an inline transactional-store skeleton: the hardened code validates
constructor state, exact input types, hashes, references, duplicate attachments, overflow, and
canonical round trips.

- [ ] **Step 1: Audit the exact checked-in store tests**

Use `tests/state/test_store.py` itself as the executable inventory. It must retain coverage for
exact non-boolean budgets; exact handle and creation-counter inputs; forged packet hashes, dangling
incidences, orphan packets, and record subclasses; atomic duplicate replacement and identity
mismatch failures; exact-budget acceptance and one-byte-under rejection; functional create/read/
replace/delete operations; uint64 read overflow; idempotent duplicate attachment; shared-packet
reclamation; and canonical round trips after every accepted transition.

- [ ] **Step 2: Verify constructor and transition validation**

Review `src/ratemem/state/store.py` against `docs/contracts/core-interface.md`. Construction
accepts only an exact `MemoryState`, an exact built-in integer byte budget, a fully
content-addressed packet set, no dangling incidence, and no orphan packet. Operations accepting a
handle apply the exact built-in nonempty-string rule before lookup. Packet and incidence inputs are
exact record instances, attachments agree with the operation identities, and replacement rejects
duplicate packet attachments before producing state.

- [ ] **Step 3: Verify functional atomicity and exact accounting**

Every accepted transition returns a newly checked store except the declared functional no-op read.
Every rejected transition leaves the old store unchanged. Create, attach/bundle, replace,
usage-updating read, and delete all flow through the same constructor validation and exact
`serialized_bytes` budget check. Shared packets remain until their last incidence is removed;
orphan packets never persist.

- [ ] **Step 4: Run the complete state gate and commit**

Run:

```bash
uv run pytest tests/state -q
uv run ruff check src/ratemem/state tests/state
uv run mypy src/ratemem/state
```

Expected: every checked-in state test passes and both static checks exit 0.

```bash
git add src/ratemem/state tests/state
git commit -m "feat: add transactional packet store"
```

### Task 4: Add the deterministic progressive CPU codec

**Files:**
- Create: `src/ratemem/codec/__init__.py`
- Create: `src/ratemem/codec/progressive.py`
- Create: `tests/codec/test_progressive.py`

This task is now a freeze-and-verify step. The exact checked-in versions of
`src/ratemem/codec/progressive.py` and `tests/codec/test_progressive.py` at the Gate 1
freeze are normative. Do not recreate either file from an abbreviated inline implementation or
starter test: earlier sketches omitted validation that is part of the frozen interface.

- [ ] **Step 1: Audit the exact checked-in codec tests**

Use `tests/codec/test_progressive.py` itself as the executable test inventory. It must retain
coverage for deterministic payloads and the planned float32 quantizer; monotone prefix
reconstruction; float16 boundaries; immutable ownership; canonical little-endian float16 NPY
headers, shape, C order, and exact data length; exact global packet cardinality; selected-prefix
hash, group, offset, body-size, and finite-value validation; rejection of repeated selected packets;
and isolation of malformed payload/hash/metadata in an existing unselected suffix.

- [ ] **Step 2: Verify the frozen codec contract**

Run:

```bash
uv run pytest tests/codec/test_progressive.py -q
```

Expected: every checked-in codec test passes. A fixed historical pass count is deliberately not
recorded because the checked-in hardened test file, rather than a stale excerpt, defines the gate.

- [ ] **Step 3: Review the exact checked-in implementation**

Review `src/ratemem/codec/progressive.py` against
`docs/contracts/core-interface.md`. In particular, preserve `EncodedCode.group_size`, exact
packet-count validation before prefix selection, canonical NPY reserialization, content-hash
validation for selected packets, exact tuple/header order and offsets, and defensive ownership of
base and packet metadata. Validation of an existing unselected suffix must remain limited to the
global tuple cardinality; its payload, hash, and metadata are not inspected by a shorter decode.

- [ ] **Step 4: Run the codec gate and commit**

Run:

```bash
uv run pytest tests/codec/test_progressive.py -q
uv run ruff check src/ratemem/codec tests/codec
uv run mypy src/ratemem/codec
if rg -q 'allow_pickle=True' src/ratemem; then
  exit 1
else
  ratemem_pickle_scan_status=$?
  if [ "$ratemem_pickle_scan_status" -ne 1 ]; then
    exit "$ratemem_pickle_scan_status"
  fi
fi
```

Expected: the checked-in tests pass, Ruff and mypy exit 0, and there is no unsafe-pickle match.

```bash
git add src/ratemem/codec tests/codec
git commit -m "feat: add progressive packet codec"
```

### Task 5: Implement the certified shared-packet coverage objective

**Files:**
- Create: `src/ratemem/allocation/__init__.py`
- Create: `src/ratemem/allocation/objective.py`
- Create: `tests/allocation/test_objective.py`

- [ ] **Step 1: Write normalization, monotonicity, submodularity, and sharing tests**

```python
# tests/allocation/test_objective.py
from fractions import Fraction
from itertools import combinations

import numpy as np

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
    assert oracle.exact_value(frozenset()) == Fraction()
    subsets = [frozenset(c) for size in range(4) for c in combinations(names, size)]
    for left in subsets:
        for right in subsets:
            if left.issubset(right):
                assert oracle.exact_value(left) <= oracle.exact_value(right)
                for item in set(names) - set(right):
                    assert oracle.exact_marginal(left, item) >= oracle.exact_marginal(
                        right, item
                    )


def test_one_payload_can_benefit_two_concepts() -> None:
    oracle = _oracle()
    assert oracle.exact_value(frozenset({"shared"})) == Fraction(2)


def test_float_methods_are_reporting_views_of_exact_methods() -> None:
    oracle = _oracle()
    selected = frozenset({"shared"})
    assert oracle.value(selected) == float(oracle.exact_value(selected))
    assert oracle.marginal(selected, "a-only") == float(
        oracle.exact_marginal(selected, "a-only")
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
```

- [ ] **Step 2: Run the objective tests and verify the import fails**

Run: `uv run pytest tests/allocation/test_objective.py -q`

Expected: collection fails importing `ratemem.allocation.objective`.

- [ ] **Step 3: Implement immutable bundles and the exact value oracle**

```python
# src/ratemem/allocation/objective.py
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
                        "oracle coefficient must be representable as a finite "
                        "float for reporting"
                    ) from error
                if not math.isfinite(reporting_coefficient):
                    raise ValueError(
                        "oracle coefficient must be representable as a finite "
                        "float for reporting"
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
                "maximum objective mass must be representable as a finite "
                "float for reporting"
            ) from error
        if not math.isfinite(reporting_maximum):
            raise ValueError(
                "maximum objective mass must be representable as a finite "
                "float for reporting"
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

    def _exact_value_from_coverage(
        self, coverage: Mapping[tuple[str, int], Fraction]
    ) -> Fraction:
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
                item_gain = (
                    item_gains[group] if group < len(item_gains) else Fraction()
                )
                remaining = Fraction(1) - coverage[(handle, group)]
                terms.append(coefficient * min(remaining, item_gain))
        return sum(terms, start=Fraction())

    def exact_value(self, selected: frozenset[str]) -> Fraction:
        """Return certified utility over exact binary-rational normalized inputs."""
        selected_ids = self._selected_ids(selected)
        return self._exact_value_from_coverage(self._exact_coverage(selected_ids))

    def exact_marginal(self, selected: frozenset[str], item: str) -> Fraction:
        """Return a direct exact marginal without rounded-value subtraction."""
        selected_ids = self._selected_ids(selected)
        self.bundles[item]
        if item in selected:
            return Fraction()
        return self._exact_marginal_from_coverage(
            self._exact_coverage(selected_ids), item
        )

    def value(self, selected: frozenset[str]) -> float:
        """Return a rounded float report; certification uses exact_value."""
        return float(self.exact_value(selected))

    def marginal(self, selected: frozenset[str], item: str) -> float:
        """Return a rounded float report; certification uses exact_marginal."""
        return float(self.exact_marginal(selected, item))

    def cost(self, selected: frozenset[str]) -> int:
        return sum(
            self.bundles[item].cost_bytes for item in self._selected_ids(selected)
        )
```

- [ ] **Step 4: Run objective tests and static checks**

Run:

```bash
uv run pytest tests/allocation/test_objective.py -q
uv run ruff check src/ratemem/allocation tests/allocation
uv run mypy src/ratemem/allocation
```

Expected: all exact-objective and reporting-view tests pass; Ruff and mypy exit 0.

- [ ] **Step 5: Commit the certified objective**

```bash
git add src/ratemem/allocation tests/allocation/test_objective.py
git commit -m "feat: add shared packet coverage objective"
```

### Task 6: Implement exact and approximation allocators

**Files:**
- Create: `src/ratemem/allocation/oracle.py`
- Create: `src/ratemem/allocation/snapshot.py`
- Create: `tests/allocation/test_snapshot_allocator.py`

- [ ] **Step 1: Write feasibility, determinism, and certified-ratio tests**

```python
# tests/allocation/test_snapshot_allocator.py
import random
from fractions import Fraction
from itertools import product

import pytest
from hypothesis import given, strategies as st

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.allocation.oracle import exhaustive_optimum
from ratemem.allocation.snapshot import (
    allocate_snapshot,
    prescreen_certified_oracle,
)

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
    reduced = prescreen_certified_oracle(oracle, budget)
    chosen = allocate_snapshot(reduced, budget)
    optimum = exhaustive_optimum(reduced, budget)
    assert reduced.cost(chosen) <= budget
    _assert_certified_ratio(reduced, chosen, optimum)
    assert allocate_snapshot(reduced, budget) == chosen


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
                reduced = prescreen_certified_oracle(oracle, budget)
                chosen = allocate_snapshot(reduced, budget)
                optimum = exhaustive_optimum(reduced, budget)
                assert reduced.cost(chosen) <= budget
                _assert_certified_ratio(reduced, chosen, optimum)


def test_allocator_factor_on_seeded_multiconcept_instances() -> None:
    rng = random.Random(20260824)
    concepts = ("a", "b", "c")
    for _ in range(40):
        bundles = {
            f"p{index}": PacketBundle(
                f"p{index}",
                rng.randint(1, 10),
                {
                    handle: tuple(rng.random() for _ in range(3))
                    for handle in concepts
                },
            )
            for index in range(rng.randint(1, 9))
        }
        oracle = CoverageOracle(
            bundles,
            {handle: rng.random() for handle in concepts},
            {
                handle: tuple(rng.random() for _ in range(3))
                for handle in concepts
            },
        )
        budget = rng.randint(1, sum(bundle.cost_bytes for bundle in bundles.values()))
        reduced = prescreen_certified_oracle(oracle, budget)
        chosen = allocate_snapshot(reduced, budget)
        optimum = exhaustive_optimum(reduced, budget)
        assert reduced.cost(chosen) <= budget
        _assert_certified_ratio(reduced, chosen, optimum)


def test_prescreen_reduces_four_concepts_with_eight_packets_each() -> None:
    concepts = tuple(f"concept-{index}" for index in range(4))
    bundles = {
        f"{concept}-packet-{packet_index}": PacketBundle(
            f"{concept}-packet-{packet_index}",
            cost_bytes=1,
            gains={concept: ((packet_index + 1) / 8,)},
        )
        for concept in concepts
        for packet_index in range(8)
    }
    oracle = CoverageOracle(
        bundles,
        request_weights={concept: 1.0 for concept in concepts},
        group_weights={concept: (1.0,) for concept in concepts},
    )

    reduced = prescreen_certified_oracle(oracle, budget_bytes=4)
    expected_ids = {
        f"{concept}-packet-{packet_index}"
        for concept in concepts
        for packet_index in range(2, 8)
    }
    assert len(oracle.bundles) == 32
    assert set(reduced.bundles) == expected_ids
    assert len(reduced.bundles) == 24
    chosen = allocate_snapshot(reduced, budget_bytes=4)
    assert chosen <= expected_ids
    assert reduced.cost(chosen) <= 4


def test_allocator_factor_on_rounding_adversarial_instance() -> None:
    oracle = CoverageOracle(
        bundles={
            "a-exact": PacketBundle("a-exact", 1, {"a": (0.5, 2**-54)}),
            "z-rounded": PacketBundle("z-rounded", 1, {"a": (0.5, 0.0)}),
        },
        request_weights={"a": 1.0},
        group_weights={"a": (1.0, 1.0)},
    )
    assert oracle.value(frozenset({"a-exact"})) == oracle.value(
        frozenset({"z-rounded"})
    )
    assert oracle.exact_value(frozenset({"a-exact"})) > oracle.exact_value(
        frozenset({"z-rounded"})
    )
    assert tuple(
        prescreen_certified_oracle(oracle, budget_bytes=1, max_bundles=1).bundles
    ) == ("a-exact",)

    chosen = allocate_snapshot(oracle, budget_bytes=1)
    optimum = exhaustive_optimum(oracle, budget_bytes=1)

    assert optimum == frozenset({"a-exact"})
    assert chosen == frozenset({"a-exact"})
    _assert_certified_ratio(oracle, chosen, optimum)


@pytest.mark.parametrize("max_bundles", [0, -1, 25])
def test_prescreen_rejects_caps_outside_one_through_24(max_bundles: int) -> None:
    oracle = CoverageOracle(
        {"p": PacketBundle("p", 1, {"a": (1.0,)})},
        {"a": 1.0},
        {"a": (1.0,)},
    )
    with pytest.raises(ValueError):
        prescreen_certified_oracle(oracle, 1, max_bundles=max_bundles)


@pytest.mark.parametrize("max_bundles", [True, 1.5, "24"])
def test_prescreen_cap_requires_an_exact_integer(max_bundles: object) -> None:
    oracle = CoverageOracle(
        {"p": PacketBundle("p", 1, {"a": (1.0,)})},
        {"a": 1.0},
        {"a": (1.0,)},
    )
    with pytest.raises(TypeError):
        prescreen_certified_oracle(
            oracle,
            1,
            max_bundles=max_bundles,  # type: ignore[arg-type]
        )
```

- [ ] **Step 2: Verify tests fail because allocator modules are absent**

Run: `uv run pytest tests/allocation/test_snapshot_allocator.py -q`

Expected: collection fails importing `ratemem.allocation.oracle`.

- [ ] **Step 3: Implement the exhaustive small-instance reference**

```python
# src/ratemem/allocation/oracle.py
from __future__ import annotations

from itertools import combinations

from ratemem.allocation.objective import CoverageOracle

DEFAULT_MAX_STATES = 2**18


def _validate_budget(budget_bytes: int) -> None:
    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer byte count")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")


def _validate_max_states(max_states: int) -> None:
    if type(max_states) is not int:
        raise TypeError("max_states must be an integer")
    if max_states <= 0:
        raise ValueError("max_states must be positive")


def exhaustive_optimum(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_states: int = DEFAULT_MAX_STATES,
) -> frozenset[str]:
    _validate_budget(budget_bytes)
    _validate_max_states(max_states)
    all_names = tuple(sorted(oracle.bundles))
    if len(all_names) > 24:
        raise ValueError("exhaustive oracle supports at most 24 packet bundles")
    names = tuple(
        item
        for item in all_names
        if oracle.bundles[item].cost_bytes <= budget_bytes
    )
    required_states = 1 << len(names)
    if required_states > max_states:
        raise ValueError(
            f"exhaustive oracle requires {required_states} states; "
            "increase max_states explicitly"
        )
    costs = {item: oracle.bundles[item].cost_bytes for item in names}
    ascending_costs = tuple(sorted(costs.values()))
    best = frozenset[str]()
    best_value = oracle.exact_value(best)
    best_ids: tuple[str, ...] = ()
    for size in range(len(names) + 1):
        if sum(ascending_costs[:size]) > budget_bytes:
            break
        for rows in combinations(names, size):
            candidate_cost = sum(costs[item] for item in rows)
            if candidate_cost > budget_bytes:
                continue
            candidate = frozenset(rows)
            candidate_value = oracle.exact_value(candidate)
            candidate_ids = tuple(sorted(candidate))
            if (candidate_value, candidate_ids) > (best_value, best_ids):
                best = candidate
                best_value = candidate_value
                best_ids = candidate_ids
    return best
```

- [ ] **Step 4: Implement partial enumeration plus exact density greedy**

```python
# src/ratemem/allocation/snapshot.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from ratemem.allocation.objective import CoverageOracle

DEFAULT_MAX_BUNDLES = 24


@dataclass(frozen=True, slots=True)
class _FillResult:
    selected: frozenset[str]
    cost_bytes: int
    exact_value: Fraction


def _validate_budget(budget_bytes: int) -> None:
    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer byte count")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")


def _validate_max_bundles(max_bundles: int) -> None:
    if type(max_bundles) is not int:
        raise TypeError("max_bundles must be an integer")
    if max_bundles <= 0:
        raise ValueError("max_bundles must be positive")


def prescreen_certified_oracle(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_bundles: int = DEFAULT_MAX_BUNDLES,
) -> CoverageOracle:
    _validate_budget(budget_bytes)
    _validate_max_bundles(max_bundles)
    if max_bundles > DEFAULT_MAX_BUNDLES:
        raise ValueError("prescreen max_bundles cannot exceed certified default 24")

    empty = frozenset[str]()
    feasible = (
        packet_id
        for packet_id, bundle in oracle.bundles.items()
        if bundle.cost_bytes <= budget_bytes
    )
    ranked = sorted(
        feasible,
        key=lambda packet_id: (
            oracle.exact_marginal(empty, packet_id)
            / oracle.bundles[packet_id].cost_bytes,
            packet_id,
        ),
        reverse=True,
    )
    selected_ids = ranked[:max_bundles]
    return CoverageOracle(
        bundles={packet_id: oracle.bundles[packet_id] for packet_id in selected_ids},
        request_weights=oracle.request_weights,
        group_weights=oracle.group_weights,
    )


def _density_fill(
    oracle: CoverageOracle,
    seed: frozenset[str],
    seed_cost: int,
    budget_bytes: int,
) -> _FillResult:
    selected = set(seed)
    selected_cost = seed_cost
    coverage = oracle._empty_exact_coverage()
    for item in sorted(seed):
        oracle._add_exact_gains(coverage, item)
    remaining = set(oracle.bundles) - selected
    while remaining:
        remaining = {
            item
            for item in remaining
            if selected_cost + oracle.bundles[item].cost_bytes <= budget_bytes
        }
        if not remaining:
            break
        item = max(
            remaining,
            key=lambda item: (
                oracle._exact_marginal_from_coverage(coverage, item)
                / oracle.bundles[item].cost_bytes,
                item,
            ),
        )
        remaining.remove(item)
        item_cost = oracle.bundles[item].cost_bytes
        selected.add(item)
        selected_cost += item_cost
        oracle._add_exact_gains(coverage, item)
    return _FillResult(
        selected=frozenset(selected),
        cost_bytes=selected_cost,
        exact_value=oracle._exact_value_from_coverage(coverage),
    )


def allocate_snapshot(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_bundles: int = DEFAULT_MAX_BUNDLES,
) -> frozenset[str]:
    _validate_budget(budget_bytes)
    _validate_max_bundles(max_bundles)
    names = tuple(sorted(oracle.bundles))
    if len(names) > max_bundles:
        raise ValueError(
            f"certified allocator ground set exceeds max_bundles={max_bundles}; "
            "raise the limit explicitly or call allocate_density_greedy_heuristic"
        )

    costs = {item: oracle.bundles[item].cost_bytes for item in names}
    best = frozenset[str]()
    best_value = oracle.exact_value(best)
    best_ids: tuple[str, ...] = ()
    for size in range(min(3, len(names)) + 1):
        for rows in combinations(names, size):
            seed = frozenset(rows)
            seed_cost = sum(costs[item] for item in rows)
            if seed_cost > budget_bytes:
                continue
            result = _density_fill(oracle, seed, seed_cost, budget_bytes)
            candidate_ids = tuple(sorted(result.selected))
            if (result.exact_value, candidate_ids) > (best_value, best_ids):
                best = result.selected
                best_value = result.exact_value
                best_ids = candidate_ids
    return best


def allocate_density_greedy_heuristic(
    oracle: CoverageOracle, budget_bytes: int
) -> frozenset[str]:
    """Return deterministic exact-density greedy output without a guarantee."""
    _validate_budget(budget_bytes)
    return _density_fill(oracle, frozenset(), 0, budget_bytes).selected
```

- [ ] **Step 5: Run exhaustive randomized tests, document the proof contract, and commit**

Create the proof contract verbatim from this starting text, then make notation match the
implemented class names:

```markdown
# Snapshot allocation proof contract

At snapshot `t`, condition on the causal history and on a fixed admitted concept cohort. Base
records and their metadata have already been reserved, leaving packet budget `b_t`. The finite
causal candidate set `G_t` contains immutable packet bundles. Bundle `p` contains one payload/hash
and its complete prespecified incidence list; selecting individual incidences is not allowed. Before
certified enumeration, `prescreen_certified_oracle` removes individually infeasible bundles and
retains the highest-density bundles in descending exact singleton marginal-density order with
deterministic packet-ID ties. Its exact non-boolean integer cap satisfies `1 <= max_bundles <= 24`,
defaults to 24, and rejects larger values. Denote this fixed reduced ground set by `C_t`; the
pre-screen has no approximation guarantee against the full `G_t`.

The on-disk format has a fixed-size state header and length-framed canonical-CBOR records. Hence
for this fixed cohort, `bundle_cost_bytes(packet, incidences)` is exactly the state-length increase
caused by installing the bundle, and costs add across bundles.

For nonnegative past-only request weights `omega[t,i]`, nonnegative locked group weights
`beta[t,i,g]`, and nonnegative locked bundle gains `v[t,i,g,p]`, the value oracle is

    F_t(X) = sum_i omega[t,i] sum_g beta[t,i,g]
             min(1, sum_{p in X} v[t,i,g,p]).

`F_t(empty)=0`. Adding a bundle cannot decrease any inner modular sum, so `F_t` is monotone.
The capped-linear function is concave and nondecreasing over a nonnegative modular sum, so every
term has diminishing returns; a nonnegative weighted sum preserves submodularity.

`allocate_snapshot` enumerates every feasible seed of cardinality zero through three from `C_t` and
completes each seed by recomputing exact marginal-gain-per-byte values after every accepted bundle.
It returns the best completed seed, with deterministic packet-id tie breaking. This is the standard
partial-enumeration knapsack algorithm used to obtain the `1 - 1/e` approximation for monotone
submodular maximization under one modular knapsack constraint. Lazy evaluation is permitted only
after a test shows it returns the identical sequence as exact recomputation.

Therefore, under the premises above,

    F_t(X_t) >= (1 - 1/e) max_{X subset C_t, cost(X) <= b_t} F_t(X).

This is a conditional per-snapshot guarantee relative only to `C_t`. Full-pool pre-screen loss,
whole-concept admission or eviction, optional incidence dropping, switching penalties, hysteresis,
learned unconstrained distortion, and any
future-aware competitive or dynamic-regret statement are outside the theorem. The future-aware
lifecycle oracle is an empirical upper reference only.

Mechanical release checks compare feasibility and the approximation ratio with exhaustive optima
on the same reduced `C_t` for enumerated and seeded-random tiny instances. They use the conservative
exact rational constant `6321205588285576 / 10**16`, strictly below `1 - 1/e`, cross-multiply
`exact_value` results, and permit no additive epsilon. If either the premises or exact ratio checks
fail, the paper removes the theorem claim rather than changing the test after observing results.
```

Run:

```bash
uv run pytest tests/allocation -q
uv run ruff check src/ratemem/allocation tests/allocation
uv run mypy src/ratemem/allocation
```

Expected: all allocation tests pass; Ruff and mypy exit 0.

```bash
git add src/ratemem/allocation tests/allocation docs/theory/snapshot-allocation.md
git commit -m "feat: add certified snapshot allocator"
```

### Task 7: Add immutable lifecycle traces and read-only probes

**Files:**
- Create: `src/ratemem/lifecycle/__init__.py`
- Create: `src/ratemem/lifecycle/events.py`
- Create: `src/ratemem/lifecycle/replay.py`
- Create: `tests/lifecycle/test_replay.py`

This task is now a freeze-and-verify step. The exact checked-in lifecycle files above are
normative. Do not replace them with an inline event or replay skeleton: the hardened code owns
bytes, validates exact identities, rejects event subclasses through exact-type dispatch, and
preserves precise error classification.

- [ ] **Step 1: Audit the exact checked-in replay tests**

Use `tests/lifecycle/test_replay.py` itself as the executable Gate 1 inventory. It currently
covers exact nonempty event identities, owned bytes-like payloads, rejection of payload and
non-payload event subclasses, read-only probes, deterministic replay, stale handles, preserved
usage on update, exact duplicate-create classification, propagation of unrelated `ValueError`,
early rejection of a non-integral budget, randomized hard-budget traces, and both sides of the
replay-initialization boundary. In particular,
`test_replay_rejects_budget_below_empty_state_before_processing_events` verifies that a budget
below `MemoryState().serialized_bytes` raises `BudgetExceeded` before event processing, while
`test_replay_records_create_failure_at_exact_empty_state_budget` verifies that a create failure
after successful empty-store initialization is recorded as the deterministic event-level
`budget-exceeded` error and leaves the empty state intact.

- [ ] **Step 2: Verify the exact checked-in event model**

Review `src/ratemem/lifecycle/events.py` against `docs/contracts/core-interface.md`. Preserve
the closed five-event union, frozen slots, exact built-in nonempty `str` validation for
`event_id` and `handle`, rejection of string subclasses and spoofed types, and owned immutable
copies of `bytes`, `bytearray`, and `memoryview` create/update payloads.

- [ ] **Step 3: Verify exact-type deterministic replay**

Review `src/ratemem/lifecycle/replay.py` and `tests/lifecycle/test_replay.py` together. Replay
must initialize `PacketStore.empty` before event handling, dispatch only on the exact five event
classes, distinguish duplicate create from unrelated validation failures, preserve prior usage on
update, leave probes read-only, and record only the declared event-level errors. It must never use
broad `isinstance` dispatch or catch every `ValueError` as a duplicate.

- [ ] **Step 4: Run lifecycle tests and commit**

Run:

```bash
uv run pytest tests/lifecycle/test_replay.py -q
uv run ruff check src/ratemem/lifecycle tests/lifecycle
uv run mypy src/ratemem/lifecycle
```

Expected: every checked-in lifecycle test, including both replay-initialization regressions, passes
and the static checks exit 0.

```bash
git add src/ratemem/lifecycle tests/lifecycle
git commit -m "feat: add deterministic lifecycle replay"
```

### Task 8: Add credential-safe artifacts and an end-to-end CPU smoke command

**Files:**
- Modify: `.gitignore`
- Create: `src/ratemem/artifacts/__init__.py`
- Create: `src/ratemem/artifacts/schema.py`
- Create: `src/ratemem/cli.py`
- Create: `tests/artifacts/test_schema.py`
- Create: `tests/test_cli.py`

The `/artifacts/` ignore rule must stay root-anchored; nested artifact source and test packages remain tracked and scanned.
This task is also a freeze-and-verify step: the exact checked-in files listed above are normative.
Do not recreate the manifest or smoke command from an inline skeleton; the earlier compact versions
omitted security and end-to-end contract checks.

- [ ] **Step 1: Audit the exact checked-in artifact and smoke tests**

Use `tests/artifacts/test_schema.py` and `tests/test_cli.py` as the executable test inventory.
The artifact suite must retain recursive credential preflight; sanitized failures that do not echo
rejected input; exact built-in scalar/container acceptance; rejection without adversarial protocol
dispatch; duplicate JSON-key rejection at every depth; guarded validation, copying, mutation, and
serialization boundaries; disabled unchecked construction; subclass rejection; and the
root-anchored artifact ignore rule.

The smoke suite must retain deterministic module and installed entry points from an external
directory, selected-prefix shape/finiteness and strict-improvement checks, lifecycle probe
execution, causal pre-screen-before-allocation order, and an `AttemptManifest` JSON round trip.

- [ ] **Step 2: Verify the hardened manifest boundary**

Review the exact checked-in `src/ratemem/artifacts/schema.py` against
`docs/contracts/core-interface.md`. `AttemptManifest.model_validate_json` is the supported
credential-safe raw-JSON entry point. Preserve duplicate-key rejection, recursive scanning,
sanitized `ValidationError` construction, exact-type input handling that avoids overrideable
protocol dispatch, frozen/extra-forbid behavior, revalidated `model_copy`, guarded outbound
serialization, and disabled `model_construct`. Do not reduce this implementation to a field
validator over a regular expression.

- [ ] **Step 3: Verify the self-contained CPU smoke boundary**

Review the exact checked-in `src/ratemem/cli.py` and `tests/test_cli.py` together. The command
must perform progressive encode/decode with strict selected-prefix improvement; transactional
storage with byte-exact accounting; `prescreen_certified_oracle` before
`allocate_snapshot`; a read-only lifecycle probe; and a supported manifest JSON round trip.
Success prints one deterministic sorted JSON line and performs no network, GPU, Modal, or credential
operation.

- [ ] **Step 4: Run the full quality and credential gate and commit**

Run:

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src/ratemem
uv run python -m ratemem.cli smoke-core
if git grep --untracked --exclude-standard -Iq -E '(ak|as)-[A-Za-z0-9_-]{20,}' -- .; then
  exit 1
else
  ratemem_tracked_scan_status=$?
  if [ "$ratemem_tracked_scan_status" -ne 1 ]; then
    exit "$ratemem_tracked_scan_status"
  fi
fi
for ratemem_scan_root in artifacts run_log logs exports; do
  if [ -e "$ratemem_scan_root" ]; then
    if rg --hidden --no-ignore -q '(ak|as)-[A-Za-z0-9_-]{20,}' "$ratemem_scan_root"; then
      exit 1
    else
      ratemem_generated_scan_status=$?
      if [ "$ratemem_generated_scan_status" -ne 1 ]; then
        exit "$ratemem_generated_scan_status"
      fi
    fi
  fi
done
```

Expected: all tests pass; Ruff and mypy exit 0; the CLI prints a JSON object with
`"status": "passed"`; both quiet credential scans exit 0. The tracked-tree scan covers root
configuration and scripts, while the explicit generated-root scan includes ignored and hidden
artifacts, logs, and exports.

```bash
git add .gitignore src/ratemem/artifacts src/ratemem/cli.py tests/artifacts tests/test_cli.py
git commit -m "feat: add core smoke artifact contract"
```

### Task 9: Verify design coverage and freeze the core interface

**Files:**
- Create: `docs/contracts/core-interface.md`
- Modify: `docs/superpowers/specs/2026-08-24-memadapter-dit-design.md`

- [ ] **Step 1: Record the exact cross-plan interface**

Create `docs/contracts/core-interface.md` with these signatures and invariants:

```text
ProgressiveCodec.encode(handle: str, code: FloatArray) -> EncodedCode
PacketStore.create(handle: str, payload: bytes, created_at: int) -> PacketStore
PacketStore.attach(packet: Packet, incidence: Incidence) -> PacketStore
PacketStore.attach_bundle(packet: Packet, bundle: tuple[Incidence, ...]) -> PacketStore
PacketStore.replace(handle: str, payload: bytes,
                    attachments: tuple[tuple[Packet, Incidence], ...]) -> PacketStore
PacketStore.read(handle: str, update_usage: bool) -> tuple[PacketStore, BaseRecord]
PacketStore.delete(handle: str) -> PacketStore
bundle_cost_bytes(packet: Packet, incidences: tuple[Incidence, ...]) -> int
CoverageOracle.value(selected: frozenset[str]) -> float
prescreen_certified_oracle(oracle: CoverageOracle, budget_bytes: int,
                           *, max_bundles: int = 24) -> CoverageOracle
allocate_snapshot(oracle: CoverageOracle, budget_bytes: int,
                  *, max_bundles: int = 24) -> frozenset[str]

All store transitions are functional. encode_state is a fixed header plus length-framed canonical
CBOR records. serialized_bytes equals the actual encoded length, and packet-bundle cost equals the
measured state-length increment for a fixed admitted cohort. Base, packet, and incidence identity
fields are exact built-in nonempty strings. Base read/creation counters are exact built-in uint64
integers and incidence gains are exact built-in int16 integers; booleans, subclasses, proxies, and
noncanonical state mapping keys are rejected. Raw Packet construction does not certify its hash;
PacketStore transitions and decode_state enforce the packet-ID-to-payload relation. Probe reads
never update usage.
Certified gains, group weights, request weights, and bundle costs are nonnegative. The release path
causally pre-screens in descending exact singleton-density order. Its exact non-boolean integer cap
must satisfy `1 <= max_bundles <= 24`; values above 24 are rejected. The certified ratio is relative
only to that reduced ground set. Full-pool pre-screen loss, admission, switching cost, and
future-aware regret remain outside the per-snapshot approximation statement.
```

- [ ] **Step 2: Mark only the paid-pilot probe items as implementation-ready in the design**

Add one sentence to Section 10.1 of the design spec:

```markdown
The core-memory contract is frozen by `docs/contracts/core-interface.md`; later plans may extend
schemas only through versioned fields and backward-compatible decoders.
```

- [ ] **Step 3: Run the final baseline verification**

Run:

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src/ratemem
git diff --check
git status --short
```

Expected: tests and static checks pass; `git diff --check` is silent; status contains only the two documentation files for this task.

- [ ] **Step 4: Commit the frozen interface**

```bash
git add docs/contracts/core-interface.md docs/superpowers/specs/2026-08-24-memadapter-dit-design.md
git commit -m "docs: freeze RateMem core interfaces"
```

- [ ] **Step 5: Record the handoff revision**

Run: `git log -1 --format='%H %s'`

Expected: one full commit hash followed by `docs: freeze RateMem core interfaces`; use that immutable revision as the starting point for the SANA/Modal plan.
