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

- [ ] **Step 1: Write canonical serialization and packet-hash tests**

```python
# tests/state/test_serialization.py
import pytest

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import (
    bundle_cost_bytes,
    decode_state,
    encode_state,
    packet_from_payload,
)


def test_packet_hash_state_bytes_and_bundle_delta_are_exact() -> None:
    packet = packet_from_payload(b"enhancement")
    assert packet.packet_id == packet_from_payload(b"enhancement").packet_id
    base = BaseRecord("concept-a", b"base", reads=2, created_at=1)
    incidence = Incidence("concept-a", packet.packet_id, gain_q=7)
    empty_packets = MemoryState(bases={"concept-a": base})
    state = MemoryState(
        bases={"concept-a": base},
        packets={packet.packet_id: packet},
        incidences={("concept-a", packet.packet_id): incidence},
    )
    encoded = encode_state(state)
    assert encode_state(decode_state(encoded)) == encoded
    assert state.serialized_bytes == len(encoded)
    assert len(encoded) - len(encode_state(empty_packets)) == bundle_cost_bytes(
        packet, (incidence,)
    )


def test_state_owns_immutable_mapping_copies() -> None:
    source = {"concept-a": BaseRecord("concept-a", b"base", reads=0, created_at=1)}
    state = MemoryState(bases=source)
    source.clear()
    assert tuple(state.bases) == ("concept-a",)
    with pytest.raises(TypeError):
        state.bases["concept-b"] = BaseRecord(  # type: ignore[index]
            "concept-b", b"base", 0, 2
        )


def test_packet_payload_and_references_are_checked_on_decode() -> None:
    packet = Packet(packet_id="0" * 64, payload=b"wrong")
    state = MemoryState(bases={}, packets={packet.packet_id: packet}, incidences={})
    with pytest.raises(ValueError, match="packet hash mismatch"):
        decode_state(encode_state(state))

    valid = packet_from_payload(b"valid")
    dangling = MemoryState(
        packets={valid.packet_id: valid},
        incidences={
            ("missing", valid.packet_id): Incidence(
                "missing", valid.packet_id, gain_q=1
            )
        },
    )
    with pytest.raises(ValueError, match="dangling packet incidence"):
        decode_state(encode_state(dangling))
```

- [ ] **Step 2: Run the focused tests and verify missing modules fail**

Run: `uv run pytest tests/state/test_serialization.py -q`

Expected: collection fails because `ratemem.state.model` does not exist.

- [ ] **Step 3: Implement the immutable state types**

```python
# src/ratemem/state/model.py
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


def _frozen_copy(values: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class BaseRecord:
    handle: str
    payload: bytes
    reads: int
    created_at: int

    def __post_init__(self) -> None:
        if not self.handle:
            raise ValueError("handle must be nonempty")
        if not 0 <= self.reads <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("reads must fit uint64")
        if not 0 <= self.created_at <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("created_at must fit uint64")


@dataclass(frozen=True, slots=True)
class Packet:
    packet_id: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class Incidence:
    handle: str
    packet_id: str
    gain_q: int

    def __post_init__(self) -> None:
        if not -0x8000 <= self.gain_q <= 0x7FFF:
            raise ValueError("gain_q must fit int16")


@dataclass(frozen=True, slots=True)
class MemoryState:
    bases: Mapping[str, BaseRecord] = field(default_factory=dict)
    packets: Mapping[str, Packet] = field(default_factory=dict)
    incidences: Mapping[tuple[str, str], Incidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bases", _frozen_copy(self.bases))
        object.__setattr__(self, "packets", _frozen_copy(self.packets))
        object.__setattr__(self, "incidences", _frozen_copy(self.incidences))

    @property
    def serialized_bytes(self) -> int:
        from ratemem.state.serialization import encode_state

        return len(encode_state(self))
```

- [ ] **Step 4: Implement canonical CBOR serialization and validation**

```python
# src/ratemem/state/serialization.py
from __future__ import annotations

import hashlib
import struct
from typing import Any, cast

import cbor2

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet

_MAGIC = b"RTMEM001"
_VERSION = 1
_HEADER = struct.Struct("<8sIQQQ")
_LENGTH = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")
_INT16 = struct.Struct("<h")


def _frame(row: list[Any]) -> bytes:
    payload = cbor2.dumps(row, canonical=True)
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("record exceeds the uint32 frame limit")
    return _LENGTH.pack(len(payload)) + payload


def _base_frame(record: BaseRecord) -> bytes:
    return _frame(
        [
            record.handle,
            record.payload,
            _UINT64.pack(record.reads),
            _UINT64.pack(record.created_at),
        ]
    )


def _packet_frame(packet: Packet) -> bytes:
    return _frame([packet.packet_id, packet.payload])


def _incidence_frame(incidence: Incidence) -> bytes:
    return _frame(
        [incidence.handle, incidence.packet_id, _INT16.pack(incidence.gain_q)]
    )


def packet_from_payload(payload: bytes) -> Packet:
    packet_id = hashlib.sha256(payload).hexdigest()
    return Packet(packet_id=packet_id, payload=payload)


def bundle_cost_bytes(packet: Packet, incidences: tuple[Incidence, ...]) -> int:
    if not incidences:
        raise ValueError("packet bundle must contain at least one incidence")
    if any(edge.packet_id != packet.packet_id for edge in incidences):
        raise ValueError("bundle incidence points at another packet")
    if len({edge.handle for edge in incidences}) != len(incidences):
        raise ValueError("packet bundle repeats a concept incidence")
    return len(_packet_frame(packet)) + sum(
        len(_incidence_frame(edge)) for edge in incidences
    )


def encode_state(state: MemoryState) -> bytes:
    bases = sorted(state.bases.values(), key=lambda item: item.handle)
    packets = sorted(state.packets.values(), key=lambda item: item.packet_id)
    incidences = sorted(
        state.incidences.values(), key=lambda item: (item.handle, item.packet_id)
    )
    output = bytearray(
        _HEADER.pack(_MAGIC, _VERSION, len(bases), len(packets), len(incidences))
    )
    for record in bases:
        output.extend(_base_frame(record))
    for packet in packets:
        output.extend(_packet_frame(packet))
    for incidence in incidences:
        output.extend(_incidence_frame(incidence))
    return bytes(output)


def decode_state(payload: bytes) -> MemoryState:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated memory-state header")
    magic, version, base_count, packet_count, incidence_count = _HEADER.unpack_from(
        payload
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("unsupported memory-state version")
    offset = _HEADER.size

    def take_row() -> list[Any]:
        nonlocal offset
        if offset + _LENGTH.size > len(payload):
            raise ValueError("truncated record length")
        (size,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        end = offset + size
        if end > len(payload):
            raise ValueError("truncated record payload")
        row = cast(list[Any], cbor2.loads(payload[offset:end]))
        offset = end
        return row

    base_rows = [take_row() for _ in range(base_count)]
    packet_rows = [take_row() for _ in range(packet_count)]
    incidence_rows = [take_row() for _ in range(incidence_count)]
    if offset != len(payload):
        raise ValueError("trailing bytes after memory state")
    bases = {
        row[0]: BaseRecord(
            row[0],
            row[1],
            _UINT64.unpack(row[2])[0],
            _UINT64.unpack(row[3])[0],
        )
        for row in base_rows
    }
    packets = {row[0]: Packet(row[0], row[1]) for row in packet_rows}
    incidences = {
        (row[0], row[1]): Incidence(row[0], row[1], _INT16.unpack(row[2])[0])
        for row in incidence_rows
    }
    if (
        len(bases) != base_count
        or len(packets) != packet_count
        or len(incidences) != incidence_count
    ):
        raise ValueError("duplicate serialized state key")
    for packet in packets.values():
        if hashlib.sha256(packet.payload).hexdigest() != packet.packet_id:
            raise ValueError("packet hash mismatch")
    for edge in incidences.values():
        if edge.handle not in bases or edge.packet_id not in packets:
            raise ValueError("dangling packet incidence")
    return MemoryState(bases=bases, packets=packets, incidences=incidences)
```

- [ ] **Step 5: Run state tests and commit**

Run: `uv run pytest tests/state/test_serialization.py -q`

Expected: `3 passed`.

```bash
git add src/ratemem/state tests/state
git commit -m "feat: add canonical memory state"
```

### Task 3: Implement atomic packet-store lifecycle operations

**Files:**
- Create: `src/ratemem/state/store.py`
- Create: `tests/state/test_store.py`

- [ ] **Step 1: Write create, deduplication, redirection, deletion, and budget tests**

```python
# tests/state/test_store.py
import pytest

from ratemem.state.model import Incidence, Packet
from ratemem.state.serialization import packet_from_payload
from ratemem.state.store import BudgetExceeded, PacketStore


def test_delete_reclaims_only_unreferenced_packets() -> None:
    store = PacketStore.empty(budget_bytes=4096)
    packet = packet_from_payload(b"shared")
    store = store.create("a", b"base-a", created_at=1)
    store = store.create("b", b"base-b", created_at=2)
    store = store.attach_bundle(
        packet,
        (
            Incidence("a", packet.packet_id, 4),
            Incidence("b", packet.packet_id, 5),
        ),
    )

    after_a = store.delete("a")
    assert packet.packet_id in after_a.state.packets
    after_b = after_a.delete("b")
    assert packet.packet_id not in after_b.state.packets
    assert after_b.state.incidences == {}


def test_failed_transaction_does_not_mutate_old_state() -> None:
    store = PacketStore.empty(budget_bytes=512).create("a", b"small", created_at=1)
    packet = packet_from_payload(b"x" * 512)
    with pytest.raises(BudgetExceeded):
        store.attach(packet, Incidence("a", packet.packet_id, 1))
    assert packet.packet_id not in store.state.packets


def test_replace_redirects_one_concept_atomically_and_preserves_shared_packet() -> None:
    shared = packet_from_payload(b"shared")
    private = packet_from_payload(b"private-a")
    store = PacketStore.empty(budget_bytes=4096)
    store = store.create("a", b"old-a", created_at=1).create("b", b"base-b", created_at=2)
    store = store.attach(shared, Incidence("a", shared.packet_id, 2))
    store = store.attach(shared, Incidence("b", shared.packet_id, 3))

    updated = store.replace(
        "a", b"new-a", ((private, Incidence("a", private.packet_id, 4)),)
    )
    assert updated.state.bases["a"].payload == b"new-a"
    assert ("a", shared.packet_id) not in updated.state.incidences
    assert ("b", shared.packet_id) in updated.state.incidences
    assert shared.packet_id in updated.state.packets
    assert private.packet_id in updated.state.packets
    assert store.state.bases["a"].payload == b"old-a"


def test_attach_rejects_forged_content_address() -> None:
    store = PacketStore.empty(budget_bytes=2048).create("a", b"base", created_at=1)
    forged = Packet("0" * 64, b"payload")
    with pytest.raises(ValueError, match="packet hash mismatch"):
        store.attach(forged, Incidence("a", forged.packet_id, 1))
```

- [ ] **Step 2: Run the tests and confirm `PacketStore` is missing**

Run: `uv run pytest tests/state/test_store.py -q`

Expected: collection fails importing `ratemem.state.store`.

- [ ] **Step 3: Implement functional store transitions and reference-count GC**

```python
# src/ratemem/state/store.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet


class BudgetExceeded(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PacketStore:
    state: MemoryState
    budget_bytes: int

    def __post_init__(self) -> None:
        if self.budget_bytes < 0:
            raise ValueError("budget_bytes must be nonnegative")
        if self.state.serialized_bytes > self.budget_bytes:
            raise BudgetExceeded(
                f"state uses {self.state.serialized_bytes} bytes, "
                f"budget is {self.budget_bytes}"
            )

    @classmethod
    def empty(cls, budget_bytes: int) -> "PacketStore":
        return cls(state=MemoryState(), budget_bytes=budget_bytes)

    def _checked(self, state: MemoryState) -> "PacketStore":
        return PacketStore(state=state, budget_bytes=self.budget_bytes)

    @staticmethod
    def _validate_attachment(packet: Packet, incidence: Incidence, handle: str) -> None:
        if hashlib.sha256(packet.payload).hexdigest() != packet.packet_id:
            raise ValueError("packet hash mismatch")
        if incidence.handle != handle:
            raise ValueError("incidence handle does not match operation")
        if incidence.packet_id != packet.packet_id:
            raise ValueError("incidence packet id does not match payload")

    @staticmethod
    def _collect_referenced(
        packets: dict[str, Packet], incidences: dict[tuple[str, str], Incidence]
    ) -> dict[str, Packet]:
        referenced = {edge.packet_id for edge in incidences.values()}
        return {key: value for key, value in packets.items() if key in referenced}

    def create(self, handle: str, payload: bytes, created_at: int) -> "PacketStore":
        if handle in self.state.bases:
            raise ValueError(f"handle already exists: {handle}")
        bases = dict(self.state.bases)
        bases[handle] = BaseRecord(handle, payload, reads=0, created_at=created_at)
        return self._checked(MemoryState(bases, self.state.packets, self.state.incidences))

    def attach(self, packet: Packet, incidence: Incidence) -> "PacketStore":
        return self.attach_bundle(packet, (incidence,))

    def attach_bundle(
        self, packet: Packet, bundle: tuple[Incidence, ...]
    ) -> "PacketStore":
        if not bundle:
            raise ValueError("packet bundle must contain at least one incidence")
        handles = [incidence.handle for incidence in bundle]
        if len(set(handles)) != len(handles):
            raise ValueError("packet bundle repeats a concept incidence")
        for incidence in bundle:
            if incidence.handle not in self.state.bases:
                raise KeyError(incidence.handle)
            self._validate_attachment(packet, incidence, incidence.handle)
        packets = dict(self.state.packets)
        packets[packet.packet_id] = packet
        incidences = dict(self.state.incidences)
        for incidence in bundle:
            incidences[(incidence.handle, incidence.packet_id)] = incidence
        return self._checked(MemoryState(self.state.bases, packets, incidences))

    def replace(
        self,
        handle: str,
        payload: bytes,
        attachments: tuple[tuple[Packet, Incidence], ...],
    ) -> "PacketStore":
        if handle not in self.state.bases:
            raise KeyError(handle)
        for packet, incidence in attachments:
            self._validate_attachment(packet, incidence, handle)
        old = self.state.bases[handle]
        bases = dict(self.state.bases)
        bases[handle] = BaseRecord(handle, payload, old.reads, old.created_at)
        packets = dict(self.state.packets)
        incidences = {
            key: value
            for key, value in self.state.incidences.items()
            if value.handle != handle
        }
        for packet, incidence in attachments:
            packets[packet.packet_id] = packet
            incidences[(handle, packet.packet_id)] = incidence
        packets = self._collect_referenced(packets, incidences)
        return self._checked(MemoryState(bases, packets, incidences))

    def read(self, handle: str, update_usage: bool = True) -> tuple["PacketStore", BaseRecord]:
        record = self.state.bases[handle]
        if not update_usage:
            return self, record
        bases = dict(self.state.bases)
        bases[handle] = BaseRecord(handle, record.payload, record.reads + 1, record.created_at)
        return self._checked(MemoryState(bases, self.state.packets, self.state.incidences)), record

    def delete(self, handle: str) -> "PacketStore":
        if handle not in self.state.bases:
            raise KeyError(handle)
        bases = {key: value for key, value in self.state.bases.items() if key != handle}
        incidences = {
            key: value for key, value in self.state.incidences.items() if value.handle != handle
        }
        packets = self._collect_referenced(dict(self.state.packets), incidences)
        return self._checked(MemoryState(bases, packets, incidences))
```

- [ ] **Step 4: Run store and serialization tests**

Run: `uv run pytest tests/state -q`

Expected: `7 passed`.

- [ ] **Step 5: Commit the packet store**

```bash
git add src/ratemem/state/store.py tests/state/test_store.py
git commit -m "feat: add transactional packet store"
```

### Task 4: Add the deterministic progressive CPU codec

**Files:**
- Create: `src/ratemem/codec/__init__.py`
- Create: `src/ratemem/codec/progressive.py`
- Create: `tests/codec/test_progressive.py`

- [ ] **Step 1: Write reconstruction and prefix-consistency tests**

```python
# tests/codec/test_progressive.py
import numpy as np

from ratemem.codec.progressive import ProgressiveCodec


def test_packets_monotonically_reduce_code_error() -> None:
    code = np.array([0.1, -1.7, 0.3, 2.2, -0.8, 0.4, 1.1, -0.2], dtype=np.float32)
    encoded = ProgressiveCodec(group_size=2).encode("a", code)
    errors = []
    for count in range(len(encoded.packets) + 1):
        decoded = encoded.decode(packet_count=count)
        errors.append(float(np.mean((decoded - code) ** 2)))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


def test_packet_payloads_are_deterministic() -> None:
    code = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    codec = ProgressiveCodec(group_size=3)
    first = codec.encode("a", code)
    second = codec.encode("a", code)
    assert first.base_payload == second.base_payload
    assert [item.packet.packet_id for item in first.packets] == [
        item.packet.packet_id for item in second.packets
    ]
```

- [ ] **Step 2: Verify tests fail because the codec is absent**

Run: `uv run pytest tests/codec/test_progressive.py -q`

Expected: collection fails importing `ratemem.codec.progressive`.

- [ ] **Step 3: Implement the base quantizer and immutable residual packets**

```python
# src/ratemem/codec/progressive.py
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ratemem.state.model import Packet
from ratemem.state.serialization import packet_from_payload

_PACKET_HEADER = struct.Struct("<II")
FloatArray: TypeAlias = NDArray[np.float32]


def _encode_residual(group: int, start: int, values: FloatArray) -> bytes:
    body = np.asarray(values, dtype="<f2").tobytes(order="C")
    return _PACKET_HEADER.pack(group, start) + body


def _decode_residual(payload: bytes) -> tuple[int, int, FloatArray]:
    if len(payload) < _PACKET_HEADER.size:
        raise ValueError("truncated residual packet")
    group, start = _PACKET_HEADER.unpack(payload[: _PACKET_HEADER.size])
    values = np.frombuffer(payload[_PACKET_HEADER.size :], dtype="<f2").astype(
        np.float32
    )
    return group, start, cast(FloatArray, values)


@dataclass(frozen=True, slots=True)
class EncodedPacket:
    group: int
    packet: Packet


@dataclass(frozen=True, slots=True)
class EncodedCode:
    handle: str
    shape: tuple[int, ...]
    base_payload: bytes
    packets: tuple[EncodedPacket, ...]

    def decode(self, packet_count: int) -> FloatArray:
        if not 0 <= packet_count <= len(self.packets):
            raise ValueError("packet_count is outside the progressive stream")
        with io.BytesIO(self.base_payload) as stream:
            base = cast(
                FloatArray, np.load(stream, allow_pickle=False).astype(np.float32)
            )
        output = base.reshape(-1).copy()
        for encoded in self.packets[:packet_count]:
            group, start, values = _decode_residual(encoded.packet.payload)
            if group != encoded.group:
                raise ValueError("packet group mismatch")
            output[start : start + len(values)] += values
        return cast(FloatArray, output.reshape(self.shape))


class ProgressiveCodec:
    def __init__(self, group_size: int) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.group_size = group_size

    def encode(self, handle: str, code: FloatArray) -> EncodedCode:
        flat = cast(FloatArray, np.asarray(code, dtype=np.float32).reshape(-1))
        if flat.size == 0 or not np.all(np.isfinite(flat)):
            raise ValueError("code must be finite and nonempty")
        scale = max(
            float(np.max(np.abs(flat))) / 127.0, np.finfo(np.float32).eps
        )
        base = (
            np.round(flat / scale).clip(-127, 127).astype(np.int8).astype(np.float32)
            * scale
        )
        base_stream = io.BytesIO()
        np.save(base_stream, base.reshape(code.shape).astype(np.float16), allow_pickle=False)
        decoded_base = base.astype(np.float16).astype(np.float32)
        residual = flat - decoded_base
        packets: list[EncodedPacket] = []
        for group, start in enumerate(range(0, len(flat), self.group_size)):
            payload = _encode_residual(group, start, residual[start : start + self.group_size])
            packets.append(EncodedPacket(group, packet_from_payload(payload)))
        return EncodedCode(handle, tuple(code.shape), base_stream.getvalue(), tuple(packets))
```

Before this step is considered green, extend the starter to the frozen Gate 1 codec contract:
`EncodedCode` owns `group_size`; every decode validates the exact canonical little-endian float16
NPY base and the global packet cardinality implied by `shape` and `group_size`; missing or extra
suffix packets invalidate even `decode(0)`. After those global checks, validate hashes, group order,
offsets, body sizes, and finite values only through the requested prefix, so malformed payload/hash/
metadata in an existing unselected suffix does not invalidate a shorter prefix. The canonical tests
also cover float16 range boundaries, defensive byte ownership, and exact reserialization of the NPY
base.

- [ ] **Step 4: Add malformed-packet rejection coverage**

Append this test to `tests/codec/test_progressive.py`:

```python
def test_truncated_packet_is_rejected() -> None:
    import pytest

    from ratemem.codec.progressive import _decode_residual

    with pytest.raises(ValueError, match="truncated residual packet"):
        _decode_residual(b"bad")
```

Expected: the new test passes and `rg 'allow_pickle=True' src/ratemem` returns no matches.

- [ ] **Step 5: Run codec tests and commit**

Run:

```bash
uv run pytest tests/codec/test_progressive.py -q
rg 'allow_pickle=True' src/ratemem && exit 1 || true
```

Expected: `3 passed`; no unsafe pickle match.

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
```

- [ ] **Step 2: Run the objective tests and verify the import fails**

Run: `uv run pytest tests/allocation/test_objective.py -q`

Expected: collection fails importing `ratemem.allocation.objective`.

- [ ] **Step 3: Implement immutable bundles and the exact value oracle**

```python
# src/ratemem/allocation/objective.py
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


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
```

- [ ] **Step 4: Run objective tests and static checks**

Run:

```bash
uv run pytest tests/allocation/test_objective.py -q
uv run ruff check src/ratemem/allocation tests/allocation
uv run mypy src/ratemem/allocation
```

Expected: `3 passed`; Ruff and mypy exit 0.

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


def test_rounding_cannot_change_certified_selection() -> None:
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
    reduced = prescreen_certified_oracle(oracle, budget_bytes=1, max_bundles=1)
    assert tuple(reduced.bundles) == ("a-exact",)


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

- [ ] **Step 1: Write the probe-isolation and deterministic-replay tests**

```python
# tests/lifecycle/test_replay.py
import pytest
from hypothesis import given, strategies as st

from ratemem.lifecycle.events import (
    CreateEvent,
    DeleteEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.lifecycle.replay import replay
from ratemem.state.model import MemoryState
from ratemem.state.store import BudgetExceeded


def test_empty_state_budget_failure_precedes_event_replay() -> None:
    minimum = MemoryState().serialized_bytes
    events = (CreateEvent(event_id="1", handle="a", base_payload=b"base"),)
    with pytest.raises(BudgetExceeded):
        replay(events, budget_bytes=minimum - 1)


def test_event_budget_failure_is_recorded_after_store_initialization() -> None:
    minimum = MemoryState().serialized_bytes
    events = (CreateEvent(event_id="1", handle="a", base_payload=b"base"),)
    result = replay(events, budget_bytes=minimum)
    assert result.state == MemoryState()
    assert result.errors == ("1:budget-exceeded:a",)


def test_probe_does_not_refresh_usage_or_change_bytes() -> None:
    events = (
        CreateEvent(event_id="1", handle="a", base_payload=b"base"),
        ReadEvent(event_id="2", handle="a"),
        ProbeEvent(event_id="3", handle="a"),
    )
    result = replay(events, budget_bytes=2048)
    assert result.state.bases["a"].reads == 1
    assert result.probe_sizes == (result.state.serialized_bytes,)


def test_replay_is_deterministic_and_probe_rejects_stale_handle() -> None:
    events = (
        CreateEvent(event_id="1", handle="a", base_payload=b"base"),
        DeleteEvent(event_id="2", handle="a"),
        ProbeEvent(event_id="3", handle="a"),
    )
    first = replay(events, budget_bytes=2048)
    second = replay(events, budget_bytes=2048)
    assert first == second
    assert first.errors == ("3:stale-handle:a",)


def test_update_preserves_usage_and_stale_delete_is_recorded() -> None:
    events = (
        CreateEvent(event_id="1", handle="a", base_payload=b"old"),
        ReadEvent(event_id="2", handle="a"),
        UpdateEvent(event_id="3", handle="a", base_payload=b"new"),
        DeleteEvent(event_id="4", handle="missing"),
    )
    result = replay(events, budget_bytes=2048)
    assert result.state.bases["a"].payload == b"new"
    assert result.state.bases["a"].reads == 1
    assert result.errors == ("4:stale-handle:missing",)


@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(("create", "update", "read", "delete", "probe")),
            st.integers(min_value=0, max_value=4),
            st.binary(min_size=0, max_size=256),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_randomized_replay_never_exceeds_budget(
    operations: list[tuple[str, int, bytes]],
) -> None:
    events = []
    for index, (kind, slot, payload) in enumerate(operations):
        event_id = str(index)
        handle = f"h{slot}"
        if kind == "create":
            events.append(CreateEvent(event_id, handle, payload))
        elif kind == "update":
            events.append(UpdateEvent(event_id, handle, payload))
        elif kind == "read":
            events.append(ReadEvent(event_id, handle))
        elif kind == "delete":
            events.append(DeleteEvent(event_id, handle))
        else:
            events.append(ProbeEvent(event_id, handle))
    result = replay(tuple(events), budget_bytes=512)
    assert result.state.serialized_bytes <= 512
    assert replay(tuple(events), budget_bytes=512) == result
```

- [ ] **Step 2: Run the tests and verify lifecycle modules are missing**

Run: `uv run pytest tests/lifecycle/test_replay.py -q`

Expected: collection fails importing `ratemem.lifecycle.events`.

- [ ] **Step 3: Implement the closed event union**

```python
# src/ratemem/lifecycle/events.py
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class CreateEvent:
    event_id: str
    handle: str
    base_payload: bytes


@dataclass(frozen=True, slots=True)
class ReadEvent:
    event_id: str
    handle: str


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    event_id: str
    handle: str
    base_payload: bytes


@dataclass(frozen=True, slots=True)
class ProbeEvent:
    event_id: str
    handle: str


@dataclass(frozen=True, slots=True)
class DeleteEvent:
    event_id: str
    handle: str


LifecycleEvent: TypeAlias = (
    CreateEvent | ReadEvent | UpdateEvent | ProbeEvent | DeleteEvent
)
```

- [ ] **Step 4: Implement deterministic replay with probe copies**

```python
# src/ratemem/lifecycle/replay.py
from __future__ import annotations

from dataclasses import dataclass

from ratemem.lifecycle.events import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.state.model import MemoryState
from ratemem.state.store import BudgetExceeded, PacketStore


@dataclass(frozen=True, slots=True)
class ReplayResult:
    state: MemoryState
    probe_sizes: tuple[int, ...]
    errors: tuple[str, ...]


def replay(events: tuple[LifecycleEvent, ...], budget_bytes: int) -> ReplayResult:
    store = PacketStore.empty(budget_bytes)
    probes: list[int] = []
    errors: list[str] = []
    for index, event in enumerate(events):
        if isinstance(event, CreateEvent):
            try:
                store = store.create(event.handle, event.base_payload, created_at=index)
            except BudgetExceeded:
                errors.append(f"{event.event_id}:budget-exceeded:{event.handle}")
            except ValueError:
                errors.append(f"{event.event_id}:duplicate-handle:{event.handle}")
        elif isinstance(event, ReadEvent):
            try:
                store, _ = store.read(event.handle, update_usage=True)
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
        elif isinstance(event, UpdateEvent):
            try:
                store = store.replace(event.handle, event.base_payload, attachments=())
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
            except BudgetExceeded:
                errors.append(f"{event.event_id}:budget-exceeded:{event.handle}")
        elif isinstance(event, ProbeEvent):
            try:
                snapshot, _ = store.read(event.handle, update_usage=False)
                probes.append(snapshot.state.serialized_bytes)
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
        elif isinstance(event, DeleteEvent):
            try:
                store = store.delete(event.handle)
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
        else:
            raise TypeError(f"unsupported event: {type(event).__name__}")
    return ReplayResult(store.state, tuple(probes), tuple(errors))
```

- [ ] **Step 5: Run lifecycle tests and commit**

Run: `uv run pytest tests/lifecycle -q`

Expected: `4 passed`.

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

The `/artifacts/` ignore rule must stay root-anchored so nested artifact source and test packages remain tracked and scanned.

- [ ] **Step 1: Write artifact redaction and CLI smoke tests**

```python
# tests/artifacts/test_schema.py
from ratemem.artifacts.schema import AttemptManifest


def test_manifest_rejects_secret_shaped_values() -> None:
    try:
        AttemptManifest(
            run_id="cpu-smoke",
            git_revision="f" * 40,
            config_hash="a" * 64,
            status="passed",
            notes="token " + "ak-" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
        )
    except ValueError as exc:
        assert "credential-shaped" in str(exc)
    else:
        raise AssertionError("credential-shaped value was accepted")
```

```python
# tests/test_cli.py
import json
import subprocess
import sys


def test_core_smoke_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ratemem.cli", "smoke-core"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["serialized_bytes"] <= payload["budget_bytes"]
```

- [ ] **Step 2: Run both tests and verify missing modules fail**

Run: `uv run pytest tests/artifacts/test_schema.py tests/test_cli.py -q`

Expected: collection fails importing `ratemem.artifacts.schema`.

- [ ] **Step 3: Implement the validated attempt manifest**

```python
# src/ratemem/artifacts/schema.py
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

_SECRET = re.compile(r"(?:ak|as)-[A-Za-z0-9_-]{20,}")


class AttemptManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    git_revision: str
    config_hash: str
    status: Literal["passed", "failed", "interrupted"]
    notes: str = ""

    @field_validator("run_id", "git_revision", "config_hash", "notes")
    @classmethod
    def reject_credentials(cls, value: str) -> str:
        if _SECRET.search(value):
            raise ValueError("credential-shaped value is forbidden")
        return value
```

- [ ] **Step 4: Implement the self-contained CPU smoke command**

```python
# src/ratemem/cli.py
from __future__ import annotations

import argparse
import json

import numpy as np

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.allocation.snapshot import allocate_snapshot
from ratemem.codec.progressive import ProgressiveCodec
from ratemem.state.model import Incidence
from ratemem.state.serialization import bundle_cost_bytes
from ratemem.state.store import PacketStore


def smoke_core() -> dict[str, int | str]:
    budget = 8192
    encoded = ProgressiveCodec(group_size=4).encode(
        "concept-a", np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    )
    store = PacketStore.empty(budget).create("concept-a", encoded.base_payload, created_at=0)
    packet = encoded.packets[0].packet
    incidence = Incidence("concept-a", packet.packet_id, gain_q=8)
    store = store.attach(packet, incidence)
    packet_bytes = bundle_cost_bytes(packet, (incidence,))
    oracle = CoverageOracle(
        {
            packet.packet_id: PacketBundle(
                packet.packet_id, packet_bytes, {"concept-a": (1.0,)}
            )
        },
        {"concept-a": 1.0},
        {"concept-a": (1.0,)},
    )
    chosen = allocate_snapshot(oracle, packet_bytes)
    if chosen != frozenset({packet.packet_id}):
        raise RuntimeError("snapshot allocator rejected the only feasible useful packet")
    return {
        "status": "passed",
        "serialized_bytes": store.state.serialized_bytes,
        "budget_bytes": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="ratemem")
    parser.add_argument("command", choices=("smoke-core",))
    args = parser.parse_args()
    if args.command == "smoke-core":
        print(json.dumps(smoke_core(), sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the full quality gate and commit**

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

Expected: all tests pass; Ruff/mypy exit 0; CLI prints a JSON object with `"status": "passed"`; both quiet credential scans exit 0. The tracked-tree scan covers root configuration and scripts, while the explicit generated-root scan includes ignored and hidden artifacts, logs, and exports.

```bash
git add src/ratemem/artifacts src/ratemem/cli.py tests/artifacts tests/test_cli.py
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
measured state-length increment for a fixed admitted cohort. Probe reads never update usage.
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
