# RateMem core interface contract

This document freezes the Gate 1 CPU-memory boundary used by later RateMem plans. Public calls
listed below remain backward compatible. Schemas may grow only through versioned fields and
backward-compatible decoders. Names beginning with `_` are implementation details and are not part
of this contract.

`FloatArray` means `numpy.typing.NDArray[numpy.float32]`. All byte budgets and byte costs below are
exact Python integers; booleans and non-integral numeric values are not valid byte counts.

## State, serialization, and functional store

```python
BaseRecord(handle: str, payload: bytes, reads: int, created_at: int)
Packet(packet_id: str, payload: bytes)
Incidence(handle: str, packet_id: str, gain_q: int)
MemoryState(
    bases: Mapping[str, BaseRecord] = ...,
    packets: Mapping[str, Packet] = ...,
    incidences: Mapping[tuple[str, str], Incidence] = ...,
)
MemoryState.serialized_bytes: int

packet_from_payload(payload: bytes | bytearray | memoryview) -> Packet
encode_state(state: MemoryState) -> bytes
decode_state(payload: bytes) -> MemoryState
bundle_cost_bytes(packet: Packet, incidences: tuple[Incidence, ...]) -> int

PacketStore(state: MemoryState, budget_bytes: int)
PacketStore.empty(budget_bytes: int) -> PacketStore
PacketStore.create(handle: str, payload: bytes, created_at: int) -> PacketStore
PacketStore.attach(packet: Packet, incidence: Incidence) -> PacketStore
PacketStore.attach_bundle(packet: Packet, bundle: tuple[Incidence, ...]) -> PacketStore
PacketStore.replace(
    handle: str,
    payload: bytes,
    attachments: tuple[tuple[Packet, Incidence], ...],
) -> PacketStore
PacketStore.read(
    handle: str,
    update_usage: bool = True,
) -> tuple[PacketStore, BaseRecord]
PacketStore.delete(handle: str) -> PacketStore
```

Records own immutable byte copies, `MemoryState` owns immutable mapping copies, and packet IDs are
SHA-256 hashes of the owned payload. `encode_state` is a fixed header followed by length-framed
canonical-CBOR records in canonical key order. A decoder accepts exactly that canonical format,
validates hashes and references, and rejects trailing data. Consequently,
`state.serialized_bytes == len(encode_state(state))`.

For a fixed admitted cohort, `bundle_cost_bytes` is the measured state-length increment for adding
one new packet and its complete, unique incidence tuple. Packet-bundle costs are exact, positive,
and additive under the fixed-cohort assumptions in the proof contract.

Every store transition is functional and transactional: it returns a checked new `PacketStore`,
does not mutate the prior store, and commits neither partial state nor over-budget state on error.
Shared packets are deduplicated by content hash and are reclaimed only after their last incidence
is removed. `read(..., update_usage=True)` increments usage in the returned store;
`read(..., update_usage=False)` returns the same store and does not alter bytes or usage.

## Progressive codec

```python
EncodedPacket(group: int, packet: Packet)
EncodedCode(
    handle: str,
    shape: tuple[int, ...],
    base_payload: bytes,
    packets: tuple[EncodedPacket, ...],
    group_size: int,
)
EncodedCode.decode(packet_count: int) -> FloatArray

ProgressiveCodec(group_size: int)
ProgressiveCodec.encode(handle: str, code: FloatArray) -> EncodedCode
```

Encoding is deterministic for finite, nonempty float32 input in the supported float16 range.
Decoding validates the selected prefix, its content hashes, canonical group order and offsets, and
the canonical little-endian float16 NPY base. `packet_count=0` decodes the base; larger valid
prefixes apply that many residual packets. Malformation in an unselected suffix does not invalidate
a shorter selected prefix.

## Coverage and allocation

```python
PacketBundle(
    packet_id: str,
    cost_bytes: int,
    gains: Mapping[str, tuple[float, ...]],
)
CoverageOracle(
    bundles: Mapping[str, PacketBundle],
    request_weights: Mapping[str, float],
    group_weights: Mapping[str, tuple[float, ...]],
)
CoverageOracle.exact_value(selected: frozenset[str]) -> Fraction
CoverageOracle.exact_marginal(selected: frozenset[str], item: str) -> Fraction
CoverageOracle.value(selected: frozenset[str]) -> float
CoverageOracle.marginal(selected: frozenset[str], item: str) -> float
CoverageOracle.cost(selected: frozenset[str]) -> int

allocate_snapshot(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_bundles: int = 24,
) -> frozenset[str]
allocate_density_greedy_heuristic(
    oracle: CoverageOracle,
    budget_bytes: int,
) -> frozenset[str]
exhaustive_optimum(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_states: int = 2**18,
) -> frozenset[str]
```

Bundle costs are positive exact integers. Gains, group weights, and request weights are finite and
nonnegative. Certification treats each normalized float as its exact binary rational value via
`Fraction.from_float`; request and group weights are converted separately and multiplied exactly.
`exact_value` and `exact_marginal` are the certification APIs. `value` and `marginal` are rounded,
reporting-only floats and must not drive certified selection or theorem checks.

For a fixed admitted cohort of immutable complete bundles with exact modular costs,
`allocate_snapshot` performs exact-density partial enumeration and satisfies the conditional
`1 - 1/e` monotone-submodular single-knapsack guarantee in
`docs/theory/snapshot-allocation.md`. It uses deterministic ties in favor of the
lexicographically larger packet ID or sorted packet-ID tuple. The certified allocator rejects more
than 24 bundles by default; callers may explicitly raise `max_bundles`, and no heuristic fallback
occurs automatically.

`exhaustive_optimum` is a verification-only tiny-instance oracle. It has a hard 24-bundle ceiling,
prefilters individually infeasible bundles, and rejects more than `2**18` remaining subset states
by default; callers may explicitly raise `max_states`. `allocate_density_greedy_heuristic` is the
separately named scalable path. It retains exact density and deterministic ties but has no
approximation guarantee.

The theorem does not cover admission or whole-base eviction, optional incidence dropping,
switching costs or hysteresis, future-aware competitive or dynamic-regret claims, learned
unconstrained distortion, or the scalable heuristic. Future-aware lifecycle selection is only an
empirical upper reference.

## Lifecycle trace and probes

```python
CreateEvent(event_id: str, handle: str, base_payload: bytes)
ReadEvent(event_id: str, handle: str)
UpdateEvent(event_id: str, handle: str, base_payload: bytes)
ProbeEvent(event_id: str, handle: str)
DeleteEvent(event_id: str, handle: str)
LifecycleEvent = CreateEvent | ReadEvent | UpdateEvent | ProbeEvent | DeleteEvent

ReplayResult(
    state: MemoryState,
    probe_sizes: tuple[int, ...],
    errors: tuple[str, ...],
)
replay(events: tuple[LifecycleEvent, ...], budget_bytes: int) -> ReplayResult
```

Events are frozen values with exact nonempty string identities; create/update events own their
bytes. Replay accepts only the closed exact event types, starts from an empty bounded store, applies
events in tuple order, and is deterministic. Reads increment usage; updates preserve prior usage;
deletes reclaim only unreferenced packets. Duplicate-create, stale-handle, and budget failures are
recorded deterministically without exceeding the byte budget. A probe is a read-only size
observation: it appends the current `serialized_bytes` to `probe_sizes` without refreshing usage or
changing state bytes.

## Attempt artifacts and CPU smoke boundary

```python
AttemptManifest(
    *,
    run_id: str,
    git_revision: str,
    config_hash: str,
    status: Literal["passed", "failed", "interrupted"],
    notes: str = "",
)
AttemptManifest.model_validate_json(
    json_data: str | bytes | bytearray,
    *,
    strict: bool | None = None,
    context: Any | None = None,
    by_alias: bool | None = None,
    by_name: bool | None = None,
) -> Self

smoke_core() -> dict[str, int | str]
python -m ratemem.cli smoke-core
```

`AttemptManifest` is frozen, forbids extra fields, revalidates instances and copies, and disables
its public unchecked constructor. Its model-level Python validation and outbound serialization
reject credential-shaped values without echoing rejected content.
`AttemptManifest.model_validate_json` is the credential-safe public boundary for untrusted raw
JSON. Raw `TypeAdapter.validate_json` and enclosing-model JSON parsing occur before this model
schema and may retain malformed raw input, so they are not credential-safe raw-JSON entry points;
this limitation does not imply that every valid adapter input is rejected.

`smoke-core` is a self-contained CPU-only deterministic contract check. It exercises progressive
encode/decode with strict prefix improvement, functional storage and exact bytes, certified packet
selection, a read-only lifecycle probe, and an `AttemptManifest` JSON round trip. Success emits one
sorted JSON line with `status`, `serialized_bytes`, and `budget_bytes`, with serialized bytes no
greater than the budget. It performs no network, GPU, Modal, or credential operation.
