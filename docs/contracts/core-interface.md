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

`BaseRecord` and `Packet` own immutable payload-byte copies, and `MemoryState` owns immutable
mapping copies. The `Packet` and `MemoryState` constructors do not themselves verify content
hashes. `packet_from_payload` creates a SHA-256 ID from the owned payload; `PacketStore`
construction and transitions, plus `decode_state`, enforce that ID-to-payload relationship.
`encode_state` is a fixed header followed by length-framed canonical-CBOR records in canonical key
order. A decoder accepts exactly that canonical format, validates hashes and references, and
rejects trailing data. Consequently, `state.serialized_bytes == len(encode_state(state))`.

For a fixed admitted cohort, `bundle_cost_bytes` is the measured state-length increment for adding
one new packet and its complete, unique incidence tuple. Packet-bundle costs are exact, positive,
and additive under the fixed-cohort assumptions in the proof contract.

Every mutating store transition is functional and transactional: it returns a checked new
`PacketStore`, does not mutate the prior store, and commits neither partial state nor over-budget
state on error. A functional no-op read may return the same already checked instance. Shared
packets are deduplicated by content hash and are reclaimed only after their last incidence is
removed. `read(..., update_usage=True)` increments usage in a returned new store;
`read(..., update_usage=False)` may return the same store and does not alter bytes or usage.

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
For every requested prefix, decoding validates the canonical little-endian float16 NPY base and the
global packet-tuple cardinality implied by `shape` and `group_size`. Missing or extra suffix records
therefore invalidate every prefix. Once those global checks pass, the decoder validates content
hashes, canonical group order, offsets, and payload shape only for the selected prefix; payload,
hash, or metadata malformation in an existing unselected suffix packet does not invalidate a
shorter selected prefix. `packet_count=0` decodes the base; larger valid prefixes apply that many
residual packets.

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

prescreen_certified_oracle(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_bundles: int = 24,
) -> CoverageOracle
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

For the release lifecycle, `prescreen_certified_oracle` first removes individually infeasible
bundles, sorts the remainder by descending exact singleton marginal-gain density, and
deterministically retains at most `max_bundles`. Ties favor the lexicographically larger packet ID.
Its cap must be an exact non-boolean integer satisfying `1 <= max_bundles <= 24`; the default is 24,
and values above 24 are rejected. Calling the input candidate pool `G_t` and the returned immutable
ground set `C_t`, this causal pre-screen has no approximation guarantee relative to `G_t`. The new
oracle preserves the request and group weights.

For a fixed admitted cohort and that fixed reduced ground set `C_t`, `allocate_snapshot` performs
exact-density partial enumeration and satisfies the conditional `1 - 1/e`
monotone-submodular single-knapsack guarantee in `docs/theory/snapshot-allocation.md`. It uses
deterministic ties in favor of the lexicographically larger packet ID or sorted packet-ID tuple. The
certified allocator rejects more than 24 bundles by default; callers may explicitly raise
`max_bundles` for bounded research instances, but the release lifecycle always pre-screens to the
default cap. No heuristic fallback occurs automatically.

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
events in tuple order, and is deterministic. After exact nonnegative-integer budget validation,
`budget_bytes` must be large enough for the canonical empty state
(`MemoryState().serialized_bytes`); otherwise store initialization raises `BudgetExceeded` before
any event is applied or a `ReplayResult` exists. Reads increment usage; updates preserve prior usage;
deletes reclaim only unreferenced packets. Duplicate-create, stale-handle, and event-level
create/update budget failures are recorded deterministically in the returned result without
exceeding the byte budget. A probe is a read-only size observation: it appends the current
`serialized_bytes` to `probe_sizes` without refreshing usage or changing state bytes.

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

`AttemptManifest` is frozen and forbids extra fields. Its supported direct construction and
`AttemptManifest.model_validate` boundaries validate exact manifest instances; its own
`AttemptManifest.model_copy` override revalidates the copied data, and its own
`AttemptManifest.model_construct` is disabled. Deliberate calls through `BaseModel` implementations
or low-level object APIs are outside those guarantees; forged state is rejected when it re-enters a
supported validation or outbound serialization boundary. Those supported model-level Python
boundaries reject credential-shaped values without echoing rejected content.
`AttemptManifest.model_validate_json` is the credential-safe public boundary for untrusted raw
JSON. It rejects duplicate keys at every object depth before a value can be shadowed. Raw
`TypeAdapter.validate_json` and enclosing-model JSON parsing occur before this model schema and may
retain malformed raw input, so they are not credential-safe raw-JSON entry points; this limitation
does not imply that every valid adapter input is rejected.

`smoke-core` is a self-contained CPU-only deterministic contract check. It exercises progressive
encode/decode with strict prefix improvement, functional storage and exact bytes, causal packet
pre-screening followed by certified selection, a read-only lifecycle probe, and an
`AttemptManifest` JSON round trip. Success emits one
sorted JSON line with `status`, `serialized_bytes`, and `budget_bytes`, with serialized bytes no
greater than the budget. It performs no network, GPU, Modal, or credential operation.
