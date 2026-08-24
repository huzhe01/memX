# Snapshot allocation proof contract

At snapshot `t`, condition on the causal history and on a fixed admitted concept cohort. Base
records and their metadata have already been reserved, leaving packet budget `b_t`. The finite
ground set `G_t` contains immutable `PacketBundle` instances. Bundle `p` contains one payload/hash
and its complete prespecified incidence list; selecting individual incidences is not allowed.

The on-disk format has a fixed-size state header and length-framed canonical-CBOR records. Hence
for this fixed cohort, `PacketBundle.cost_bytes` is exactly the state-length increase caused by
installing the payload/hash and all of the bundle's incidences, and costs add across bundles.

For nonnegative past-only `CoverageOracle.request_weights` `omega[t,i]`, nonnegative locked
`CoverageOracle.group_weights` `beta[t,i,g]`, and nonnegative locked `PacketBundle.gains`
`v[t,i,g,p]`, the value returned by `CoverageOracle.value` is

    F_t(X) = sum_i omega[t,i] sum_g beta[t,i,g]
             min(1, sum_{p in X} v[t,i,g,p]).

`F_t(empty)=0`. Adding a bundle cannot decrease any inner modular sum, so `F_t` is monotone.
The capped-linear function is concave and nondecreasing over a nonnegative modular sum, so every
term has diminishing returns; a nonnegative weighted sum preserves submodularity.

`allocate_snapshot` enumerates every feasible seed of cardinality zero through three and completes
each seed by recomputing exact marginal-gain-per-byte values after every accepted bundle. It returns
the best completed seed, with deterministic tie breaking in favor of the lexicographically larger
packet-ID tuple. This is the standard partial-enumeration knapsack algorithm used to obtain the
`1 - 1/e` approximation for monotone submodular maximization under one modular knapsack constraint.
Lazy evaluation is permitted only after a test shows it returns the identical sequence as exact
recomputation.

Therefore, under the premises above,

    F_t(X_t) >= (1 - 1/e) max_{X subset G_t, cost(X) <= b_t} F_t(X).

This is a conditional per-snapshot guarantee. Whole-concept admission or eviction, optional
incidence dropping, switching penalties, hysteresis, learned unconstrained distortion, and any
future-aware competitive or dynamic-regret statement are outside the theorem. The future-aware
lifecycle oracle is an empirical upper reference only.

Mechanical release checks compare feasibility and the approximation ratio with exhaustive optima
on enumerated and seeded-random tiny instances. If either the premises or ratio checks fail, the
paper removes the theorem claim rather than changing the test after observing results.
