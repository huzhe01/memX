# Snapshot allocation proof contract

At snapshot `t`, condition on the causal history and on a fixed admitted concept cohort. Base
records and their metadata have already been reserved, leaving packet budget `b_t`. The finite
causal candidate set `G_t` contains immutable `PacketBundle` instances constructed only from the
available history. Bundle `p` contains one payload/hash and its complete prespecified incidence
list; selecting individual incidences is not allowed.

The on-disk format has a fixed-size state header and length-framed canonical-CBOR records. Hence,
for this fixed cohort, `PacketBundle.cost_bytes` is the exact integer state-length increase caused
by installing the payload/hash and every incidence in the bundle. These costs add across bundles.

## Certified objective

`CoverageOracle` first validates and normalizes every request weight, group weight, and gain to a
finite nonnegative Python `float`. The certified inputs are the exact binary rational numbers
represented by those normalized floats, obtained with `Fraction.from_float`. In particular,
request and group weights are converted separately and their coefficient product is then computed
exactly; an underflowed float product is never used. Exact gains and exact coefficient products are
cached when the oracle is constructed.

For exact normalized request weights `omega[t,i]`, group weights `beta[t,i,g]`, and bundle gains
`v[t,i,g,p]`, `CoverageOracle.exact_value` returns the `Fraction`

    F_t(X) = sum_i omega[t,i] sum_g beta[t,i,g]
             min(1, sum_{p in X} v[t,i,g,p]).

`F_t(empty)=0`. Adding a bundle cannot decrease an inner nonnegative modular sum, so `F_t` is
monotone. The capped-linear function is concave and nondecreasing over a nonnegative modular sum,
so each term has diminishing returns; a nonnegative weighted sum preserves submodularity.
`CoverageOracle.exact_marginal` computes this diminishing return directly as a `Fraction` rather
than subtracting rounded objective reports.

`CoverageOracle.value` and `CoverageOracle.marginal` convert their exact counterparts to `float`
only for reporting and backward compatibility. Constructor validation requires the exact maximum
objective mass to have a finite float representation. Reporting conversion can still round or
underflow, so neither certified selection nor a theorem check may use these reporting methods.
The theorem is about `exact_value` over the exact binary-rational normalized inputs.

## Deterministic causal prescreen

The controller passes its history-only candidate oracle to `prescreen_certified_oracle`. The
prescreen first removes every bundle whose individual cost exceeds `b_t`. It sorts the remaining
bundles in descending order by the exact singleton density

    (F_t({p}) - F_t(empty)) / cost_bytes(p),

using `Fraction` arithmetic and ties in favor of the lexicographically larger packet ID. Let `I_t`
be the individually feasible subset of `G_t`. Given an exact positive integer cap `L <= 24`, the
prescreen retains the first `min(L, |I_t|)` bundles in that descending order—exactly the
highest-density bundles. Call the resulting reduced ground set `C_t`. The operation is deterministic
and depends only on its caller-provided causal inputs; it neither reads future events nor mutates the
input oracle.

Filtering an individually infeasible bundle does not change a feasible optimum. Truncating the
remaining candidates by singleton density can change it. Consequently, the prescreen has **no
approximation guarantee relative to the full set `G_t`**. Every certified ratio below is stated
only relative to the reduced set `C_t`; comparisons against the full causal set are empirical.

## Certified algorithms and resource boundaries

`allocate_snapshot` enumerates every feasible seed of cardinality zero through three over `C_t` and
completes each seed by recomputing `exact_marginal / cost_bytes` for every remaining feasible bundle
after each accepted bundle. Fill state caches exact per-group coverage and selected integer cost.
Density, completed-candidate value, and final-candidate value comparisons use `Fraction` throughout,
with deterministic ties in favor of the lexicographically larger packet ID or packet-ID tuple.

Sviridenko's first phase keeps the best feasible singleton or pair; its second phase completes every
feasible triple by marginal-density greedy. This implementation greedily completes seeds of sizes
zero through three. Monotonicity means a completed singleton or pair is worth at least its seed, and
the triple phase is the cited procedure, so the returned best candidate dominates the cited
two-phase output. See Maxim Sviridenko, “A note on maximizing a submodular set function subject to a
knapsack constraint,” *Operations Research Letters* 32(1):41–43, 2004,
doi:10.1016/S0167-6377(03)00062-2. For a normalized monotone submodular value oracle and one modular
knapsack constraint, it gives

    F_t(X_t) >= (1 - 1/e) max_{X subset C_t, cost(X) <= b_t} F_t(X).

The prescreen defaults to `max_bundles=24`, accepts a smaller positive cap, and rejects a cap above
24. Therefore its output can be passed to the default certified allocator. Independently,
`allocate_snapshot` defaults to `max_bundles=24` and rejects a larger input ground set before
enumeration; a direct caller may raise that allocator resource limit explicitly, and there is no
automatic fallback. The exact test oracle `exhaustive_optimum` retains a hard 24-bundle ceiling,
prefilters bundles that are individually infeasible, and defaults to `max_states=2**18` over the
remaining ground set. It rejects before enumeration when that conservative state bound is exceeded
unless the caller explicitly raises `max_states`.

`allocate_density_greedy_heuristic` is the separately named scalable option for larger ground sets.
It uses the same cached exact coverage, exact density comparison, integer cost, and deterministic
ties, but it performs no seed enumeration and has **no `1 - 1/e` guarantee**. Refusal by the
certified allocator never invokes this heuristic implicitly.

## Mechanical and lifecycle boundary

Mechanical release checks compare `exact_value` against `exhaustive_optimum` on exhaustive-grid,
Hypothesis, seeded-random, multigroup, subnormal, and rounding-adversarial instances. They use the
conservative rational constant `6321205588285576 / 10**16`, which is strictly below `1 - 1/e`, and
cross-multiply exact values without an additive epsilon. Feasibility is always checked in exact
integer bytes. The rounding-adversarial allocator regression makes two singleton reports tie as
`float` even though their exact binary-rational values differ. It explicitly pins the exact
`exhaustive_optimum` identity before applying the cross-multiplied ratio check to the certified
output. That focused two-item case isolates rounded comparison; the exhaustive-grid, Hypothesis,
and seeded multigroup checks exercise multi-item greedy completion. A separate regression matching
the controller shape builds four concepts with eight distinct packets each (32 candidates), verifies
a deterministic reduction to 24, and passes the reduced oracle to `allocate_snapshot`. If the
premises or exact ratio checks fail, the paper removes the theorem claim rather than weakening a test
after observing results.

This is a conditional per-snapshot guarantee for fixed-cohort selection from the reduced set `C_t`
of immutable bundles with exact modular costs and nonnegative past-only weights and gains. It is not
a guarantee against the full set `G_t`. Whole-concept admission or eviction, optional incidence
dropping, switching penalties, hysteresis, learned unconstrained distortion, and future-aware
competitive or dynamic-regret statements are outside the theorem. The future-aware lifecycle
oracle is an empirical upper reference only.
