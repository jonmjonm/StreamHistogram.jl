# The `MomentAccumulator`: online, mergeable central moments

`MomentAccumulator` tracks the mean and the central moments
`M_p = Σᵢ (xᵢ - mean)^p` for `p = 2..maxpower`, updatable one point or one
batch at a time, without ever revisiting old data. This document explains
why the naive approach doesn't work, and the algorithm actually used
(`src/moments.jl`).

## The problem with the obvious approach

The textbook way to get, say, variance from a running total is

```
variance = E[X²] - E[X]²
```

This is numerically dangerous: if the mean is large relative to the spread
of the data, `E[X²]` and `E[X]²` are two large, nearly-equal numbers, and
their difference — which is what you actually want — has lost most of its
significant digits to cancellation. The error gets worse as more points
accumulate and as the moment order goes up (an 8th-power raw moment has a
much larger dynamic range than the 8th central moment itself). Streaming in
batches makes it worse still, if the naive approach is applied by simply
re-deriving central moments from raw power sums at the end.

## Welford's fix, and its generalization

Welford (1962) showed that variance can be tracked without ever computing
`E[X²]`: keep a running mean and `M2`, and update both together so that `M2`
is always defined relative to the *current* running mean — no re-centering
of old data, no subtraction of independently large quantities.

Terriberry (2007) extended this to skewness/kurtosis (`M3`, `M4`). Pébay
(2008, *"Formulas for Robust, One-Pass Parallel Computation of Covariances
and Arbitrary-Order Statistical Moments"*) generalized it twice more, in the
two directions this package actually needs:

1. **Arbitrary order** — a single recurrence gives the update for `M_p` in
   terms of `M_2 .. M_{p-1}`, for any `p`.
2. **Merging two partitions, not just appending one point** — given two
   independently-computed accumulators, a closed-form formula combines them
   into the accumulator for their union, expressed in terms of the *shift*
   between their means rather than the means' absolute values.

That merge is the whole implementation. A single point is just a batch of
size 1 whose own central moments are trivially zero, so `add!` for a point
and `add!` for a batch are literally the same code path.

## The merge formula

Given two partitions `A` (count `n_A`, mean `mean_A`, central moments
`M_p^A`) and `B` (`n_B`, `mean_B`, `M_p^B`), let

```
n     = n_A + n_B
δ     = mean_B - mean_A
mean  = mean_A + δ · n_B / n
```

Then for each order `p ≥ 2`:

```
M_p = M_p^A + M_p^B
    + Σ_{k=1}^{p-2} C(p,k) · [ M_{p-k}^A · (-n_B·δ/n)^k  +  M_{p-k}^B · (n_A·δ/n)^k ]
    + δ^p · n_A·n_B · (n_A^{p-1} - (-n_B)^{p-1}) / n^p
```

(`C(p,k)` is the binomial coefficient.) This is implemented generically in
`mergeaccumulators` for whatever `maxpower` the accumulator was built with —
no order-specific formulas are hand-derived, so it works the same way for
`p=2` as for `p=8` or any other order requested via `momentPowers`.

### Sanity check against the known p=2 and p=3 cases

For `p = 2` the sum over `k` is empty (`1..0`), leaving

```
M_2 = M_2^A + M_2^B + δ² · n_A n_B (n_A + n_B) / n²
    = M_2^A + M_2^B + δ² · n_A n_B / n
```

which is exactly Chan, Golub & LeVeque's (1979) parallel variance formula.

For `p = 3`, the `k=1` term gives `3·δ/n·(n_A M_2^B - n_B M_2^A)`, and the
last term simplifies (via `n_A² - n_B² = (n_A-n_B)(n_A+n_B) = (n_A-n_B)·n`)
to `δ³ n_A n_B (n_A - n_B) / n²` — matching the standard combined-skewness
formula quoted in the online-moments literature. Both reduce correctly, and
`mergeaccumulators` is order-independent (`merge(merge(A,B),C) ==
merge(A,merge(B,C))`, verified in the test suite) as any correct
associative combination rule must be.

### The empty-partition edge case

If `n_A = 0`, then `δ = mean_B`, `n_B/n = 1`, so `mean = mean_B`; every term
in the sum and the final term carries a factor of `n_A = 0` (and no term
ever needs `0^0`, since `k ≥ 1` throughout), so `M_p` reduces to `M_p^B`
exactly. Merging with an empty accumulator is therefore a no-op, which is
what lets a freshly-constructed `MomentAccumulator(maxpower)` — all zeros —
serve as the identity element that batches merge into.

## How it's used

- `batchaccumulator(xs, maxpower)` computes a fresh accumulator for one
  bounded batch directly (two-pass: mean, then `Σ(x-mean)^p`) — safe to do
  naively because it's a single batch, not the whole stream.
- `addpoint(A, x)` merges `A` with a trivial one-point accumulator.
- `addbatch(A, xs)` merges `A` with `batchaccumulator(xs, maxpower(A))`.
- `centralmoment(A, p)` returns `mean(A)` for `p == 1` (the true first
  central moment is trivially zero, so order 1 is defined to mean the mean
  itself) and `M[p] / n` for `p ≥ 2` (the population moment; no `n-1`
  correction).

Because `moment_powers` can be a sparse set (e.g. `[1, 2, 4, 8]`), the
accumulator internally still carries the full ladder `M_2 .. M_max` — the
recurrence for `M_p` depends on all the lower orders, not just the ones a
caller asked to see. That's a handful of extra `Float64`s, not a real cost.

## How accurate is this, really?

Chan, Golub & LeVeque (1983) showed the `p=2` online/parallel update has
error independent of `n` — comparable to Kahan-style compensated summation,
and far better than the naive `E[X²]-E[X]²` formula (whose error grows with
both `n` and the data's distance from zero). The same structural argument —
always working with deviations from a *current* mean, never differencing
two independently-large quantities — carries over to the higher orders, but
there isn't as clean a universal error bound at, say, `p=8`. In practice
(see `test/runtests.jl`), merging a stream in chunks agrees with a
whole-batch computation to about 1 part in 10⁷ at `p=7`-`8` for data with a
mean offset large relative to its spread — plenty for the moments' intended
use (`densityQuality`'s comparison against the ASH is far coarser than
that), but worth knowing it's not the ~15-digit agreement you'd get at
`p=2`. High-order central moments are just inherently more sensitive to
floating-point error than low-order ones, independent of the algorithm.
