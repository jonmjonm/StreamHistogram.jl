# StreamHistogram.jl

[![CI](https://github.com/jonmjonm/StreamHistogram.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jonmjonm/StreamHistogram.jl/actions/workflows/CI.yml)

`StreamHistogram` collects a histogram — and the statistics needed to judge
how much to trust it — from data arriving online, one point or one batch at
a time. It's meant for the case where you don't have all the data up front
(a stream, a simulation, a large file you don't want to hold in memory) but
still want a proper histogram, a smoothed density estimate, and moments
precise enough to check the two against each other. Points may carry
weights (e.g. importance weights), and every statistic is then available
both unweighted and weighted.

A `StreamHist` maintains, incrementally and simultaneously:

1. a traditional histogram (`StatsBase.Histogram`)
2. an Average Shifted Histogram (ASH) density estimate
3. the exact min and max of every point seen
4. online central moments at a configurable set of orders, computed with a
   numerically stable, mergeable algorithm (see
   [momentAccumulator.md](momentAccumulator.md))

and, for weighted points, each of 1, 2 and 4 again with every point counted
by its weight (see [Weights](#weights)).

## Trusting the numbers: exact moments, out-of-range counts, density quality

The moment accumulator is computed directly from the raw values as they
arrive — never from the binned histogram, never from the smoothed ASH — using
Pébay's mergeable generalization of Welford's algorithm (the full derivation,
its merge formula, and its precision properties are in
[momentAccumulator.md](momentAccumulator.md)). That makes it, for practical
purposes, exact and independent of every lossy choice the other two
representations have to make (bin width, smoothing bandwidth, range). Which
is exactly what makes it useful as ground truth to check those other
representations against:

- **`densityQuality(oh)`** numerically integrates the ASH density against
  `(x-mean)^p` for each requested power and compares the result to the
  moment accumulator's value for that same power. A large relative error
  means the ASH — the thing you'd actually integrate, sample from, or
  compare to a model — has drifted from the true shape of the data (too wide
  a smoothing bandwidth, or a range that's clipping the tails), even though
  nothing about the ASH's own bookkeeping looks obviously wrong.
- **`outofrange(oh)`** covers the complementary failure mode: points that
  fall outside the fixed histogram/ASH range entirely. These aren't dropped
  or clamped into the edge bins (clamping would distort the edge bins' shape
  and mask the exact thing you're trying to catch) — they're tallied
  separately as `(underflow=, overflow=)` counts, ROOT/HEP-style, so
  `sum(exactHistogram(oh).weights) + outofrange(oh).underflow +
  outofrange(oh).overflow == nobs(oh)` always holds exactly. A nonzero count
  here is a direct signal that the chosen (or learned) range missed part of
  the data. The moment accumulator, meanwhile, is never affected by this —
  it always reflects every point ever added, in range or not, since it never
  goes through the histogram/ASH's binning in the first place.

Together, `densityQuality` catches a range/bandwidth that's *technically*
covering the data but misrepresenting its shape, and `outofrange` catches a
range that's missing data outright — both checked against the same
independent, exact source of truth.

## Why both a histogram and an ASH?

They answer different questions. The traditional histogram gives you exact
bin counts — useful for reporting, and for sanity-checking the smoothed
estimate. The ASH gives you a continuous density estimate, which is what you
actually want to integrate, sample from, or compare against a model. Keeping
both, plus moments computed independently of either, lets `densityQuality`
tell you when the smoothed density has drifted from the true shape of the
data (e.g. because the smoothing bandwidth is too wide, or the histogram
range no longer covers the tails).

## Weights

`add!(oh, x, w)` and `add!(oh, xs, ws)` add points with weights (finite and
nonnegative; `add!(oh, x)` means weight 1). Weights never change the
unweighted results: each tally is kept twice, side by side.

| unweighted (every point counts 1) | weighted (every point counts its weight) |
|---|---|
| `nobs(oh)` | `totalweight(oh)` |
| `exactHistogram(oh)` (counts per bin) | `weightedHistogram(oh)` (sum of weights per bin) |
| `density(oh)`, `cdf(oh)`, `histogram(oh, bins)` | `weightedDensity(oh)`, `weightedCdf(oh)`, `weightedHistogram(oh, bins)` |
| `moment(oh, p)`, `mean`, `variance`, `std`, `skewness`, `kurtosis` | `weightedMoment(oh, p)`, `weightedMean`, `weightedVariance`, `weightedStd`, `weightedSkewness`, `weightedKurtosis` |
| `outofrange(oh)` | `outofrangeweight(oh)` |
| `densityQuality(oh)` | `weightedDensityQuality(oh)` |

Weighted moments are the population moments of the weighted data,
`Σw(x - mean)^p / Σw` with `mean = Σwx / Σw` -- the right reading for
importance weights, where the weighted points stand for the target
distribution. They use the same mergeable accumulator as the unweighted
moments, with counts replaced by sums of weights, which keeps the formulas
exact (Pébay et al. 2016).

**Weighted = count × mean weight.** The ASH grid keeps, for each grid cell,
both the number of points and their sum of weights. The weighted density is
the second smoothed with the same kernel, grid and width as the first. So it
equals the count density times the kernel-smoothed mean weight `E[w | x]`,
exactly, and equals the count density when all weights are equal. The same
holds bin by bin in the traditional histograms: `meanWeight(oh)` gives each
bin's mean weight, and `weightedHistogram(oh)`'s weights are the counts times
it. A flat `meanWeight` means the weights don't depend on the value; a steep
one means the weighting reshapes this statistic.

**How many points is a weighted sample worth?**
`effectivesamplesize(oh)` is Kish's `(Σw)² / Σw²`: `nobs(oh)` when all
weights are equal, and much less when a few points carry most of the
weight. A small value means the weighted statistics are noisy, however many
points went in.

A point of weight 0 still counts in `nobs`, the min/max and every unweighted
result; it just adds no weight.

## The learn phase

Both the histogram and the ASH need a fixed bin range up front — neither
can be cheaply re-ranged after points have already been binned. But you
often don't know the right range before you've seen the data. So by
default, `StreamHist` defers deciding the range: the first `learnLength`
points (default 10,000) are just buffered while the exact min/max and
moments are already being tracked. Once the buffer fills, the range is
picked from the buffer's `extrema`, padded by `paddingPct` on each side
(default 5%), and the histogram/ASH are initialized and fed the buffered
points. From then on every `add!` goes straight into the live histogram/ASH.

If you already know the range, skip the learn phase entirely by passing
`binRange` or `bins` — the range is fixed at construction and every point is
live from the start.

Points that arrive after the range is fixed and fall outside it are neither
dropped nor clamped into the outermost bin — clamping would distort the edge
bins' shape and mask exactly the "range turned out too narrow" signal this
is meant to help catch. Instead they're tallied separately in
`outofrange(oh)` (ROOT/HEP-style under/overflow counts), so
`sum(exactHistogram(oh).weights) + outofrange(oh).underflow +
outofrange(oh).overflow == nobs(oh)` always holds exactly. The exact min/max
and moments are always computed from the true values regardless of range, so
between that, `outofrange`, and `densityQuality`, a too-narrow range is easy
to spot even though the binned/smoothed views are lossy out there.

## Integer mode

When `integer=true`, the data is known to be integer-valued: bins are
centered exactly on the integers in the observed (or given) range, and there
is no ASH (`density`/`histogram`/`densityQuality` are unavailable —
there's no smoothing to do, and no bandwidth to get wrong, for exact integer
counts).

`bins` and a non-default `binNum` conflict with this (edges are always one
bin per integer) and raise `ArgumentError` rather than being silently
ignored. `closed` is always overridden to `:left`, since that's the only
convention under which the one-bin-per-integer construction is correct — it
doesn't error, since unlike `bins`/`binNum` there's no way to accidentally
lose information by passing it, just no visible effect. The moment
accumulator is unaffected by `integer` — it's already exact and O(1) per
point regardless of mode, so there's nothing to gain by treating it
differently here.

`integer=:auto` defers the decision to the learn phase: once the learn buffer
fills, `oh.integer` is set to `true` iff every buffered point `isinteger`,
otherwise `false`. Because that decision needs a learn-phase sample to
inspect, `:auto` requires `learn=true` and neither `bins` nor `binRange`
(both skip the learn phase and fix the range immediately, leaving nothing to
inspect) — passing either alongside `integer=:auto` raises `ArgumentError` at
construction. The `bins`/`binNum` conflict check above still applies; it's
just deferred until the buffer fills and `integer` is actually resolved, so
`StreamHist(integer=:auto, binNum=99)` doesn't error until the first `add!`
that completes the learn phase, and only if the buffer turns out to be
integer-valued.

## Construction options

```julia
StreamHist(;
    integer      = false,           # true/false/:auto; integer-valued data has no ASH density
    momentPowers = [1, 2, 4, 8],    # which moment orders to track/expose
    learn        = true,            # auto-pick the range from the first points
    learnLength  = 10_000,          # how many points to buffer before deciding
    paddingPct   = 0.05,            # range padding, as a fraction of observed span
    binRange     = nothing,         # (lo, hi); skips the learn phase if given
    binNum       = 50,              # number of traditional-histogram bins
    bins         = nothing,         # explicit edges; overrides binRange/binNum
    closed       = :left,           # StatsBase.Histogram bin closedness
    kernel       = StreamHistogram.biweight,   # default ASH kernel, any function of u
    m            = 5,               # default ASH smoothing width, in grid cells
    ashNGrid     = 500,             # the ASH grid has ashNGrid + 1 points over the range
)
```

`closed` is passed to `StatsBase.Histogram`; everything about *where* the
bins/range sit is owned by `integer`/`learn`/`binRange`/`binNum`/`bins`.

### How the ASH is computed

Adding a point only increments its nearest grid cell's count and weight sum
(O(1)); the density is smoothed from the grid when it is read
(`density`, `weightedDensity`, `densityGrid`), each cell `k`'s total spread
over its neighbours `i` by `kernel((i - k) / m)` and then normalized. That
is the computation of `AverageShiftedHistograms.jl` (the test suite checks
the two agree to 1e-12), but on our own grid, which holds weights as well as
counts and needs no batching of single-point `add!`s: reading is always
current.

`density(oh)` interpolates linearly between the grid points and is
normalized so that it integrates to exactly 1 over the range; `cdf(oh)` is
its exact integral (the interpolation is piecewise linear, so the integral is
piecewise quadratic), and `histogram(oh, bins)` takes differences of it.

Because the raw grid is kept, the smoothing can be changed when reading,
without rebuilding: the density readers (`densityGrid`, `density`,
`weightedDensity`, `cdf`, `weightedCdf`, `histogram(oh, bins)`,
`weightedHistogram(oh, bins)`) take `m=` and `kernel=`, defaulting to the
ones `oh` was built with:

```julia
density(oh; m=20)                          # smoother
cdf(oh; kernel=u -> exp(-u^2 / 2), m=3)    # a gaussian kernel
```

## Functions

- `add!(oh, x[, w])`, `add!(oh, xs[, ws])` — add a point or a vector of
  points, optionally weighted.
- `finalize!(oh)` — if still mid-learn, fixes the range now from whatever's
  in the buffer, so the histograms and densities can be read. A no-op once
  the range is fixed; points can still be added afterwards.
- `density(oh)` — a callable `x -> density(x)`, linearly interpolating the
  ASH; it integrates to 1. Errors in integer mode, and while the range is not
  yet fixed.
- `cdf(oh)` — a callable `x -> cdf(x)`, the exact integral of `density(oh)`
  up to `x`: 0 below the range, 1 from its top up.
- `densityGrid(oh; weighted=false)` — `(grid=, density=)`, the ASH at its grid
  points.
- The density readers take `m=` and `kernel=` to smooth with another width or
  kernel than the ones `oh` was built with; see
  [How the ASH is computed](#how-the-ash-is-computed).
- `densityQuality(oh)` — for each requested moment power (in the order of
  `oh.momentPowers`), numerically integrates the ASH density against
  `(x-mean)^p` and compares it to the true moment accumulator value; returns
  a `Vector{Float64}` of the relative error between the two estimates.
- `exactHistogram(oh)` — the underlying `StatsBase.Histogram` (edges + raw
  counts).
- `histogram(oh, bins)` — integrates `density(oh)` exactly between an
  arbitrary set of edges (scaled by the number of points in range) to produce
  expected counts over bins that needn't match the ones `oh` was built with.
- `moment(oh, p)` / `moments(oh)` — the order-`p` moment (mean for `p==1`,
  population central moment for `p>=2`), or all requested moments as a
  `Dict`.
- `mean(oh)`, `variance(oh)`, `std(oh)`, `skewness(oh)`, `kurtosis(oh)` —
  convenience wrappers over the moment accumulator.
- `nobs(oh)` — total number of points ever added.
- `datarange(oh)` — `(exactMin, exactMax)` over all points ever added (not
  to be confused with the fixed histogram/ASH bin range).
- `outofrange(oh)` — `(underflow=, overflow=)` counts of points that fell
  outside the fixed histogram range; see above.
- `isinitialized(oh)` — whether the range has been fixed yet (`false` while
  still learning).
- `lp_distance(a, b; p=2, weighted=false)`, `tv_distance(a, b; weighted=false)`
  — L^p and total-variation distances between two histograms' ASH densities
  (weighted densities if `weighted`).
- The weighted counterparts (`totalweight`, `weightedHistogram`,
  `weightedDensity`, `weightedCdf`, `weightedMoment`, `weightedMoments`, `weightedMean`, ...,
  `outofrangeweight`, `weightedDensityQuality`), `meanWeight(oh)` and
  `effectivesamplesize(oh)` — see [Weights](#weights).

Not yet: merging two independently-built `StreamHist`s. The moments, the
traditional histograms and the ASH grid are all mergeable (same range
required), but it isn't written.

## Example

```julia
using StreamHistogram

oh = StreamHist(learnLength=5_000, binNum=100)
for batch in datastream
    add!(oh, batch)
end
finalize!(oh)

mean(oh), variance(oh), skewness(oh)
exactHistogram(oh)
outofrange(oh)  # (underflow=, overflow=) -- check this if the learned range looks too narrow
d = density(oh)                 # d(x) at any x
F = cdf(oh)                     # F(b) - F(a): the estimated fraction of points in (a, b]
density(oh; m=20)               # the same data, smoothed more
densityQuality(oh)

# weighted points, e.g. importance weights
ow = StreamHist(binRange=(-5.0, 5.0))
add!(ow, xs, ws)
weightedMean(ow), effectivesamplesize(ow)
weightedDensity(ow), meanWeight(ow)
```
