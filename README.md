# StreamHistogram.jl

[![CI](https://github.com/jonmjonm/StreamHistogram.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jonmjonm/StreamHistogram.jl/actions/workflows/CI.yml)

`StreamHistogram` builds a histogram, a smooth density and exact moments from
data that arrives one point or one batch at a time: a stream, a simulation, a
file too large to hold in memory. Points may carry weights, such as importance
weights, and every statistic is then available both unweighted and weighted.
The moments are exact, so they can be used to check how well the binned and
smoothed views represent the data.

A `StreamHist` keeps, updated as each point arrives:

1. a traditional histogram (a `StatsBase.Histogram`);
2. an average shifted histogram (ASH) grid, from which a continuous density,
   its cumulative distribution and histograms over any bins are computed;
3. the exact minimum and maximum;
4. exact central moments up to a chosen order, computed with a numerically
   stable, mergeable algorithm (see [momentAccumulator.md](momentAccumulator.md));
5. counts of the points that fell outside the histogram's range.

Each of 1, 2, 4 and 5 is kept twice: once counting every point once, and once
counting every point by its weight.

## Installation

```julia
import Pkg
Pkg.add("StreamHistogram")       # once registered
Pkg.add(url="https://github.com/jonmjonm/StreamHistogram.jl")   # until then
```

Julia 1.10 or later.

## Example

```julia
using StreamHistogram

oh = StreamHist()                     # the range is learned from the first 10,000 points
for _ in 1:100
    add!(oh, randn(1_000))            # a batch; add!(oh, x) adds one point
end

nobs(oh), mean(oh), std(oh), kurtosis(oh)      # exact, from every point
outofrange(oh)                        # (underflow=, overflow=): points outside the learned range
exactHistogram(oh)                    # bin counts

f = density(oh)                       # a function: f(x) at any x
F = densityCdf(oh)                    # F(b) - F(a) estimates the fraction of points in (a, b]
histogram(oh, -3:0.5:3)               # expected counts in any bins
density(oh; m=20)                     # the same data, smoothed more
densityQuality(oh)                    # how far the density's moments are from the exact ones

# weighted points
ow = StreamHist(binRange=(-5.0, 5.0))  # a fixed range: no learn phase
xs = randn(10_000)
add!(ow, xs, exp.(xs))                 # weights exp(x): the weighted data is N(1, 1)
weightedMean(ow), effectivesamplesize(ow)
weightedDensity(ow), weightedDensityCdf(ow), meanWeight(ow)
```

## The range: learned or given

The histogram and the ASH grid need a fixed range. You can give one with
`binRange = (lo, hi)` or with explicit edges `bins`, and every point is then
counted from the start. Otherwise `StreamHist` learns it. It buffers the first
`learnLength` points (default 10,000), then sets the range to their minimum
and maximum, widened by `paddingPct` (default 5%) of the span on each side, and
counts the buffered points. Exact moments, the minimum and maximum are tracked
during the learn phase too. Call `finalize!(oh)` to end the learn phase early;
the histograms and densities can't be read before the range is set.

A point outside the range is not dropped or put into the edge bin, which
would distort the edge bins and hide the problem. It is counted separately:
`outofrange(oh)` gives `(underflow=, overflow=)`, and

```julia
sum(exactHistogram(oh).weights) + outofrange(oh).underflow + outofrange(oh).overflow == nobs(oh)
```

always holds. A nonzero count means the range misses part of the data. The
moments, minimum and maximum always include every point.

## The density

Adding a point inside the range increments its nearest ASH grid point's count
and weight sum; points outside the range are only in `outofrange`, so the
grid holds exactly the points the histogram does.
The grid has `ashNGrid + 1` points (default 501) across the range. The density
is computed from the grid when it is read: each grid point's count is spread
over its neighbours `i` by `kernel((i - k) / m)`, a kernel `kernel` (default
`StreamHistogram.biweight`, `(1 - u²)²`) reaching `m` grid points (default 5)
each way. This is the computation of AverageShiftedHistograms.jl, done on a
grid that also holds weights. The test suite checks the two agree to 1e-12.

- `density(oh)` returns a function that interpolates linearly between the grid
  points. It is 0 outside the range and integrates to exactly 1 over it.
- `densityCdf(oh)` returns its exact integral up to `x`: 0 below the range, 1 from its
  top up.
- `histogram(oh, bins)` gives expected counts in any bins: the number of
  points in range times the differences of the CDF at the edges.
- `densityGrid(oh)` gives the grid and the density's values on it,
  `(grid=, density=)`.

Each returns a snapshot: points added later are included in the next call.

Because the grid itself is kept, the smoothing can be changed when reading,
without rebuilding. Each of these readers, and their weighted versions, takes
`m=` and `kernel=`, defaulting to the ones the `StreamHist` was built with:

```julia
density(oh; m=20)                          # smoother
densityCdf(oh; kernel=u -> exp(-u^2 / 2), m=3)    # a gaussian kernel
```

## Checking the results

The moments are computed from the raw values, never from the bins or the
grid, with Pébay's mergeable generalization of Welford's algorithm. They are
exact for practical purposes, so they serve as the reference for the two views
that do depend on a bin width, a smoothing width and a range:

- `densityQuality(oh)` integrates `(x - mean)^p` against the density for each
  tracked order `p` and returns the relative error from the exact moment. A
  large error means the density misrepresents the data, for example because
  the smoothing is too wide or the range cuts off a tail.
- `outofrange(oh)` catches a range that misses data outright.

`moment(oh, p)` is the mean for `p == 1` and the population central moment
`Σ(x - mean)^p / n` for `p >= 2`; `p` must be in `momentPowers` (default
`[1, 2, 4, 8]`). `variance`, `std`, `skewness` and `kurtosis` are population
values (no `n - 1` correction), and `kurtosis` is the excess kurtosis: 0 for a
normal distribution. They need the highest tracked order to be at least 2, 2,
3 and 4 respectively. Every moment, the mean included, is `NaN` before any
point is added.

`moment`, `skewness` and `kurtosis` are methods of the StatsBase functions of
those names (as `mean`, `std` and `quantile` are of the Statistics ones), so
`using StatsBase, StreamHistogram` works without clashes, and they agree with
StatsBase on the same data. The one difference: StatsBase's `moment(x, 1)` is
the first central moment, always 0, where `moment(oh, 1)` is the mean.

## Quantiles

`quantile(oh, p)` estimates the value below which a fraction `p` of the points
lie (`p` a number or a vector); `weightedQuantile(oh, p)` does the same for a
fraction of the total weight. Both come from the traditional histogram, not the
density, so they work in integer mode too:

- within a bin the points are taken to be spread evenly, so the estimate is
  interpolated between the bin's edges; in integer mode it is the bin's integer
  (the inverse of the empirical CDF);
- `p = 0` and `p = 1` give the exact minimum and maximum;
- a quantile that falls among the points outside the range is interpolated
  across all of them, from the exact minimum to the bottom edge (or the top edge
  to the exact maximum): a much coarser estimate, so one outside the histogram's
  edges deserves less trust.

## Weights

`add!(oh, x, w)` and `add!(oh, xs, ws)` add points with weights, which must be
finite and nonnegative; `add!(oh, x)` means weight 1. Weights never change the
unweighted results, because each tally is kept twice:

| unweighted (each point counts 1) | weighted (each point counts its weight) |
|---|---|
| `nobs(oh)` | `totalweight(oh)` |
| `exactHistogram(oh)` (count per bin) | `weightedHistogram(oh)` (sum of weights per bin) |
| `density(oh)`, `densityCdf(oh)`, `histogram(oh, bins)` | `weightedDensity(oh)`, `weightedDensityCdf(oh)`, `weightedHistogram(oh, bins)` |
| `quantile(oh, p)` | `weightedQuantile(oh, p)` |
| `moment(oh, p)`, `moments`, `mean`, `variance`, `std`, `skewness`, `kurtosis` | `weightedMoment(oh, p)`, `weightedMoments`, `weightedMean`, `weightedVariance`, `weightedStd`, `weightedSkewness`, `weightedKurtosis` |
| `outofrange(oh)` | `outofrangeweight(oh)` |
| `densityQuality(oh)` | `weightedDensityQuality(oh)` |
| `lp_distance(a, b)`, `tv_distance(a, b)` | the same with `weighted=true` |

Weighted moments are the population moments of the weighted data,
`Σw(x - mean)^p / Σw` with `mean = Σwx / Σw`. This is the right reading for
importance weights, where the weighted points stand for the target
distribution. They use the same exact merge formulas as the unweighted
moments, with counts replaced by sums of weights (Pébay et al. 2016).

**Weighted = count × mean weight.** The weighted density is the grid's weight
sums smoothed with the same kernel, grid and width as the counts. So it equals
the count density times the kernel-smoothed mean weight `E[w | x]`, exactly,
and equals the count density when all weights are equal. The same holds bin by
bin: `meanWeight(oh)` gives each bin's mean weight, and
`weightedHistogram(oh)`'s weights are the counts times it. A flat
`meanWeight` means the weights don't depend on the value.

**How many points is a weighted sample worth?** `effectivesamplesize(oh)` is
Kish's `(Σw)² / Σw²`. It equals `nobs(oh)` when all weights are equal and is
much smaller when a few points carry most of the weight. A small value means
the weighted statistics are noisy, however many points went in.

A point of weight 0 still counts in `nobs`, the minimum and maximum and every
unweighted result; it just adds no weight. Weighted results are `NaN` while
the total weight is 0.

## Integer mode

With `integer=true` the data is taken to be integer-valued: there is one bin
per integer in the range, and no ASH grid. `density`, `densityCdf`, `densityGrid`,
`histogram(oh, bins)`, `densityQuality`, the distances and their weighted
versions raise `ArgumentError`; use `exactHistogram` and `weightedHistogram`.

`bins` and a non-default `binNum` conflict with one bin per integer and raise
`ArgumentError`. `closed` is always `:left`, since that is the only
convention under which each bin `[k, k+1)` holds exactly the integer `k`.

`integer=:auto` decides during the learn phase: when the buffer fills, the
data is integer-valued if every buffered point `isinteger`. It needs the learn
phase, so it can't be combined with `bins`, `binRange` or `learn=false`. The
`binNum` check then happens when the decision is made.

## Construction options

```julia
StreamHist(;
    integer      = false,           # true, false or :auto; integer-valued data has no density
    momentPowers = [1, 2, 4, 8],    # the moment orders moment(oh, p) reports
    learn        = true,            # learn the range from the first points
    learnLength  = 10_000,          # how many points to buffer before setting the range
    paddingPct   = 0.05,            # widen the learned range by this fraction of its span, each side
    binRange     = nothing,         # (lo, hi): a fixed range, skipping the learn phase
    binNum       = 50,              # the number of traditional-histogram bins
    bins         = nothing,         # explicit edges; overrides binRange and binNum
    closed       = :left,           # StatsBase.Histogram's bin closedness
    kernel       = StreamHistogram.biweight,   # the default smoothing kernel, any function of u
    m            = 5,               # the default smoothing width, in grid points
    ashNGrid     = 500,             # the ASH grid has ashNGrid + 1 points across the range
)
```

`learn=false` needs `bins` or `binRange`. With `bins`, the ASH grid spans
`first(bins)` to `last(bins)`.

## Functions

| | |
|---|---|
| **Building** | |
| `StreamHist(; ...)` | a new, empty histogram; see [Construction options](#construction-options) |
| `add!(oh, x[, w])`, `add!(oh, xs[, ws])` | add a point or a vector of points, optionally weighted |
| `finalize!(oh)` | end the learn phase now; a no-op once the range is set |
| `isinitialized(oh)` | whether the range is set |
| **Counts and range** | |
| `nobs(oh)`, `totalweight(oh)` | the number of points, and their total weight |
| `effectivesamplesize(oh)` | Kish's effective sample size of the weights |
| `datarange(oh)` | the exact `(min, max)` of every point added |
| `outofrange(oh)`, `outofrangeweight(oh)` | `(underflow=, overflow=)` counts and weights outside the range |
| **Histograms** | |
| `exactHistogram(oh)`, `weightedHistogram(oh)` | the bin counts and the bin weight sums, as `StatsBase.Histogram`s |
| `meanWeight(oh)` | `(edges=, meanweight=)`: each bin's mean weight |
| `quantile(oh, p)`, `weightedQuantile(oh, p)` | estimated quantiles, from the bins (see [Quantiles](#quantiles)) |
| `histogram(oh, bins)`, `weightedHistogram(oh, bins)` | `(edges=, weights=)`: expected counts or weights in any bins, from the density |
| **Density** (not in integer mode; take `m=` and `kernel=`) | |
| `density(oh)`, `weightedDensity(oh)` | the density, as a function of `x` |
| `densityCdf(oh)`, `weightedDensityCdf(oh)` | its exact integral up to `x`, as a function |
| `densityGrid(oh; weighted=false)` | `(grid=, density=)`: the density at the grid points |
| `densityQuality(oh)`, `weightedDensityQuality(oh)` | relative error of the density's moments, one per tracked order |
| `lp_distance(a, b; p=2, weighted=false)` | the L^p distance between two densities |
| `tv_distance(a, b; weighted=false)` | the total-variation distance, half the L^1 distance |
| **Moments** | |
| `moment(oh, p)`, `moments(oh)` | the order-`p` moment, or a `Dict` of every tracked order |
| `mean`, `variance`, `std`, `skewness`, `kurtosis` | the usual summaries, from the exact moments |
| `weightedMoment`, `weightedMoments`, `weightedMean`, ... | the same for the weighted data |
| `MomentAccumulator` | the mergeable moment accumulator itself; see [momentAccumulator.md](momentAccumulator.md) |

Not yet written: merging two independently built `StreamHist`s. The moments,
the histograms and the grid are all mergeable when the ranges match.

## Related packages

[OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) covers much of the same ground for unweighted
streams, among many other online statistics: `Hist`, `KHist` and `ExpandingHist` histograms, `Ash`, `Moments`,
`Extrema` and `Quantile`. Choose StreamHistogram when the points carry real-valued weights, such as importance
weights or the weights of an MCMC sample. OnlineStats' `fit!` takes only a whole-number repeat count per
observation, and its weight objects (`EqualWeight`, `ExponentialWeight`) set how fast older observations are
forgotten, not how much each one counts. StreamHistogram also offers:

- every statistic both unweighted and weighted from the same pass, with the total weight and the effective
  sample size;
- exact central moments of any order (OnlineStats' `Moments` tracks the first four);
- the moments as a check on the histogram and density, with `densityQuality`, and the count and weight outside
  the range;
- a range learned from the first points, then a fixed grid.

Choose OnlineStats for its wider set of statistics, for exponential weighting of a stream that changes over
time, or to combine several statistics with `Series`.

## Releasing

1. Bump `version` in `Project.toml` and add a CHANGELOG entry; merge to `main`.
2. Comment `@JuliaRegistrator register` on the merge commit (release notes may follow on a `Release notes:` line).
3. When General merges the registration, TagBot (`.github/workflows/TagBot.yml`) tags the commit `vX.Y.Z` and
   creates the GitHub release. It can't do either for a commit that changes `.github/workflows/`, so keep workflow
   changes out of release commits.
