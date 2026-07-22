# StreamHistogram.jl

`StreamHistogram` collects a histogram — and the statistics needed to judge
how much to trust it — from data arriving online, one point or one batch at
a time. It's meant for the case where you don't have all the data up front
(a stream, a simulation, a large file you don't want to hold in memory) but
still want a proper histogram, a smoothed density estimate, and moments
precise enough to check the two against each other.

A `StreamHist` maintains, incrementally and simultaneously:

1. a traditional histogram (`StatsBase.Histogram`)
2. an Average Shifted Histogram (ASH) density estimate
   (`AverageShiftedHistograms.jl`)
3. the exact min and max of every point seen
4. online central moments at a configurable set of orders, computed with a
   numerically stable, mergeable algorithm (see `momentAccumulator.md`)

## Why both a histogram and an ASH?

They answer different questions. The traditional histogram gives you exact
bin counts — useful for reporting, and for sanity-checking the smoothed
estimate. The ASH gives you a continuous density estimate, which is what you
actually want to integrate, sample from, or compare against a model. Keeping
both, plus moments computed independently of either, lets `densityQuality`
tell you when the smoothed density has drifted from the true shape of the
data (e.g. because the smoothing bandwidth is too wide, or the histogram
range no longer covers the tails).

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
centered exactly on the integers in the observed (or given) range, and the
ASH is disabled (`density`/`histogram`/`densityQuality` are unavailable —
there's no smoothing to do, and no bandwidth to get wrong, for exact integer
counts).

## Construction options

```julia
StreamHist(;
    integer      = false,           # integer-valued data; disables the ASH
    momentPowers = [1, 2, 4, 8],    # which moment orders to track/expose
    learn        = true,            # auto-pick the range from the first points
    learnLength  = 10_000,          # how many points to buffer before deciding
    paddingPct   = 0.05,            # range padding, as a fraction of observed span
    binRange     = nothing,         # (lo, hi); skips the learn phase if given
    binNum       = 50,              # number of traditional-histogram bins
    bins         = nothing,         # explicit edges; overrides binRange/binNum
    closed       = :left,           # StatsBase.Histogram bin closedness
    kernel       = AverageShiftedHistograms.Kernels.biweight,  # ASH kernel
    m            = 5,               # ASH smoothing width
    ashNGrid     = 500,             # resolution of the ASH's internal grid
    ashBatchSize = 256,             # scalar add!s flush to the ASH in batches this big
)
```

`closed`, `kernel`, and `m` are passed straight through to the underlying
packages; everything about *where* the bins/range sit is owned by
`integer`/`learn`/`binRange`/`binNum`/`bins`.

### Why `ashBatchSize`

`AverageShiftedHistograms.ash!` is dramatically cheaper per point when called
once on a batch than called once per point (its cost is dominated by
smoothing over the whole grid, not by how many new points it's given) — in a
one-point-at-a-time streaming benchmark, calling it on a 100k-point batch
took ~0.0001s total, versus ~0.37s calling it 100k times on single points.
So a scalar `add!` doesn't call `ash!` immediately: it holds the point in an
internal buffer and flushes to the ASH once `ashBatchSize` points have
accumulated. Batch `add!` calls bypass this entirely and go straight to the
ASH, since they're already batched. `finalize!` always flushes whatever's
still pending, so nothing is silently missing from the final density — which
is exactly why `density`/`histogram`/`densityQuality` require `oh` to be
finalized (see below): between flushes, the ASH can lag behind the
histogram/moments by up to `ashBatchSize` points.

## Functions

- `add!(oh, data)` — add a single point or a vector of points.
- `finalize!(oh)` — marks `oh` ready to read; if still mid-learn, forces the
  range to be fixed now from whatever's in the buffer. Any later `add!`
  un-finalizes `oh` again, requiring another `finalize!` before reading.
- `density(oh)` — a callable `x -> density(x)`, linearly interpolating the
  ASH. Errors in integer mode, and requires `oh` to be finalized (raises
  `ArgumentError` otherwise) since the ASH can lag behind by up to
  `ashBatchSize` points until a `finalize!`/flush.
- `densityQuality(oh)` — for each requested moment power (in the order of
  `oh.momentPowers`), numerically integrates the ASH density against
  `(x-mean)^p` and compares it to the true moment accumulator value; returns
  a `Vector{Float64}` of the relative error between the two estimates.
- `exactHistogram(oh)` — the underlying `StatsBase.Histogram` (edges + raw
  counts).
- `histogram(oh, bins)` — integrates `density(oh)` between an arbitrary set
  of edges (scaled by `nobs(oh)`) to produce expected counts over bins that
  needn't match the ones `oh` was built with.
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

Not in v1: merging two independently-built `StreamHist`s. The moments and
the traditional histogram are both natively mergeable, but
`AverageShiftedHistograms.jl` doesn't expose a public merge for two `Ash`
objects, so a faithful `merge!` isn't possible without reaching into its
internals — left out for now rather than done halfway.

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
d = density(oh)
densityQuality(oh)
```
