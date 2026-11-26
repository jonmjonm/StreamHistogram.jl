# Changelog

## 0.3.0

- **The type `StreamHist` is renamed `StreamingHistogram`.** There is no alias for the old name. Breaking. The
  package and module are still `StreamHistogram`.

## 0.2.1

- **`kurtosis` and `weightedKurtosis` are the excess kurtosis** (the fourth central moment over the variance
  squared, minus 3; 0 for a normal), as `StatsBase.kurtosis`. Before, they did not subtract 3. Breaking.
- **`cdf` and `weightedCdf` are renamed `densityCdf` and `weightedDensityCdf`**: they are the CDFs of `density`
  and `weightedDensity`, and the old name clashed with `Distributions.cdf`, which evaluates a distribution's CDF at
  a point. Breaking.
- **`moment`, `skewness` and `kurtosis` are methods of the StatsBase functions** of those names, instead of
  functions of their own, so `using StatsBase, StreamHistogram` no longer leaves those three names undefined.
  They agree with StatsBase on the same data, except that `moment(oh, 1)` is the mean (StatsBase's first central
  moment is always 0).
- **`quantile(oh, p)` and `weightedQuantile(oh, p)`**: estimated quantiles (`p` a number or a vector), from the
  traditional histogram's bins, interpolated linearly within a bin (the bin's integer in integer mode), with the
  exact minimum and maximum at `p = 0` and `1`. A quantile among the out-of-range points is interpolated across
  all of them, a coarser estimate. Moved here from AtlasUtilities (`histQuantile`).

## 0.2.0

- **Weighted points.** `add!(oh, x, w)` and `add!(oh, xs, ws)` add points with finite, nonnegative weights;
  `add!(oh, x)` means weight 1. The unweighted results are unchanged by weights, and each has a weighted
  counterpart: `totalweight`, `weightedHistogram(oh)` (sum of weights per bin), `weightedDensity`,
  `weightedHistogram(oh, bins)`, `weightedMoment`/`weightedMoments`/`weightedMean`/`weightedVariance`/
  `weightedStd`/`weightedSkewness`/`weightedKurtosis`, `outofrangeweight`, `weightedDensityQuality`. Also
  `meanWeight(oh)` (each bin's mean weight) and `effectivesamplesize(oh)` (Kish's `(Σw)²/Σw²`). The weighted
  density equals the count density times the kernel-smoothed mean weight, exactly. `lp_distance` and
  `tv_distance` take `weighted=true`.
- **`cdf(oh)` and `weightedCdf(oh)`**: the exact integrals of `density(oh)` and `weightedDensity(oh)`.
  `histogram(oh, bins)` and `weightedHistogram(oh, bins)` now take differences of them instead of integrating
  numerically. The densities are normalized so their linear interpolation integrates to exactly 1 (before, the
  grid values summed to 1, which differs slightly when the density is not 0 at the ends of the range).
- **`m=` and `kernel=` when reading**: `densityGrid`, `density`, `weightedDensity`, `cdf`, `weightedCdf`,
  `histogram(oh, bins)` and `weightedHistogram(oh, bins)` smooth the same grid with another width or kernel;
  the constructor's `m` and `kernel` are the defaults.
- `ashNGrid` must be positive.
- `MomentAccumulator{T}`: `T = Int` counts points, `T = Float64` sums weights (the same exact merge formulas).
- **Own ASH.** The density is smoothed from StreamHist's own grid of counts and weight sums when it is read,
  reproducing AverageShiftedHistograms.jl's density (now only a test dependency) to 1e-12. Breaking:
  - the `ashBatchSize` option, the single-point batching, and `oh.ash` are gone; `densityGrid(oh)` gives the
    density at the grid points;
  - `lp_distance(::Ash, ::Ash)`/`tv_distance(::Ash, ::Ash)` are gone (use the `StreamHist` methods);
  - the `finalized` state is gone: the densities are always current, readable as soon as the range is fixed.
    `finalize!` still ends the learn phase early, and is a no-op afterwards.
  - the default kernel is `StreamHistogram.biweight` (the same function).
- Fix: the ASH grid holds only the points inside the histogram's range. Before, points up to half a grid step
  outside it, and a point exactly at the top with `closed=:left`, were on the grid but counted as out of range by
  the histogram, so the density described slightly different points than `histogram(oh, bins)` scaled it by.
- Every moment of an empty `StreamHist` or `MomentAccumulator` is `NaN`; `mean` was 0.0.
- Fix: `histogram(oh, bins)` scales the density by the number of points inside the range (what the density is
  normalized over), not by `nobs(oh)`, which also counts points outside it.

## 0.1.2

- `densityQuality` and `histogram(oh, bins)` no longer error when no point is inside the range.
- Julia 1.10 or later; `[compat]` bounds for every dependency.
