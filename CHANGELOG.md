# Changelog

## 0.2.0

- **Weighted points.** `add!(oh, x, w)` and `add!(oh, xs, ws)` add points with finite, nonnegative weights;
  `add!(oh, x)` means weight 1. The unweighted results are unchanged by weights, and each has a weighted
  counterpart: `totalweight`, `weightedHistogram(oh)` (sum of weights per bin), `weightedDensity`,
  `weightedHistogram(oh, bins)`, `weightedMoment`/`weightedMoments`/`weightedMean`/`weightedVariance`/
  `weightedStd`/`weightedSkewness`/`weightedKurtosis`, `outofrangeweight`, `weightedDensityQuality`. Also
  `meanWeight(oh)` (each bin's mean weight) and `effectivesamplesize(oh)` (Kish's `(Σw)²/Σw²`). The weighted
  density equals the count density times the kernel-smoothed mean weight, exactly. `lp_distance` and
  `tv_distance` take `weighted=true`.
- `MomentAccumulator{T}`: `T = Int` counts points, `T = Float64` sums weights (the same exact merge formulas).
- **Own ASH.** The density is smoothed from StreamHist's own grid of counts and weight sums when it is read,
  reproducing AverageShiftedHistograms.jl's density (now only a test dependency) to 1e-12. Breaking:
  - the `ashBatchSize` option, the single-point batching, and `oh.ash` are gone; `densityGrid(oh)` gives the
    density at the grid points;
  - `lp_distance(::Ash, ::Ash)`/`tv_distance(::Ash, ::Ash)` are gone (use the `StreamHist` methods);
  - the `finalized` state is gone: the densities are always current, readable as soon as the range is fixed.
    `finalize!` still ends the learn phase early, and is a no-op afterwards.
  - the default kernel is `StreamHistogram.biweight` (the same function).
- Fix: `histogram(oh, bins)` scales the density by the number of points inside the range (what the density is
  normalized over), not by `nobs(oh)`, which also counts points outside it.

## 0.1.2

- `densityQuality` and `histogram(oh, bins)` no longer error when no point is inside the range.
- Julia 1.10 or later; `[compat]` bounds for every dependency.
