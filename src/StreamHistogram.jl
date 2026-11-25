"""
    StreamHistogram

Histograms, densities and exact moments of data arriving one point or one
batch at a time, optionally weighted. Start with `StreamHist`, `add!` and the
README.
"""
module StreamHistogram

using Statistics
using StatsBase
using StatsBase: Histogram
using QuadGK

import Statistics: mean, std, quantile
# Methods of the StatsBase functions of the same names, so that `using StatsBase, StreamHistogram` leaves no
# name ambiguous. They mean what StatsBase's do: population values, and `kurtosis` is the excess kurtosis.
import StatsBase: nobs, moment, skewness, kurtosis

export StreamHist, add!, finalize!, isinitialized, MomentAccumulator,
    nobs, datarange, outofrange, exactHistogram, density, densityGrid, histogram, densityQuality,
    moment, moments, mean, variance, std, skewness, kurtosis, lp_distance, tv_distance,
    totalweight, effectivesamplesize, outofrangeweight, weightedHistogram, weightedDensity, meanWeight,
    densityCdf, weightedDensityCdf, quantile, weightedQuantile,
    weightedDensityQuality, weightedMoment, weightedMoments, weightedMean, weightedVariance, weightedStd,
    weightedSkewness, weightedKurtosis

include("moments.jl")
include("onlinehistogram.jl")

end # module StreamHistogram
