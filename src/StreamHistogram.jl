module StreamHistogram

using Statistics
using StatsBase
using StatsBase: Histogram
using AverageShiftedHistograms
using AverageShiftedHistograms: Ash, ash, ash!
using QuadGK

import Statistics: mean, std
import StatsBase: nobs

export StreamHist, add!, finalize!, density, densityQuality, exactHistogram, histogram,
    moment, moments, nobs, datarange, mean, variance, std, skewness, kurtosis,
    isinitialized, MomentAccumulator

include("moments.jl")
include("onlinehistogram.jl")

end # module StreamHistogram
