const DEFAULT_MOMENT_POWERS = [1, 2, 4, 8]
const DEFAULT_BIN_NUM = 50

"The biweight kernel `(1 - u²)²` on `[-1, 1]`, the default ASH kernel."
biweight(u) = abs(u) <= 1 ? (1.0 - u^2)^2 : 0.0

# Every point carries a weight (1 unless `add!` is given one). Each tally comes in two forms side by side: the
# unweighted one (counts) and the weighted one (sums of weights). Weights never change the unweighted tallies.
mutable struct StreamHist
    # config, fixed at construction
    integer::Bool
    # true iff `integer=:auto` was requested and not yet resolved from the learn buffer; `integer` itself is a
    # placeholder (`false`) until then
    integerAuto::Bool
    momentPowers::Vector{Int}
    learn::Bool
    learnLength::Int
    paddingPct::Float64
    binNum::Int
    closed::Symbol
    kernel::Any
    m::Int
    ashNGrid::Int

    # learn-phase buffers of points and their weights; nothing once the range has been fixed
    learnBuffer::Union{Vector{Float64},Nothing}
    learnWeights::Union{Vector{Float64},Nothing}

    # The traditional histogram (counts) and one with the same edges holding each bin's sum of weights; nothing
    # until the range is known.
    hist::Union{Histogram,Nothing}
    whist::Union{Histogram,Nothing}

    # The ASH grid (nothing in integer mode or until the range is known) and each grid cell's count and sum of
    # weights. The densities are these, smoothed by `kernel` when read (see `smoothgrid`).
    grid::Union{AbstractRange{Float64},Nothing}
    gridCounts::Vector{Int}
    gridWeights::Vector{Float64}

    # Online moments of the points (`moments.n` = the number of points) and of the weighted points
    # (`wmoments.n` = the total weight), and the sum of squared weights (for `effectivesamplesize`).
    moments::MomentAccumulator{Int}
    wmoments::MomentAccumulator{Float64}
    sumW2::Float64

    exactMin::Float64
    exactMax::Float64

    # Points outside the histogram's range are not clamped into the edge bins (that would distort the edge bins
    # and hide a range that turned out too narrow). They are tallied here instead, ROOT/HEP-style, so
    # sum(exactHistogram(oh).weights) + under + over == nobs(oh) always holds exactly (and likewise for weights).
    underflowCount::Int
    overflowCount::Int
    underflowWeight::Float64
    overflowWeight::Float64
end

"""
    StreamHist(; integer=false, momentPowers=[1,2,4,8], learn=true,
               learnLength=10_000, paddingPct=0.05, binRange=nothing,
               binNum=50, bins=nothing, closed=:left,
               kernel=StreamHistogram.biweight, m=5, ashNGrid=500)

An empty online histogram. Points, optionally weighted, are added with `add!`,
one at a time or in batches. It keeps a traditional histogram
(`StatsBase.Histogram`), an average shifted histogram (ASH) grid from which
the density is computed, the exact minimum and maximum, exact central
moments, and counts of the points outside the range: each both unweighted
(every point counts once) and weighted (every point counts its weight).

The range:
- `binRange = (lo, hi)`: a fixed range for the histogram and the ASH grid.
- `bins`: explicit histogram edges, overriding `binRange` and `binNum`; the
  ASH grid spans `first(bins)` to `last(bins)`.
- Otherwise, with `learn=true`, the first `learnLength` points are buffered
  and the range is set from their minimum and maximum, widened by
  `paddingPct` of the span on each side (`finalize!` sets it early).
  `learn=false` requires `bins` or `binRange`.

The histogram:
- `binNum`: the number of bins (when `bins` isn't given).
- `closed`: passed to `StatsBase.Histogram`.

The density:
- `ashNGrid`: the ASH grid has `ashNGrid + 1` points across the range.
- `kernel`, `m`: the default smoothing: each grid point's count is spread over
  its neighbours `i` by `kernel((i - k) / m)`. The density readers take
  `kernel=` and `m=` to smooth the same grid differently.

The moments:
- `momentPowers`: the orders `moment(oh, p)` reports. The accumulator tracks
  every order up to the highest, so `skewness` needs it to be at least 3 and
  `kurtosis` at least 4.

Integer-valued data:
- `integer=true`: one bin per integer in the range, and no ASH grid or density.
  `bins` and a non-default `binNum` raise `ArgumentError`; `closed` is always
  `:left`, so that bin `[k, k+1)` holds exactly the integer `k`.
- `integer=:auto`: decided when the learn phase ends, `true` if every buffered
  point `isinteger`. Needs the learn phase, so not with `bins`, `binRange` or
  `learn=false`.
"""
function StreamHist(;
    integer::Union{Bool,Symbol}=false,
    momentPowers::AbstractVector{<:Integer}=DEFAULT_MOMENT_POWERS,
    learn::Bool=true,
    learnLength::Integer=10_000,
    paddingPct::Real=0.05,
    binRange::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    binNum::Integer=DEFAULT_BIN_NUM,
    bins::Union{Nothing,AbstractVector{<:Real}}=nothing,
    closed::Symbol=:left,
    kernel=biweight,
    m::Integer=5,
    ashNGrid::Integer=500,
)
    m > 0 || throw(ArgumentError("the smoothing parameter `m` must be positive"))
    ashNGrid > 0 || throw(ArgumentError("`ashNGrid` must be positive"))
    integerAuto = integer === :auto
    if integerAuto
        bins === nothing && binRange === nothing || throw(ArgumentError(
            "`integer=:auto` requires the learn phase to pick the range; " *
            "don't pass `bins` or `binRange`, which skip it"))
        learn || throw(ArgumentError("`integer=:auto` requires `learn=true`"))
        integer = false # placeholder; resolved from the learn buffer once it fills
    elseif integer isa Symbol
        throw(ArgumentError("`integer` must be `true`, `false`, or `:auto`, got `:$integer`"))
    elseif integer
        bins === nothing || throw(ArgumentError(
            "`bins` has no effect when `integer=true` (edges are always one bin per integer); " *
            "pass `binRange` instead, or drop `integer=true`"))
        binNum == DEFAULT_BIN_NUM || throw(ArgumentError(
            "`binNum` has no effect when `integer=true` (edges are always one bin per integer)"))
    end

    momentPowers = sort(unique(Int.(momentPowers)))
    maxp = max(1, maximum(momentPowers))

    oh = StreamHist(
        integer, integerAuto, momentPowers, learn, Int(learnLength), Float64(paddingPct),
        Int(binNum), closed, kernel, Int(m), Int(ashNGrid),
        Float64[], Float64[],
        nothing, nothing,
        nothing, Int[], Float64[],
        MomentAccumulator(maxp), MomentAccumulator(Float64, maxp), 0.0,
        Inf, -Inf,
        0, 0, 0.0, 0.0,
    )

    if bins !== nothing
        initializerange!(oh, (Float64(first(bins)), Float64(last(bins))); bins=collect(Float64.(bins)))
    elseif binRange !== nothing
        initializerange!(oh, (Float64(binRange[1]), Float64(binRange[2])))
    elseif !learn
        throw(ArgumentError("must provide `bins` or `binRange` when `learn=false`"))
    end

    return oh
end

# ---------------------------------------------------------------------------------------------------------------
# Counts, weights and moments

"""
    nobs(oh)

The number of points added, in the range or not, whatever their weights.
"""
StatsBase.nobs(oh::StreamHist) = oh.moments.n

"""
    datarange(oh)

`(min, max)` of every point added, in the range or not (`(Inf, -Inf)` before
any point). Not the histogram's range, which is `extrema` of
`exactHistogram(oh)`'s edges.
"""
datarange(oh::StreamHist) = (oh.exactMin, oh.exactMax)

"""
    isinitialized(oh)

Whether the range is set: `false` during the learn phase, when the histograms
and densities can't be read yet.
"""
isinitialized(oh::StreamHist) = oh.hist !== nothing

"""
    totalweight(oh)

The sum of the weights of every point added (`nobs(oh)` when no weights were
given, since each point then has weight 1).
"""
totalweight(oh::StreamHist) = oh.wmoments.n

"""
    effectivesamplesize(oh)

Kish's effective sample size `(Σw)² / Σw²`: the number of equally weighted
points that would give the weighted statistics the same precision. Equals
`nobs(oh)` when every weight is the same; much smaller when a few points
carry most of the weight. `0.0` while the total weight is 0.
"""
effectivesamplesize(oh::StreamHist) = oh.sumW2 == 0 ? 0.0 : totalweight(oh)^2 / oh.sumW2

"""
    outofrange(oh)

`(underflow=, overflow=)`: the numbers of points below and above the
histogram's range, which are not in `exactHistogram(oh)`'s bins. (With
`closed=:left`, a point exactly at the top of the range is above it.)
`sum(exactHistogram(oh).weights) + underflow + overflow == nobs(oh)` always
holds exactly. A nonzero count means the range, learned or given, misses part
of the data.
"""
outofrange(oh::StreamHist) = (underflow=oh.underflowCount, overflow=oh.overflowCount)

"""
    outofrangeweight(oh)

`(underflow=, overflow=)`: the sums of the weights of the points counted by
`outofrange(oh)`. `sum(weightedHistogram(oh).weights) + underflow + overflow
== totalweight(oh)` (up to rounding).
"""
outofrangeweight(oh::StreamHist) = (underflow=oh.underflowWeight, overflow=oh.overflowWeight)

"Number of points counted inside the histogram's range."
ninrange(oh::StreamHist) = nobs(oh) - oh.underflowCount - oh.overflowCount

"Sum of the weights of the points inside the histogram's range."
inrangeweight(oh::StreamHist) = totalweight(oh) - oh.underflowWeight - oh.overflowWeight

"The exact mean of every point added; `NaN` before any point."
Statistics.mean(oh::StreamHist) = mean(oh.moments)
"The exact population variance `Σ(x - mean)² / n` of every point added (no `n - 1` correction); `NaN` before any point."
variance(oh::StreamHist) = variance(oh.moments)
"The square root of `variance(oh)`."
Statistics.std(oh::StreamHist) = std(oh.moments)
"""
    skewness(oh)

The third central moment over `variance(oh)^1.5`. Needs the highest order in
`momentPowers` to be at least 3.
"""
skewness(oh::StreamHist) = skewness(oh.moments)
"""
    kurtosis(oh)

The fourth central moment over `variance(oh)^2`; not the excess kurtosis, so
3 for a normal distribution. Needs the highest order in `momentPowers` to be
at least 4.
"""
kurtosis(oh::StreamHist) = kurtosis(oh.moments)

"""
    moment(oh, p)

Order-`p` moment (mean for `p == 1`, population central moment for `p >= 2`).
`p` must be one of `oh.momentPowers`. `NaN` before any point.
"""
function moment(oh::StreamHist, p::Integer)
    p in oh.momentPowers || throw(ArgumentError("power $p was not requested in momentPowers"))
    return centralmoment(oh.moments, p)
end

"`Dict{Int,Float64}`: each order in `momentPowers` => `moment(oh, p)`."
moments(oh::StreamHist) = Dict(p => centralmoment(oh.moments, p) for p in oh.momentPowers)

"""
    weightedMoment(oh, p)

The order-`p` moment of the weighted data: the weighted mean `Σwx / Σw` for
`p == 1`, else the weighted population central moment `Σw(x - mean)^p / Σw`.
`p` must be one of `oh.momentPowers`. With no weights given (all weights 1)
this is `moment(oh, p)`. `NaN` while the total weight is 0.
"""
function weightedMoment(oh::StreamHist, p::Integer)
    p in oh.momentPowers || throw(ArgumentError("power $p was not requested in momentPowers"))
    return centralmoment(oh.wmoments, p)
end

"`Dict` of every requested power => its weighted moment (see `weightedMoment`)."
weightedMoments(oh::StreamHist) = Dict(p => centralmoment(oh.wmoments, p) for p in oh.momentPowers)

"The weighted mean `Σwx / Σw`; `NaN` while the total weight is 0."
weightedMean(oh::StreamHist) = mean(oh.wmoments)
"The weighted population variance `Σw(x - mean)² / Σw`; `NaN` while the total weight is 0."
weightedVariance(oh::StreamHist) = variance(oh.wmoments)
"The square root of `weightedVariance(oh)`."
weightedStd(oh::StreamHist) = std(oh.wmoments)
"The weighted skewness: the third weighted central moment over `weightedVariance(oh)^1.5`."
weightedSkewness(oh::StreamHist) = skewness(oh.wmoments)
"The weighted kurtosis (not excess): the fourth weighted central moment over `weightedVariance(oh)^2`."
weightedKurtosis(oh::StreamHist) = kurtosis(oh.wmoments)

# ---------------------------------------------------------------------------------------------------------------
# The range: learned or given

"""
    integeredges(lo, hi)

An `Int`-typed range `lo_i:1:(hi_i+1)`, so that with `closed=:left` each bin
`[k, k+1)` holds exactly one integer value `k` for `k` in `lo_i:hi_i` —
i.e. one bin per observed integer, no half-integer shifting needed.
"""
function integeredges(lo::Real, hi::Real)
    lo_i = floor(Int, lo)
    hi_i = ceil(Int, hi)
    return lo_i:1:(hi_i + 1)
end

"""
    resolveintegerauto!(oh)

If `integer=:auto` was requested, decide it now from the (about to be
consumed) learn buffer: `true` iff every buffered point `isinteger`. Applies
the same `binNum` conflict check that `integer=true` gets at construction,
just deferred until the decision is actually made. No-op once resolved (or
if `:auto` was never requested).
"""
function resolveintegerauto!(oh::StreamHist)
    oh.integerAuto || return oh
    oh.integerAuto = false
    oh.integer = all(isinteger, oh.learnBuffer)
    if oh.integer && oh.binNum != DEFAULT_BIN_NUM
        throw(ArgumentError(
            "`binNum` has no effect when `integer=true` (edges are always one bin per integer); " *
            "the learn-phase sample was auto-detected (`integer=:auto`) as integer-valued"))
    end
    return oh
end

function initializerange!(oh::StreamHist, range::Tuple{Float64,Float64}; bins::Union{Nothing,Vector{Float64}}=nothing)
    lo, hi = range
    if oh.integer
        edges = integeredges(lo, hi)
        oh.hist = Histogram(edges, :left)
        oh.whist = Histogram(edges, Float64, :left)
    else
        edges = bins === nothing ? collect(range_(lo, hi, oh.binNum)) : bins
        oh.hist = Histogram(edges, oh.closed)
        oh.whist = Histogram(edges, Float64, oh.closed)
        oh.grid = Base.range(lo, hi; length=oh.ashNGrid + 1)  # lo, hi = extrema(bins) when bins are given
        oh.gridCounts = zeros(Int, length(oh.grid))
        oh.gridWeights = zeros(Float64, length(oh.grid))
    end
    return oh
end

range_(lo, hi, n) = n <= 1 ? (lo:1.0:hi) : range(lo, hi; length=n + 1)

function paddedrange(mn::Real, mx::Real, pct::Real)
    span = mx - mn
    pad = span == 0 ? (abs(mx) == 0 ? 1.0 : abs(mx) * pct) : span * pct
    return (mn - pad, mx + pad)
end

function learnedrange(oh::StreamHist)
    mn, mx = extrema(oh.learnBuffer)
    return oh.integer ? (mn, mx) : paddedrange(mn, mx, oh.paddingPct)
end

"End the learn phase: fix the range from the buffer and feed the buffered points (and weights) in."
function endlearning!(oh::StreamHist)
    resolveintegerauto!(oh)
    initializerange!(oh, learnedrange(oh))
    feed!(oh, oh.learnBuffer, oh.learnWeights)
    oh.learnBuffer = nothing
    oh.learnWeights = nothing
    return oh
end

"""
    finalize!(oh)

End the learn phase now if it is still going on (fewer than `learnLength`
points have arrived), fixing the range from whatever is in the buffer, so
that the histograms and densities can be read. A no-op once the range is
fixed; points can still be added afterwards.
"""
function finalize!(oh::StreamHist)
    if !isinitialized(oh)
        isempty(oh.learnBuffer) && throw(ArgumentError("cannot finalize an empty StreamHist"))
        endlearning!(oh)
    end
    return oh
end

# ---------------------------------------------------------------------------------------------------------------
# Adding points

"""
    add!(oh, x[, w])
    add!(oh, xs[, ws])

Add a point `x` with weight `w` (default 1), or a collection of points `xs`
with a vector of weights `ws` (default all 1). Weights must be finite and
nonnegative; a bad weight raises `ArgumentError` before anything is added,
and a `ws` of the wrong length raises `DimensionMismatch`.

Every point updates the exact minimum, maximum and moments. During the learn
phase points are buffered; when `learnLength` have arrived the range is set
from them and they are counted. After that each point goes straight into the
histograms and the ASH grid.

Weights never change the unweighted results (`exactHistogram`, `density`,
`moment`, `nobs`, ...); they feed the weighted ones beside them
(`weightedHistogram`, `weightedDensity`, `weightedMoment`, `totalweight`,
...). A point of weight 0 still counts in `nobs`, the min/max and the
unweighted results.
"""
function add!(oh::StreamHist, x::Real, w::Real=1)
    x = Float64(x)
    w = checkweight(w)

    oh.moments = addpoint(oh.moments, x)
    oh.wmoments = w == 1 ? addpoint(oh.wmoments, x) : addpoint(oh.wmoments, x, w)
    oh.sumW2 += w * w
    oh.exactMin = min(oh.exactMin, x)
    oh.exactMax = max(oh.exactMax, x)

    if isinitialized(oh)
        tally!(oh, x, w)
    else
        push!(oh.learnBuffer, x)
        push!(oh.learnWeights, w)
        length(oh.learnBuffer) >= oh.learnLength && endlearning!(oh)
    end
    return oh
end

function add!(oh::StreamHist, xs, ws::Union{Nothing,AbstractVector{<:Real}}=nothing)
    isempty(xs) && return oh
    fxs = tofloatvec(xs)
    fws = ws === nothing ? ones(Float64, length(fxs)) : checkweights(ws, length(fxs))

    oh.moments = addbatch(oh.moments, fxs)
    oh.wmoments = addbatch(oh.wmoments, fxs, fws)
    oh.sumW2 += sum(abs2, fws)
    mn, mx = extrema(fxs)
    oh.exactMin = min(oh.exactMin, mn)
    oh.exactMax = max(oh.exactMax, mx)

    if isinitialized(oh)
        feed!(oh, fxs, fws)
    else
        append!(oh.learnBuffer, fxs)
        append!(oh.learnWeights, fws)
        length(oh.learnBuffer) >= oh.learnLength && endlearning!(oh)
    end
    return oh
end

function checkweight(w::Real)
    w = Float64(w)
    (isfinite(w) && w >= 0) || throw(ArgumentError("weights must be finite and nonnegative, got $w"))
    return w
end

function checkweights(ws::AbstractVector{<:Real}, n::Integer)
    length(ws) == n || throw(DimensionMismatch("got $(length(ws)) weights for $n points"))
    fws = tofloatvec(ws)
    foreach(checkweight, fws)
    return fws
end

# Avoid a full copy when the caller already passed a Vector{Float64}.
tofloatvec(xs::Vector{Float64}) = xs
tofloatvec(xs) = collect(Float64, xs)

# The histograms and grid are passed as arguments (a function barrier), so the loop runs on their concrete types.
feed!(oh::StreamHist, xs::Vector{Float64}, ws::Vector{Float64}) = feed!(oh, oh.hist, oh.whist, oh.grid, xs, ws)
function feed!(oh::StreamHist, h, wh, grid, xs::Vector{Float64}, ws::Vector{Float64})
    for i in eachindex(xs, ws)
        tally!(oh, h, wh, grid, xs[i], ws[i])
    end
    return oh
end

tally!(oh::StreamHist, x::Float64, w::Float64) = tally!(oh, oh.hist, oh.whist, oh.grid, x, w)

"""
    tally!(oh, x, w)

Count point `x` (weight `w`) in the histograms and, unless `integer`, at the
nearest ASH grid point. A point outside the histogram's range goes into the
underflow or overflow count and weight instead, and not on the grid
(`StatsBase.Histogram`'s `push!` would silently drop it). So the grid holds
exactly the points the histogram does, and the density describes the same
points that `histogram(oh, bins)` scales it by.
"""
function tally!(oh::StreamHist, h::Histogram, wh::Histogram, grid, x::Float64, w::Float64)
    idx = StatsBase.binindex(h, x)
    if idx < 1
        oh.underflowCount += 1
        oh.underflowWeight += w
    elseif idx > length(h.weights)
        oh.overflowCount += 1
        oh.overflowWeight += w
    else
        h.weights[idx] += 1
        wh.weights[idx] += w
        if grid !== nothing                            # no grid in integer mode
            # the nearest grid point; clamped only against rounding, since the grid spans the histogram's range
            k = clamp(floor(Int, (x - first(grid)) * inv(step(grid)) + 1.5), 1, length(grid))
            oh.gridCounts[k] += 1
            oh.gridWeights[k] += w
        end
    end
    return oh
end

# ---------------------------------------------------------------------------------------------------------------
# Reading the histograms and densities

function requireinitialized(oh::StreamHist)
    isinitialized(oh) || throw(ArgumentError(
        "StreamHist range not yet initialized (still learning); call finalize!(oh) first"))
    return oh
end

function requiredensity(oh::StreamHist)
    oh.integer && throw(ArgumentError("there is no ASH density in integer mode"))
    return requireinitialized(oh)
end

"""
    exactHistogram(oh)

The traditional histogram: a `StatsBase.Histogram` of the bin counts. Needs
the range set (see `finalize!`). Points outside the range are in
`outofrange(oh)`, not in the edge bins.
"""
exactHistogram(oh::StreamHist) = requireinitialized(oh).hist

"""
    weightedHistogram(oh)

A `StatsBase.Histogram` with `exactHistogram(oh)`'s edges whose weights are
each bin's sum of point weights (equal to the counts when no weights were
given).
"""
weightedHistogram(oh::StreamHist) = requireinitialized(oh).whist

"""
    meanWeight(oh)

`(edges=, meanweight=)`: the average weight of the points in each bin of
`exactHistogram(oh)` (bin weight sum / bin count), `NaN` for an empty bin.
How the weights depend on the value: flat when they don't, so the weighted
and unweighted histograms then agree in shape. `weightedHistogram(oh)`'s
weights are `exactHistogram(oh)`'s counts times these.
"""
function meanWeight(oh::StreamHist)
    h, wh = exactHistogram(oh), weightedHistogram(oh)
    mw = [c == 0 ? NaN : w / c for (c, w) in zip(h.weights, wh.weights)]
    return (edges=collect(only(h.edges)), meanweight=mw)
end

"""
    smoothgrid(cellsums, grid, kernel, m)

The ASH of per-grid-cell totals `cellsums` (counts, or sums of weights) on
`grid`: each cell `k`'s total spread over its neighbours `i` by
`kernel((i - k) / m)`, then normalized so that its linear interpolation
(`density`) integrates to 1. All `NaN` if every total is 0.
"""
function smoothgrid(cellsums::AbstractVector{<:Real}, grid::AbstractRange, kernel, m::Integer)
    m > 0 || throw(ArgumentError("the smoothing parameter `m` must be positive"))
    b = length(grid)
    dens = zeros(Float64, b)
    for k in 1:b
        c = cellsums[k]
        c == 0 && continue
        for i in max(1, k - m + 1):min(b, k + m - 1)
            @inbounds dens[i] += c * kernel((i - k) / m)
        end
    end
    dens .*= 1 / trapezoid(dens, step(grid))
    return dens
end

"The integral of the linear interpolation of `y`, given at points `Δ` apart."
trapezoid(y, Δ) = Δ * (sum(y) - (first(y) + last(y)) / 2)

"""Linearly interpolate density values `dens` given on `grid`; zero outside it."""
function interpolated(grid::AbstractRange, dens::AbstractVector{Float64})
    lo, hi = first(grid), last(grid)
    Δ = step(grid)
    n = length(grid)
    return function (x::Real)
        (x < lo || x > hi) && return 0.0
        i = clamp(floor(Int, (x - lo) / Δ) + 1, 1, n - 1)
        return dens[i] + (dens[i + 1] - dens[i]) * (x - grid[i]) / (grid[i + 1] - grid[i])
    end
end

"""
The integral from `first(grid)` to `x` of `interpolated(grid, dens)`, exactly
(it is piecewise quadratic): 0 below the grid, the total above it.
"""
function cumulative(grid::AbstractRange, dens::AbstractVector{Float64})
    lo, hi = first(grid), last(grid)
    Δ = step(grid)
    n = length(grid)
    upto = zeros(Float64, n)                     # the integral up to each grid point
    for i in 2:n
        upto[i] = upto[i - 1] + (grid[i] - grid[i - 1]) * (dens[i - 1] + dens[i]) / 2
    end
    return function (x::Real)
        x < lo && return 0.0
        x >= hi && return upto[n]
        i = clamp(floor(Int, (x - lo) / Δ) + 1, 1, n - 1)
        t = x - grid[i]
        return upto[i] + t * (dens[i] + (dens[i + 1] - dens[i]) * t / (2 * (grid[i + 1] - grid[i])))
    end
end

"""
    densityGrid(oh; weighted=false, m=oh.m, kernel=oh.kernel)

`(grid=, density=)`: the ASH density at the ASH grid points (of the weighted
data if `weighted`), the values `density(oh)` / `weightedDensity(oh)`
interpolate between. `m` and `kernel` smooth with another width or kernel
than the ones `oh` was built with, without changing `oh`.
"""
function densityGrid(oh::StreamHist; weighted::Bool=false, m::Integer=oh.m, kernel=oh.kernel)
    requiredensity(oh)
    cells = weighted ? oh.gridWeights : oh.gridCounts
    return (grid=oh.grid, density=smoothgrid(cells, oh.grid, kernel, m))
end

"""
    density(oh; m=oh.m, kernel=oh.kernel)

A callable `x -> density(x)` linearly interpolating the ASH density of the
points (each counted once) between the grid points, zero outside the range;
it integrates to 1. Unavailable when `integer=true`; needs the range fixed
(`finalize!(oh)` ends the learn phase). It is a snapshot: points added later
are in the next `density(oh)`. Every value inside the range is `NaN` if no
point is inside it. `m` and `kernel`: see `densityGrid`.
"""
density(oh::StreamHist; m::Integer=oh.m, kernel=oh.kernel) = interpolated(densityGrid(oh; m, kernel)...)

"""
    weightedDensity(oh; m=oh.m, kernel=oh.kernel)

`density(oh)` for the weighted data: the ASH density with each point counted
with its weight. With the same grid, kernel and smoothing width as
`density(oh)`, it equals the count density times the kernel-smoothed mean
weight `E[w | x]`, and equals `density(oh)` when no weights were given. Every
value inside the range is `NaN` if the points in range have total weight 0.
"""
weightedDensity(oh::StreamHist; m::Integer=oh.m, kernel=oh.kernel) =
    interpolated(densityGrid(oh; weighted=true, m, kernel)...)

"""
    cdf(oh; m=oh.m, kernel=oh.kernel)

A callable `x -> cdf(x)`: the exact integral of `density(oh)` up to `x`, the
estimated fraction of the points in range that are at most `x`. It is 0 below
the range and 1 from its top up, and `NaN` from the bottom of the range up if
no point is inside it. Needs what `density` needs; a snapshot like it.
"""
cdf(oh::StreamHist; m::Integer=oh.m, kernel=oh.kernel) = cumulative(densityGrid(oh; m, kernel)...)

"""
    weightedCdf(oh; m=oh.m, kernel=oh.kernel)

`cdf(oh)` for the weighted data: the exact integral of `weightedDensity(oh)`
up to `x`, the estimated fraction of the in-range weight at values at most `x`.
"""
weightedCdf(oh::StreamHist; m::Integer=oh.m, kernel=oh.kernel) =
    cumulative(densityGrid(oh; weighted=true, m, kernel)...)

"""
    histogram(oh, bins; m=oh.m, kernel=oh.kernel)

The expected counts over an arbitrary set of edges `bins`: the integral of
`density(oh)` between consecutive edges (exactly, from `cdf(oh)`), scaled by
the number of points inside the histogram's range, which is what the density
is normalized over.
"""
histogram(oh::StreamHist, bins::AbstractVector{<:Real}; m::Integer=oh.m, kernel=oh.kernel) =
    binintegrals(cdf(oh; m, kernel), ninrange(oh), bins)

"""
    weightedHistogram(oh, bins; m=oh.m, kernel=oh.kernel)

`histogram(oh, bins)` for the weighted data: the integral of
`weightedDensity(oh)` between consecutive `bins` edges, scaled by the total
weight of the points in range, giving the expected weight in each bin.
"""
weightedHistogram(oh::StreamHist, bins::AbstractVector{<:Real}; m::Integer=oh.m, kernel=oh.kernel) =
    binintegrals(weightedCdf(oh; m, kernel), inrangeweight(oh), bins)

# With nothing in range the density is undefined (NaN), and nothing falls in any bin.
function binintegrals(F, total, bins)
    weights = total == 0 ? zeros(length(bins) - 1) :
              [total * (F(bins[i + 1]) - F(bins[i])) for i in 1:(length(bins) - 1)]
    return (edges=collect(bins), weights=weights)
end

"""
    densityQuality(oh)

For each requested moment power (in the order of `oh.momentPowers`),
numerically integrates the ASH density against `(x - mean)^p` and compares
it to the moment accumulator's value. Returns a `Vector{Float64}` of the
relative error `abs(fromdensity - exact) / (abs(exact) + eps())` between the
two estimates, one entry per power. If no point fell inside the histogram's
range (all under/overflow), the density is undefined and every entry is `NaN`.
"""
densityQuality(oh::StreamHist) = densityquality(oh, density(oh), ninrange(oh), mean(oh), p -> moment(oh, p))

"""
    weightedDensityQuality(oh)

`densityQuality` for the weighted data: compares `weightedDensity(oh)`'s
moments with the exact weighted moments (`weightedMoment`).
"""
weightedDensityQuality(oh::StreamHist) =
    densityquality(oh, weightedDensity(oh), inrangeweight(oh), weightedMean(oh), p -> weightedMoment(oh, p))

function densityquality(oh::StreamHist, d, inrange, μ, exactmoment)
    inrange == 0 && return fill(NaN, length(oh.momentPowers))
    lo, hi = first(oh.grid), last(oh.grid)
    return map(oh.momentPowers) do p
        exact = exactmoment(p)
        fromdensity = p == 1 ? first(quadgk(x -> x * d(x), lo, hi)) : first(quadgk(x -> (x - μ)^p * d(x), lo, hi))
        abs(fromdensity - exact) / (abs(exact) + eps())
    end
end

# ---------------------------------------------------------------------------------------------------------------
# Distances between densities

"""
    lp_distance(a::StreamHist, b::StreamHist; p=2, weighted=false, rtol=sqrt(eps(Float64)), atol=0.0)

Return the finite-``p`` L^p distance between the ASH densities of `a` and `b`
(their weighted densities if `weighted`):

```math
\\lVert f_a-f_b\\rVert_p =
\\left(\\int |f_a(x)-f_b(x)|^p\\,dx\\right)^{1/p}.
```

The integral is evaluated over the two densities' finite ASH-grid supports.
It is split into the left non-overlapping tail, the overlapping region, and
the right non-overlapping tail.  This avoids evaluating either density
outside its support.  If the supports are disjoint, the middle integral is
empty. Both histograms must be non-integer, with their ranges fixed and some
point (weight, if `weighted`) in range. `rtol` and `atol` are passed to
`QuadGK.quadgk`.
"""
function lp_distance(a::StreamHist, b::StreamHist;
    p::Real=2, weighted::Bool=false, rtol::Real=sqrt(eps(Float64)), atol::Real=0.0)
    isfinite(p) && p > 0 || throw(ArgumentError("p must be finite and positive"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    for (which, oh) in (("first", a), ("second", b))
        requiredensity(oh)
        (weighted ? inrangeweight(oh) : ninrange(oh)) > 0 ||
            throw(ArgumentError("lp_distance: the $which StreamHist has nothing in its range"))
    end

    fa, fb = weighted ? (weightedDensity(a), weightedDensity(b)) : (density(a), density(b))
    alo, ahi = first(a.grid), last(a.grid)
    blo, bhi = first(b.grid), last(b.grid)

    integral(f, lo, hi) = lo < hi ? first(quadgk(f, lo, hi; rtol=rtol, atol=atol)) : 0.0

    left_density = alo <= blo ? fa : fb
    left_tail = integral(x -> abs(left_density(x))^p, min(alo, blo), max(alo, blo))
    overlap = integral(x -> abs(fa(x) - fb(x))^p, max(alo, blo), min(ahi, bhi))
    right_density = ahi >= bhi ? fa : fb
    right_tail = integral(x -> abs(right_density(x))^p, min(ahi, bhi), max(ahi, bhi))

    return (left_tail + overlap + right_tail)^(inv(p))
end

"""
    tv_distance(a, b; weighted=false, kwargs...)

Return the total-variation distance between two ASH density estimates, equal
to one half of `lp_distance(a, b; p=1, kwargs...)`.
"""
tv_distance(a::StreamHist, b::StreamHist; kwargs...) = lp_distance(a, b; p=1, kwargs...) / 2

function Base.show(io::IO, oh::StreamHist)
    print(io, "StreamHist(n=", nobs(oh), ", totalweight=", totalweight(oh), ", initialized=", isinitialized(oh),
        ", integer=", oh.integer, ")")
end
