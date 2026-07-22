const DEFAULT_MOMENT_POWERS = [1, 2, 4, 8]

mutable struct StreamHist
    # config, fixed at construction
    integer::Bool
    momentPowers::Vector{Int}
    learn::Bool
    learnLength::Int
    paddingPct::Float64
    binNum::Int
    closed::Symbol
    kernel::Any
    m::Int
    ashNGrid::Int
    ashBatchSize::Int

    # learn-phase buffer; set to nothing once the range has been fixed
    learnBuffer::Union{Vector{Float64},Nothing}

    # live state; both nothing until the range is known
    hist::Union{Histogram,Nothing}
    ash::Union{Ash,Nothing}

    # scalar add! points destined for the ASH, held back and flushed in
    # batches of ashBatchSize (or on finalize!/a batch add!) -- ash! is
    # dramatically cheaper per point when called on a batch than called
    # once per point.
    ashPending::Vector{Float64}

    moments::MomentAccumulator
    exactMin::Float64
    exactMax::Float64

    finalized::Bool
end

"""
    StreamHist(; integer=false, momentPowers=[1,2,4,8], learn=true,
               learnLength=10_000, paddingPct=0.05, binRange=nothing,
               binNum=50, bins=nothing, closed=:left,
               kernel=AverageShiftedHistograms.Kernels.biweight, m=5,
               ashNGrid=500, ashBatchSize=256)

An online histogram: maintains a traditional (`StatsBase.Histogram`)
histogram, an Average Shifted Histogram density estimate, exact min/max, and
online central moments, all updatable one point or one batch at a time.

- `integer`: data is integer-valued; bins are centered on the integers in the
  observed/given range and the ASH is disabled (`ash(oh) === nothing`).
- `momentPowers`: which moment orders to expose via `moment(oh, p)`.
- `binRange = (lo, hi)`: fix the histogram/ASH range up front.
- `binNum`: number of traditional-histogram bins (ignored if `bins` given).
- `bins`: explicit bin edges for the traditional histogram; overrides
  `binRange`/`binNum`. The ASH range is taken from `extrema(bins)`.
- `learn`/`learnLength`: if no `bins`/`binRange` is given, the first
  `learnLength` points are buffered, then used to pick a range
  (`extrema(buffer)` padded by `paddingPct` on each side).
- `closed`, `kernel`, `m`: passed straight through to
  `StatsBase.Histogram`/`AverageShiftedHistograms.ash`.
- `ashBatchSize`: single-point `add!` calls hold their point back from the
  ASH and flush in batches of this size (`ash!` is far cheaper per point
  called on a batch than called once per point); `finalize!` always flushes
  whatever is pending. Batch `add!` calls bypass this and go straight to the
  ASH regardless of size, since they're already batches.
"""
function StreamHist(;
    integer::Bool=false,
    momentPowers::AbstractVector{<:Integer}=DEFAULT_MOMENT_POWERS,
    learn::Bool=true,
    learnLength::Integer=10_000,
    paddingPct::Real=0.05,
    binRange::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    binNum::Integer=50,
    bins::Union{Nothing,AbstractVector{<:Real}}=nothing,
    closed::Symbol=:left,
    kernel=AverageShiftedHistograms.Kernels.biweight,
    m::Integer=5,
    ashNGrid::Integer=500,
    ashBatchSize::Integer=256,
)
    momentPowers = sort(unique(Int.(momentPowers)))
    maxp = max(1, maximum(momentPowers))

    oh = StreamHist(
        integer, momentPowers, learn, Int(learnLength), Float64(paddingPct),
        Int(binNum), closed, kernel, Int(m), Int(ashNGrid), Int(ashBatchSize),
        Float64[], nothing, nothing, Float64[],
        MomentAccumulator(maxp), Inf, -Inf, false,
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

nmoments(oh::StreamHist) = maxpower(oh.moments)
StatsBase.nobs(oh::StreamHist) = oh.moments.n
datarange(oh::StreamHist) = (oh.exactMin, oh.exactMax)
Statistics.mean(oh::StreamHist) = mean(oh.moments)
variance(oh::StreamHist) = variance(oh.moments)
Statistics.std(oh::StreamHist) = std(oh.moments)
skewness(oh::StreamHist) = skewness(oh.moments)
kurtosis(oh::StreamHist) = kurtosis(oh.moments)

"""
    moment(oh, p)

Order-`p` moment (mean for `p == 1`, population central moment for `p >= 2`).
`p` must be one of `oh.momentPowers`.
"""
function moment(oh::StreamHist, p::Integer)
    p in oh.momentPowers || throw(ArgumentError("power $p was not requested in momentPowers"))
    return centralmoment(oh.moments, p)
end

"""
    moments(oh)

NamedTuple-like `Dict{Int,Float64}` of every requested power => its moment.
"""
moments(oh::StreamHist) = Dict(p => centralmoment(oh.moments, p) for p in oh.momentPowers)

isinitialized(oh::StreamHist) = oh.hist !== nothing

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

function initializerange!(oh::StreamHist, range::Tuple{Float64,Float64}; bins::Union{Nothing,Vector{Float64}}=nothing)
    lo, hi = range
    if oh.integer
        edges = integeredges(lo, hi)
        oh.hist = Histogram(edges, :left)
        oh.ash = nothing
    else
        edges = bins === nothing ? collect(range_(lo, hi, oh.binNum)) : bins
        oh.hist = Histogram(edges, oh.closed)
        ashlo, ashhi = bins === nothing ? (lo, hi) : (first(bins), last(bins))
        ashrng = range_(ashlo, ashhi, oh.ashNGrid)
        oh.ash = ash(Float64[]; rng=ashrng, kernel=oh.kernel, m=oh.m)
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
    buf = oh.learnBuffer
    mn, mx = extrema(buf)
    return oh.integer ? (mn, mx) : paddedrange(mn, mx, oh.paddingPct)
end

"""
    add!(oh, data)

Add a single point or a vector of points to the histogram. `data` updates the
exact min/max and the online moments unconditionally; while still in the
learn phase it is buffered until `learnLength` points have accumulated, at
which point the range is fixed and the traditional histogram / ASH are
initialized from the buffer (plus padding, unless `integer=true`).
"""
function add!(oh::StreamHist, x::Real)
    x = Float64(x)
    oh.finalized = false

    oh.moments = addpoint(oh.moments, x)
    oh.exactMin = min(oh.exactMin, x)
    oh.exactMax = max(oh.exactMax, x)

    if !isinitialized(oh)
        push!(oh.learnBuffer, x)
        if length(oh.learnBuffer) >= oh.learnLength
            initializerange!(oh, learnedrange(oh))
            feed!(oh, oh.learnBuffer)
            oh.learnBuffer = nothing
        end
    else
        push!(oh.hist, x)
        if !oh.integer
            push!(oh.ashPending, x)
            length(oh.ashPending) >= oh.ashBatchSize && flushash!(oh)
        end
    end
    return oh
end

function add!(oh::StreamHist, xs)
    isempty(xs) && return oh
    oh.finalized = false

    fxs = tofloatvec(xs)
    oh.moments = addbatch(oh.moments, fxs)
    mn, mx = extrema(fxs)
    oh.exactMin = min(oh.exactMin, mn)
    oh.exactMax = max(oh.exactMax, mx)

    if !isinitialized(oh)
        append!(oh.learnBuffer, fxs)
        if length(oh.learnBuffer) >= oh.learnLength
            initializerange!(oh, learnedrange(oh))
            feed!(oh, oh.learnBuffer)
            oh.learnBuffer = nothing
        end
    else
        feed!(oh, fxs)
    end
    return oh
end

# Avoid a full copy when the caller already passed a Vector{Float64}.
tofloatvec(xs::Vector{Float64}) = xs
tofloatvec(xs) = collect(Float64, xs)

function feed!(oh::StreamHist, xs::Vector{Float64})
    for x in xs
        push!(oh.hist, x)
    end
    if !oh.integer
        # Batches already amortize ash!'s per-call overhead over many
        # points, so they bypass ashPending entirely -- but flush whatever
        # scalar add!s left pending first, to keep it from accumulating
        # across mixed scalar/batch usage.
        flushash!(oh)
        ash!(oh.ash, xs)
    end
    return oh
end

function flushash!(oh::StreamHist)
    isempty(oh.ashPending) && return oh
    ash!(oh.ash, oh.ashPending)
    empty!(oh.ashPending)
    return oh
end

"""
    finalize!(oh)

Marks `oh` ready to read. If still in the learn phase (fewer than
`learnLength` points ever arrived), forces the range to be fixed now from
whatever is in the buffer. Any subsequent `add!` un-finalizes `oh` again.
"""
function finalize!(oh::StreamHist)
    if !isinitialized(oh)
        isempty(oh.learnBuffer) && throw(ArgumentError("cannot finalize an empty StreamHist"))
        initializerange!(oh, learnedrange(oh))
        feed!(oh, oh.learnBuffer)
        oh.learnBuffer = nothing
    else
        oh.integer || flushash!(oh)
    end
    oh.finalized = true
    return oh
end

"""
    exactHistogram(oh)

The underlying `StatsBase.Histogram` (edges + raw counts).
"""
function exactHistogram(oh::StreamHist)
    isinitialized(oh) || throw(ArgumentError("StreamHist range not yet initialized (still learning); call finalize!(oh) first"))
    return oh.hist
end

"""
    density(oh)

A callable `x -> density(x)` linearly interpolating the ASH density.
Unavailable when `integer=true`. Requires `oh` to be finalized (via
`finalize!(oh)`) -- single-point `add!` calls hold points back from the ASH
and batch them for efficiency (see `ashBatchSize`), so the ASH is only
guaranteed current once finalized; any `add!` after that un-finalizes `oh`
again.
"""
function density(oh::StreamHist)
    oh.integer && throw(ArgumentError("density(::StreamHist) is unavailable in integer mode"))
    oh.finalized || throw(ArgumentError("StreamHist must be finalize!(oh)d before calling density(oh)"))
    rng = oh.ash.rng
    dens = oh.ash.density
    lo, hi = first(rng), last(rng)
    Δ = step(rng)
    n = length(rng)
    return function (x::Real)
        (x < lo || x > hi) && return 0.0
        i = clamp(floor(Int, (x - lo) / Δ) + 1, 1, n - 1)
        x0, x1 = rng[i], rng[i + 1]
        y0, y1 = dens[i], dens[i + 1]
        x1 == x0 && return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    end
end

"""
    histogram(oh, bins)

Integrates `density(oh)` between consecutive `bins` edges (scaled by
`nobs(oh)`) to produce expected counts over an arbitrary set of edges.
"""
function histogram(oh::StreamHist, bins::AbstractVector{<:Real})
    d = density(oh)
    n = nobs(oh)
    weights = [n * first(quadgk(d, bins[i], bins[i + 1])) for i in 1:(length(bins) - 1)]
    return (edges=collect(bins), weights=weights)
end

"""
    densityQuality(oh)

For each requested moment power (in the order of `oh.momentPowers`),
numerically integrates the ASH density against `(x - mean)^p` and compares
it to the moment accumulator's value. Returns a `Vector{Float64}` of the
relative error `abs(fromdensity - exact) / (abs(exact) + eps())` between the
two estimates, one entry per power.
"""
function densityQuality(oh::StreamHist)
    d = density(oh)
    rng = oh.ash.rng
    lo, hi = first(rng), last(rng)
    μ = mean(oh)
    relerrs = Float64[]
    for p in oh.momentPowers
        exact = moment(oh, p)
        fromdensity = if p == 1
            first(quadgk(x -> x * d(x), lo, hi))
        else
            first(quadgk(x -> (x - μ)^p * d(x), lo, hi))
        end
        push!(relerrs, abs(fromdensity - exact) / (abs(exact) + eps()))
    end
    return relerrs
end

function Base.show(io::IO, oh::StreamHist)
    print(io, "StreamHist(n=", nobs(oh), ", initialized=", isinitialized(oh),
        ", finalized=", oh.finalized, ", integer=", oh.integer, ")")
end
