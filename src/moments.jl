# Online, mergeable central-moment accumulator (Pébay 2008 / Chan-Golub-LeVeque /
# Terriberry generalization to arbitrary order). Avoids ever subtracting two
# independently-large quantities (e.g. E[X^p] - mean^p), which is the source of
# catastrophic cancellation in the naive two-pass formulas.
#
# M.M[p] holds Σ(xᵢ - mean)^p for p = 2:maxpower. M.M[1] is unused (always 0,
# since the first central moment is trivially zero) — the mean is tracked
# separately and stands in for "moment order 1".
#
# Weighted data (Pébay, Terriberry, Kolla & Bennett 2016): the same formulas
# hold with every count replaced by a sum of weights. A `MomentAccumulator{Int}`
# counts points (each of weight 1); a `MomentAccumulator{Float64}` has `n` =
# the total weight Σwᵢ, `mean` = Σwᵢxᵢ / Σwᵢ and `M[p]` = Σwᵢ(xᵢ - mean)^p.

struct MomentAccumulator{T<:Real}
    n::T
    mean::Float64
    M::Vector{Float64}
    # binom[p, k] = binomial(p, k), for p in 2:maxpower, k in 1:(p-2); built
    # once per accumulator "lineage" and carried forward by reference through
    # every merge, rather than recomputed on every single-point/batch update.
    binom::Matrix{Float64}
end

function binomtable(maxp::Integer)
    T = zeros(Float64, maxp, maxp)
    @inbounds for p in 2:maxp, k in 1:(p - 2)
        T[p, k] = binomial(p, k)
    end
    return T
end

MomentAccumulator(maxpower::Integer) = MomentAccumulator(Int, maxpower)
MomentAccumulator(::Type{T}, maxpower::Integer) where {T<:Real} =
    MomentAccumulator{T}(zero(T), 0.0, zeros(Float64, maxpower), binomtable(maxpower))

maxpower(A::MomentAccumulator) = length(A.M)

function batchaccumulator(xs::AbstractVector{<:Real}, maxp::Integer)
    n = length(xs)
    n == 0 && return MomentAccumulator(maxp)
    μ = sum(x -> Float64(x), xs) / n
    M = zeros(Float64, maxp)
    # One pass over xs, building every power's deviation d^p from d^(p-1)
    # (d *= d each step) instead of maxp-1 separate full traversals each
    # recomputing d^p from scratch via `^`.
    @inbounds for x in xs
        d = Float64(x) - μ
        dp = d * d
        M[2] += dp
        for p in 3:maxp
            dp *= d
            M[p] += dp
        end
    end
    return MomentAccumulator(n, μ, M, binomtable(maxp))
end

"""
    batchaccumulator(xs, ws, maxp)

The weighted accumulator (`MomentAccumulator{Float64}`) of points `xs` with
weights `ws`: total weight Σw, mean Σwx/Σw, and M[p] = Σw(x - mean)^p.
"""
function batchaccumulator(xs::AbstractVector{<:Real}, ws::AbstractVector{<:Real}, maxp::Integer)
    W = sum(Float64, ws; init=0.0)
    W == 0 && return MomentAccumulator(Float64, maxp)
    μ = sum(Float64(w) * Float64(x) for (x, w) in zip(xs, ws)) / W
    M = zeros(Float64, maxp)
    @inbounds for (x, w) in zip(xs, ws)
        d = Float64(x) - μ
        dp = d * d
        M[2] += w * dp
        for p in 3:maxp
            dp *= d
            M[p] += w * dp
        end
    end
    return MomentAccumulator{Float64}(W, μ, M, binomtable(maxp))
end

"""
    mergeaccumulators(A, B)

Combine two independently-computed `MomentAccumulator`s into the accumulator
for the union of their data, using Pébay's generalization of the
Chan/Golub/LeVeque parallel variance formula to arbitrary order.
`add!`ing a batch of points reduces to this merge (`A` merged with a fresh
`batchaccumulator` of the new batch); single points use the specialized
`addpoint` below instead, which is this same formula algebraically
simplified for a size-1, all-zero-moment `B`.
"""
function mergeaccumulators(A::MomentAccumulator{T}, B::MomentAccumulator{T}) where {T}
    maxp = maxpower(A)
    maxp == maxpower(B) || throw(ArgumentError("accumulators must track the same moment orders"))
    nA, nB = A.n, B.n
    n = nA + nB
    n == 0 && return MomentAccumulator(T, maxp)
    nA == 0 && return B
    nB == 0 && return A

    δ = B.mean - A.mean
    μ = A.mean + δ * nB / n
    M = zeros(Float64, maxp)
    binom = A.binom

    negnBδ_over_n = -nB * δ / n
    nAδ_over_n = nA * δ / n
    # Running powers for the p-loop's own terms, built incrementally
    # (one multiplication per p) instead of via `^` from scratch each time.
    nApow = Float64(nA)      # nA^(p-1), starting at p=2
    nBpow = Float64(-nB)     # (-nB)^(p-1), starting at p=2
    deltapow = δ * δ         # δ^p, starting at p=2
    npow = Float64(n) * n    # n^p, starting at p=2

    @inbounds for p in 2:maxp
        s = A.M[p] + B.M[p]
        kpow1 = negnBδ_over_n  # (-nB δ/n)^k, starting at k=1
        kpow2 = nAδ_over_n     # (nA δ/n)^k, starting at k=1
        for k in 1:(p - 2)
            s += binom[p, k] * (A.M[p - k] * kpow1 + B.M[p - k] * kpow2)
            kpow1 *= negnBδ_over_n
            kpow2 *= nAδ_over_n
        end
        s += deltapow * nA * nB * (nApow - nBpow) / npow
        M[p] = s

        nApow *= nA
        nBpow *= -nB
        deltapow *= δ
        npow *= n
    end
    return MomentAccumulator(n, μ, M, binom)
end

"""
    addpoint(A, x, w)

Add one point `x` of weight `w` to a weighted accumulator: the merge with a
one-point accumulator of total weight `w` (and all central moments zero).
"""
addpoint(A::MomentAccumulator{Float64}, x::Real, w::Real) =
    mergeaccumulators(A, MomentAccumulator{Float64}(Float64(w), Float64(x), zeros(Float64, maxpower(A)), A.binom))

"""
    addpoint(A, x)

`mergeaccumulators(A, B)` specialized for a single new point (`B` of size 1,
all of whose own central moments are trivially zero): the `B.M[...]` terms
and the `k`-power built from `nB`'s side both drop out algebraically, and
`(-nB)^(p-1)` with `nB=1` collapses to a plain sign flip. Verified against
the general `mergeaccumulators` path in the test suite.
"""
function addpoint(A::MomentAccumulator, x::Real)
    maxp = maxpower(A)
    x = Float64(x)
    A.n == 0 && return MomentAccumulator(oneunit(A.n), x, zeros(Float64, maxp), A.binom)

    nA = A.n
    n = nA + 1
    δ = x - A.mean
    μ = A.mean + δ / n

    M = zeros(Float64, maxp)
    binom = A.binom
    negδ_over_n = -δ / n

    nApow = Float64(nA)   # nA^(p-1), starting at p=2
    deltapow = δ * δ      # δ^p, starting at p=2
    npow = Float64(n) * n # n^p, starting at p=2
    signB = -1.0          # (-1)^(p-1), starting at p=2

    @inbounds for p in 2:maxp
        s = A.M[p]
        kpow = negδ_over_n  # (-δ/n)^k, starting at k=1
        for k in 1:(p - 2)
            s += binom[p, k] * A.M[p - k] * kpow
            kpow *= negδ_over_n
        end
        s += deltapow * nA * (nApow - signB) / npow
        M[p] = s

        nApow *= nA
        deltapow *= δ
        npow *= n
        signB = -signB
    end
    return MomentAccumulator(n, μ, M, binom)
end

addbatch(A::MomentAccumulator, xs::AbstractVector{<:Real}) =
    mergeaccumulators(A, batchaccumulator(xs, maxpower(A)))
addbatch(A::MomentAccumulator{Float64}, xs::AbstractVector{<:Real}, ws::AbstractVector{<:Real}) =
    mergeaccumulators(A, batchaccumulator(xs, ws, maxpower(A)))

"""
    centralmoment(A, p)

Return the order-`p` moment: the mean for `p == 1`, else the population
central moment `M[p] / n`.
"""
function centralmoment(A::MomentAccumulator, p::Integer)
    p == 1 && return A.mean
    (2 <= p <= maxpower(A)) || throw(ArgumentError("moment order $p not tracked (max = $(maxpower(A)))"))
    A.n == 0 && return NaN
    return A.M[p] / A.n
end

Statistics.mean(A::MomentAccumulator) = A.mean
StatsBase.nobs(A::MomentAccumulator) = A.n
variance(A::MomentAccumulator) = centralmoment(A, 2)
Statistics.std(A::MomentAccumulator) = sqrt(variance(A))
function skewness(A::MomentAccumulator)
    v = variance(A)
    return centralmoment(A, 3) / v^1.5
end
function kurtosis(A::MomentAccumulator)
    v = variance(A)
    return centralmoment(A, 4) / v^2
end
