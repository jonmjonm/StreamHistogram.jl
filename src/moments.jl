# Online, mergeable central-moment accumulator (Pébay 2008 / Chan-Golub-LeVeque /
# Terriberry generalization to arbitrary order). Avoids ever subtracting two
# independently-large quantities (e.g. E[X^p] - mean^p), which is the source of
# catastrophic cancellation in the naive two-pass formulas.
#
# M.M[p] holds Σ(xᵢ - mean)^p for p = 2:maxpower. M.M[1] is unused (always 0,
# since the first central moment is trivially zero) — the mean is tracked
# separately and stands in for "moment order 1".

struct MomentAccumulator
    n::Int
    mean::Float64
    M::Vector{Float64}
end

MomentAccumulator(maxpower::Integer) = MomentAccumulator(0, 0.0, zeros(Float64, maxpower))

maxpower(A::MomentAccumulator) = length(A.M)

function batchaccumulator(xs::AbstractVector{<:Real}, maxp::Integer)
    n = length(xs)
    n == 0 && return MomentAccumulator(maxp)
    μ = sum(x -> Float64(x), xs) / n
    M = zeros(Float64, maxp)
    @inbounds for p in 2:maxp
        M[p] = sum(x -> (Float64(x) - μ)^p, xs)
    end
    return MomentAccumulator(n, μ, M)
end

"""
    mergeaccumulators(A, B)

Combine two independently-computed `MomentAccumulator`s into the accumulator
for the union of their data, using Pébay's generalization of the
Chan/Golub/LeVeque parallel variance formula to arbitrary order. Both
`add!`ing a single point and `add!`ing a batch of points reduce to this same
merge (a single point is just a batch of size 1, all of whose central moments
are trivially zero).
"""
function mergeaccumulators(A::MomentAccumulator, B::MomentAccumulator)
    maxp = maxpower(A)
    maxp == maxpower(B) || throw(ArgumentError("accumulators must track the same moment orders"))
    nA, nB = A.n, B.n
    n = nA + nB
    n == 0 && return MomentAccumulator(maxp)
    nA == 0 && return B
    nB == 0 && return A

    δ = B.mean - A.mean
    μ = A.mean + δ * nB / n
    M = zeros(Float64, maxp)
    @inbounds for p in 2:maxp
        s = A.M[p] + B.M[p]
        for k in 1:(p - 2)
            c = binomial(p, k)
            s += c * (A.M[p - k] * (-nB * δ / n)^k + B.M[p - k] * (nA * δ / n)^k)
        end
        s += δ^p * nA * nB * (nA^(p - 1) - (-nB)^(p - 1)) / n^p
        M[p] = s
    end
    return MomentAccumulator(n, μ, M)
end

addpoint(A::MomentAccumulator, x::Real) =
    mergeaccumulators(A, MomentAccumulator(1, Float64(x), zeros(Float64, maxpower(A))))

addbatch(A::MomentAccumulator, xs::AbstractVector{<:Real}) =
    mergeaccumulators(A, batchaccumulator(xs, maxpower(A)))

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
