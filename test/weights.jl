# Weighted points: the weighted tallies kept beside the unweighted ones.
using QuadGK

"A finalized histogram of `xs` (with weights `ws`, if given) on a fixed range."
function whist(xs, ws=nothing; kw...)
    oh = StreamHist(; learn=false, binRange=(-5.0, 5.0), binNum=40, kw...)
    ws === nothing ? add!(oh, xs) : add!(oh, xs, ws)
    return finalize!(oh)
end

gridpoints(oh) = collect(oh.grid)

@testset "weights: no weights given means the weighted results equal the unweighted ones" begin
    xs = randn(3000)
    oh = whist(xs)
    @test totalweight(oh) == nobs(oh) == 3000
    @test effectivesamplesize(oh) ≈ 3000
    @test weightedHistogram(oh).weights == exactHistogram(oh).weights
    @test weightedDensity(oh).(gridpoints(oh)) == density(oh).(gridpoints(oh))
    for p in oh.momentPowers
        @test weightedMoment(oh, p) ≈ moment(oh, p) rtol=1e-10
    end
    @test all(isequal(1.0), filter(!isnan, meanWeight(oh).meanweight))
    @test outofrangeweight(oh) == (underflow=0.0, overflow=0.0)
end

@testset "weights: an integer weight k is the same as adding the point k times" begin
    xs = randn(2000) .* 1.5
    ks = rand(0:3, 2000)                                    # including weight 0: the point adds no mass
    rep = reduce(vcat, [fill(x, k) for (x, k) in zip(xs, ks)])
    a, b = whist(xs, ks), whist(rep)
    @test totalweight(a) == nobs(b) == sum(ks)
    @test nobs(a) == 2000                                   # nobs still counts points
    @test weightedHistogram(a).weights == exactHistogram(b).weights
    @test weightedDensity(a).(gridpoints(a)) ≈ density(b).(gridpoints(b)) rtol=1e-10
    for p in a.momentPowers
        @test weightedMoment(a, p) ≈ moment(b, p) rtol=1e-8
    end
    @test weightedSkewness(a) ≈ skewness(b) rtol=1e-8
    @test weightedKurtosis(a) ≈ kurtosis(b) rtol=1e-8
    @test weightedStd(a) ≈ std(b) rtol=1e-10

    # the same through the learn phase (weights >= 1, so both buffers have the same extrema)
    ks = rand(1:3, 2000)
    rep = reduce(vcat, [fill(x, k) for (x, k) in zip(xs, ks)])
    a, b = StreamHist(learnLength=500), StreamHist(learnLength=500)
    add!(a, xs[1:300], ks[1:300])                           # still learning
    @test !isinitialized(a)
    add!(a, xs[301:end], ks[301:end])
    add!(b, xs)                                             # the same range: learned from the same points
    @test exactHistogram(a).edges == finalize!(b).hist.edges
    finalize!(a)
    @test totalweight(a) == sum(ks)
    c = StreamHist(learn=false, bins=collect(only(exactHistogram(a).edges)))   # a's learned bins, fixed
    add!(c, rep)
    finalize!(c)
    @test weightedHistogram(a).weights == exactHistogram(c).weights
    @test a.grid ≈ c.grid
    @test weightedDensity(a).(gridpoints(a)) ≈ density(c).(gridpoints(c)) rtol=1e-8
end

@testset "weights: scaling every weight changes only the totals" begin
    xs, ws = randn(2000), rand(2000)
    a, b = whist(xs, ws), whist(xs, 7.5 .* ws)
    @test totalweight(b) ≈ 7.5 * totalweight(a)
    @test effectivesamplesize(b) ≈ effectivesamplesize(a)
    @test weightedHistogram(b).weights ≈ 7.5 .* weightedHistogram(a).weights
    @test weightedDensity(b).(gridpoints(b)) ≈ weightedDensity(a).(gridpoints(a)) rtol=1e-10
    for p in a.momentPowers
        @test weightedMoment(b, p) ≈ weightedMoment(a, p) rtol=1e-8
    end
    @test meanWeight(b).meanweight[.!isnan.(meanWeight(b).meanweight)] ≈
          7.5 .* meanWeight(a).meanweight[.!isnan.(meanWeight(a).meanweight)]
end

@testset "weights: one point at a time equals one batch, and the moments match the direct formulas" begin
    xs, ws = randn(1500) .* 2 .+ 10, rand(1500) .^ 3          # skewed weights
    a = whist(xs, ws)
    b = StreamHist(learn=false, binRange=(-5.0, 5.0), binNum=40)
    for (x, w) in zip(xs, ws)
        add!(b, x, w)
    end
    finalize!(b)
    @test weightedHistogram(b).weights ≈ weightedHistogram(a).weights
    @test weightedDensity(b).(gridpoints(b)) ≈ weightedDensity(a).(gridpoints(a))
    @test totalweight(b) ≈ totalweight(a)
    W = sum(ws)
    μ = sum(ws .* xs) / W
    @test weightedMean(a) ≈ μ && weightedMean(b) ≈ μ
    for p in 2:8                                             # every order tracked (up to 8, the largest power)
        direct = sum(ws .* (xs .- μ) .^ p) / W
        @test StreamHistogram.centralmoment(a.wmoments, p) ≈ direct rtol=1e-6
        @test StreamHistogram.centralmoment(b.wmoments, p) ≈ direct rtol=1e-6
    end
    @test weightedMoments(a) == Dict(p => weightedMoment(a, p) for p in a.momentPowers)
    @test weightedVariance(a) ≈ sum(ws .* (xs .- μ) .^ 2) / W
    @test effectivesamplesize(a) ≈ W^2 / sum(abs2, ws)
    @test effectivesamplesize(a) < nobs(a)
end

@testset "weights: weighted = count x mean weight (exactly in the bins, and for the smoothed density)" begin
    xs = randn(4000)
    ws = exp.(0.8 .* xs)                                     # the weights depend on the value
    oh = whist(xs, ws)
    counts, wsum = exactHistogram(oh).weights, weightedHistogram(oh).weights
    mw = meanWeight(oh)
    @test mw.edges == collect(only(exactHistogram(oh).edges))
    full = counts .> 0
    @test wsum[full] ≈ counts[full] .* mw.meanweight[full]
    @test all(isnan, mw.meanweight[.!full])
    @test issorted(filter(!isnan, mw.meanweight)[2:end-1])  # increasing with x, as exp(0.8x) is

    # The smoothed version: count density times the kernel-smoothed mean weight (Nadaraya-Watson), on the grid.
    rng, K, m = oh.grid, oh.kernel, oh.m
    smooth(v) = [sum(v[k] * K((i - k) / m) for k in max(1, i - m + 1):min(length(v), i + m - 1)) for i in eachindex(v)]
    Kn, Ks = smooth(Float64.(oh.gridCounts)), smooth(oh.gridWeights)
    keep = Kn .> 0
    prod = zeros(length(Kn))
    prod[keep] = (Kn ./ (sum(Kn) * step(rng)))[keep] .* (Ks[keep] ./ Kn[keep])
    prod ./= sum(prod) * step(rng)
    @test weightedDensity(oh).(collect(rng)) ≈ prod rtol=1e-10
    @test density(oh).(collect(rng)) ≈ Kn ./ (sum(Kn) * step(rng)) rtol=1e-10

    # the weighted density moves toward large x; the unweighted one does not
    @test weightedMean(oh) > mean(oh) + 0.5
    @test first(quadgk(x -> x * weightedDensity(oh)(x), -5, 5)) ≈ weightedMean(oh) rtol=0.02
    @test all(<(0.02), weightedDensityQuality(oh)[1:2])
    h = weightedHistogram(oh, [-5.0, 0.0, 5.0])
    @test sum(h.weights) ≈ totalweight(oh) rtol=1e-6
end

@testset "weights: out of range, integer mode, zero total weight" begin
    oh = StreamHist(binRange=(0.0, 1.0), binNum=4)
    add!(oh, [-1.0, 0.5, 0.6, 2.0, 3.0], [2.0, 1.0, 0.5, 4.0, 0.25])
    finalize!(oh)
    @test outofrange(oh) == (underflow=1, overflow=2)
    @test outofrangeweight(oh) == (underflow=2.0, overflow=4.25)
    @test sum(weightedHistogram(oh).weights) + 2.0 + 4.25 ≈ totalweight(oh)
    @test sum(weightedHistogram(oh, [0.0, 0.5, 1.0]).weights) ≈ 1.5 rtol=1e-6   # the in-range weight
    @test sum(histogram(oh, [0.0, 0.5, 1.0]).weights) ≈ 2 rtol=1e-6            # the points in range

    ints = StreamHist(integer=true, binRange=(1, 4))
    add!(ints, [1, 2, 2, 3, 4, 4], [0.5, 1.0, 1.0, 2.0, 0.25, 0.25])
    finalize!(ints)
    @test weightedHistogram(ints).weights == [0.5, 2.0, 2.0, 0.5]
    @test exactHistogram(ints).weights == [1, 2, 1, 2]
    @test meanWeight(ints).meanweight == [0.5, 1.0, 2.0, 0.25]
    @test_throws ArgumentError weightedDensity(ints)

    z = StreamHist(binRange=(0.0, 1.0), binNum=4)
    add!(z, [0.2, 0.7], [0.0, 0.0])
    finalize!(z)
    @test totalweight(z) == 0 && nobs(z) == 2 && effectivesamplesize(z) == 0
    @test isnan(weightedMean(z))
    @test all(isnan, weightedDensityQuality(z))
    @test weightedHistogram(z, [0.0, 1.0]).weights == [0.0]
    @test exactHistogram(z).weights == [1, 0, 1, 0]          # the points still count
end

@testset "weights: bad weights are refused before anything is added" begin
    oh = StreamHist(binRange=(0.0, 1.0))
    add!(oh, 0.5, 2.0)
    for bad in (-1.0, NaN, Inf)
        @test_throws ArgumentError add!(oh, 0.3, bad)
        @test_throws ArgumentError add!(oh, [0.3, 0.4], [1.0, bad])
    end
    @test_throws DimensionMismatch add!(oh, [0.3, 0.4], [1.0])
    @test nobs(oh) == 1 && totalweight(oh) == 2.0
    @test_throws ArgumentError weightedMoment(oh, 3)          # not a requested power
end

@testset "weights: the weighted moment accumulator" begin
    MA, bacc, merge = StreamHistogram.MomentAccumulator, StreamHistogram.batchaccumulator,
                      StreamHistogram.mergeaccumulators
    xs, ws = randn(3000) .* 4 .+ 1e3, rand(3000)            # far from zero, to stress cancellation
    whole = bacc(xs, ws, 8)
    @test whole isa MA{Float64} && whole.n ≈ sum(ws)
    acc = MA(Float64, 8)                                     # merging chunks = one batch
    for r in (1:700, 701:1600, 1601:1601, 1602:3000)
        acc = merge(acc, bacc(xs[r], ws[r], 8))
    end
    @test acc.n ≈ whole.n && acc.mean ≈ whole.mean
    @test acc.M[2:8] ≈ whole.M[2:8] rtol=1e-8
    one = MA(Float64, 8)                                     # point by point = one batch
    for (x, w) in zip(xs, ws)
        one = StreamHistogram.addpoint(one, x, w)
    end
    @test one.M[2:8] ≈ whole.M[2:8] rtol=1e-6
    # an integer weight k = the point k times; weight 0 = nothing; all weights 1 = the unweighted accumulator
    @test StreamHistogram.addpoint(MA(Float64, 4), 2.5, 3).M ≈
          foldl((A, _) -> StreamHistogram.addpoint(A, 2.5), 1:3; init=MA(Float64, 4)).M
    @test StreamHistogram.addpoint(whole, 7.0, 0) === whole
    @test bacc(xs, ones(3000), 8).M ≈ bacc(xs, 8).M rtol=1e-10
    @test bacc(xs, zeros(3000), 8).n == 0
    @test_throws MethodError merge(MA(8), MA(Float64, 8))    # counts and weights don't mix
end

@testset "weights: weights never change the unweighted results" begin
    xs, ws = randn(2500), rand(2500) .* 5
    for kw in ((learn=false, binRange=(-5.0, 5.0)), (learnLength=1000,), (learnLength=10_000,))
        a, b = StreamHist(; kw...), StreamHist(; kw...)
        add!(a, xs[1:600], ws[1:600])                       # mixed batch and scalar adds
        for i in 601:900
            add!(a, xs[i], ws[i])
        end
        add!(a, xs[901:end], ws[901:end])
        add!(b, xs)
        finalize!(a); finalize!(b)
        @test exactHistogram(a).edges == exactHistogram(b).edges
        @test exactHistogram(a).weights == exactHistogram(b).weights
        @test density(a).(gridpoints(a)) ≈ density(b).(gridpoints(b)) rtol=1e-10
        @test all(moment(a, p) ≈ moment(b, p) for p in a.momentPowers)
        @test nobs(a) == nobs(b) && datarange(a) == datarange(b) && outofrange(a) == outofrange(b)
        @test totalweight(a) ≈ sum(ws) && effectivesamplesize(a) ≈ sum(ws)^2 / sum(abs2, ws)
        # and the weighted results are the same however the points arrived
        c = StreamHist(; kw...)                              # all in one batch
        add!(c, xs, ws)
        finalize!(c)
        @test weightedHistogram(a).weights ≈ weightedHistogram(c).weights
        @test weightedMean(a) ≈ weightedMean(c)
    end
end

@testset "weights: integer=:auto decides from the values, not the weights" begin
    a = StreamHist(integer=:auto, learnLength=50)
    add!(a, collect(1:60), fill(0.5, 60))
    @test isinitialized(a) && a.integer
    @test weightedHistogram(finalize!(a)).weights == 0.5 .* exactHistogram(a).weights
    b = StreamHist(integer=:auto, learnLength=50)
    add!(b, collect(1:60) ./ 7, fill(2, 60))
    @test isinitialized(b) && !b.integer
end

@testset "weights: distances between weighted densities" begin
    xs = randn(4000)
    a = whist(xs, exp.(0.8 .* xs))                           # reweighted toward large x
    b = whist(xs)
    c = whist(xs .+ 0.64)                                    # the same tilt, as a shift (exact for a Gaussian)
    @test tv_distance(a, b) ≈ 0 atol=1e-10                   # unweighted: the same points
    @test tv_distance(a, b; weighted=true) > 0.2
    @test tv_distance(a, c; weighted=true) < tv_distance(a, b; weighted=true) / 3
    @test lp_distance(a, b; p=2, weighted=true) ≈ lp_distance(b, a; p=2, weighted=true)
    z = whist([0.1, 0.2], [0.0, 0.0])                        # points in range, but no weight
    @test tv_distance(z, b) ≥ 0
    @test_throws ArgumentError tv_distance(z, b; weighted=true)
    @test_throws ArgumentError tv_distance(StreamHist(integer=true, binRange=(1, 3)), b)
    @test_throws ArgumentError StreamHist(m=0)
end
