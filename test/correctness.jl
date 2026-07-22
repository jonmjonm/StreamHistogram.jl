# End-to-end correctness tests: does StreamHist's output agree with
# independently-computed ground truth (StatsBase on the raw data, known
# analytic distribution properties, a directly-built reference Histogram/Ash)
# -- as opposed to runtests.jl's tests of the internal mechanics
# (merge/addpoint/batchaccumulator agreement, API surface, error paths).

import StatsBase
using StatsBase: fit, Histogram  # StatsBase also exports skewness/kurtosis,
                                  # which would clash with StreamHistogram's
                                  # own -- call those as StatsBase.skewness/kurtosis
using AverageShiftedHistograms
using QuadGK: quadgk

@testset "moments agree with StatsBase computed on the same raw data" begin
    Random.seed!(11)
    data = randn(50_000) .* 2.5 .+ 7.0

    oh = StreamHist(binRange=(-20.0, 40.0), binNum=200)
    add!(oh, data)
    finalize!(oh)

    @test mean(oh) ≈ mean(data) rtol=1e-10
    @test variance(oh) ≈ var(data, corrected=false) rtol=1e-8
    @test std(oh) ≈ std(data, corrected=false) rtol=1e-8
    @test skewness(oh) ≈ StatsBase.skewness(data) rtol=1e-4
    # StatsBase.kurtosis is *excess* kurtosis (kurtosis - 3); ours is raw.
    @test kurtosis(oh) ≈ StatsBase.kurtosis(data) + 3 rtol=1e-3
end

@testset "known-distribution moments: Uniform(0,1)" begin
    Random.seed!(12)
    data = rand(200_000)  # Uniform(0,1): mean=1/2, var=1/12, skew=0, raw kurtosis=9/5

    oh = StreamHist(binRange=(0.0, 1.0), binNum=100, momentPowers=[1, 2, 3, 4])
    add!(oh, data)
    finalize!(oh)

    @test mean(oh) ≈ 0.5 atol=0.01
    @test variance(oh) ≈ 1 / 12 atol=0.001
    @test moment(oh, 3) ≈ 0.0 atol=0.01          # symmetric -> odd central moment ~0
    @test kurtosis(oh) ≈ 9 / 5 atol=0.05
end

@testset "known-distribution moments: standard Normal" begin
    Random.seed!(13)
    data = randn(200_000)  # mean=0, var=1, skew=0, raw kurtosis=3

    oh = StreamHist(binRange=(-6.0, 6.0), binNum=300)
    add!(oh, data)
    finalize!(oh)

    @test mean(oh) ≈ 0.0 atol=0.02
    @test variance(oh) ≈ 1.0 atol=0.02
    @test skewness(oh) ≈ 0.0 atol=0.05
    @test kurtosis(oh) ≈ 3.0 atol=0.1
end

@testset "exactHistogram matches a directly-built reference Histogram, bin-for-bin" begin
    Random.seed!(14)
    data = randn(30_000) .* 3 .+ 5.0
    edges = -10.0:0.2:20.0

    reference = fit(Histogram, data, edges)

    # Feed the same data through StreamHist three different ways: as one
    # batch, one point at a time, and mixed batches -- all must agree with
    # the ground-truth histogram exactly (integer bin counts), since binning
    # is deterministic regardless of how the points arrived.
    oh_batch = StreamHist(bins=collect(edges))
    add!(oh_batch, data)

    oh_scalar = StreamHist(bins=collect(edges))
    for x in data
        add!(oh_scalar, x)
    end

    oh_mixed = StreamHist(bins=collect(edges))
    add!(oh_mixed, data[1:10_000])
    for x in data[10_001:20_000]
        add!(oh_mixed, x)
    end
    add!(oh_mixed, data[20_001:end])

    for oh in (oh_batch, oh_scalar, oh_mixed)
        finalize!(oh)
        h = exactHistogram(oh)
        @test h.weights == reference.weights
        @test sum(h.weights) == length(data)
    end
end

@testset "incrementally-batched ASH matches a directly-built reference ASH" begin
    Random.seed!(15)
    data = randn(40_000) .* 2 .+ 3.0
    rng = -10.0:0.02:16.0

    reference = ash(data; rng=rng, kernel=AverageShiftedHistograms.Kernels.biweight, m=5)

    oh = StreamHist(bins=collect(-10.0:2.0:16.0), ashNGrid=length(rng) - 1, ashBatchSize=333)
    # deliberately uneven batch sizes, mixing scalar and batch add!, to
    # stress the ashPending flush logic against a range that doesn't line
    # up evenly with ashBatchSize
    add!(oh, data[1:7])
    for x in data[8:5000]
        add!(oh, x)
    end
    add!(oh, data[5001:end])
    finalize!(oh)

    @test oh.ash.rng == reference.rng
    @test nobs(oh.ash) == nobs(reference)
    @test oh.ash.density ≈ reference.density rtol=1e-8
end

@testset "density integrates to ~1 and moments computed from it match the accumulator" begin
    Random.seed!(16)
    data = randn(60_000) .* 1.5 .+ 2.0

    oh = StreamHist(binRange=(-8.0, 12.0), binNum=200)
    add!(oh, data)
    finalize!(oh)

    d = density(oh)
    total_mass = first(quadgk(d, -8.0, 12.0))
    @test total_mass ≈ 1.0 atol=1e-3

    relerrs = densityQuality(oh)
    @test all(relerrs .< 0.1)
end

@testset "learn phase loses no points and matches an equivalent binRange run" begin
    Random.seed!(17)
    data = randn(20_000) .* 3 .+ 1.0

    # Feed exactly learnLength points first, as their own add! call, so the
    # range is fixed from precisely that subset -- a single add! call with
    # a batch bigger than learnLength buffers (and learns from) the whole
    # batch at once, per the "collects until *at least* learnLength" spec.
    oh_learn = StreamHist(learn=true, learnLength=2_000, binNum=150, paddingPct=0.0)
    add!(oh_learn, data[1:2_000])
    add!(oh_learn, data[2_001:end])
    finalize!(oh_learn)

    lo, hi = extrema(data[1:2_000])
    oh_direct = StreamHist(binRange=(lo, hi), binNum=150)
    add!(oh_direct, data)
    finalize!(oh_direct)

    # every point that was ever added must be reflected in nobs/moments,
    # even though only the first 2000 informed the range
    @test nobs(oh_learn) == length(data)
    @test mean(oh_learn) ≈ mean(data) rtol=1e-10
    @test variance(oh_learn) ≈ var(data, corrected=false) rtol=1e-8

    # points beyond the (unpadded) learned range are tallied as
    # under/overflow, not dropped or clamped -- the exact invariant holds
    o = outofrange(oh_learn)
    @test sum(exactHistogram(oh_learn).weights) + o.underflow + o.overflow == length(data)
    @test o.underflow + o.overflow > 0  # near-certain with 18000 more draws from the same distribution

    # and the resulting histogram (and overflow accounting) must be
    # identical to fixing the same (unpadded) range directly from
    # construction
    @test exactHistogram(oh_learn).weights == exactHistogram(oh_direct).weights
    @test outofrange(oh_learn) == outofrange(oh_direct)
end

@testset "integer mode counts match an exact manual tally" begin
    Random.seed!(18)
    data = rand(1:20, 10_000)

    oh = StreamHist(integer=true, binRange=(1, 20))
    add!(oh, data)
    finalize!(oh)

    h = exactHistogram(oh)
    for k in 1:20
        @test h.weights[searchsortedlast(h.edges[1], k)] == count(==(k), data)
    end
    @test sum(h.weights) == length(data)
end

@testset "degenerate zero-variance data doesn't crash (NaN, not an error)" begin
    oh = StreamHist(binRange=(-1.0, 1.0), binNum=10)
    add!(oh, fill(0.5, 100))
    finalize!(oh)

    @test mean(oh) == 0.5
    @test variance(oh) == 0.0
    @test isnan(skewness(oh))  # 0/0
    @test isnan(kurtosis(oh))  # 0/0
end
