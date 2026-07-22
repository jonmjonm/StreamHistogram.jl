using Test
using Random
using Statistics
using StreamHistogram

Random.seed!(1234)

@testset "MomentAccumulator vs naive (Float64 batch)" begin
    xs = randn(5000) .* 3 .+ 100.0  # away from zero, to stress the naive formula
    maxp = 8
    A = StreamHistogram.batchaccumulator(xs, maxp)
    μ = mean(xs)
    @test A.mean ≈ μ
    for p in 2:maxp
        naive = sum((x - μ)^p for x in xs)
        @test A.M[p] ≈ naive rtol=1e-5
    end
end

@testset "merge equals whole-batch accumulator" begin
    xs = randn(3000) .* 2 .+ 50.0
    maxp = 8
    whole = StreamHistogram.batchaccumulator(xs, maxp)

    # split into several chunks, merge sequentially
    chunks = [xs[1:500], xs[501:1200], xs[1201:2000], xs[2001:end]]
    acc = StreamHistogram.MomentAccumulator(maxp)
    for c in chunks
        acc = StreamHistogram.mergeaccumulators(acc, StreamHistogram.batchaccumulator(c, maxp))
    end

    @test acc.n == whole.n
    @test acc.mean ≈ whole.mean rtol=1e-10
    for p in 2:maxp
        @test acc.M[p] ≈ whole.M[p] rtol=1e-5
    end
end

@testset "merge order independence (parallel-style)" begin
    xs = randn(2000) .+ 10.0
    maxp = 6
    a = StreamHistogram.batchaccumulator(xs[1:800], maxp)
    b = StreamHistogram.batchaccumulator(xs[801:1500], maxp)
    c = StreamHistogram.batchaccumulator(xs[1501:end], maxp)

    left = StreamHistogram.mergeaccumulators(StreamHistogram.mergeaccumulators(a, b), c)
    right = StreamHistogram.mergeaccumulators(a, StreamHistogram.mergeaccumulators(b, c))

    @test left.n == right.n
    @test left.mean ≈ right.mean rtol=1e-10
    for p in 2:maxp
        @test left.M[p] ≈ right.M[p] rtol=1e-5
    end
end

@testset "single-point add matches batch add" begin
    xs = randn(500) .+ 5.0
    maxp = 4
    viaPoints = StreamHistogram.MomentAccumulator(maxp)
    for x in xs
        viaPoints = StreamHistogram.addpoint(viaPoints, x)
    end
    viaBatch = StreamHistogram.batchaccumulator(xs, maxp)

    @test viaPoints.n == viaBatch.n
    @test viaPoints.mean ≈ viaBatch.mean rtol=1e-10
    for p in 2:maxp
        @test viaPoints.M[p] ≈ viaBatch.M[p] rtol=1e-5
    end
end

@testset "StreamHist basic add!/finalize! with binRange given" begin
    oh = StreamHist(binRange=(-5.0, 5.0), binNum=20)
    @test isinitialized(oh)  # initialized immediately since binRange given

    data = randn(2000)
    add!(oh, data)
    @test nobs(oh) == 2000
    @test mean(oh) ≈ mean(data) atol=1e-9
    @test variance(oh) ≈ var(data, corrected=false) rtol=1e-5

    mn, mx = datarange(oh)
    @test mn == minimum(data)
    @test mx == maximum(data)

    finalize!(oh)
    h = exactHistogram(oh)
    @test sum(h.weights) == 2000
    @test length(h.weights) == 20

    d = density(oh)
    @test d(0.0) > 0
    @test d(100.0) == 0.0  # far outside range

    # add! after finalize un-finalizes
    add!(oh, 0.5)
    @test !oh.finalized
    finalize!(oh)
    @test oh.finalized
    @test nobs(oh) == 2001
end

@testset "learn phase auto-ranging" begin
    oh = StreamHist(learn=true, learnLength=1000, binNum=25, paddingPct=0.1)
    @test !isinitialized(oh)

    data1 = randn(999) .* 2 .+ 3.0
    add!(oh, data1)
    @test !isinitialized(oh)  # still short of learnLength
    @test nobs(oh) == 999      # moments/min/max still tracked during learn

    add!(oh, [3.0])  # crosses the threshold
    @test isinitialized(oh)
    @test nobs(oh) == 1000

    mn, mx = datarange(oh)
    h = exactHistogram(oh)
    @test first(h.edges[1]) < mn   # padded below observed min
    @test last(h.edges[1]) > mx    # padded above observed max

    # further points feed straight into hist/ash
    add!(oh, randn(500) .* 2 .+ 3.0)
    @test nobs(oh) == 1500
    @test sum(exactHistogram(oh).weights) == 1500
end

@testset "finalize! forces early range init" begin
    oh = StreamHist(learn=true, learnLength=10_000)
    add!(oh, randn(50))
    @test !isinitialized(oh)
    finalize!(oh)
    @test isinitialized(oh)
    @test oh.finalized
    @test nobs(oh) == 50
end

@testset "integer mode" begin
    oh = StreamHist(integer=true, binRange=(1, 10))
    @test isinitialized(oh)
    add!(oh, [1, 1, 2, 10, 10, 10, 5])
    finalize!(oh)
    h = exactHistogram(oh)
    @test sum(h.weights) == 7
    @test_throws ArgumentError density(oh)
end

@testset "densityQuality on a Gaussian" begin
    oh = StreamHist(binRange=(-6.0, 6.0), binNum=200, momentPowers=[1, 2, 4])
    add!(oh, randn(20_000))
    finalize!(oh)
    relerrs = densityQuality(oh)
    @test length(relerrs) == length(oh.momentPowers)
    @test all(relerrs .<= 0.15)
end

@testset "histogram(oh, bins) integrates density into custom bins" begin
    oh = StreamHist(binRange=(-6.0, 6.0), binNum=200)
    add!(oh, randn(20_000))
    finalize!(oh)
    custom = collect(-6.0:1.0:6.0)
    res = histogram(oh, custom)
    @test length(res.weights) == length(custom) - 1
    @test sum(res.weights) ≈ nobs(oh) rtol=0.05
end

@testset "moment()/moments() respect momentPowers restriction" begin
    oh = StreamHist(binRange=(-5.0, 5.0), momentPowers=[1, 2])
    add!(oh, randn(100))
    @test_throws ArgumentError moment(oh, 4)
    m = moments(oh)
    @test Set(keys(m)) == Set([1, 2])
end
