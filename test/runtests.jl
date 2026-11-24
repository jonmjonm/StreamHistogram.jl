using Test
using Random
using Statistics
using StreamHistogram
using QuadGK

Random.seed!(1234)

@testset "tv_distance integrates ASH densities over their supports" begin
    function finalized_hist(data, range)
        oh = StreamHist(learn=false, binRange=range, binNum=40)
        add!(oh, data)
        finalize!(oh)
        return oh
    end

    left = finalized_hist(collect(range(0.05, 0.95; length=400)), (0.0, 1.0))
    same = finalized_hist(collect(range(0.05, 0.95; length=400)), (0.0, 1.0))
    right = finalized_hist(collect(range(2.05, 2.95; length=400)), (2.0, 3.0))

    @test tv_distance(left, left) ≈ 0.0 atol=1e-10
    @test tv_distance(left, same) ≈ 0.0 atol=1e-10

    fleft, fright = density(left), density(right)
    expected = first(QuadGK.quadgk(x -> abs(fleft(x) - fright(x)), 0.0, 3.0)) / 2
    @test tv_distance(left, right) ≈ expected atol=1e-8
    @test tv_distance(left, right; weighted=true) ≈ tv_distance(left, right) atol=1e-10   # no weights given
    @test lp_distance(left, right; p=1) ≈ 2 * tv_distance(left, right) atol=1e-10
    expected_l2 = sqrt(first(QuadGK.quadgk(x -> abs(fleft(x) - fright(x))^2, 0.0, 3.0)))
    @test lp_distance(left, right; p=2) ≈ expected_l2 atol=1e-8
    @test_throws ArgumentError lp_distance(left, right; p=0)
    @test tv_distance(left, right) ≈ tv_distance(right, left) atol=1e-10
    @test_throws ArgumentError tv_distance(StreamHist(learn=false, binRange=(0.0, 1.0)), left)
end

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

@testset "specialized addpoint matches general mergeaccumulators" begin
    xs = randn(400) .* 4 .+ 7.0
    maxp = 8
    viaSpecialized = StreamHistogram.MomentAccumulator(maxp)
    viaGeneral = StreamHistogram.MomentAccumulator(maxp)
    for x in xs
        viaSpecialized = StreamHistogram.addpoint(viaSpecialized, x)
        trivialB = StreamHistogram.MomentAccumulator(1, Float64(x), zeros(Float64, maxp), viaGeneral.binom)
        viaGeneral = StreamHistogram.mergeaccumulators(viaGeneral, trivialB)
    end

    @test viaSpecialized.n == viaGeneral.n
    @test viaSpecialized.mean ≈ viaGeneral.mean rtol=1e-12
    for p in 2:maxp
        @test viaSpecialized.M[p] ≈ viaGeneral.M[p] rtol=1e-10
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

    # points can be added after reading; the density read next includes them
    before = density(oh)(0.5)
    add!(oh, fill(0.5, 500))
    @test nobs(oh) == 2500
    @test density(oh)(0.5) > before
    @test finalize!(oh) === oh                    # a no-op once the range is fixed
end

@testset "single-point add! and batch add! give the same grid and density" begin
    data = randn(3000)
    a, b = StreamHist(binRange=(-6.0, 6.0), binNum=100), StreamHist(binRange=(-6.0, 6.0), binNum=100)
    foreach(x -> add!(a, x), data)
    add!(b, data)
    @test a.gridCounts == b.gridCounts && sum(a.gridCounts) == count(x -> -6.0 <= x <= 6.0, data)
    @test densityGrid(a).density == densityGrid(b).density
    @test density(a)(0.0) > 0                     # readable straight away: no finalize! needed with a range
end

@testset "density/histogram/densityQuality need the range fixed (finalize! ends the learn phase)" begin
    oh = StreamHist(learnLength=10_000, binNum=50)
    add!(oh, randn(1000))
    @test !isinitialized(oh)
    for read in (density, weightedDensity, densityQuality, densityGrid, exactHistogram, weightedHistogram, meanWeight)
        @test_throws ArgumentError read(oh)
    end
    @test_throws ArgumentError histogram(oh, collect(-6.0:1.0:6.0))

    finalize!(oh)
    @test density(oh)(0.0) > 0
    @test !isempty(histogram(oh, collect(-6.0:1.0:6.0)).weights)
    @test !isempty(densityQuality(oh))
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

@testset "single-point add! finishes the learn phase like a batch add!" begin
    # Feeding points one at a time (the streaming use) must cross learnLength inside the scalar add!,
    # fix the same range, replay the buffer, and end up identical to feeding the same data as one batch.
    data = [sin(1.7i) * 3 + 0.01i for i in 1:1500]
    one = StreamHist(learn=true, learnLength=1000, binNum=30, paddingPct=0.1)
    for (k, x) in enumerate(data)
        add!(one, x)
        @test isinitialized(one) == (k >= 1000)   # initialized exactly when the buffer fills
    end
    batch = StreamHist(learn=true, learnLength=1000, binNum=30, paddingPct=0.1)
    add!(batch, data[1:1000]); add!(batch, data[1001:end])
    finalize!(one); finalize!(batch)
    @test nobs(one) == nobs(batch) == length(data)
    @test exactHistogram(one).edges == exactHistogram(batch).edges
    @test exactHistogram(one).weights == exactHistogram(batch).weights
    @test outofrange(one) == outofrange(batch)
    @test sum(exactHistogram(one).weights) + outofrange(one).underflow + outofrange(one).overflow == length(data)
    @test mean(one) ≈ mean(batch) rtol=1e-12
    @test density(one)(0.0) ≈ density(batch)(0.0) rtol=1e-10

    # integer=:auto is resolved by the scalar path too
    ints = StreamHist(integer=:auto, learnLength=50)
    for i in 1:60
        add!(ints, i % 7)
    end
    @test isinitialized(ints) && ints.integer
    floats = StreamHist(integer=:auto, learnLength=50)
    for i in 1:60
        add!(floats, i / 7)
    end
    @test isinitialized(floats) && !floats.integer
end

@testset "no point inside the range: NaN density quality, zero counts, no error" begin
    # e.g. one --bin-range applied to many fields, one of which lies entirely outside it
    oh = StreamHist(binRange=(100.0, 250.0), binNum=3)
    add!(oh, [838.0, 839.5, 840.1])
    finalize!(oh)
    @test outofrange(oh) == (underflow=0, overflow=3)
    @test exactHistogram(oh).weights == [0, 0, 0]
    @test isnan(density(oh)(125.0))
    q = densityQuality(oh)
    @test length(q) == length(oh.momentPowers) && all(isnan, q)
    @test histogram(oh, [100.0, 175.0, 250.0]).weights == [0.0, 0.0]
end

@testset "learn=false needs a range" begin
    @test_throws ArgumentError StreamHist(learn=false)
    @test isinitialized(StreamHist(learn=false, binRange=(0, 1)))
end

@testset "finalize! forces early range init" begin
    oh = StreamHist(learn=true, learnLength=10_000)
    add!(oh, randn(50))
    @test !isinitialized(oh)
    finalize!(oh)
    @test isinitialized(oh)
    @test nobs(oh) == 50
end

@testset "integer mode" begin
    oh = StreamHist(integer=true, binRange=(1, 10))
    @test isinitialized(oh)
    add!(oh, [1, 1, 2, 10, 10, 10, 5])
    finalize!(oh)
    h = exactHistogram(oh)
    @test sum(h.weights) == 7
    @test eltype(h.edges[1]) == Int
    @test h.edges[1] == 1:1:11
    @test h.weights[searchsortedlast(1:1:11, 1)] == 2  # two 1's
    @test h.weights[searchsortedlast(1:1:11, 10)] == 3  # three 10's
    @test_throws ArgumentError density(oh)
end

@testset "integer mode rejects conflicting bins/binNum" begin
    @test_throws ArgumentError StreamHist(integer=true, binRange=(1, 10), bins=[0.5, 5.5, 10.5])
    @test_throws ArgumentError StreamHist(integer=true, binRange=(1, 10), binNum=99)
    # binNum at its default is fine (not a conflict)
    @test StreamHist(integer=true, binRange=(1, 10)) isa StreamHist
end

@testset "integer=:auto detects integer-valued learn buffer" begin
    oh = StreamHist(integer=:auto, learnLength=50)
    add!(oh, collect(1:50))
    @test oh.integer
    @test !oh.integerAuto
    @test isinitialized(oh)
    @test eltype(exactHistogram(oh).edges[1]) == Int
    add!(oh, [51, 52])
    finalize!(oh)
    @test nobs(oh) == 52
    @test_throws ArgumentError density(oh)
end

@testset "integer=:auto detects non-integer learn buffer" begin
    oh = StreamHist(integer=:auto, learnLength=50)
    add!(oh, [collect(1:49); 3.5])
    @test !oh.integer
    @test !oh.integerAuto
    @test isinitialized(oh)
    add!(oh, randn(10))
    finalize!(oh)
    @test density(oh) isa Function
end

@testset "integer=:auto defers the binNum conflict check until resolution" begin
    # No conflict raised yet -- integer isn't decided until the buffer fills.
    oh = StreamHist(integer=:auto, binNum=99, learnLength=20)
    @test oh.integerAuto

    # Resolves to non-integer: binNum=99 is actually used, no conflict.
    add!(oh, randn(20))
    @test !oh.integer

    # Resolves to integer: binNum=99 conflicts, raised at resolution time.
    oh2 = StreamHist(integer=:auto, binNum=99, learnLength=20)
    @test_throws ArgumentError add!(oh2, collect(1:20))
end

@testset "integer=:auto resolves in finalize! when the learn buffer never fills" begin
    # learnLength is never reached via add!, so resolution can only happen in
    # finalize!'s early buffer-flush path.
    oh = StreamHist(integer=:auto, learnLength=1000)
    add!(oh, collect(1:10))
    @test oh.integerAuto              # not yet resolved
    finalize!(oh)
    @test oh.integer
    @test !oh.integerAuto
    @test eltype(exactHistogram(oh).edges[1]) == Int

    oh2 = StreamHist(integer=:auto, learnLength=1000)
    add!(oh2, [collect(1:9); 3.5])
    finalize!(oh2)
    @test !oh2.integer
    @test density(oh2) isa Function
end

@testset "integer=:auto requires learn=true and rejects bins/binRange" begin
    @test_throws ArgumentError StreamHist(integer=:auto, binRange=(1, 10))
    @test_throws ArgumentError StreamHist(integer=:auto, bins=[0.5, 5.5, 10.5])
    @test_throws ArgumentError StreamHist(integer=:auto, learn=false)
    @test_throws ArgumentError StreamHist(integer=:notauto)
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

include("correctness.jl")
include("weights.jl")
