# quantile and weightedQuantile: estimated from the traditional histogram's bins and the out-of-range tallies.
using Statistics: quantile

@testset "quantile: close to the sample quantiles, exact at 0 and 1" begin
    Random.seed!(51)
    xs = randn(20_000)
    oh = StreamingHistogram(binRange=(-5.0, 5.0), binNum=200)             # bins 0.05 wide
    add!(oh, xs)
    ps = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    q = quantile(oh, ps)
    @test q[1] == minimum(xs) && q[end] == maximum(xs)
    @test all(abs.(q .- quantile(xs, ps)) .< 0.05)                 # within a bin
    @test issorted(q)
    @test quantile(oh, 0.5) == q[5]                                 # the vector form is the scalar one
    @test issorted(quantile(oh, range(0, 1; length=1001)))
end

@testset "quantile: interpolated evenly within a bin" begin
    oh = StreamingHistogram(binRange=(0.0, 4.0), binNum=4)
    add!(oh, [0.5, 1.2, 1.4, 1.6, 2.5, 3.5])                        # bins hold 1, 3, 1, 1
    @test quantile(oh, 1 / 6) == 1.0                                # the top of the first bin
    @test quantile(oh, 2.5 / 6) ≈ 1.5                               # half way through the second
    @test quantile(oh, 4 / 6) == 2.0
end

@testset "quantile: among the out-of-range points, interpolated across all of them" begin
    oh = StreamingHistogram(binRange=(10.0, 95.0), binNum=50)
    add!(oh, collect(1.0:100.0))                                    # 9 below 10; 6 at or above 95
    @test outofrange(oh) == (underflow=9, overflow=6)
    low = quantile(oh, 0.03)                                         # rank 3 of the 9 below
    @test 1.0 <= low < 10.0 && low ≈ 1.0 + (3 / 9) * 9.0
    high = quantile(oh, 0.97)
    @test 95.0 < high <= 100.0
    @test quantile(oh, 0.09) == 10.0                                 # the last underflow rank: the bottom edge
end

@testset "quantile in integer mode: the integer of the bin (the inverse CDF)" begin
    Random.seed!(52)
    xs = rand(1:9, 5000)
    oh = StreamingHistogram(integer=true, binRange=(1, 9))
    add!(oh, xs)
    sorted = sort(xs)
    for p in (0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999)
        @test quantile(oh, p) == sorted[ceil(Int, p * length(xs))]
    end
    @test all(isinteger, quantile(oh, range(0, 1; length=101)))
    out = StreamingHistogram(integer=true, binRange=(1, 5))
    add!(out, [1, 2, 3, 8, 9])
    @test isinteger(quantile(out, 0.9)) && 5 <= quantile(out, 0.9) <= 9   # out of range: still an integer
end

@testset "weightedQuantile" begin
    xs = collect(0.5:1.0:99.5)
    tilted = StreamingHistogram(binRange=(0.0, 100.0), binNum=100)
    add!(tilted, xs, xs)                       # weight ∝ x: the weighted CDF is (x/100)², the median 100/√2
    @test isapprox(weightedQuantile(tilted, 0.5), 100 / sqrt(2); atol=0.5)
    @test isapprox(quantile(tilted, 0.5), 50; atol=0.5)
    @test weightedQuantile(tilted, [0.0, 1.0]) == [0.5, 99.5]

    equal = StreamingHistogram(binRange=(0.0, 100.0), binNum=100)
    add!(equal, xs, fill(2.5, length(xs)))
    ps = range(0, 1; length=41)
    @test weightedQuantile(equal, ps) ≈ quantile(equal, ps)

    # an integer weight k is the point added k times, out-of-range points included
    Random.seed!(53)
    ys = 3 .* randn(500)
    ks = rand(1:4, 500)
    a = StreamingHistogram(binRange=(-4.0, 4.0), binNum=40)
    add!(a, ys, ks)
    b = StreamingHistogram(binRange=(-4.0, 4.0), binNum=40)
    add!(b, reduce(vcat, fill(y, k) for (y, k) in zip(ys, ks)))
    @test sum(outofrange(a)) > 0
    @test weightedQuantile(a, ps) ≈ quantile(b, ps)
end

@testset "quantile: NaN with nothing to take it of; bad p; the learn phase" begin
    empty = StreamingHistogram(binRange=(0.0, 1.0))
    @test isnan(quantile(empty, 0.5)) && isnan(weightedQuantile(empty, 0.5))
    zero = StreamingHistogram(binRange=(0.0, 1.0))
    add!(zero, [0.2, 0.7], [0.0, 0.0])
    @test isnan(weightedQuantile(zero, 0.5)) && quantile(zero, 1.0) == 0.7
    @test_throws ArgumentError quantile(zero, 1.5)
    @test_throws ArgumentError weightedQuantile(zero, -0.1)
    learning = StreamingHistogram(learnLength=100)
    add!(learning, randn(10))
    @test_throws ArgumentError quantile(learning, 0.5)
    finalize!(learning)
    @test quantile(learning, 0.0) == datarange(learning)[1]
end
