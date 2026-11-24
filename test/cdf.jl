# cdf/weightedCdf, and smoothing an existing grid with another `m` or `kernel` when reading.
using QuadGK
using AverageShiftedHistograms: ash, Kernels

"A histogram of `xs` (with weights `ws`, if given) on the fixed range `lo..hi`."
function rangedhist(xs, ws=nothing; lo=-5.0, hi=5.0, kw...)
    oh = StreamHist(; learn=false, binRange=(lo, hi), binNum=40, kw...)
    ws === nothing ? add!(oh, xs) : add!(oh, xs, ws)
    return oh
end

"The integral of `f` from the start of `grid` to `x`, split at the grid points (`f` is linear between them)."
function integral(f, grid, x)
    x <= first(grid) && return 0.0
    cuts = [filter(<(x), collect(grid)); min(x, last(grid))]
    return first(quadgk(f, cuts...; rtol=1e-14))
end

const POINTS = [-6.0, -5.0, -4.99, -3.3, -1.0, -0.37, 0.0, 0.004, 1.5, 2.71, 4.999, 5.0, 7.0]

@testset "cdf is the exact integral of density, and weightedCdf of weightedDensity" begin
    Random.seed!(41)
    xs = randn(5000)
    oh = rangedhist(xs, exp.(xs))
    for (F, f) in ((cdf(oh), density(oh)), (weightedCdf(oh), weightedDensity(oh)))
        for x in POINTS
            @test F(x) ≈ integral(f, oh.grid, x) atol=1e-13
        end
        grid = collect(oh.grid)
        @test issorted(F.(grid))
        @test F(-5.0) == 0.0 && F(-100.0) == 0.0
        @test F(5.0) ≈ 1 atol=1e-12
        @test F(100.0) == F(5.0)
    end
    @test cdf(oh)(0.0) ≈ 0.5 atol=0.02                              # the median of N(0, 1)
    @test weightedCdf(oh)(1.0) ≈ 0.5 atol=0.03                      # the weights exp(x) tilt N(0, 1) to N(1, 1)
end

@testset "the density integrates to exactly 1, even when it is not 0 at the range's ends" begin
    # uniform data filling the whole range: the density is far from 0 at both ends
    oh = rangedhist(collect(0.0:0.0005:0.9995); lo=0.0, hi=1.0)      # 2000 points
    d = densityGrid(oh)
    @test first(d.density) > 0.5 && last(d.density) > 0.5
    @test integral(density(oh), oh.grid, 1.0) ≈ 1 atol=1e-13
    @test cdf(oh)(1.0) ≈ 1 atol=1e-12
    # so expected counts over edges spanning the range add up to the points in range
    @test sum(histogram(oh, [-1.0, 0.25, 0.5, 2.0]).weights) ≈ 2000 rtol=1e-12
end

@testset "histogram(oh, bins) is differences of the cdf" begin
    Random.seed!(42)
    xs = randn(3000)
    ws = rand(3000)
    oh = rangedhist([xs; [-9.0, 8.0]], [ws; [2.0, 3.0]])           # two points out of range
    edges = [-5.0, -1.5, 0.0, 0.25, 3.0, 5.0]
    F, W = cdf(oh), weightedCdf(oh)
    @test histogram(oh, edges).weights ≈ 3000 .* diff(F.(edges)) rtol=1e-14
    @test weightedHistogram(oh, edges).weights ≈ sum(ws) .* diff(W.(edges)) rtol=1e-12
    @test sum(histogram(oh, edges).weights) ≈ 3000 rtol=1e-12      # the points in range
    @test sum(weightedHistogram(oh, edges).weights) ≈ sum(ws) rtol=1e-12
end

@testset "m= and kernel= smooth the same grid as a StreamHist built with them would" begin
    Random.seed!(43)
    xs = randn(4000)
    ws = rand(4000)
    gauss = Kernels.gaussian
    a = rangedhist(xs, ws)                                          # the defaults: biweight, m = 5
    b = rangedhist(xs, ws; m=12, kernel=gauss)
    @test densityGrid(a; m=12, kernel=gauss) == densityGrid(b)
    @test densityGrid(a; weighted=true, m=12, kernel=gauss) == densityGrid(b; weighted=true)
    grid = collect(a.grid)
    for read in (density, weightedDensity, cdf, weightedCdf)
        @test read(a; m=12, kernel=gauss).(grid) == read(b).(grid)
        @test read(b; m=a.m, kernel=a.kernel).(grid) == read(a).(grid)
    end
    edges = [-5.0, -1.0, 0.0, 2.0, 5.0]
    @test histogram(a, edges; m=12, kernel=gauss) == histogram(b, edges)
    @test weightedHistogram(a, edges; m=12, kernel=gauss) == weightedHistogram(b, edges)

    # one keyword alone; and the AverageShiftedHistograms reference on this grid
    @test densityGrid(a; m=12) == densityGrid(rangedhist(xs, ws; m=12))
    @test densityGrid(a; kernel=gauss) == densityGrid(rangedhist(xs, ws; kernel=gauss))
    @test densityGrid(a; m=12, kernel=gauss).density ≈ ash(xs; rng=a.grid, m=12, kernel=gauss).density rtol=1e-12

    # reading with other options leaves the histogram as it was
    before = densityGrid(a)
    density(a; m=30); cdf(a; kernel=gauss); histogram(a, edges; m=2)
    @test a.m == 5 && a.kernel === StreamHistogram.biweight
    @test densityGrid(a) == before

    # a wider smoothing width gives a smoother, flatter estimate of the peak
    @test density(a; m=50)(0.0) < density(a; m=2)(0.0)
    @test_throws ArgumentError density(a; m=0)
    @test_throws ArgumentError cdf(a; m=-1)
end

@testset "cdf: integer mode, the learn phase, nothing in range" begin
    ints = StreamHist(integer=true, binRange=(1, 4))
    add!(ints, [1, 2, 3])
    @test_throws ArgumentError cdf(ints)
    @test_throws ArgumentError weightedCdf(ints)

    learning = StreamHist(learnLength=1000)
    add!(learning, randn(10))
    @test_throws ArgumentError cdf(learning)
    finalize!(learning)
    @test cdf(learning)(Inf) ≈ 1 atol=1e-12

    empty = StreamHist(binRange=(100.0, 250.0))
    add!(empty, [838.0, 839.5])
    @test cdf(empty)(50.0) == 0.0
    @test isnan(cdf(empty)(125.0)) && isnan(cdf(empty)(300.0))
    @test histogram(empty, [100.0, 175.0, 250.0]).weights == [0.0, 0.0]

    @test_throws ArgumentError StreamHist(ashNGrid=0)
end

@testset "the grid holds exactly the points the histogram does" begin
    lo, hi = 0.0, 1.0
    half = (hi - lo) / 500 / 2                                     # half a grid step (ashNGrid = 500)
    edge = [lo - half / 2, hi, hi + half / 2]                      # just outside, or at the top (closed=:left)
    oh = StreamHist(binRange=(lo, hi))
    add!(oh, edge, [1.0, 2.0, 3.0])
    @test outofrange(oh) == (underflow=1, overflow=2)
    @test sum(oh.gridCounts) == 0 && sum(oh.gridWeights) == 0
    @test isnan(density(oh)(0.5))                                  # no point in range: no density, as histogram says
    @test histogram(oh, [lo, hi]).weights == [0.0]

    right = StreamHist(binRange=(lo, hi), closed=:right)           # now lo is outside and hi inside
    add!(right, [lo, hi])
    @test outofrange(right) == (underflow=1, overflow=0)
    @test oh.grid == right.grid && right.gridCounts[end] == 1 && sum(right.gridCounts) == 1

    # many points around the range, one at a time and in a batch, weighted
    Random.seed!(44)
    xs = 1.2 .* randn(5000)
    ws = rand(5000)
    for (bins, closed) in ((nothing, :left), (nothing, :right), ([-1.5, -0.2, 0.7, 2.0], :left))
        a = bins === nothing ? StreamHist(binRange=(-2.0, 2.0); closed) : StreamHist(; bins, closed)
        add!(a, xs[1:2000], ws[1:2000])
        foreach(i -> add!(a, xs[i], ws[i]), 2001:5000)
        under, over = outofrange(a)
        @test under > 0 && over > 0
        @test sum(a.gridCounts) == nobs(a) - under - over == sum(exactHistogram(a).weights)
        @test sum(a.gridWeights) ≈ totalweight(a) - sum(outofrangeweight(a))
        @test sum(histogram(a, [first(a.grid), last(a.grid)]).weights) ≈ sum(exactHistogram(a).weights)
    end
end

@testset "every moment, the mean included, is NaN before any point" begin
    oh = StreamHist(binRange=(0.0, 1.0))
    @test isnan(mean(oh)) && isnan(variance(oh)) && isnan(std(oh)) && isnan(skewness(oh)) && isnan(kurtosis(oh))
    @test all(isnan, values(moments(oh))) && all(p -> isnan(moment(oh, p)), oh.momentPowers)
    @test isnan(weightedMean(oh)) && all(isnan, values(weightedMoments(oh)))
    @test isnan(mean(MomentAccumulator(4))) && isnan(mean(MomentAccumulator(Float64, 4)))
    @test_throws ArgumentError StreamHistogram.centralmoment(MomentAccumulator(4), 5)
    add!(oh, 0.25)
    @test mean(oh) == 0.25 && variance(oh) == 0.0
    # a point of weight 0 counts, but adds no weight
    z = StreamHist(binRange=(0.0, 1.0))
    add!(z, 0.5, 0.0)
    @test mean(z) == 0.5 && isnan(weightedMean(z))
end
