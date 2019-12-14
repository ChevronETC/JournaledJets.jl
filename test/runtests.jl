using Revise
using Distributed, Jets, JournaledJets, LinearAlgebra, Schedulers, Test

foo(iblock) = iblock * ones(2)

@testset "DJArray construct" begin
    x = DJArray(foo, 3, workers())
    @test typeof(x) == DJArray{Float64, Vector{Float64}}
end

@testset "DJArray size" begin
    x = DJArray(foo, 3, workers())
    @test size(x) == (6,)
    @test length(x) == 6
end

@testset "DJArray getblock" begin
    x = DJArray(foo, 3, workers())
    x₁ = getblock(x, 3)

    for iblock = 1:3
        xᵢ = getblock(x, iblock)
        @test xᵢ ≈ iblock*ones(2)
    end
end

@testset "DJArray setblock!" begin
    x = DJArray(foo, 3, workers())
    for iblock = 1:3
        setblock!(x, iblock, 2*iblock*ones(2))
        xᵢ = getblock(x, iblock)
        @test xᵢ ≈ 2*iblock*ones(2)
    end
end

@testset "DJArray getindex" begin
    x = DJArray(foo, 3, workers())
    for i = 1:length(x)
        if i <= 2
            @test x[i] ≈ 1
        elseif 2 < i <= 4
            @test x[i] ≈ 2
        elseif 4 < i <= 6
            @test x[i] ≈ 3
        end
    end
end

@testset "DJArray similar" begin
    addprocs(3)
    @everywhere using JournaledJets, Jets
    x = DJArray(i->rand(2), 3, workers())
    y = similar(x)
    z = similar(x, Float32)
    @test length(x) == length(y)
    @test length(z) == length(y)
    @test eltype(z) == Float32
    @test indices(z) == indices(x)
    @test indices(y) == indices(x)
    rmprocs(workers())
end

@testset "DJArray procs" begin
    addprocs(3)
    @everywhere using JournaledJets
    x = DJArray(i->rand(2), 3, workers());
    @test procs(x) == workers()
    rmprocs(workers())
end

@testset "DJArray collect/convert" begin
    addprocs(3)
    @everywhere using JournaledJets
    x = DJArray(i->rand(2), 3, workers())
    y = collect(x)
    z = convert(Jets.BlockArray, x)
    u = convert(Array, x)

    @test isa(y, Jets.BlockArray)
    @test isa(z, Jets.BlockArray)
    @test isa(u, Array)
    for i = 1:6
        @test x[i] ≈ y[i]
        @test x[i] ≈ z[i]
        @test x[i] ≈ u[i]
    end
    rmprocs(workers())
end

@testset "DJArray norm" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets, LinearAlgebra
    x = rand(90)
    y = DJArray(i->x[(i-1)*30+1:i*30], 3, workers())
    z = collect(y)
    @test x ≈ z

    @test norm(x) ≈ norm(y)
    for p in (0,1,2,Inf,-Inf)
        @test norm(x,p) ≈ norm(y,p)
    end
    rmprocs(workers())
end

@testset "DJArray dot" begin
    addprocs(3)
    @everywhere using JournaledJets, LinearAlgebra
    x₁ = rand(90)
    x₂ = rand(90)
    y₁ = DJArray(i->x₁[(i-1)*30+1:i*30], 3, workers())
    y₂ = DJArray(i->x₂[(i-1)*30+1:i*30], 3, workers())
    z₁ = collect(y₁)
    z₂ = collect(y₂)
    @test x₁ ≈ z₁
    @test x₂ ≈ z₂
    @test dot(x₁,x₂) ≈ dot(y₁,y₂)
    rmprocs(workers())
end

@testset "DJArray extrema" begin
    addprocs(3)
    @everywhere using JournaledJets, LinearAlgebra
    x = rand(90)
    y = DJArray(i->x[(i-1)*30+1:i*30], 3, workers())
    z = collect(y)
    @test x ≈ y
    mnmx = extrema(y)
    _mnmx = extrema(x)
    @test mnmx[1] ≈ _mnmx[1]
    @test mnmx[2] ≈ _mnmx[2]
end

@testset "DJArray fill!" begin
    addprocs(3)
    @everywhere using JournaledJets, LinearAlgebra
    y = DJArray(i->zeros(30), 3, workers())
    y .= 3
    z = collect(y)
    for i = 1:90
        z[i] ≈ 3
    end
    rmprocs(workers())
end

@testset "DJArray broadcasting" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    u = DJArray(i->rand(10), 4, workers())
    v = DJArray(i->rand(10), 4, workers())
    w = DJArray(i->rand(10), 4, workers())
    a = rand(Float64)
    b = rand(Float64)
    c = rand(Float64)
    x = a*u .+ b*v .+ c*w
    @test typeof(x) == JournaledJets.DJArray{Float64,Array{Float64,1}}

    _u = collect(u)
    _v = collect(v)
    _w = collect(w)
    @test a*_u .+ b*_v .+ c*_w ≈ collect(x)

    y = DJArray(i->zeros(10), 4, workers())
    y .= x
    @test typeof(y) == JournaledJets.DJArray{Float64,Array{Float64,1}}
    @test a*_u .+ b*_v .+ c*_w ≈ collect(y)

    z = zeros(40)
    z .= x
    @test a*_u .+ b*_v .+ c*_w ≈ z

    x = rand(Int32, 40)
    y = DJArray(i->rand(10), 4, workers())
    z = x .* y
    @test isa(z, JournaledJets.DJArray)
    @test convert(Array, z) ≈ x .* convert(Array, y)
    x = similar(y, Int)
    @test isa(x, DJArray{Int})

    rmprocs(workers())
end

@testset "JetJSpace construct" begin
    addprocs(3)
    workers()
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], 5, workers()))
    @test isa(R, JetJSpace)
    rmprocs(workers())
end

@testset "JetJSpace size" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], 5, workers()))
    size(R)
    @test size(R) == (50,)
    rmprocs(workers())
end

@testset "JetJSpace eltype" begin
    addprocs(3)
    workers()
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], 5, workers()))
    @test eltype(R) == Float64
    rmprocs(workers())
end

@testset "JetJSpace indices" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], 5, workers()))
    @test indices(R) == [1:10, 11:20, 21:30, 31:40, 41:50]
    rmprocs(workers())
end

@testset "JetJSpace space" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], 5, workers()))
    space(R,2)
    @test space(R, 2) == JetSpace(Float64,10)
    rmprocs(workers())
end

@testset "JetJSpace nblocks" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], 5, workers()))
    @test nblocks(R) == 5
    rmprocs(workers())
end
