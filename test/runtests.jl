using Revise
using Distributed, JetPack, Jets, JournaledJets, LinearAlgebra, Schedulers, Test

foo(iblock) = iblock * ones(2)
bar(iblock) = LinearIndices((3,2))[iblock[1],iblock[2]] * ones(2,3)

@testset "DJArray construct, 1D" begin
    x = DJArray(foo, (3,), workers())
    @test typeof(x) == DJArray{Float64, 1, Vector{Float64}}
end

@testset "DJArray construct, 2D" begin
    x = DJArray(bar, (3,2), workers())
    @test typeof(x) == DJArray{Float64, 2, Array{Float64,2}}
end

@testset "DJArray size, 1D" begin
    x = DJArray(foo, (3,), workers())
    @test size(x) == (6,)
    @test length(x) == 6
end

@testset "DJArray size, 2D" begin
    x = DJArray(bar, (3,2), workers())
    @test size(x) == (6,6)
    @test length(x) == 36
end

@testset "DJArray getblock, 1D" begin
    x = DJArray(foo, (3,), workers())
    x₁ = getblock(x, 3)

    for iblock = 1:3
        xᵢ = getblock(x, iblock)
        @test xᵢ ≈ iblock*ones(2)
    end
end

@testset "DJArray getblock, 2D" begin
    x = DJArray(bar, (3,2), workers())
    x₁₁ = getblock(x, 1, 1)

    for jblock = 1:2, iblock = 1:3
        xᵢⱼ = getblock(x, iblock, jblock)
        @test xᵢⱼ ≈ LinearIndices((3,2))[iblock,jblock]*ones(2,3)
    end
end

@testset "DJArray setblock!, 1D" begin
    x = DJArray(foo, (3,), workers())
    for iblock = 1:3
        setblock!(x, 2*iblock*ones(2), iblock)
        xᵢ = getblock(x, iblock)
        @test xᵢ ≈ 2*iblock*ones(2)
    end
end

@testset "DJArray setblock!, 2D" begin
    x = DJArray(bar, (3,2), workers())
    for jblock = 1:2, iblock = 1:3
        setblock!(x, 2*LinearIndices((3,2))[iblock,jblock]*ones(2,3), iblock, jblock)
        xᵢⱼ = getblock(x, iblock, jblock)
        @test xᵢⱼ ≈ 2*LinearIndices((3,2))[iblock,jblock]*ones(2,3)
    end
end

@testset "DJArray getindex, 1D" begin
    x = DJArray(foo, (3,), workers())
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

@testset "DJArray getindex, 2D" begin
    x = DJArray(bar, (3,2), workers())
    for j = 1:size(x,2), i = 1:size(x,1)
        if 1 <= i <= 2 && 1 <= j <= 3
            @test x[i,j] ≈ 1
        elseif 3 <= i <= 4 && 1 <= j <= 3
            @test x[i,j] ≈ 2
        elseif 5 <= i <= 6 && 1 <= j <= 3
            @test x[i,j] ≈ 3
        elseif 1 <= i <= 2 && 4 <= j <= 6
            @test x[i,j] ≈ 4
        elseif 3 <= i <= 4 && 4 <= j <= 6
            @test x[i,j] ≈ 5
        elseif 5 <= i <= 6 && 4 <= j <= 6
            @test x[i,j] ≈ 6
        end
    end
end

@testset "DJArray similar, 1D" begin
    addprocs(3)
    @everywhere using JournaledJets, Jets
    x = DJArray(i->rand(2), (3,), workers())
    y = similar(x)
    z = similar(x, Float32)
    @test length(x) == length(y)
    @test length(z) == length(y)
    @test eltype(z) == Float32
    @test indices(z) == indices(x)
    @test indices(y) == indices(x)
    rmprocs(workers())
end

@testset "DJArray similar, 2D" begin
    addprocs(3)
    @everywhere using JournaledJets, Jets
    x = DJArray(i->rand(2,3), (3,2), workers())
    y = similar(x)
    z = similar(x, Float32)
    @test length(x) == length(y)
    @test length(z) == length(y)
    @test size(x) == size(y)
    @test size(z) == size(y)
    @test eltype(z) == Float32
    @test indices(z) == indices(x)
    @test indices(y) == indices(x)
    rmprocs(workers())
end

@testset "DJArray procs" begin
    addprocs(3)
    @everywhere using JournaledJets
    x = DJArray(i->rand(2), (3,), workers());
    @test procs(x) == workers()
    rmprocs(workers())
end

@testset "DJArray collect/convert, 1D" begin
    addprocs(3)
    @everywhere using JournaledJets
    x = DJArray(i->rand(2), (3,), workers())
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

@testset "DJArray norm, 1D" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets, LinearAlgebra
    x = rand(90)
    y = DJArray(i->x[(i-1)*30+1:i*30], (3,), workers())
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
    y₁ = DJArray(i->x₁[(i-1)*30+1:i*30], (3,), workers())
    y₂ = DJArray(i->x₂[(i-1)*30+1:i*30], (3,), workers())
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
    y = DJArray(i->x[(i-1)*30+1:i*30], (3,), workers())
    z = collect(y)
    mnmx = extrema(y)
    _mnmx = extrema(x)
    @test mnmx[1] ≈ _mnmx[1]
    @test mnmx[2] ≈ _mnmx[2]
    rmprocs(workers())
end

@testset "DJArray fill!" begin
    addprocs(3)
    @everywhere using JournaledJets, LinearAlgebra
    y = DJArray(i->zeros(30), (3,), workers())
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
    u = DJArray(i->rand(10), (4,), workers())
    v = DJArray(i->rand(10), (4,), workers())
    w = DJArray(i->rand(10), (4,), workers())
    a = rand(Float64)
    b = rand(Float64)
    c = rand(Float64)
    x = a*u .+ b*v .+ c*w
    @test typeof(x) == JournaledJets.DJArray{Float64,1,Array{Float64,1}}

    _u = collect(u)
    _v = collect(v)
    _w = collect(w)
    @test a*_u .+ b*_v .+ c*_w ≈ collect(x)

    y = DJArray(i->zeros(10), (4,), workers())
    y .= x
    @test typeof(y) == JournaledJets.DJArray{Float64,1,Array{Float64,1}}
    @test a*_u .+ b*_v .+ c*_w ≈ collect(y)

    z = zeros(40)
    z .= x
    @test a*_u .+ b*_v .+ c*_w ≈ z

    x = rand(Int32, 40)
    y = DJArray(i->rand(10), (4,), workers())
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
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    @test isa(R, JetJSpace)
    rmprocs(workers())
end

@testset "JetJSpace size" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    size(R)
    @test size(R) == (50,)
    rmprocs(workers())
end

@testset "JetJSpace eltype" begin
    addprocs(3)
    workers()
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    @test eltype(R) == Float64
    rmprocs(workers())
end

@testset "JetJSpace indices" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    @test indices(R) == [1:10, 11:20, 21:30, 31:40, 41:50]
    rmprocs(workers())
end

@testset "JetJSpace space" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    space(R,2)
    @test space(R, 2) == JetSpace(Float64,10)
    rmprocs(workers())
end

@testset "JetJSpace nblocks" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    @test nblocks(R) == 5
    rmprocs(workers())
end

@testset "JetJSpace Array" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    x = Array(R)
    @test isa(x, DJArray)
    @test size(x) == (50,)
    rmprocs(workers())
end

@testset "JetJSpace ones" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    x = ones(R)
    @test isa(x, DJArray)
    @test size(x) == (50,)
    @test collect(x) ≈ ones(50)
    rmprocs(workers())
end

@testset "JetJSpace zeros" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    x = zeros(R)
    @test isa(x, DJArray)
    @test size(x) == (50,)
    @test collect(x) ≈ zeros(50)
    rmprocs(workers())
end

@testset "JetJSpace rand" begin
    addprocs(3)
    @everywhere using Jets, JournaledJets
    R = JetJSpace(DJArray(_->[JetSpace(Float64,10)], (5,), workers()))
    x = rand(R)
    @test isa(x, DJArray)
    @test size(x) == (50,)
    rmprocs(workers())
end

@testset "block operator, tall and skinny" begin
    A = @blockop DJArray(i->[JopDiagonal(rand(2))], (2,3), workers())
    @test isa(A, Jop)
end

#@testset "JopJBlock, homogeneous, tall and skinny" begin
using Distributed
addprocs(3)
@everywhere using Jets, JournaledJets, Logging, Test, LinearAlgebra

@everywhere Logging.global_logger(Logging.ConsoleLogger(stdout, Logging.Debug))

@everywhere JopBar_f!(d,m) = d .= m.^2
@everywhere JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
@everywhere function JopBar(n) spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, df′! = JopBar_df!, dom = spc, rng = spc)
end

_F  = DJArray(_->[JopBar(10)], (4,1), workers())
F = @blockop _F

_G = [_F[i] for i in 1:4, j in 1:1]
G = @blockop _G
@test isa(F, JopNl{<:Jet{<:JetSpace,<:JetJSpace,typeof(JournaledJets.JetJBlock_f!)}})

m = rand(domain(F))
d = F*m;
_d = G*m;
__d = collect(d)

@test _d ≈ __d

JournaledJets.blocks(d)[4].where

@test collect(F*m) ≈ G*m

F*m

J = jacobian!(F, m)
_J = jacobian!(G, m)

δm = rand(domain(J))
@test collect(J*δm) ≈ _J*δm

δd = rand(range(J))
@test J'*δd ≈ _J'*collect(δd)

rmprocs(workers())
#end
