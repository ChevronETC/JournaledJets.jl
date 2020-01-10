using Distributed
addprocs(3)
@everywhere using Distributed, Jets, JournaledJets, LinearAlgebra, Test

@everywhere foo(iblock) = prod(iblock.I) * ones(2)
@everywhere bar(iblock) = LinearIndices((3,4))[iblock[1],iblock[2]] * ones(2,3)

@everywhere JopBar_f!(d,m) = d .= m.^2
@everywhere JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
@everywhere function JopBar(n) spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, df′! = JopBar_df!, dom = spc, rng = spc)
end

@testset "JArray construct, 1D" begin
    x = JArray(foo, (3,), workers(), Array{Float64,1});
    @test typeof(x) == JArray{Float64, 1, Vector{Float64}}
end

@testset "JArray construct, 2D" begin
    x = JArray(bar, (3,2), workers(), Array{Float64,2});
    @test typeof(x) == JArray{Float64, 2, Array{Float64,2}}
end

@testset "JArray size, 1D" begin
    x = JArray(foo, (3,), workers(), Vector{Float64});
    @test size(x) == (6,)
    @test length(x) == 6
end

@testset "JArray size, 2D" begin
    x = JArray(bar, (3,4), workers(), Array{Float64,2});
    @test size(x) == (6,12)
end

@testset "JArray getblock, 1D" begin
    x = JArray(foo, (3,), workers(), Vector{Float64});
    getblock(x,1)
    for i = 1:3
        @test getblock(x, i) ≈ i*ones(2)
    end
end

@testset "JArray getblock, 2D" begin
    x = JArray(bar, (3,4), workers(), Array{Float64,2});
    for jblock = 1:4, iblock = 1:3
        xᵢⱼ = getblock(x, iblock, jblock)
        @test xᵢⱼ ≈ LinearIndices((3,4))[iblock,jblock]*ones(2,3)
    end
end

@testset "JArray setblock!, 1D" begin
    x = JArray(foo, (3,), workers(), Vector{Float64});
    for iblock = 1:3
        setblock!(x, 2*iblock*ones(2), iblock)
        xᵢ = getblock(x, iblock)
        @test xᵢ ≈ 2*iblock*ones(2)
    end
end

@testset "JArray setblock!, 2D" begin
    x = JArray(bar, (3,4), workers(), Array{Float64,2});
    for jblock = 1:4, iblock = 1:3
        setblock!(x, 2*LinearIndices((3,4))[iblock,jblock]*ones(2,3), iblock, jblock)
        xᵢⱼ = getblock(x, iblock, jblock)
        @test xᵢⱼ ≈ 2*LinearIndices((3,4))[iblock,jblock]*ones(2,3)
    end
end

@testset "JArray getindex, 1D" begin
    x = JArray(foo, (3,), workers(), Vector{Float64})
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

@testset "JArray getindex, 2D" begin
    x = JArray(bar, (3,2), workers(), Array{Float64,2})
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

@testset "JArray similar, 1D" begin
    x = JArray(i->rand(2), (3,), workers(), Vector{Float64})
    y = similar(x)
    z = similar(x, Float32)
    @test length(x) == length(y)
    @test length(z) == length(y)
    @test eltype(z) == Float32
    @test z.indices == x.indices
    @test y.indices == x.indices
end

@testset "JArray similar, 2D" begin
    x = JArray(i->rand(2,3), (3,2), workers(), Array{Float64,2})
    y = similar(x)
    z = similar(x, Float32)
    @test length(x) == length(y)
    @test length(z) == length(y)
    @test size(x) == size(y)
    @test size(z) == size(y)
    @test eltype(z) == Float32
    @test z.indices == x.indices
    @test y.indices == x.indices
end

@testset "JArray procs" begin
    x = JArray(i->rand(2), (3,), workers(), Vector{Float64})
    @test procs(x) == workers()
end

@testset "DJArray collect/convert, 1D" begin
    x = JArray(i->rand(2), (3,), workers(), Vector{Float64})
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
end

@testset "JArray norm, 1D" begin
    x = rand(90)
    y = JArray(i->x[(i[1]-1)*30+1:i[1]*30], (3,), workers(), Vector{Float64})
    z = collect(y)
    @test x ≈ z

    @test norm(x) ≈ norm(y)
    for p in (0,1,2,Inf,-Inf)
        @test norm(x,p) ≈ norm(y,p)
    end
end

@testset "JArray dot, 1D" begin
    x₁ = rand(90)
    x₂ = rand(90)
    y₁ = JArray(i->x₁[(i[1]-1)*30+1:i[1]*30], (3,), workers(), Vector{Float64})
    y₂ = JArray(i->x₂[(i[1]-1)*30+1:i[1]*30], (3,), workers(), Vector{Float64})
    z₁ = collect(y₁)
    z₂ = collect(y₂)
    @test x₁ ≈ z₁
    @test x₂ ≈ z₂
    @test dot(x₁,x₂) ≈ dot(y₁,y₂)
end

@testset "JArray extrema, 1D" begin
    x = rand(90)
    y = JArray(i->x[(i[1]-1)*30+1:i[1]*30], (3,), workers(), Vector{Float64})
    z = collect(y)
    mnmx = extrema(y)
    _mnmx = extrema(x)
    @test mnmx[1] ≈ _mnmx[1]
    @test mnmx[2] ≈ _mnmx[2]
end

@testset "JArray fill!, 1D" begin
    y = JArray(i->zeros(30), (3,), workers(), Vector{Float64})
    y .= 3
    z = collect(y)
    for i = 1:90
        @test z[i] ≈ 3
    end
end

@testset "JArray broadcasting" begin
    u = JArray(i->rand(10), (4,), workers(), Vector{Float64})
    v = JArray(i->rand(10), (4,), workers(), Vector{Float64})
    w = JArray(i->rand(10), (4,), workers(), Vector{Float64})
    a = rand(Float64)
    b = rand(Float64)
    c = rand(Float64)
    x = a*u .+ b*v .+ c*w

    _u = collect(u)
    _v = collect(v)
    _w = collect(w)
    @test a*_u .+ b*_v .+ c*_w ≈ collect(x)

    y = JArray(i->zeros(10), (4,), workers(), Vector{Float64})
    y .= x
    @test typeof(y) == JournaledJets.JArray{Float64,1,Array{Float64,1}}
    @test a*_u .+ b*_v .+ c*_w ≈ collect(y)

    z = zeros(40)
    z .= x
    @test a*_u .+ b*_v .+ c*_w ≈ z

    x = rand(Int32, 40)
    y = JArray(i->rand(10), (4,), workers(), Vector{Float64})
    z = x .* y
    @test isa(z, JournaledJets.JArray)

    @test convert(Array, z) ≈ x .* convert(Array, y)
    x = similar(y, Int)
    @test isa(x, JArray{Int})
end

@testset "JetJSpace construct" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    @test isa(R, JetJSpace)
end

@testset "JetJSpace size" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    @test size(R) == (50,)
end

@testset "JetJSpace eltype" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    @test eltype(R) == Float64
end

@testset "JetJSpace indices" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    @test indices(R) == [1:10, 11:20, 21:30, 31:40, 41:50]
end

@testset "JetJSpace space" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    @test space(R, 2) == JetSpace(Float64,10)
end

@testset "JetJSpace nblocks" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    @test nblocks(R) == 5
end

@testset "JetJSpace Array" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    x = Array(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
end

@testset "JetJSpace ones" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    x = ones(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
    @test collect(x) ≈ ones(50)
end

@testset "JetJSpace zeros" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    x = zeros(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
    @test collect(x) ≈ zeros(50)
end

@testset "JetJSpace rand" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,), workers(), Vector{JetSpace{Float64,1}}))
    x = rand(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
end

@testset "block operator, tall and skinny" begin
    A = @blockop JArray(i->[JopBar(10)], (4,1), workers(), Vector{typeof(JopBar(10))})
    @test isa(A, Jop)

    _F  = JArray(_->[JopBar(10)], (4,1), workers(), Vector{typeof(JopBar(10))})
    F = @blockop _F

    _G = [_F[i,1] for i in 1:4, j in 1:1]
    G = @blockop _G

    m = rand(domain(F))
    @test collect(F*m) ≈ G*m

    J = jacobian!(F, m)
    _J = jacobian!(G, m)

    δm = rand(domain(J))
    @test collect(J*δm) ≈ _J*δm
    δd = rand(range(J))
    @test J'*δd ≈ _J'*collect(δd)
end
