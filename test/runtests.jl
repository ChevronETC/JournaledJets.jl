using Revise
using Distributed
addprocs(3)
@everywhere using Distributed, Jets, JournaledJets, LinearAlgebra, Test

@everywhere foo(iblock) = prod(iblock.I) * ones(2)
@everywhere bar(iblock) = LinearIndices((3,4))[iblock[1],iblock[2]] * ones(2,3)

@everywhere JopBar_f!(d,m;x) = d .= x[1] .* m.^2
@everywhere JopBar_df!(δd,δm;mₒ,x,kwargs...) = δd .= 2 .* x[1] .* mₒ .* δm
@everywhere function JopBar(n,x=[1.0]) spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, df′! = JopBar_df!, dom = spc, rng = spc, s=(x=x,))
end

@testset "JArray construct, 1D" begin
    x = JArray(foo, (3,))
    @test typeof(x) == JArray{Float64, 1, Vector{Float64}}
end

@testset "JArray construct, 2D" begin
    x = JArray(bar, (3,2))
    @test typeof(x) == JArray{Float64, 2, Array{Float64,2}}
end

@testset "JArray size, 1D" begin
    x = JArray(foo, (3,))
    @test size(x) == (6,)
    @test length(x) == 6
end

@testset "JArray size, 2D" begin
    x = JArray(bar, (3,4))
    @test size(x) == (6,12)
end

@testset "JArray getblock, 1D" begin
    x = JArray(foo, (3,))
    getblock(x,1)
    for i = 1:3
        @test getblock(x, i) ≈ i*ones(2)
    end
end

@testset "JArray getblock, 2D" begin
    x = JArray(bar, (3,4))
    for jblock = 1:4, iblock = 1:3
        xᵢⱼ = getblock(x, iblock, jblock)
        @test xᵢⱼ ≈ LinearIndices((3,4))[iblock,jblock]*ones(2,3)
    end
end

@testset "JArray setblock!, 1D" begin
    x = JArray(foo, (3,))
    for iblock = 1:3
        setblock!(x, 2*iblock*ones(2), iblock)
        xᵢ = getblock(x, iblock)
        @test xᵢ ≈ 2*iblock*ones(2)
    end
end

@testset "JArray setblock!, 2D" begin
    x = JArray(bar, (3,4))
    for jblock = 1:4, iblock = 1:3
        setblock!(x, 2*LinearIndices((3,4))[iblock,jblock]*ones(2,3), iblock, jblock)
        xᵢⱼ = getblock(x, iblock, jblock)
        @test xᵢⱼ ≈ 2*LinearIndices((3,4))[iblock,jblock]*ones(2,3)
    end
end

@testset "JArray getindex, 1D" begin
    x = JArray(foo, (3,))
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
    x = JArray(bar, (3,2))
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
    x = JArray(i->rand(2), (3,))
    y = similar(x)
    z = similar(x, Float32)
    @test length(x) == length(y)
    @test length(z) == length(y)
    @test eltype(z) == Float32
    @test z.indices == x.indices
    @test y.indices == x.indices
end

@testset "JArray similar, 2D" begin
    x = JArray(i->rand(2,3), (3,2))
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
    x = JArray(i->rand(2), (3,))
    @test procs(x) == workers()
end

@testset "JArray collect/convert, 1D" begin
    x = JArray(i->rand(2), (3,))
    workers()
    x.blockmap
    y = collect(x)
    workers()

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
    y = JArray(i->x[(i[1]-1)*30+1:i[1]*30], (3,))
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
    y₁ = JArray(i->x₁[(i[1]-1)*30+1:i[1]*30], (3,))
    y₂ = JArray(i->x₂[(i[1]-1)*30+1:i[1]*30], (3,))
    z₁ = collect(y₁)
    z₂ = collect(y₂)
    @test x₁ ≈ z₁
    @test x₂ ≈ z₂
    @test dot(x₁,x₂) ≈ dot(y₁,y₂)
end

@testset "JArray extrema, 1D" begin
    x = rand(90)
    y = JArray(i->x[(i[1]-1)*30+1:i[1]*30], (3,))
    z = collect(y)
    mnmx = extrema(y)
    _mnmx = extrema(x)
    @test mnmx[1] ≈ _mnmx[1]
    @test mnmx[2] ≈ _mnmx[2]
end

@testset "JArray fill!, 1D" begin
    y = JArray(i->zeros(30), (3,))
    y .= 3
    z = collect(y)
    for i = 1:90
        @test z[i] ≈ 3
    end
end

@testset "JArray broadcasting" begin
    u = JArray(i->rand(10), (4,))
    v = JArray(i->rand(10), (4,))
    w = JArray(i->rand(10), (4,))
    a = rand(Float64)
    b = rand(Float64)
    c = rand(Float64)
    x = a*u .+ b*v .+ c*w

    _u = collect(u)
    _v = collect(v)
    _w = collect(w)
    @test a*_u .+ b*_v .+ c*_w ≈ collect(x)

    y = JArray(i->zeros(10), (4,))
    y .= x
    @test typeof(y) == JournaledJets.JArray{Float64,1,Array{Float64,1}}
    @test a*_u .+ b*_v .+ c*_w ≈ collect(y)

    z = zeros(40)
    z .= x
    @test a*_u .+ b*_v .+ c*_w ≈ z

    x = rand(Int32, 40)
    y = JArray(i->rand(10), (4,))
    z = x .* y
    @test isa(z, JournaledJets.JArray)

    @test convert(Array, z) ≈ x .* convert(Array, y)
    x = similar(y, Int)
    @test isa(x, JArray{Int})
end

@testset "JetJSpace construct" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    @test isa(R, JetJSpace)
end

@testset "JetJSpace size" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    @test size(R) == (50,)
end

@testset "JetJSpace eltype" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    @test eltype(R) == Float64
end

@testset "JetJSpace indices" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    @test indices(R) == [1:10, 11:20, 21:30, 31:40, 41:50]
end

@testset "JetJSpace space" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    @test space(R, 2) == JetSpace(Float64,10)
end

@testset "JetJSpace nblocks" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    @test nblocks(R) == 5
end

@testset "JetJSpace Array" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    x = Array(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
end

@testset "JetJSpace ones" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    x = ones(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
    @test collect(x) ≈ ones(50)
end

@testset "JetJSpace zeros" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    x = zeros(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
    @test collect(x) ≈ zeros(50)
end

@testset "JetJSpace rand" begin
    R = JetJSpace(JArray(_->[JetSpace(Float64,10)], (5,)))
    x = rand(R)
    @test isa(x, JArray)
    @test size(x) == (50,)
end

@testset "block operator, tall and skinny" begin
    A = @blockop JArray(i->[JopBar(10)], (4,1))
    @test isa(A, Jop)

    _F  = JArray(_->[JopBar(10)], (4,1))
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

@testset "scale-up" begin
    x = JArray(_->rand(100), (10,1))
    y = JArray(_->rand(100), (10,1))
    a = dot(x,y)
    _x = convert(Array, x)

    addprocs(3)
    @everywhere using JournaledJets

    @everywhere _getpids_fromid(id) = JournaledJets.registry[id].pids
    @everywhere function getpids(x)
        pids = Dict{Any,Any}()
        for pid in x.pids
            pids[pid] = remotecall_fetch(_getpids_fromid, pid, x.id)
        end
        pids
    end

    @everywhere _getblkmap_fromid(id) = JournaledJets.registry[id].blockmap
    @everywhere function getblkmaps(x)
        bm = [1:10;]
        for pid in x.pids
            bm = [bm remotecall_fetch(_getblkmap_fromid, pid, x.id)]
        end
        bm
    end

    _a = dot(x,y)
    @test a ≈ _a

    p = getpids(x)
    bm = getblkmaps(x)

    for key in keys(p)
        @test p[key] ≈ p[first(keys(p))]
    end

    for i = 2:size(bm,2)
        @test bm[:,i] ≈ bm[:,2]
    end

    @test _x ≈ convert(Array, x)

    close(x)
    close(y)
    rmprocs(workers()[4:6])
end

#@testset "journal, broadcasting" begin # TODO - does not work inside a test-set
######
######
___x = @journal JArray(i->i.I[1]*ones(2), (2,))
___y = @journal JArray(i->2*i.I[1]*ones(2), (2,))
@journal ___y ___y .*= 2
@journal ___y ___y .*= 2*___x .+ ___y
@journal ___y ___y .*= 2*___x .+ 3*___y

____y = copy(___y)
___y .= 0
@replay ___y 1
@test getblock(___y, 1) ≈ getblock(____y, 1)
@test (getblock(___y, 2) ≈ getblock(____y, 1)) == false
@replay ___y 2
@test getblock(___y, 2) ≈  getblock(____y, 2)

___y .= @journal :reset ___y 2*___x
@test length(___y.journal) == 2

____y = copy(___y)
___y .= 0
@replay ___y 1
@test getblock(___y, 1) ≈ getblock(____y, 1)
@test (getblock(___y, 2) ≈ getblock(____y, 1)) == false
@replay ___y 2
@test getblock(___y, 2) ≈  getblock(____y, 2)
######
######
#end

#@testset "journal, single block" begin  # TODO - does not work inside a test-set
######
######
x = @journal JArray(i->ones(2), (2,))

@journal x begin
    _x = getblock(x,1)
    _x[1] = π
end

x .= 0
@replay x 1
@test getblock(x, 1) ≈ [π, 1]

@replay x 2
@test getblock(x, 2) ≈ [1, 1]
######
######
#end

#@testset "journal, mul!, single operator"
######
######
A = @blockop @journal JArray(_->[JopBar(2)], (2,1))
x = rand(domain(A))

y = @journal zeros(range(A))
@journal y mul!(y, A, x)

_y = copy(y)
y .= 0

@test (y ≈ _y) == false
@replay y 1
@replay y 2
@test y ≈ _y

A₁₁ = getblock(A, 1, 1)
state(A₁₁).x[1] = 2.0
mul!(y, A, x)
@test (y ≈ _y) == false

@replay A 1

mul!(y, A, x)
@test y ≈ _y
######
######
#end

#@testset "journal, mul!, composite operator" begin
######
######
A = @blockop @journal JArray(_->[JopBar(2) ∘ JopBar(2)], (2,1))
x = rand(domain(A))

y = @journal zeros(range(A))
y = @journal y mul!(y, A, x)

_y = copy(y)
y .= 0

@replay y 1
@replay y 2
@test y ≈ _y

A₁₁ = getblock(A, 1, 1)
A₁₁₁ = state(A₁₁).ops[1]
state(A₁₁₁).x[1] = 2.0
mul!(y, A, x)
@test (y ≈ _y) == false

@replay A 1

mul!(y, A, x)
@test y ≈ _y
######
######
#end

rmprocs(workers())
