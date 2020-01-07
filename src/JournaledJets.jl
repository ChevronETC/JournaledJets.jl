module JournaledJets

using Distributed, Jets, LinearAlgebra, ParallelOperations, Schedulers

struct DJArray{T,N,A<:AbstractArray{T}} <: AbstractArray{T,N}
    blocks::Array{Future,N}
    indices::Array{NTuple{N,UnitRange{Int}},N}
    journal::Vector{Expr}
end

DJArray_construct_block(iblock, f) = remotecall(f, myid(), iblock)
DJArray_block_length(block) = length(fetch(block))
DJArray_block_size(block, i) = size(fetch(block), i)
DJArray_block_typeof(block) = typeof(fetch(block))

function DJArray(f::Function, nblocks::NTuple{1,Int}, pids)
    blocks = cvxpmap(Future, DJArray_construct_block, 1:nblocks[1], f)

    indices = Vector{NTuple{1,UnitRange{Int}}}(undef, nblocks)
    i1 = 1
    for (iblock,block) in enumerate(blocks)
        i2 = i1 + remotecall_fetch(DJArray_block_length, block.where, block) - 1
        indices[iblock] = (i1:i2,)
        i1 = i2 + 1
    end

    A = remotecall_fetch(DJArray_block_typeof, blocks[1].where, blocks[1])
    T = eltype(A)

    DJArray{T,1,A}(blocks, indices, Expr[])
end

function DJArray(f::Function, nblock::NTuple{N,Int}, pids) where {N}
    blocks = cvxpmap(Future, DJArray_construct_block, CartesianIndices(nblock), f)

    i1 = [Int[] for i=1:N]
    i2 = [Int[] for i=1:N]
    for idim = 1:N
        _i1 = 1
        for iblock = 1:nblock[idim]
            block = blocks[ntuple(_idim->_idim==idim ? iblock : 1, N)...]
            _i2 = _i1 + remotecall_fetch(DJArray_block_size, block.where, block, idim) - 1
            push!(i1[idim], _i1)
            push!(i2[idim], _i2)
            _i1 = _i2 + 1
        end
    end

    indices = Array{NTuple{N,UnitRange{Int}},N}(undef, nblock)
    for (i,idx) in enumerate(CartesianIndices(nblock))
        indices[idx] = ntuple(idim->i1[idim][idx[idim]]:i2[idim][idx[idim]], N)
    end

    A = remotecall_fetch(DJArray_block_typeof, blocks[1].where, blocks[1])
    T = eltype(A)

    DJArray{T,N,A}(blocks, indices, Expr[])
end

# DJArray interface implementation <--
function _getblock(x::DJArray{T,N,A}, iblock::Int) where {T,N,A}
    xblock = fetch(blocks(x)[iblock])::A
    _where = blocks(x)[iblock].where
    _myid = myid()
    if _myid != _where
        x.blocks[iblock] = remotecall(identity, _myid, xblock)
        # send the future back to the master pid .. how do we do this ??
        # do we need x to contain some reference to the master-held blocks (i.e. a Future)
        @info "_myid=$_myid, iblock=$iblock, where=$(blocks(x)[iblock].where)"
        @info "before fetch, ex=$(extrema(xblock))"
        xblock = fetch(blocks(x)[iblock])
        @info "after fetch, ex=$(extrema(xblock))"
        @debug "Owner of block $iblock has switched from process $_where to $_myid, eltype(xblock)=$(eltype(xblock))"
    end
    xblock
end

blocks(x::DJArray) = x.blocks
Jets.indices(x::DJArray) = x.indices
size_dim(x::DJArray{T,N}, idim) where {T,N} = x.indices[ntuple(_idim->_idim==idim ? size(x.indices)[idim] : 1, N)...][idim][end]
Base.size(x::DJArray{T,N}) where {T,N} = ntuple(idim->size_dim(x, idim), N)
Jets.getblock(x::DJArray, iblock::Int...) = _getblock(x, LinearIndices(size(blocks(x)))[iblock...])
Jets.getblock(x::DJArray, iblock::CartesianIndex) = getblock(x, iblock.I...)
DJArray_setblock_local!(block, xblock, ::Type{A}) where {A} = fetch(block)::A .= xblock
_setblock!(x::DJArray{T,N,A}, xblock, iblock::Int) where {T,N,A} = remotecall_fetch(DJArray_setblock_local!, blocks(x)[iblock].where, blocks(x)[iblock], xblock, A)
Jets.setblock!(x::DJArray, xblock, iblock::Int...) = _setblock!(x, xblock, LinearIndices(size(blocks(x)))[iblock...])
Jets.setblock!(x::DJArray, xblock, iblock::CartesianIndex) = setblock!(x, xblock, iblock.I...)
size_blocks(x::DJArray) = size(blocks(x))
length_blocks(x::DJArray) = prod(size(blocks(x)))
Jets.nblocks(x::DJArray) = length_blocks(x)

DJArray_getindex(::Type{A}, block_future, δi) where {A} = getindex(fetch(block_future)::A, δi...)
function Base.getindex(x::DJArray{T,N,A}, i::Vararg{Int,N}) where {T,N,A}
    iblock = findfirst(rng->mapreduce(idim->i[idim]∈rng[idim], &, 1:N), indices(x))
    block_future = x.blocks[iblock]
    iₒ = ntuple(idim->indices(x)[iblock][idim][1], N)
    δi = ntuple(idim->i[idim]-iₒ[idim]+1, N)
    remotecall_fetch(DJArray_getindex, block_future.where, A, block_future, δi)
end

_DJArray_similar_block(iblock, x::DJArray, ::Type{S}) where {S} = similar(getblock(x, iblock), S)
DJArray_similar_block(iblock, x::DJArray, ::Type{S}) where {S} = remotecall(_DJArray_similar_block, myid(), iblock, x, S)
DJArray_similar_typeof(block) = typeof(fetch(block))
function Base.similar(x::DJArray{T,N,A}, ::Type{S}) where {T,N,A,S}
    _indices = copy(indices(x))
    _block_futures = cvxpmap(DJArray_similar_block, CartesianIndices(size_blocks(x)), x, S)
    _A = remotecall_fetch(DJArray_similar_typeof, _block_futures[1].where, _block_futures[1])
    DJArray{S,N,_A}(_block_futures, _indices, Expr[])
end

DJArray_local_norm(iblock, x::DJArray, p) = norm(getblock(x, iblock), p)

function LinearAlgebra.norm(x::DJArray{T}, p::Real=2) where {T}
    _T = float(real(T))
    z = cvxpmap(DJArray_local_norm, 1:length_blocks(x), x, p)
    if p == Inf
        maximum(z)
    elseif p == -Inf
        minimum(z)
    elseif p == 0 || p == 1
        sum(z)
    else
        _p = _T(p)
        mapreduce(_z->_z^p, +, z)^(one(_T)/_p)
    end
end

DJArray_local_dot(iblock, x, y) = dot(getblock(x, iblock), getblock(y, iblock))

function LinearAlgebra.dot(x::DJArray, y::DJArray)
    z = cvxpmap(DJArray_local_dot, 1:length_blocks(x), x, y)
    sum(z)
end

DJArray_local_extrema(iblock, x::DJArray) = extrema(getblock(x, iblock))
function Base.extrema(x::DJArray{T}) where {T}
    mnmx = cvxpmap(DJArray_local_extrema, 1:length_blocks(x), x)
    mn, mx = mnmx[1]
    for i = 2:length(mnmx)
        _mn, _mx = extrema(mnmx[i])
        _mn < mn && (mn = _mn)
        _mx > mx && (mx = _mx)
    end
    mn,mx
end

DJArray_local_fill!(block, a) = fill!(fetch(block), a)
function Base.fill!(x::DJArray, a)
    for block in blocks(x)
        remotecall_fetch(DJArray_local_fill!, block.where, block, a)
    end
end

Distributed.procs(x::DJArray) = unique([block.where for block in blocks(x)])

function Base.collect(x::DJArray{T,1,A}) where {T,A}
    _x = Vector{A}(undef, length_blocks(x))
    n = 0
    for iblock = CartesianIndices(size_blocks(x))
        _x[iblock] = fetch(x.blocks[iblock])
    end
    indicesx = indices(x)
    Jets.BlockArray(_x, [indicesx[i][1] for i=1:length(indicesx)])
end

Base.convert(::Type{S}, x::DJArray{T,1,A}) where {S<:Jets.BlockArray,T,A} = collect(x)
Base.convert(::Array, x::DJArray{T,1,A}) where {T,A} = convert(Array, collect(x))
# -->

# DJArray broadcasting implementation --<
struct DJArrayStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:DJArray}) = DJArrayStyle()
DJArrayStyle(::Val{1}) = DJArrayStyle()

Base.similar(bc::Broadcast.Broadcasted{DJArrayStyle}, ::Type{T}) where {T} = similar(find_djarray(bc), T)
find_djarray(bc::Broadcast.Broadcasted) = find_djarray(bc.args)
find_djarray(args::Tuple) = find_djarray(find_djarray(args[1]), Base.tail(args))
find_djarray(x) = x
find_djarray(a::DJArray, rest) = a
find_djarray(::Any, rest) = find_djarray(rest)

Jets.getblock(A::DJArray, ::Type{<:Any}, iblock, indices) = getblock(A, iblock)

function DJArray_bcast_local_copyto!(iblock, dest, bc, S)
    setblock!(dest, getblock(bc, S, iblock, indices(dest)[iblock][1]), iblock)
    nothing
end
function Base.copyto!(dest::DJArray{T,1,<:AbstractArray{T,N}}, bc::Broadcast.Broadcasted{DJArrayStyle}) where {T,N}
    S = Broadcast.DefaultArrayStyle{N}
    cvxpmap(DJArray_bcast_local_copyto!, 1:length_blocks(dest), dest, bc, S)
    dest
end
# -->

#
# JetJSpace
#
struct JetJSpace{T,S<:Jets.JetAbstractSpace{T}} <: Jets.JetAbstractSpace{T,1}
    spaces::DJArray{S,1,Array{S,1}}
    indices::Vector{UnitRange{Int}}
end

JetJSpace_length(iblock, spaces) = length(getblock(spaces, iblock)[1])

function JetJSpace(spaces::DJArray{S,1,A}) where {S<:JetAbstractSpace, A<:AbstractArray{S}}
    n = cvxpmap(JetJSpace_length, 1:length_blocks(spaces), spaces)
    iₒ = 1
    indices = Vector{UnitRange{Int}}(undef, length(spaces))
    for (iblock,block) in enumerate(blocks(spaces))
        indices[iblock] = iₒ:(iₒ+n[iblock]-1)
        iₒ = indices[iblock][end] + 1
    end
    JetJSpace(spaces, indices)
end

Base.size(R::JetJSpace) = (indices(R)[end][end],)
Base.eltype(R::JetSpace{T}) where {T} = T
Base.eltype(R::Type{JetJSpace{T}}) where {T} = T
Base.eltype(R::Type{JetJSpace{T,S}}) where {T,S} = T

Jets.indices(R::JetJSpace) = R.indices
Jets.space(R::JetJSpace, iblock::Integer) = getblock(R.spaces, iblock)[1]
Jets.nblocks(R::JetJSpace) = length(R.spaces)
Distributed.procs(R::JetJSpace) = procs(R.spaces)

for f in (:Array, :ones, :rand, :zeros)
    @eval (Base.$f)(R::JetJSpace) = DJArray(iblock->($f)(space(R, iblock)), (nblocks(R),), procs(R))
end

#
# Block operator
#
JetJBlock_dom(iblock, ops) = domain(getblock(ops, 1, iblock)[1])
JetJBlock_rng(iblock, ops) = range(getblock(ops, iblock, 1)[1])
JetJBlock_spc_length(iblock, blkspaces) = length(getblock(blkspaces, iblock)[1])
function Jets.JetBlock(ops::DJArray{T,2}) where {T<:Jop}
    n1,n2 = size(ops)

    local dom
    if n2 == 1
        dom = remotecall_fetch(JetJBlock_dom, blocks(ops)[1,1].where, 1, ops)
    else
        indices_dom = Vector{UnitRange{Int}}(undef, n2)
        blkspaces_dom = DJArray(iblock->[JetJBlock_dom(iblock, ops)], (n2,), workers())
        n = cvxpmap(Int, JetJBlock_spc_length, 1:n2, blkspaces_dom)
        i1 = 1
        for i = 1:length(n)
            i2 = i1 + n[i] - 1
            indices_dom[i] = i1:i2
            i1 = i2 + 1
        end
        dom = JetJSpace(blkspaces_dom, indices_dom)
    end

    local rng
    if n1 == 1
        rng = remotecall_fetch(JetDJBlock_rng, block(ops)[1,1].where, 1, ops)
    else
        indices_rng = Vector{UnitRange{Int}}(undef, n1)
        blkspaces_rng = DJArray(iblock->[JetJBlock_rng(iblock, ops)], (n1,), workers())
        n = cvxpmap(Int, JetJBlock_spc_length, 1:n1, blkspaces_rng)
        i1 = 1
        for i = 1:length(n)
            i2 = i1 + n[i] - 1
            indices_rng[i] = i1:i2
            i1 = i2 + 1
        end
        rng = JetJSpace(blkspaces_rng, indices_rng)
    end
    Jet(dom = dom, rng = rng, f! = JetJBlock_f!, df! = JetJBlock_df!, df′! = JetJBlock_df′!, s = (ops=ops, dom=dom, rng=rng))
end

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function JetJBlock_local_f!(iblock, d, _m, ops)
    @info "^^^^^before getblock, iblock=$iblock, d.blocks[4].where=$(d.blocks[4].where), myid()=$(myid())"
    dᵢ = getblock(d, iblock)
    @info ">>>>>after getblock, iblock=$iblock, d.blocks[4].where=$(d.blocks[4].where), myid()=$(myid())"
    opsᵢ = getblock(ops, iblock)[1]
    m = localpart(_m)
    @info "***before, iblock=$iblock, extrema(dᵢ)=$(extrema(dᵢ)), myid()=$(myid())"
    mul!(dᵢ, opsᵢ, m)
    @info "---after, iblock=$iblock, extrema(dᵢ)=$(extrema(dᵢ)), myid()=$(myid())"
    nothing
end

function JetJBlock_f!(d::DJArray, m::AbstractArray; ops, dom, rng)
    pids = procs(d)
    _m = bcast(m, addmasterpid(pids))
    cvxpmap(JetJBlock_local_f!, 1:nblocks(d), d, _m, ops)
    d
end

function JetJBlock_local_df!(iblock, d, _m, ops)
    dᵢ = getblock(d, iblock)
    opsᵢ = getblock(ops, iblock)[1]
    m = localpart(_m)
    mul!(dᵢ, JopLn(opsᵢ), m)
    nothing
end

function JetJBlock_df!(d::DJArray, m::AbstractArray; ops, kwargs...)
    pids = procs(d)
    _m = bcast(m, addmasterpid(pids))
    cvxpmap(JetJBlock_local_df!, 1:nblocks(d), d, _m, ops)
    d
end

function JetJBlock_local_df′!(iblock, _m, d, ops)
    m = localpart(_m)
    dᵢ = getblock(d, iblock)
    opsᵢ = JopLn(getblock(ops, iblock)[1])
    m .+= opsᵢ' * dᵢ
    nothing
end

function JetJBlock_df′!(m::AbstractArray, d::DJArray; ops, kwargs...)
    pids = procs(d)
    m .= 0
    cvxpmapreduce!(m, JetJBlock_local_df′!, 1:nblocks(d), (d, ops))
end

function JetJBlock_local_point!(i, j, mₒ)
    op = getblock(state(j).ops, i)[1]
    Jets.point!(jet(op), getblock(mₒ, i))
    nothing
end

function Jets.point!(jet::Jet{D,R,typeof(JetJBlock_f!)}, mₒ::AbstractArray) where {D<:Jets.JetAbstractSpace, R<:Jets.JetAbstractSpace}
    cvxpmap(JetJBlock_local_point!, CartesianIndices(size(state(jet).ops)), jet, mₒ)
    jet
end

export DJArray, JetJSpace

end
