module JournaledJets

using Distributed, Jets, LinearAlgebra, Schedulers

struct DJArray{T,A<:AbstractArray{T,1}} <: AbstractArray{T,1}
    blocks::Vector{Future}
    indices::Vector{UnitRange{Int}}
    journal::Vector{Expr}
end

DJArray_construct_block(iblock, f) = remotecall(f, myid(), iblock)
DJArray_block_length(block) = length(fetch(block))
DJArray_block_typeof(block) = typeof(fetch(block))

function DJArray(f::Function, nblocks::Integer, pids)
    blocks = cvxpmap(DJArray_construct_block, 1:nblocks, f)

    indices = Vector{UnitRange{Int}}(undef, nblocks)
    i1 = 1
    for (iblock,block) in enumerate(blocks)
        i2 = i1 + remotecall_fetch(DJArray_block_length, block.where, block) - 1
        indices[iblock] = i1:i2
        i1 = i2 + 1
    end

    A = remotecall_fetch(DJArray_block_typeof, blocks[1].where, blocks[1])
    T = eltype(A)

    DJArray{T,A}(blocks, indices, Expr[])
end

# DJArray interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:DJArray} = IndexLinear()
Base.size(x::DJArray) = (x.indices[end][end],)
Jets.indices(x::DJArray) = x.indices
Jets.getblock(x::DJArray{T,A}, iblock::Int) where {T,A} = fetch(x.blocks[iblock])::A
DJArray_setblock_local!(block, xblock, ::Type{A}) where {A} = fetch(block)::A .= xblock
Jets.setblock!(x::DJArray{T,A}, iblock::Int, xblock) where {T,A} = remotecall_fetch(DJArray_setblock_local!, blocks(x)[iblock].where, blocks(x)[iblock], xblock, A)
blocks(x::DJArray) = x.blocks
nblocks(x::DJArray) = length(blocks(x))

DJArray_getindex(::Type{A}, block_future, j) where {A} = getindex(fetch(block_future)::A, j)
function Base.getindex(x::DJArray{T,A}, i::Int) where {T,A}
    iblock = findfirst(rng->i∈rng, indices(x))
    block_future = x.blocks[iblock]
    iₒ = indices(x)[iblock][1]
    remotecall_fetch(DJArray_getindex, block_future.where, A, block_future, i-iₒ+1)
end

_DJArray_similar_block(iblock, x::DJArray, ::Type{S}) where {S} = similar(getblock(x, iblock), S)
DJArray_similar_block(iblock, x::DJArray, ::Type{S}) where {S} = remotecall(_DJArray_similar_block, myid(), iblock, x, S)
DJArray_similar_typeof(block) = typeof(fetch(block))
function Base.similar(x::DJArray{T,A}, ::Type{S}) where {T,A,S}
    _indices = copy(indices(x))
    _block_futures = cvxpmap(DJArray_similar_block, 1:nblocks(x), x, S)
    _A = remotecall_fetch(DJArray_similar_typeof, _block_futures[1].where, _block_futures[1])
    DJArray{S,_A}(_block_futures, _indices, Expr[])
end

DJArray_local_norm(iblock, x::DJArray, p) = norm(getblock(x, iblock), p)

function LinearAlgebra.norm(x::DJArray{T}, p::Real=2) where {T}
    _T = float(real(T))
    z = cvxpmap(DJArray_local_norm, 1:nblocks(x), x, p)
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
    z = cvxpmap(DJArray_local_dot, 1:nblocks(x), x, y)
    sum(z)
end

DJArray_local_extrema(iblock, x::DJArray) = extrema(getblock(x, iblock))
function Base.extrema(x::DJArray{T}) where {T}
    mnmx = cvxpmap(DJArray_local_extrema, 1:nblocks(x), x)
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

function Base.collect(x::DJArray{T,A}) where {T,A}
    _x = A[]
    n = 0
    for iblock = 1:nblocks(x)
        push!(_x, getblock(x, iblock))
    end
    Jets.BlockArray(_x, indices(x))
end

Base.convert(::Type{T}, x::DJArray) where {T<:Jets.BlockArray} = collect(x)
Base.convert(::Array, x::DJArray) = convert(Array, collect(x))
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
    setblock!(dest, iblock, getblock(bc, S, iblock, indices(dest)[iblock]))
    nothing
end
function Base.copyto!(dest::DJArray{T,<:AbstractArray{T,N}}, bc::Broadcast.Broadcasted{DJArrayStyle}) where {T,N}
    S = Broadcast.DefaultArrayStyle{N}
    cvxpmap(DJArray_bcast_local_copyto!, 1:nblocks(dest), dest, bc, S)
    dest
end
# -->

#
# JetJSpace
#
struct JetJSpace{T,S<:Jets.JetAbstractSpace{T}} <: Jets.JetAbstractSpace{T,1}
    spaces::DJArray{S,Array{S,1}}
    indices::Vector{UnitRange{Int}}
end

JetJSpace_length(iblock, spaces) = length(getblock(spaces, iblock)[1])

function JetJSpace(spaces::DJArray{S,A}) where {S<:JetAbstractSpace, A<:AbstractArray{S}}
    n = cvxpmap(JetJSpace_length, 1:nblocks(spaces), spaces)
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

#
# Block operator
#
function Jets.JetBlock(ops::DJArray{T,2}) where {T<:Jop}
    n1,n2 = length(indices(ops, 1)),length(indices(ops,2))
    Jet(dom = dom, rng = rng, f! = JetJBlock_f!, df! = JetJBlock_df!, df′! = JetJBlock_df′!, s = (ops=_ops, dom=dom, rng=rng))
end

export DJArray, JetJSpace

end
