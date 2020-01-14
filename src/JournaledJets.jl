module JournaledJets

using Distributed, Jets, LinearAlgebra, ParallelOperations, Serialization, Schedulers

const registry = Dict{Tuple, Any}()
let JID::Int = 1
    global next_jid
    next_jid() = (id = JID; JID += 1; (myid(),id))
end

mutable struct JArray{T,N,A<:AbstractArray{T}} <: AbstractArray{T,N}
    id::Tuple{Int,Int}
    pids::Vector{Int}
    blockmap::Array{Int,N}
    localblocks::Array{Union{A,Nothing}, N}
    indices::Array{NTuple{N,UnitRange{Int}},N}
    journal::Vector{Expr}
end

getblockmap_from_id(id) = registry[id].blockmap

function getblockmap_from_id!(id, _blockmap)
    blockmap = localpart(_blockmap)
    blockmap .= registry[id].blockmap
end

setblockmap_from_id!(id, _blockmap) = registry[id].blockmap .= localpart(_blockmap)

indices(A::JArray) = A.indices
indices_from_id!(id, _indices) = registry[id].indices = _indices

blocklength_from_id(id, iblock) = length(registry[id].localblocks[iblock])
blocksize_from_id(id, iblock, idim) = size(registry[id].localblocks[iblock], idim)

function initialize_local_part(id, pids, nblocks::NTuple{N,Int}, ::Type{A}) where {N,A}
    blockmap = zeros(Int, nblocks)
    localblocks = Union{Nothing,A}[nothing for idx in CartesianIndices(nblocks)]
    indices = Array{NTuple{N,UnitRange{Int}},N}(undef, nblocks)
    journal = Expr[]
    x = JArray{eltype(A),N,A}(id, pids, blockmap, localblocks, indices, journal)
    registry[id] = x
    nothing
end

function fill_local_part(iblock, id, f)
    x = registry[id]
    x.blockmap[iblock] = myid()
    x.localblocks[iblock] = f(iblock)
    nothing
end

function finish_local_part(id, indices)
    x = registry[id]
    x.indices .= indices
    nothing
end

function JArray(f::Function, nblocks::NTuple{N,Int}, pids) where {N}
    id = next_jid()

    A = Base.return_types(f, (CartesianIndex{N},))[1]
    T = eltype(A)
    @sync for pid in pids
        @async remotecall_fetch(initialize_local_part, pid, id, pids, nblocks, A)
    end

    cvxpmap(fill_local_part, CartesianIndices(nblocks), id, f)

    blockmap = zeros(Int, nblocks)
    x = ArrayFutures(blockmap)
    @sync for pid in pids
        @async remotecall_fetch(getblockmap_from_id!, pid, id, x)
    end
    reduce!(x)
    x = bcast(blockmap)
    @sync for pid in pids
        @async remotecall_fetch(setblockmap_from_id!, pid, id, x)
    end

    indices = getindices(id, nblocks, blockmap)

    @sync for pid in pids
        @async remotecall_fetch(finish_local_part, pid, id, indices)
    end

    local x
    if myid() ∈ pids
        x = registry[id]
    else
        localblocks = Union{Nothing,A}[nothing for idx in CartesianIndices(nblocks)]
        x = JArray{T,N,A}(id, pids, blockmap, localblocks, indices, Expr[])
        registry[id] = x
    end

    finalizer(close, x)

    x
end

function close_by_id(id)
    #ccall(:printf, Cvoid, (Cstring,), "close_by_id\n")
    if id ∈ keys(registry)
        delete!(registry, id)
    end
    nothing
end

function Base.close(x::JArray)
    #ccall(:printf, Cvoid, (Cstring,), "close\n")
    @sync for pid in x.pids
        if pid ∈ workers()
            @async remotecall_fetch(close_by_id, pid, x.id)
        end
    end
    if x.id ∈ keys(registry)
        delete!(registry, x.id)
    end
    nothing
end

function getindices(id, nblocks::NTuple{1}, blockmap)
    indices = Vector{NTuple{1,UnitRange{Int}}}(undef, nblocks)
    i1 = 1
    for iblock in CartesianIndices(nblocks)
        i2 = i1 + remotecall_fetch(blocklength_from_id, blockmap[iblock], id, iblock) - 1
        indices[iblock] = (i1:i2,)
        i1 = i2 + 1
    end
    indices
end

function getindices(id, nblocks::NTuple{N}, blockmap) where {N}
    i1 = [Int[] for i=1:N]
    i2 = [Int[] for i=1:N]
    for idim = 1:N
        _i1 = 1
        for iblock = 1:nblocks[idim]
            _i2 = _i1 + remotecall_fetch(blocksize_from_id, blockmap[iblock], id, iblock, idim) - 1
            push!(i1[idim], _i1)
            push!(i2[idim], _i2)
            _i1 = _i2 + 1
        end
    end

    indices = Array{NTuple{N,UnitRange{Int}},N}(undef, nblocks)
    for (i,idx) in enumerate(CartesianIndices(nblocks))
        indices[idx] = ntuple(idim->i1[idim][idx[idim]]:i2[idim][idx[idim]], N)
    end

    indices
end

# JArray serialization implementation <--
function Serialization.serialize(S::AbstractSerializer, x::JArray{T,N,A}) where {T,N,A}
    _where = worker_id_from_socket(S.io)
    Serialization.serialize_type(S, typeof(x))
    if (_where ∈ x.pids) || (_where == x.id[1])
        serialize(S, (true, x.id))
    else
        serialize(S, (false, x.id))
        for n in [:pids, :blockmap, :indices, :journal]
            serialize(S, getfield(x, n))
        end
        serialize(S, A)
    end
end

function Serialization.deserialize(S::AbstractSerializer, ::Type{T}) where {T<:JArray}
    what = deserialize(S)
    id_only = what[1]
    id = what[2]

    local x
    if id_only
        x = registry[id]
    else
        pids = deserialize(S)
        blockmap = deserialize(S)
        indices = deserialize(S)
        journal = deserialize(S)
        A = deserialize(S)
        localblocks = Union{A,Nothing}[nothing for idx in CartesianIndices(blockmap)]
        x = T(id, pids, blockmap, localblocks, indices, journal)
    end
    x
end
# -->

# JArray interface implementation <--
function getblock_and_delete_fromid!(_id, _whence, iblock)
    x = registry[_id]
    blocks = x.localblocks
    block = copy(blocks[iblock])
    blocks[iblock] = nothing

    blockmap = x.blockmap
    blockmap[iblock] = _whence

    block
end

function update_blockmap_fromid!(_id, _whence, iblock)
    x = registry[_id]
    x.blockmap[iblock] = _whence
    nothing
end

function Jets.getblock(x::JArray{T,N,A}, iblock::CartesianIndex) where {T,N,A}
    _where = x.blockmap[iblock]
    _myid = myid()

    if _myid == _where
        return x.localblocks[iblock]
    end

    _id = x.id

    block = remotecall_fetch(getblock_and_delete_fromid!, _where, _id, _myid, iblock)::A
    x.blockmap[iblock] = _myid
    x.localblocks[iblock] = block

    pids = x.id[1] ∈ x.pids ? x.pids : [x.pids;x.id[1]]
    for pid in pids
        if pid != _myid && pid != _where
            begin
                remotecall_fetch(update_blockmap_fromid!, pid, _id, _myid, iblock)
            end
        end
    end

    block
end
Jets.getblock(x::JArray, iblock::Int...) = getblock(x, CartesianIndex(iblock))

function setblock_from_id!(id, xblock, iblock)
    x = registry[id]
    _xblock = x.localblocks[iblock]
    _xblock .= xblock
    nothing
end

function Jets.setblock!(x::JArray, xblock, iblock::CartesianIndex)
    if myid() == x.blockmap[iblock]
        _xblock = x.localblocks[iblock]
        _xblock .= xblock
        return nothing
    end

    remotecall_fetch(setblock_from_id!, x.blockmap[iblock], x.id, xblock, iblock)
    nothing
end
Jets.setblock!(x::JArray, xblock, iblock::Int...) = setblock!(x, xblock, CartesianIndex(iblock))

size_dim(x::JArray{T,N}, idim) where {T,N} = x.indices[ntuple(_idim->_idim==idim ? size(x.indices)[idim] : 1, N)...][idim][end]
Base.size(x::JArray{T,N}) where {T,N} = ntuple(idim->size_dim(x, idim), N)

function getindex_from_id(id, iblock, i, N)
    x = registry[id]
    iₒ = ntuple(idim->indices(x)[iblock][idim][1], N)
    δ = CartesianIndex(ntuple(idim->i[idim]-iₒ[idim]+1, N))
    a = x.localblocks[iblock][δ]
    a
end

function Base.getindex(x::JArray{T,N}, i::CartesianIndex) where {T,N}
    iblock = findfirst(rng->mapreduce(idim->i[idim]∈rng[idim], &, 1:N), x.indices)
    remotecall_fetch(getindex_from_id, x.blockmap[iblock], x.id, iblock, i, N)
end
Base.getindex(x::JArray, i::Int...) = getindex(x, CartesianIndex(i))

Base.similar(x::Nothing, ::Type{S}) where {S} = nothing
function similar_localpart_from_id(id, similar_id, ::Type{A}, N, ::Type{S}) where {A,S}
    x = registry[id]
    pids = copy(x.pids)
    blockmap = copy(x.blockmap)
    localblocks = Union{Nothing,A}[similar(x.localblocks[idx], S) for idx in CartesianIndices(size(x.blockmap))]
    indices = copy(x.indices)
    _x = JArray{S,N,A}(similar_id, pids, blockmap, localblocks, indices, Expr[])
    registry[similar_id] = _x
    nothing
end

function Base.similar(x::JArray{T,N,A}, ::Type{S}) where {T,N,A,S}
    id = x.id
    similar_id = next_jid()
    _A = typeof(similar(A(undef,ntuple(_->0,N)), S))
    for pid in x.pids
        remotecall_fetch(similar_localpart_from_id, pid, id, similar_id, _A, N, S)
    end

    if myid() ∈ x.pids
        @info "similar, 1"
        _x = registry[similar_id]
    else
        pids = copy(x.pids)
        blockmap = copy(x.blockmap)
        localblocks = Union{Nothing,_A}[similar(x.localblocks[idx], S) for idx in CartesianIndices(size(x.blockmap))]
        indices = copy(x.indices)
        _x = JArray{S,N,_A}(similar_id, pids, blockmap, localblocks, indices, Expr[])
        registry[similar_id] = _x
    end

    #finalizer(close, _x)

    _x
end

Distributed.procs(x::JArray) = x.pids
Jets.nblocks(x::JArray) = length(x.localblocks)

function Base.collect(x::JArray{T,1,A}) where {T,A}
    _x = Vector{A}(undef, length(x.localblocks))
    n = 0
    @sync for iblock = 1:length(_x)
        @async begin
            _x[iblock] = remotecall_fetch(getblock, x.blockmap[iblock], x, iblock)
        end
    end
    Jets.BlockArray(_x, [x.indices[i][1] for i=1:length(x.indices)])
end

Base.convert(::Type{S}, x::JArray{T,1,A}) where {S<:Jets.BlockArray,T,A} = collect(x)
Base.convert(::Array, x::JArray{T,1,A}) where {T,A} = convert(Array, collect(x))

JArray_local_norm(iblock, x::JArray, p) = norm(getblock(x, iblock), p)
function LinearAlgebra.norm(x::JArray{T}, p::Real=2) where {T}
    _T = float(real(T))
    z = cvxpmap(JArray_local_norm, 1:length(x.localblocks), x, p)
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

JArray_local_dot(iblock, x, y) = dot(getblock(x, iblock), getblock(y, iblock))
function LinearAlgebra.dot(x::JArray, y::JArray)
    z = cvxpmap(JArray_local_dot, 1:length(x.localblocks), x, y)
    sum(z)
end

JArray_local_extrema(iblock, x) = extrema(getblock(x, iblock))
function Base.extrema(x::JArray)
    mnmx = cvxpmap(JArray_local_extrema, 1:length(x.localblocks), x)
    mn, mx = mnmx[1]
    for i = 2:length(mnmx)
        _mn, _mx = mnmx[i]
        _mn < mn && (mn = _mn)
        _mx > mx && (mx = _mx)
    end
    mn,mx
end

JArray_local_fill!(iblock, x, a) = fill!(getblock(x, iblock), a)
Base.fill!(x::JArray, a) = cvxpmap(JArray_local_fill!, 1:length(x.localblocks), x, a)
# -->

# JArray broadcasting implementation --<
struct JArrayStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:JArray}) = JArrayStyle()
JArrayStyle(::Val{1}) = JArrayStyle()

Base.similar(bc::Broadcast.Broadcasted{JArrayStyle}, ::Type{T}) where {T} = similar(find_jarray(bc), T)
find_jarray(bc::Broadcast.Broadcasted) = find_jarray(bc.args)
find_jarray(args::Tuple) = find_jarray(find_jarray(args[1]), Base.tail(args))
find_jarray(x) = x
find_jarray(a::JArray, rest) = a
find_jarray(::Any, rest) = find_jarray(rest)

Jets.getblock(A::JArray, ::Type{<:Any}, iblock, indices) = getblock(A, iblock)

function JArray_bcast_local_copyto!(iblock, dest, bc, S)
    _x = getblock(bc, S, iblock, dest.indices[iblock][1])
    setblock!(dest, _x, iblock)
    nothing
end
function Base.copyto!(dest::JArray{T,1,<:AbstractArray{T,N}}, bc::Broadcast.Broadcasted{JArrayStyle}) where {T,N}
    S = Broadcast.DefaultArrayStyle{N}
    cvxpmap(JArray_bcast_local_copyto!, 1:length(dest.localblocks), dest, bc, S)
    dest
end
# -->

#
# JetJSpace
#
struct JetJSpace{T,S<:Jets.JetAbstractSpace{T}} <: Jets.JetAbstractSpace{T,1}
    spaces::JArray{S,1,Array{S,1}}
    indices::Vector{UnitRange{Int}}
end

JetJSpace_length(iblock, spaces) = length(getblock(spaces, iblock)[1])

function JetJSpace(spaces::JArray{S,1,A}) where {S<:JetAbstractSpace, A<:AbstractArray{S}}
    n = cvxpmap(JetJSpace_length, 1:length(spaces.localblocks), spaces)
    iₒ = 1
    indices = Vector{UnitRange{Int}}(undef, length(spaces))
    for iblock = 1:length(indices)
        indices[iblock] = iₒ:(iₒ+n[iblock]-1)
        iₒ = indices[iblock][end] + 1
    end
    JetJSpace(spaces, indices)
end

Jets.indices(R::JetJSpace) = R.indices

Base.size(R::JetJSpace) = (R.indices[end][end],)
Base.eltype(R::JetSpace{T}) where {T} = T
Base.eltype(R::Type{JetJSpace{T}}) where {T} = T
Base.eltype(R::Type{JetJSpace{T,S}}) where {T,S} = T

Jets.space(R::JetJSpace, iblock::Integer) = getblock(R.spaces, iblock)[1]
Jets.nblocks(R::JetJSpace) = length(R.spaces)
Distributed.procs(R::JetJSpace) = procs(R.spaces)

for f in (:Array, :ones, :rand, :zeros)
    @eval (Base.$f)(R::JetJSpace) = JArray(iblock->($f)(space(R, iblock[1])), (nblocks(R),), procs(R))
end

#
# Block operator
#
JetJBlock_dom(iblock, ops) = domain(getblock(ops, 1, iblock[1])[1])
JetJBlock_rng(iblock, ops) = range(getblock(ops, iblock[1], 1)[1])
JetJBlock_spc_length(iblock, blkspaces) = length(getblock(blkspaces, iblock)[1])
function Jets.JetBlock(ops::JArray{T,2}) where {T<:Jop}
    n1,n2 = size(ops)

    local dom
    if n2 == 1
        dom = remotecall_fetch(JetJBlock_dom, ops.blockmap[1,1], 1, ops)
    else
        indices_dom = Vector{UnitRange{Int}}(undef, n2)
        blkspaces_dom = JArray(iblock->[JetJBlock_dom(iblock, ops)], (n2,), workers())
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
        rng = remotecall_fetch(JetJBlock_rng, ops.blockmap[1,1], 1, ops)
    else
        indices_rng = Vector{UnitRange{Int}}(undef, n1)
        blkspaces_rng = JArray(iblock->[JetJBlock_rng(iblock, ops)], (n1,), workers())
        n = cvxpmap(Int, JetJBlock_spc_length, 1:n1, blkspaces_rng)
        i1 = 1
        for i = 1:length(n)
            i2 = i1 + n[i] - 1
            indices_rng[i] = i1:i2
            i1 = i2 + 1
        end
        rng = JetJSpace(blkspaces_rng, indices_rng)
    end
    Jet(dom = dom, rng = rng, f! = JetJBlock_f!, df! = JetJBlock_df!, df′! = JetJBlock_df′!, s = (ops=ops,))
end

function addmasterpid(pids)
    if myid() ∉ pids
        return [myid();pids]
    end
    pids
end

function JetJBlock_local_f!(iblock, d, _m, ops)
    dᵢ = getblock(d, iblock)
    opsᵢ = getblock(ops, iblock)[1]
    m = localpart(_m)
    mul!(dᵢ, opsᵢ, m)
    nothing
end

function JetJBlock_f!(d::JArray, m::AbstractArray; ops, kwargs...)
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

function JetJBlock_df!(d::JArray, m::AbstractArray; ops, kwargs...)
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

function JetJBlock_df′!(m::AbstractArray, d::JArray; ops, kwargs...)
    pids = procs(d)
    m .= 0
    cvxpmapreduce!(m, JetJBlock_local_df′!, 1:nblocks(d), (d, ops))
end

function JetJBlock_local_point!(iblock, j, mₒ)
    op = getblock(state(j).ops, iblock)[1]
    Jets.point!(jet(op), getblock(mₒ, iblock))
    nothing
end

function Jets.point!(jet::Jet{D,R,typeof(JetJBlock_f!)}, mₒ::AbstractArray) where {D<:Jets.JetAbstractSpace, R<:Jets.JetAbstractSpace}
    cvxpmap(JetJBlock_local_point!, CartesianIndices(size(state(jet).ops)), jet, mₒ)
    jet
end

export JArray, JetJSpace

end
