module JournaledJets

using Distributed, Jets, LinearAlgebra, DistributedOperations, Serialization, Schedulers

const registry = Dict{Tuple, Any}()
let JID::Int = 1
    global next_jid
    next_jid() = (id = JID; JID += 1; (myid(),id))
end

"""
    struct JArray

Journal Array Struct for collecting Jet blocks in a 'Journal' for fault tolerance
"""
mutable struct JArray{T,N,A<:AbstractArray{T}} <: AbstractArray{T,N}
    id::Tuple{Int,Int}
    pids::Vector{Int}
    blockmap::Array{Int,N}
    localblocks::Array{Union{A,Nothing}, N}
    indices::Array{NTuple{N,UnitRange{Int}},N}
    journal::Vector{Expr}
end

getblockmap_from_id(id) = registry[id].blockmap

"""
    getblockmap_from_id!(id, _blockmap)

Pull out a block map from the registry based on id
"""
function getblockmap_from_id!(id, _blockmap)
    blockmap = localpart(_blockmap)
    blockmap .= registry[id].blockmap
end

"""
    setblockmap_from_id!(id, _blockmap)

Add a block map to the registry based using id as the 'key'
"""
setblockmap_from_id!(id, _blockmap) = registry[id].blockmap .= localpart(_blockmap)

"""
    indices(A::JArray)

return indices from JArray passed in
"""
indices(A::JArray) = A.indices

"""
    indices_from_id!(id, _indices)

return indices from registry based on passed in id
"""
indices_from_id!(id, _indices) = registry[id].indices = _indices

"""
    blocklength_from_id(id, iblock)

compute local block length based on blockmap id and local block id
"""
blocklength_from_id(id, iblock) = length(registry[id].localblocks[iblock])

"""
    blocksize_from_id(id, iblock, idim)

compute local block size based on blockmap id and local block id and a given dimension
"""
blocksize_from_id(id, iblock, idim) = size(registry[id].localblocks[iblock], idim)

"""
    initialize_local_part(id, pids, nblocks::NTuple{N,Int}, ::Type{A}) where {N,A}

Create an empty block map for a given id in the registry
"""
function initialize_local_part(id, pids, nblocks::NTuple{N,Int}, ::Type{A}) where {N,A}
    blockmap = zeros(Int, nblocks)
    localblocks = Union{Nothing,A}[nothing for idx in CartesianIndices(nblocks)]
    indices = Array{NTuple{N,UnitRange{Int}},N}(undef, nblocks)
    journal = Expr[]
    x = JArray{eltype(A),N,A}(id, pids, blockmap, localblocks, indices, journal)
    registry[id] = x
    nothing
end

"""
    initialize_local_part(id, pids, nblocks::NTuple{N,Int}, ::Type{A}) where {N,A}

Create an empty block map for a given id in the registry
"""
function fill_local_part(iblock, id, f)
    x = registry[id]
    x.blockmap[iblock] = myid()
    x.localblocks[iblock] = f(iblock)
    nothing
end

"""
    finish_local_part(id, indices)
"""
function finish_local_part(id, indices)
    x = registry[id]
    x.indices .= indices
    nothing
end

"""
    JArray(f::Function, nblocks::NTuple{N,Int}) where {N}

"""
function JArray(f::Function, nblocks::NTuple{N,Int}) where {N}
    pids = workers()
    id = next_jid()

    A = Base.return_types(f, (CartesianIndex{N},))[1]
    T = eltype(A)
    @sync for pid in pids
        @async remotecall_fetch(initialize_local_part, pid, id, pids, nblocks, A)
    end

    epmap(fill_local_part, CartesianIndices(nblocks), id, f)

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

"""
    close_by_id(id)

Delete the work id from the registry if it hasnt already been deleted. Used when work finishes
"""
function close_by_id(id)
    #ccall(:printf, Cvoid, (Cstring,), "close_by_id\n")
    if haskey(registry, id)
        delete!(registry, id)
    end
    nothing
end

"""
    Base.close(x::JArray)
"""
function Base.close(x::JArray)
    #ccall(:printf, Cvoid, (Cstring,), "close\n")
    @sync for pid in workers()
        @async remotecall_fetch(close_by_id, pid, x.id)
    end
    if haskey(registry, x.id)
        delete!(registry, x.id)
    end
    nothing
end

"""
    getindices(id, nblocks::NTuple{1}, blockmap)
"""
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

"""
    getindices(id, nblocks::NTuple{N}, blockmap) where {N}
"""
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

"""
    Serialization.serialize(S::AbstractSerializer, x::JArray{T,N}) where {T,N}

Serilization implementation for JArray struct
"""
function Serialization.serialize(S::AbstractSerializer, x::JArray{T,N}) where {T,N}
    _where = worker_id_from_socket(S.io)

    Serialization.serialize_type(S, typeof(x))
    if _where ∈ x.pids || _where == x.id[1]
        serialize(S, (true, x.id))
    else
        serialize(S, (false, x.id))
        for n in [:pids, :blockmap, :indices, :journal]
            serialize(S, getfield(x, n))
        end
    end
end

"""
    Serialization.deserialize(S::AbstractSerializer, ::Type{J}) where {T,N,A,J<:JArray{T,N,A}}

Deserilization implementation for JArray struct
"""
function Serialization.deserialize(S::AbstractSerializer, ::Type{J}) where {T,N,A,J<:JArray{T,N,A}}
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
        localblocks = Union{A,Nothing}[nothing for idx in CartesianIndices(blockmap)]
        x = J(id, pids, blockmap, localblocks, indices, journal)
        registry[id] = x
    end

    x
end

"""
    getblock_and_delete_fromid!(_id, _whence, iblock)

Pop a block off the registry based on the block map id and block key
"""
function getblock_and_delete_fromid!(_id, _whence, iblock)
    x = registry[_id]
    blocks = x.localblocks
    block = copy(blocks[iblock]) # TODO... is this copy needed?
    blocks[iblock] = nothing

    blockmap = x.blockmap
    blockmap[iblock] = _whence

    if _whence ∉ x.pids
        sort!(push!(x.pids, _whence))
    end

    block
end

"""
    block_delete_fromid!(_id, _whence, iblock)

Remove a block from the registry based on the block map id and block key
"""
function block_delete_fromid!(_id, _whence, iblock)
    x = registry[_id]
    blocks = x.localblocks
    blocks[iblock] = nothing

    blockmap = x.blockmap
    blockmap[iblock] = _whence

    if _whence ∉ x.pids
        sort!(push!(x.pids, _whence))
    end
    nothing
end

"""
    update_blockmap_and_pids_fromid!(_id, _whence, iblock)

Update a block in the registry based on the block map id and block key with _whence
"""
function update_blockmap_and_pids_fromid!(_id, _whence, iblock)
    if haskey(registry, _id)
        x = registry[_id]
        x.blockmap[iblock] = _whence
        if _whence ∉ x.pids
            sort!(push!(x.pids, _whence))
        end
    end
    nothing
end

"""
    Jets.getblock(x::JArray{T,N,A}, iblock::CartesianIndex) where {T,N,A}

Get a block from a remote worked based on the iblock index
"""
function Jets.getblock(x::JArray{T,N,A}, iblock::CartesianIndex) where {T,N,A}
    _where = x.blockmap[iblock]
    _myid = myid()

    if _myid == _where
        return x.localblocks[iblock]::A
    end

    _id = x.id

    block = remotecall_fetch(getblock_and_delete_fromid!, _where, _id, _myid, iblock)::A
    x.blockmap[iblock] = _myid
    x.localblocks[iblock] = block

    if _myid ∉ x.pids
        sort!(push!(x.pids, _myid))
    end

    for pid in procs()
        if pid != _myid && pid != _where
            remotecall_fetch(update_blockmap_and_pids_fromid!, pid, _id, _myid, iblock)
        end
    end

    block
end
Jets.getblock(x::JArray, iblock::Int...) = getblock(x, CartesianIndex(iblock))

"""
    setblock_from_id!(id, xblock, iblock)

Set a block in the registry based on the blockmap id, and block index
"""
function setblock_from_id!(id, xblock, iblock)
    x = registry[id]
    _xblock = x.localblocks[iblock]
    _xblock .= xblock
    nothing
end

"""
    setblock_from_id!(x::JArray, xblock, iblock::CartesianIndex)

Set a block from remote worker based on the blockmap id, and block index
"""
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

"""
    getindex_from_id(id, iblock, i, N)

"""
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
function similar_localpart_from_id(id, pids, similar_id, ::Type{A}, N, ::Type{S}) where {A,S}
    if haskey(registry, id)
        x = registry[id]
        blockmap = copy(x.blockmap)
        localblocks = Union{Nothing,A}[similar(x.localblocks[idx], S) for idx in CartesianIndices(size(x.blockmap))]
        indices = copy(x.indices)
        _x = JArray{S,N,A}(similar_id, pids, blockmap, localblocks, indices, Expr[])
        registry[similar_id] = _x
    end
    nothing
end

function Base.similar(x::JArray{T,N,A}, ::Type{S}) where {T,N,A,S}
    id = x.id
    pids = x.pids
    similar_id = next_jid()
    _A = typeof(similar(A(undef,ntuple(_->0,N)), S))
    for pid in workers()
        remotecall_fetch(similar_localpart_from_id, pid, id, pids, similar_id, _A, N, S)
    end

    if haskey(registry, similar_id)
        _x = registry[similar_id]
    else
        blockmap = copy(x.blockmap)
        localblocks = Union{Nothing,_A}[similar(x.localblocks[idx], S) for idx in CartesianIndices(size(x.blockmap))]
        indices = copy(x.indices)
        _x = JArray{S,N,_A}(similar_id, pids, blockmap, localblocks, indices, Expr[])
        registry[similar_id] = _x
    end

    finalizer(close, _x)

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
    z = epmap(JArray_local_norm, 1:length(x.localblocks), x, p)
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
    z = epmap(JArray_local_dot, 1:length(x.localblocks), x, y)
    sum(z)
end

JArray_local_extrema(iblock, x) = extrema(getblock(x, iblock))
function Base.extrema(x::JArray)
    mnmx = epmap(JArray_local_extrema, 1:length(x.localblocks), x)
    mn, mx = mnmx[1]
    for i = 2:length(mnmx)
        _mn, _mx = mnmx[i]
        _mn < mn && (mn = _mn)
        _mx > mx && (mx = _mx)
    end
    mn,mx
end

JArray_local_fill!(iblock, x, a) = fill!(getblock(x, iblock), a)
function Base.fill!(x::JArray, a)
    epmap(JArray_local_fill!, 1:length(x.localblocks), x, a)
    x
end

function Base.copy(x::JArray{T,N,A}) where {T,N,A}
    _x = JArray(i->copy(getblock(x,i))::A, size(x.localblocks))
    _x.journal = deepcopy(x.journal)
    _x
end
Base.deepcopy(x::JArray) = copy(x)
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
    epmap(JArray_bcast_local_copyto!, 1:length(dest.localblocks), dest, bc, S)
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
    n = epmap(JetJSpace_length, 1:length(spaces.localblocks), spaces)
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
    @eval (Base.$f)(R::JetJSpace) = JArray(iblock->($f)(space(R, iblock[1])), (nblocks(R),))
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
        blkspaces_dom = JArray(iblock->[JetJBlock_dom(iblock, ops)], (n2,))
        n = epmap(Int, JetJBlock_spc_length, 1:n2, blkspaces_dom)
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
        blkspaces_rng = JArray(iblock->[JetJBlock_rng(iblock, ops)], (n1,))
        n = epmap(Int, JetJBlock_spc_length, 1:n1, blkspaces_rng)
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
    _m = bcast(m, procs())
    epmap(JetJBlock_local_f!, 1:nblocks(d), d, _m, ops)
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
    _m = bcast(m, procs())
    epmap(JetJBlock_local_df!, 1:nblocks(d), d, _m, ops)
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
    m .= 0
    epmapreduce!(m, JetJBlock_local_df′!, 1:nblocks(d), d, ops)
end

function JetJBlock_local_point!(iblock, j, mₒ)
    op = getblock(state(j).ops, iblock)[1]
    Jets.point!(jet(op), getblock(mₒ, iblock))
    nothing
end

function Jets.point!(jet::Jet{D,R,typeof(JetJBlock_f!)}, mₒ::AbstractArray) where {D<:Jets.JetAbstractSpace, R<:Jets.JetAbstractSpace}
    epmap(JetJBlock_local_point!, CartesianIndices(size(state(jet).ops)), jet, mₒ)
    jet
end

Jets.getblock(jet::Jet{D,R,typeof(JetJBlock_f!)}, i, j) where {D,R} = state(jet).ops[i,j]
Jets.getblock(A::JopLn{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)}} = JopLn(getblock(jet(A), i, j))
Jets.getblock(A::JopNl{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)}} = getblock(jet(A), i, j)
Jets.getblock(A::T, i, j) where {J<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)},T<:JopAdjoint{J}} = getblock(A.op, j, i)'
Jets.getblock(::Type{JopNl}, A::Jop{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)}} = getblock(jet(A), i, j)::JopNl
Jets.getblock(::Type{JopLn}, A::Jop{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)}} = JopLn(getblock(jet(A), i, j))

#
# Journal
#

# partial operator for journal replay
function JetJSubBlock_f!(d::AbstractArray, m::AbstractArray; ops, iblock, kwargs...)
    opsᵢ = getblock(ops, iblock, 1)[1]
    mul!(d, opsᵢ, m)
    d
end

function JetJSubBlock_df!(d::AbstractArray, m::AbstractArray; ops, iblock, kwargs...)
    opsᵢ = getblock(ops, iblock, 1)[1]
    mul!(d, JopLn(jet(opsᵢ)), m)
    d
end

journal_getblock(x::JArray{T}, iblock::CartesianIndex) where {T} = getblock(x, iblock)
journal_getblock(x::JopNl, iblock::CartesianIndex) = JopNl(journal_getblock(jet(x), iblock))
journal_getblock(x::JopLn, iblock::CartesianIndex) = JopLn(journal_getblock(jet(x), iblock))
#journal_getblock(x, iblock::CartesianIndex) = x
journal_getblock(x, iblock::Int) = journal_getblock(x, CartesianIndex(iblock))

function journal_getblock(x::Jet, iblock::CartesianIndex)
    dom = domain(x)
    rng = range(getblock(state(x).ops, iblock.I[1], 1)[1])
    Jet(;dom = dom, rng = rng, f! = JetJSubBlock_f!, df! = JetJSubBlock_df!, df′! = _->@info("not implemented"), s=(ops=state(x).ops, iblock=iblock.I[1]))
end

journal!(x::JArray, expr) = push!(x.journal, expr)
journal!(x::Jop, expr) = journal!(state(x).ops, expr)

emptyjournal!(x::JArray) = x.journal = x.journal[1:1]
emptyjournal!(x::Jop) = emptyjournal!(state(x).ops)

journal(x::JArray) = x.journal
journal(x::Jop) = state(x).ops.journal

macro journal(expr::Expr)
    sexpr = string(expr)

    quote
        x = $(esc(expr))
        journal!(x, Meta.parse($sexpr))
        x
    end
end

macro journal(x::Symbol, expr::Expr)
    sexpr = string(expr)
    quote
        journal!($(esc(x)), Meta.parse($sexpr))
        $(esc(expr))
    end
end

macro journal(r::QuoteNode, x::Symbol, expr::Expr)
    sexpr = string(expr)
    quote
        $(esc(r)) == :reset && emptyjournal!($(esc(x)))
        journal!($(esc(x)), Meta.parse($sexpr))
        $(esc(expr))
    end
end

function replay_construct!(mod, x::JArray, iblock::CartesianIndex)
    expr = x.journal[1]

    # using the JArray constructor is a special case
    for iexpr = 1:length(expr.args)
        if expr.args[iexpr] == :JArray
            _where = x.blockmap[iblock]
            _myid = myid()
            _id = x.id

            if _myid != _where
                remotecall_fetch(block_delete_fromid!, _where, _id, _myid, iblock)
                x.blockmap[iblock] = _myid
            end

            if _myid ∉ x.pids
                sort!(push!(x.pids, _myid))
            end

            for pid in procs()
                if pid != _myid && pid != _where
                    remotecall_fetch(update_blockmap_and_pids_fromid!, pid, _id, _myid, iblock)
                end
            end

            _expr = expr.args[iexpr+1]
            x.localblocks[iblock] = @eval mod $(_expr)($iblock)
            return nothing
        end
    end

    _expr = metablocks(mod, expr, iblock)
    x.localblocks[iblock] = @eval mod $(_expr)

    nothing
end
replay_construct!(mod, x::JArray, iblock) = replay_construct!(mod, x, CartesianIndex(iblock))

function replay!(mod, symx::String, x::JArray, iblock)
    replay_construct!(mod, x, iblock)
    for expr in x.journal[2:end]
        if block_in_play(Symbol(symx), expr, iblock)
            _expr = metablocks(mod, expr, iblock)
            @eval mod $(_expr)
        end
    end
end
replay!(mod, symx, x::Jop, iblock::Int) = replay!(mod, symx, state(x).ops, iblock)

macro replay(x, iblock)
    _x = string(x)
    :(replay!(@__MODULE__, $_x, $(esc(x)), $(esc(iblock))))
end

function block_in_play(symx::Symbol, expr::Expr, iblock)
    inplay = true
    args = expr.args
    for (iarg, arg) in enumerate(args)
        if arg == :getblock
            isx = args[iarg+1] == symx
            isi = CartesianIndex(iblock) == CartesianIndex(eval(args[iarg+2]))
            (isx && isi) && (return true)
            (isx && !isi) && (return false)
        end
        inplay = block_in_play(symx, arg, iblock)
        inplay == false && (return inplay)
    end
    true
end
block_in_play(symx::Symbol, notexpr, iblock) = true

function metablocks!(mod::Module, expr::Expr, iblock)
    if expr.head == :call
        _args = @view expr.args[2:end]
        metablocks!(mod, _args, iblock)
    else
        metablocks!(mod, expr.args, iblock)
    end
    expr
end

function metablocks!(mod::Module, arg::Symbol, iblock)
    e = :(@isdefined $arg)
    isdefined = @eval mod $e
    if isdefined
        _arg = @eval mod $arg
        T = Union{
            JArray,
            Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)},
            JopNl{<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)}},
            JopLn{<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetJBlock_f!)}}
            }
        isblocked = isa(_arg, T)
        if isblocked
            return :(JournaledJets.journal_getblock($arg, $iblock))
        end
    end
    arg
end

function metablocks!(mod::Module, args::AbstractArray, iblock)
    for iarg in 1:length(args)
        args[iarg] = metablocks!(mod, args[iarg], iblock)
    end
    args
end

metablocks!(mod, arg, iblock) = arg

function metablocks(mod::Module, expr::Expr, iblock)
    _expr = deepcopy(expr)
    metablocks!(mod, _expr, iblock)
    _expr
end

export JArray, JetJSpace, @journal, @replay

end
