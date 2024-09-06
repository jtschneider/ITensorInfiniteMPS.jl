
mutable struct InfiniteBlockMPO <: AbstractInfiniteMPS
  data::CelledVector{Matrix{ITensor}}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteBlockMPO) = mpo.data.translator
data(mpo::InfiniteBlockMPO) = mpo.data

# TODO better printing?
function Base.show(io::IO, M::InfiniteBlockMPO)
  print(io, "$(typeof(M))")
  (length(M) > 0) && print(io, "\n")
  for i in eachindex(M)
    if !isassigned(M, i)
      println(io, "#undef")
    else
      A = M[i]
      println(io, "Matrix tensor of size $(size(A))")
      for k in 1:size(A, 1), l in 1:size(A, 2)
        if !isassigned(A, k + (size(A, 1) - 1) * l)
          println(io, "[($k, $l)] #undef")
        elseif isempty(A[k, l])
          println(io, "[($k, $l)] empty")
        else
          println(io, "[($k, $l)] $(inds(A[k, l]))")
        end
      end
    end
  end
end

function getindex(ψ::InfiniteBlockMPO, n::Integer)
  return ψ.data[n]
end

function InfiniteBlockMPO(arrMat::Vector{Matrix{ITensor}})
  return InfiniteBlockMPO(arrMat, 0, size(arrMat, 1), false)
end

function InfiniteBlockMPO(data::Vector{Matrix{ITensor}}, translator::Function)
  return InfiniteBlockMPO(CelledVector(data, translator), 0, size(data, 1), false)
end

function InfiniteBlockMPO(data::CelledVector{Matrix{ITensor}}, m::Int64, n::Int64)
  return InfiniteBlockMPO(data, m, n, false)
end

function InfiniteBlockMPO(data::CelledVector{Matrix{ITensor}})
  return InfiniteBlockMPO(data, 0, size(data, 1), false)
end

function ITensors.siteinds(A::InfiniteBlockMPO)
  data = [dag(only(filterinds(uniqueinds(A[1][1, 1], A[2][1, 1]); plev=0)))]
  for x in 2:(nsites(A) - 1)
    append!(
      data,
      [
        dag(
          only(filterinds(uniqueinds(A[x][1, 1], A[x - 1][1, 1], A[x + 1][1, 1]); plev=0))
        ),
      ],
    )
  end
  append!(
    data,
    [dag(only(filterinds(uniqueinds(A[nsites(A)][1, 1], A[nsites(A) - 1][1, 1]); plev=0)))],
  )
  return CelledVector(data, translator(A))
end

function ITensors.splitblocks(H::InfiniteBlockMPO)
  H = copy(H)
  N = nsites(H)
  for j in 1:N
    for n in 1:length(H)
      H[j][n] = splitblocks(H[j][n])
    end
  end
  return H
end

function find_all_links(Hm::Matrix{ITensor})
  is = inds(Hm[1, 1]) #site inds
  lx, ly = size(Hm)
  #We extract the links from the order-3 tensors on the first column and line
  #We add dummy indices if there is no relevant indices
  ir = only(uniqueinds(Hm[1, 2], is))
  ir0 = Index(ITensors.trivial_space(ir); dir=dir(ir), tags="Link,extra")
  il0 = dag(ir0)
  left_links = typeof(ir)[]
  for x in 1:lx
    temp = uniqueinds(Hm[x, 1], is)
    if length(temp) == 0
      append!(left_links, [il0])
    elseif length(temp) == 1
      append!(left_links, temp)
    else
      error("ITensor does not seem to be of the correct order")
    end
  end
  right_links = typeof(ir)[]
  for x in 1:lx
    temp = uniqueinds(Hm[1, x], is)
    if length(temp) == 0
      append!(right_links, [ir0])
    elseif length(temp) == 1
      append!(right_links, temp)
    else
      error("ITensor does not seem to be of the correct order")
    end
  end
  return left_links, right_links
end

"""
    local_mpo_block_projectors(is::Index; new_tags = tags(is))

  Build the projectors on the three parts of the itensor used to split a MPO into an InfiniteBlockMPO
  More precisely, create projectors on the first dimension, the 2:end-1 and the last dimension of the index
  Input: is the Index to split
  Output: the triplet of projectors (first, middle, last)
  Optional arguments: new_tags: if we want to change the tags of the index.
"""
function local_mpo_block_projectors(is::Index; tags=tags(is))
  old_dim = dim(is)
  #Build the local projectors.
  #We have to differentiate between the dense and the QN case
  #Note that as far as I know, the MPO even dense is always guaranteed to have identities at both corners
  #If it is not the case, my construction will not work
  top = onehot(dag(is) => 1)
  bottom = onehot(dag(is) => old_dim)
  if length(is.space) == 1
    new_ind = Index(is.space - 2; tags=tags)
    mat = zeros(new_ind.space, is.space)
    for x in 1:(new_ind.space)
      mat[x, x + 1] = 1
    end
    middle = ITensor(copy(mat), new_ind, dag(is))
  else
    new_ind = Index(is.space[2:(end - 1)]; dir=dir(is), tags=tags)
    middle = ITensors.BlockSparseTensor(
      Float64,
      undef,
      Block{2}[Block(x, x + 1) for x in 1:length(new_ind.space)],
      (new_ind, dag(is)),
    )
    for x in 1:length(new_ind.space)
      dim_block = new_ind.space[x][2]
      ITensors.blockview(middle, Block(x, x + 1)) .= diagm(0 => ones(dim_block))
    end
    middle = itensor(middle)
  end
  return top, middle, bottom
end

"""
    local_mpo_blocks(tensor::ITensor, left_ind::Index, right_ind::Index; left_tags = tags(inds[1]), right_tags = tags(inds[2]), ...)

  Converts a 4-legged tensor (coming from an (infinite) MPO) with two site indices and a left and a right leg into a 3 x 3 matrix of ITensor.
  We assume the normal form for MPO (before full compression) where the top left and bottom right corners are identity matrices.
  The goal is to write the tensor in the form
         1      M_12   M_13
         M_21   M_22   M_23
         M_31   M_32   1
  such that we can then easily compress it. Note that for most of our tensors, the upper triangular part will be 0.
  Input: tensor the four leg tensors and the pair of Index (left_ind, right_ind)
  Output: the 3x3 matrix of tensors
  Optional arguments: left_tags: if we want to change the tags of the left indices.
                      right_tags: if we want to change the tags of the right indices.
"""
function local_mpo_blocks(
  t::ITensor,
  left_ind::Index,
  right_ind::Index;
  left_tags=tags(left_ind),
  right_tags=tags(right_ind),
  kwargs...,
)
  @assert order(t) == 4

  left_dim = dim(left_ind)
  right_dim = dim(right_ind)
  #Build the local projectors.
  top_left, middle_left, bottom_left = local_mpo_block_projectors(left_ind; tags=left_tags)
  top_right, middle_right, bottom_right = local_mpo_block_projectors(
    right_ind; tags=right_tags
  )

  matrix = Matrix{ITensor}(undef, 3, 3)
  for (idx_left, proj_left) in enumerate([top_left, middle_left, bottom_left])
    for (idx_right, proj_right) in enumerate([top_right, middle_right, bottom_right])
      matrix[idx_left, idx_right] = proj_left * t * proj_right
    end
  end
  return matrix
end

"""
    local_mpo_blocks(t::ITensor, ind::Index; new_tags = tags(ind), position = :first, ...)

  Converts a 3-legged tensor (the extremity of a MPO) with two site indices and one leg into a 3 Vector of ITensor.
  We assume the normal form for MPO (before full compression) where the top left and bottom right corners are identity matrices in the bulk.

  Input: tensor the three leg tensors and the index connecting to the rest of the MPO
  Output: the 3x1 or 1x3 vector of tensors
  Optional arguments: new_tags: if we want to change the tags of the indices.
                      position: whether we consider the first term in the MPO or the last.
"""
function local_mpo_blocks(
  t::ITensor, ind::Index; new_tags=tags(ind), position=:first, kwargs...
)
  @assert order(t) == 3
  top, middle, bottom = local_mpo_block_projectors(ind; tags=new_tags)

  if position == :first
    vector = Matrix{ITensor}(undef, 1, 3)
  else
    vector = Matrix{ITensor}(undef, 3, 1)
  end
  for (idx, proj) in enumerate([top, middle, bottom])
    vector[idx] = proj * t
  end
  return vector
end

"""
combineblocks_linkinds_auxiliary(Hcl::InfiniteBlockMPO)

The workhorse of combineblocks_linkinds. We separated them for ease of maintenance.
Fuse the non-site legs of the infiniteBlockMPO Hcl and the corresponding left L and right R environments.
Preserve the corner structure.
Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
Input: Hcl the infiniteBlockMPO
Output: a copy of Hcl fused, and the two array of combiners to apply to left and right environments if needed.
"""
function combineblocks_linkinds_auxiliary(H::InfiniteBlockMPO)
  H = copy(H)
  N = nsites(H)
  for j in 1:(N - 1)
    right_dim = size(H[j], 2)
    for d in 2:(right_dim - 1)
      right_link = only(commoninds(H[j][1, d], H[j + 1][d, 1]))
      comb = combiner(right_link; tags=tags(right_link))
      comb_ind = combinedind(comb)
      for k in 1:size(H[j], 1)
        if isempty(H[j][k, d])
          H[j][k, d] = ITensor(Float64, uniqueinds(H[j][k, d], right_link)..., comb_ind)
        else
          H[j][k, d] = H[j][k, d] * comb
        end
      end
      for k in 1:size(H[j + 1], 2)
        if isempty(H[j + 1][d, k])
          H[j + 1][d, k] = ITensor(
            Float64, uniqueinds(H[j + 1][d, k], dag(right_link))..., dag(comb_ind)
          )
        else
          H[j + 1][d, k] = H[j + 1][d, k] * dag(comb)
        end
      end
    end
  end
  right_dim = size(H[N], 2)
  left_combs = []
  right_combs = []
  for d in 2:(right_dim - 1)
    right_link = only(commoninds(H[N][1, d], H[N + 1][d, 1]))
    comb = combiner(right_link; tags=tags(right_link))
    comb_ind = combinedind(comb)
    comb2 = translatecell(translator(H), comb, -1)
    comb_ind2 = translatecell(translator(H), comb_ind, -1)
    for k in 1:size(H[N], 1)
      if isempty(H[N][k, d])
        H[N][k, d] = ITensor(Float64, uniqueinds(H[N][k, d], right_link)..., comb_ind)
      else
        H[N][k, d] = H[N][k, d] * comb
      end
    end
    for k in 1:size(H[1], 2)
      if isempty(H[1][d, k])
        H[1][d, k] = ITensor(
          Float64,
          uniqueinds(H[1][d, k], dag(translatecell(translator(H), right_link, -1)))...,
          dag(comb_ind2),
        )
      else
        H[1][d, k] = H[1][d, k] * dag(comb2)
      end
    end
    append!(left_combs, [comb2])
    append!(right_combs, [dag(comb)])
  end
  return H, left_combs, right_combs
end

"""
combineblocks_linkinds(Hcl::InfiniteBlockMPO, left_environment, right_environment)

Fuse the non-site legs of the infiniteBlockMPO Hcl and the corresponding left L and right R environments.
Preserve the corner structure.
Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
Input: Hcl the infiniteBlockMPO, the left environment and right environment
Output: Hcl, left_environment, right_environment the updated MPO and environments
"""
function combineblocks_linkinds(H::InfiniteBlockMPO, left_environment, right_environment)
  H, left_combs, right_combs = combineblocks_linkinds_auxiliary(H)
  left_environment = copy(left_environment)
  for j in 1:length(left_combs)
    if isempty(left_environment[j + 1])
      left_environment[j + 1] = ITensor(
        uniqueinds(left_environment[j + 1], left_combs[j])...,
        uniqueinds(left_combs[j], left_environment[j + 1])...,
      )
    else
      left_environment[j + 1] = left_environment[j + 1] * left_combs[j]
    end
  end
  right_environment = copy(right_environment)
  for j in 1:length(right_combs)
    if isempty(right_environment[j + 1])
      right_environment[j + 1] = ITensor(
        uniqueinds(right_environment[j + 1], right_combs[j])...,
        uniqueinds(right_combs[j], right_environment[j + 1])...,
      )
    else
      right_environment[j + 1] = right_environment[j + 1] * right_combs[j]
    end
  end
  return H, left_environment, right_environment
end

"""
combineblocks_linkinds(Hcl::InfiniteBlockMPO)

Fuse the non-site legs of the infiniteBlockMPO Hcl.
Preserve the corner structure.
Essentially the inverse of splitblocks. It becomes useful for the very dense MPOs once get after compression sometimes.
Input: Hcl the infiniteBlockMPO,
Output: the updated MPO
"""
function combineblocks_linkinds(H::InfiniteBlockMPO)
  H, _ = combineblocks_linkinds_auxiliary(H)
  return H
end

function InfGenericHamiltonianMPO(
  sites::ITensorInfiniteMPS.CelledVector,
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
)

  @assert (length(AStrings) == length(JAs)) &&
    (length(BStrings) == length(JBs)) &&
    (length(CStrings) == length(JCs))

  @assert (length(JBs) == length(JAs)) &&
    (length(JCs) == length(JBs)) &&
    (length(JAs) == length(JCs))



  NrOfTerms = length(AStrings)
  # A is possible exponential decay so test for "0"
  As = if prod(AStrings .== "0")
    map(x -> 0.0 * op("Id", sites[2]), zeros(NrOfTerms))
  else
    map(x -> x[1] * op(x[2], sites[2]), zip(JAs, AStrings))
  end
  
  #### check if A is zero
  if iszero(As) 
    return InfGenericHamiltonianMPO_NN(sites, BStrings, JBs, CStrings, JCs, DString, JD)
  else
    return InfGenericHamiltonianMPO_3Form(sites, AStrings, JAs, BStrings, JBs, CStrings, JCs, DString, JD)
  end
end

function InfGenericHamiltonianMPO_NN(
  sites::ITensorInfiniteMPS.CelledVector,
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
)
  @assert length(sites) == 2

  link = get_linkindices(sites, 1, BStrings, CStrings)[1]
  link_trans = get_linkindices(sites, 1, BStrings, CStrings)[1]
  
  W_left   = fill_left_W(1, sites, link, CStrings, JCs, DString, JD;)
  W_right = fill_right_W(2, sites, dag(link), BStrings, JBs;)

  W_left_trans   = fill_left_W(2, sites, link_trans, CStrings, JCs, DString, JD;)
  W_right_trans = fill_right_W(3, sites, dag(link_trans), BStrings, JBs;)

  # blocked_mats_ofW = [
  #   ITensorInfiniteMPS.local_mpo_blocks(W_left, dag(link); position=:last),
  #   ITensorInfiniteMPS.local_mpo_blocks(W_right, link; position=:first),
  # ]
  # return InfiniteBlockMPO(blocked_mats_ofW)
  
  return InfiniteSum{MPO}([MPO([W_left,W_right]),MPO([W_left_trans,W_right_trans])], translator(sites))


end

function InfGenericHamiltonianMPO_3Form(
  sites::ITensorInfiniteMPS.CelledVector,
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number,
)
  @assert length(sites) == 2

  linkindices = get_linkindices(sites, 2, BStrings, CStrings)
  linkindices_trans = get_linkindices(sites, 2, BStrings, CStrings)

  
  W_left   = fill_left_W(1, sites, linkindices[1], CStrings, JCs, DString, JD;)
  W_bulk   = fill_bulk_W(2, sites, dag(linkindices[1]), linkindices[2], AStrings, JAs;)
  W_right  = fill_right_W(3, sites, dag(linkindices[2]), BStrings, JBs;)

  W_left_trans   = fill_left_W(2, sites, linkindices_trans[1], CStrings, JCs, DString, JD;)
  W_bulk_trans   = fill_bulk_W(3, sites, dag(linkindices_trans[1]), linkindices_trans[2], AStrings, JAs;)
  W_right_trans  = fill_right_W(4, sites, dag(linkindices_trans[2]), BStrings, JBs;)

  # blocked_mats_ofW = [
  #   ITensorInfiniteMPS.local_mpo_blocks(W_left, dag(linkindices[1]); position=:first),
  #   ITensorInfiniteMPS.local_mpo_blocks(W_bulk, linkindices[1], dag(linkindices[2])),
  #   ITensorInfiniteMPS.local_mpo_blocks(W_right, linkindices[2]; position=:last),
  # ]
  # return InfiniteBlockMPO(blocked_mats_ofW, translator(sites))
  return InfiniteSum{MPO}([MPO([W_left,W_bulk,W_right]), MPO([W_left_trans, W_bulk_trans, W_right_trans])], translator(sites))

end

function get_linkindices(
  sites::ITensorInfiniteMPS.CelledVector,
  n_links::Int64,
  BStrings::Vector{String},
  CStrings::Vector{String},
)
  if length(BStrings) != length(CStrings)
    throw(
      ArgumentError(
        "Cannot have unequal length of BStrings and CStrings!\n$(@show length(BStrings)) $(@show length(CStrings))",
      ),
    )
  end
  nTerms = length(BStrings)

  link_dimension = nTerms + 2

  linkindices = if hasqns(sites.data)
    Vector{Index{Vector{Pair{QN,Int64}}}}(undef, n_links)
  else
    Vector{Index{Int64}}(undef, n_links)
  end

  if hasqns(sites.data)
    # save QN flux of each operator, note that multiple operators may have same flux,
    # thus beloning to the same block and increasing the local dimension of this block
    # reuse the vector to save memory
    QNFlux_vector = Vector{QN}(undef, length(BStrings))

    for n in 1:n_links
      # save the flux and the corresponding dimension in a dynamically sized vector as it is ordered in historical order
      # same as the corresponding operators
      QN_local_index_dim = Vector{Pair{QN,Int64}}(undef, 1)
      nameQN = String(qn(sites[n][1]).data[1].name)
      QNmodulus = qn(sites[n][1]).data[1].modulus
      # loop over all interaction operators
      for (indexOP, (BString, CString)) in enumerate(zip(BStrings, CStrings))
        # get the flux of interaction
        # QNFlux_vector[indexOP] = flux(sites, Bstring, n, CString, n + 1)
        QNFlux_vector[indexOP] = flux(op(BString, sites[n]))
        checkflux = flux(op(CString, sites[n]))
        if !(checkflux == -QNFlux_vector[indexOP])
          error(
            "Operators B and C are not conserving the total QN in the system as their flux is not opposite",
          )
        end
        # local dimension of flux is at least 1
        if indexOP == 1
          QN_local_index_dim[indexOP] = Pair(QNFlux_vector[indexOP], 1)
        elseif indexOP > 1 && QNFlux_vector[indexOP - 1] == QNFlux_vector[indexOP]
          # increase tuple of number and dimension by one in dimension
          QN_local_index_dim[end] = Pair(
            QN_local_index_dim[end][1], QN_local_index_dim[end][2] + 1
          )
        elseif indexOP > 1
          # if new flux is encountered, save in vector
          push!(QN_local_index_dim, Pair(QNFlux_vector[indexOP], 1))
        end
      end # loop over interaction operators
      # # construct the local link index from all flux blocks and their dimension
      linkindices[n] = Index(
        [QN() => 1, QN_local_index_dim..., QN(nameQN, 0, QNmodulus) => 1], "Link,l=1"
      )
    end
    # do not forget about last index
    # linkindices[n_links + 1] = sim(linkindices[n_links]; tags="Link,l=1")
  else
    linkindices[:] = [Index(link_dimension, "Link,l=1") for n in 1:(n_links)]
  end
  return linkindices
end

function fill_right_W(
  n::Int,
  sites,
  ll,
  BStrings::Vector{String},
  JBs::Vector{<:Number};
  endState = 1
  )

  s = sites[n]

  Bs = map(x -> x[1] * op(x[2], sites[n]), zip(JBs, BStrings)) # JB * op(sites, BString, n)

  ElType = promote_itensor_eltype(Bs)
  # Init ITensor inside MPO
  W_store = ITensor(ElType, s', dag(s), ll)

  # first element
  W_store += setelt(ll[endState]) * op("Id", sites[n])
  for (iM, B,) in enumerate(Bs)
    # CHECK FOR NILL-POTENT OPERATORS or operators proportional to zero
    # avoid setting entries explicitly zero because that counters the purpose of sparse matrices and 
    # heavily reduces the runtime efficiency
    iszero(B) ? nothing : W_store += setelt(ll[1 + iM]) * B
  end
  return W_store
end

function fill_left_W(
  n::Int,
  sites,
  rl,
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number;
  startState = dim(rl),
  endState = 1)

  s = sites[n]
  
  Cs = map(x -> x[1] * op(x[2], sites[n]), zip(JCs, CStrings)) # JC * op(sites, CString, n)
  D = JD * op(DString, sites[n])

  ElType = promote_itensor_eltype([Cs..., D])
  # Init ITensor inside MPO
  W_store = ITensor(ElType, s', dag(s), rl)
  W_store += setelt(rl[startState]) * op("Id", sites[n])
  # CHECK FOR NILL-POTENT OPERATORS or operators proportional to zero
  # avoid setting entries explicitly zero because that counters the purpose of sparse matrices and 
  # heavily reduces the runtime efficiency
  iszero(D) ? nothing : W_store += setelt(rl[endState]) * D
  for (iM, C) in enumerate(Cs)
    iszero(C) ? nothing : W_store += setelt(rl[1 + iM]) * C
  end
  return W_store
end

function fill_bulk_W(
  n::Int,
  sites,
  ll,
  rl,
  AStrings::Vector{String}, JAs::Vector{<:Number};
  startState = dim(ll),
  endState = 1
)
  s = sites[n]

  As = if prod(AStrings .== "0")
    map(x -> 0.0 * op("Id", sites[n]), zeros(length(JAs)))
  else
    map(x -> x[1] * op(x[2], sites[n]), zip(JAs, AStrings))
  end

  ElType = promote_itensor_eltype(As)
  # Init ITensor inside MPO
  W_store = ITensor(ElType, s', dag(s), ll, rl)

  # first element
  W_store += setelt(ll[startState]) * (setelt(rl[startState])) * op("Id", sites[n])
  W_store += setelt(ll[endState]) * (setelt(rl[endState])) * op("Id", sites[n])
  for (iM, A) in enumerate(As)
    iszero(A) ? nothing : W_store += setelt(ll[1 + iM]) * (setelt(rl[1 + iM])) * A
  end
  return W_store
end


function fill_bulk_finite_W(
  n::Int,
  sites,
  ll,
  rl,
  AStrings::Vector{String},
  JAs::Vector{<:Number},
  BStrings::Vector{String},
  JBs::Vector{<:Number},
  CStrings::Vector{String},
  JCs::Vector{<:Number},
  DString::String,
  JD::Number;
  startState = dim(ll),
  endState = 1
)

  # siteindex s
  # n_cell = length(sites)
  s = sites[n]
  # ll = (linkindices[n])
  # rl = dag(linkindices[n + 1])


  As = if prod(AStrings .== "0")
    map(x -> 0.0 * op("Id", sites[2]), zeros(length(JAs)))
  else
    map(x -> x[1] * op(x[2], sites[2]), zip(JAs, AStrings))
  end
  Bs = map(x -> x[1] * op(x[2], sites[n]), zip(JBs, BStrings)) 
  Cs = map(x -> x[1] * op(x[2], sites[n]), zip(JCs, CStrings)) # JC * op(sites, CString, n)
  D = JD * op(DString, sites[n])

  ElType = promote_itensor_eltype([As...,Bs...,Cs...,D])
  # Init ITensor inside MPO
  W_store = ITensor(ElType, s', dag(s), ll, rl)

  # first element
  W_store += setelt(ll[startState]) * (setelt(rl[startState])) * op("Id", sites[n])
  W_store += setelt(ll[endState]) * (setelt(rl[endState])) * op("Id", sites[n])
  iszero(D) ? nothing : W_store += setelt(rl[endState]) * D
  for (iM, (A, B, C)) in enumerate(zip(As, Bs, Cs))
    iszero(A) ? nothing : W_store += setelt(ll[1+iM]) * setelt(rl[1+iM]) * A
    iszero(B) ? nothing : W_store += setelt(ll[1+iM]) * setelt(rl[endState]) * B
    iszero(C) ? nothing : W_store += setelt(ll[startState]) * setelt(rl[1+iM]) * C
  end
  return W_store
end