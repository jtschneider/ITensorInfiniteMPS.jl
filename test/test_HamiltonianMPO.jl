using ITensors, ITensorInfiniteMPS

include("../examples/vumps/src/vumps_subspace_expansion.jl")


n_cell = 2 

initstate(n) = "Up"

zero_mag(n) = isodd(n) ? "↑" : "↓"

sites = infsiteinds("S=1/2", n_cell; initstate, conserve_szparity=true)

sitesH = infsiteinds("S=1/2", n_cell; initstate=zero_mag, conserve_qns=true, )


function ITensorInfiniteMPS.unit_cell_terms(::Model"ising_NN"; J=3.0, h=2.0, J₂=5.0)
  opsum = OpSum()
  opsum += -J, "X", 1, "X", 2
  opsum += -J₂, "X", 1, "Id", 2, "X", 3
  opsum += -h, "Z", 1
  return opsum
end

### defining the Ising model at criticality as, cf. "../src/models/ising.jl"
## H = ∑ -J⋅σˣₙσˣₙ₊₁ - h⋅σᶻₙ
## 

J = 1.0
h = 1.0

H_ref = InfiniteSum{MPO}(Model("ising"), sites; J=J, h=h)
H_INN = InfiniteSum{MPO}(Model("ising_NN"), sites;)
H_refH = InfiniteSum{MPO}(Model("heisenberg"), sitesH)
# H_ref2 = InfiniteBlockMPO(Model("ising"), sites, )

test_HINN = H_INN[0][3]'' * H_INN[1][2]' * H_INN[2][1]
links = inds(test_HINN,tags="Link") ## note they are ordered IN,IN,OUT,OUT i.e. right,right,left,left
combiner_L = combiner(links[1:2]...)
combiner_R = combiner(links[3:4]...)

cW_INN = combiner_L * test_HINN * combiner_R 

W_INN = array(cW_INN, inds(cW_INN)[[3,4,2,1]])


array(H_INN[1][1])
array(H_INN[1][2])
array(H_INN[1][3])

f_XL = array(H_INN[1][1])[2,1,2]
f_XBulk = array(H_INN[1][2])[2,1,2,1]
f_IdBulk = array(H_INN[1][2])[2,2,2,2]


f_XL/(f_IdBulk * f_XBulk)
 f_IdBulk / f_XBulk
  f_XBulk /f_IdBulk

AStrings = ["Id"]
JAs = [0.01]
BStrings = ["X"]
JBs = [1.0]
CStrings = ["X"]
JCs = [-J]
DString = "Z"
JD = -h


linkindices = ITensorInfiniteMPS.get_linkindices(sites, n_cell, BStrings, CStrings)

startState = dim(linkindices[1])
endState = 1


n = 2
As = if prod(AStrings .== "0")
  map(x -> 0.0 * op(sites.data, "Id", n), zeros(NrOfTerms))
else
  map(x -> x[1] * op(sites.data, x[2], n), zip(JAs, AStrings))
end
Bs = map(x -> x[1] * op(sites.data, x[2], n), zip(JBs, BStrings)) # JB * op(sites, BString, n)
Cs = map(x -> x[1] * op(sites.data, x[2], n), zip(JCs, CStrings)) # JC * op(sites, CString, n)
D = JD * op(sites.data, DString, n)

W_left = ITensorInfiniteMPS.fill_left_W(1, sites, dag(linkindices[1]), CStrings, JCs, DString, JD;)
# W_bulk_ITensor = ITensorInfiniteMPS.fill_bulk_W(2, sites, linkindices[1], dag(linkindices[2]), As; startState=startState , endState=endState)
W_right = ITensorInfiniteMPS.fill_right_W(2, sites, linkindices[1], BStrings, JBs;)


top_L, middle_L, bottom_L = ITensorInfiniteMPS.local_mpo_block_projectors(dag(linkindices[1]); tags=tags(linkindices[1]))

top_R, middle_R, bottom_R = ITensorInfiniteMPS.local_mpo_block_projectors(linkindices[1]; tags=tags(linkindices[1]))


vector_right = Matrix{ITensor}(undef, 3, 1)
vector_left = Matrix{ITensor}(undef, 1, 3)

for (idx, proj) in enumerate(zip([top_L, middle_L, bottom_L],[top_R, middle_R, bottom_R]))
  vector_left[idx] = proj[1] * W_left
  vector_right[idx] = proj[2] * W_right
end
# W_stores[n_cell] =
@show array(vector_left[1])
@show array(vector_left[2])[1,:,:]
@show array(vector_left[3])

@show array(vector_right[1])
@show array(vector_right[2])[1,:,:]
@show array(vector_right[3])

test = InfiniteBlockMPO([vector_left,vector_right], translator(sites))


test[1]
test[2]
@show array(test[2][1])
@show array(test[2][2])
@show array(test[2][3])


inds.(H_ref[1])
inds.(H_ref[2])

H_ref_test = InfiniteSum{MPO}([H_ref.data[1], H_ref.data[2]], translator(sites))

H_test = InfGenericHamiltonianMPO(sites,
       AStrings, JAs,
       BStrings, JBs,
       CStrings, JCs,
       DString, JD
)


inds(H_test[0][3])


W_test = reshape(permutedims(array(H_test[0][3]'' * H_test[1][2]' * H_test[2][1]), [1,2,3,4,6,5]),(2,9,9,2))

W_test[:,1,1,:]


H_test[1]

linkinds(H_test[1])
linkinds(H_ref[1])



ψ0 = InfMPS(sites, initstate)

# Form the Hamiltonian
# H = infiniteIsingMPO(sites; J=1.0, h=1.0, localham_type=ITensor)



vumps_kwargs = (
  tol=1e-6,
  maxiter=10,
  solver_tol=(x -> x / 100),
  multisite_update_alg="parallel",
)
subspace_expansion_kwargs = (cutoff=1e-6, maxdim=30)

ψ_ref = vumps_subspace_expansion(H_ref, ψ0; outer_iters=5, subspace_expansion_kwargs, vumps_kwargs)

ψ_test = vumps_subspace_expansion(H_test, ψ0; outer_iters=5, subspace_expansion_kwargs, vumps_kwargs)

# Check translational invariance
@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))


# Check translational invariance
@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))

