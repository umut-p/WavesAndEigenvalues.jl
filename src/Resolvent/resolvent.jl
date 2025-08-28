# Resolvent.jl
# Utilities to compute resolvent operators, estimate their norm and compute SVDs

module Resolvent


using SparseArrays
using WavesAndEigenvalues.NLEVP
using WavesAndEigenvalues.Helmholtz


# Attempt to use KrylovKit if available for large-scale problems
const _has_krylovkit = try
@eval using KrylovKit
true
catch
false
end

export compute_resolvent, apply_resolvent!, resolvent_norm, resolvent_svd, mslp_eigensolve

"""
compute_resolvent(L, ω; solver=:lu, checkfinite=true) -> R

Compute the resolvent operator R(ω) = A(ω)^(-1) where A(ω) is provided either
as a matrix or a function L(ω) that returns a matrix-like operator.
"""
function compute_resolvent(L, ω; solver=:lu, checkfinite=true)
A = isa(L, Function) ? L(ω) : L

if checkfinite && any(!isfinite, Array(A))
error("Matrix A(ω) contains non-finite values")
end

if solver == :inv
return inv(Matrix(A))
elseif solver == :lu
return lu(Matrix(A))
else
error("Unknown solver $(solver). Use :lu or :inv")
end
end
"""
apply_resolvent!(out, R, f)


Apply the resolvent represented by `R` to the forcing vector `f` and write the result to `out`.
Works with LU factorizations or explicit inverse matrices.
"""
function apply_resolvent!(out::AbstractVector, R, f::AbstractVector)
if isa(R, LU)
copyto!(out, R \\ f)
elseif isa(R, AbstractMatrix)
mul!(out, R, f)
else
out[:] = R * f
end
return out
end

"""
resolvent_norm(L, ω; tol=1e-6, use_krylov=true)

Estimate the operator norm (largest singular value) of the resolvent R(ω).
Uses WavesAndEigenvalues.Helmholtz operators.
"""
function resolvent_norm(L, ω; tol=1e-6, use_krylov=true)
A = isa(L, Function) ? L(ω) : L
n = size(A,1)


if n <= 500 || !_has_krylovkit || !use_krylov
s = svdvals(Matrix(A))
smin = minimum(s)
return iszero(smin) ? Inf : 1.0 / smin
else
if _has_krylovkit
sv = KrylovKit.svds(Matrix(A); nev=1, which=:SM, tol=tol)
smin = sv.s[1]
return iszero(smin) ? Inf : 1.0 / smin
else
s = svdvals(Matrix(A))
smin = minimum(s)
return iszero(smin) ? Inf : 1.0 / smin
end
end
end

"""
resolvent_svd(L, ω; nev=1, return_vecs=true, tol=1e-6)

Compute the top `nev` singular values (and optionally vectors) of the resolvent operator R(ω).
"""
function resolvent_svd(L, ω; nev=1, return_vecs=true, tol=1e-6)
A = isa(L, Function) ? L(ω) : L
n = size(A,1)

if n <= 500 || !_has_krylovkit
R = inv(Matrix(A))
if return_vecs
U, S, Vt = svd(R)
svals = diag(S)[1:nev]
return svals, U[:,1:nev], Vt[:,1:nev]
else
return svdvals(R)[1:nev]
end
else
op = (x) -> A \\ x
S = KrylovKit.svds(op, n; nev=nev, which=:LM, tol=tol)
if return_vecs
return S.s, S.U, S.V
else
return S.s
end
end
end

"""
mslp_eigensolve(nep; λ0=nothing, tol=1e-8, maxiter=50, verbose=false)

Run a Method of Successive Linear Problems (MSLP) iteration to compute eigenpairs of a nonlinear eigenproblem `nep`.
"""
function mslp_eigensolve(nep; λ0=nothing, tol=1e-10, maxiter=50, verbose=false)
return NVLEP.mslp(nep; tol=tol, maxit=maxiter, λ=λ0)
end

end # module
