# Resolvent.jl
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
Delegates to WavesAndEigenvalues.NVLEP.mslp if available.
"""
function mslp_eigensolve(nep; λ0=nothing, tol=1e-10, maxiter=50, verbose=false)
return NVLEP.mslp(nep; tol=tol, maxit=maxiter, λ=λ0)
end

end # module
