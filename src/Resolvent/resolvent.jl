module Resolvent

using LinearAlgebra
using Optim
using LinearOperators
using WavesAndEigenvalues
using WavesAndEigenvalues.Helmholtz

# ===============================
# Compute the resolvent operator
# ===============================
function compute_resolvent(L, M, ω)
    # Solve the nonlinear eigenvalue problem
    nlevp = NLEVP(L, M, ω)
    Ω, P = solve(nlevp)

    # Compute the resolvent operator
    R = inv(L - ω^2 * M)
    return R, Ω, P
end

# ===============================
# Estimate the resolvent norm
# ===============================
function resolvent_norm(R)
    svals = svdvals(R)
    return maximum(svals)
end

# ===============================
# Perform singular value decomposition
# ===============================
function resolvent_svd(R)
    U, S, V = svd(R)
    return U, S, V
end

# ===============================
# Define linear operator family
# ===============================
function linear_operator_family(L, M, ω)
    L_family = LinearOperator(size(L)..., false) do x
        return L*x - ω^2*M*x
    end
    M_family = LinearOperator(size(M)..., false) do x
        return M*x
    end
    return L_family, M_family
end

# ===============================
# MSLP iteration for eigenvalues
# ===============================
function mslp_iteration(L, M, ω)
    nlevp = NLEVP(L, M, ω)
    Ω, P = mslp(nlevp)
    return Ω, P
end

# ===============================
# Objective function for optimal forcing
# ===============================
function objective_function(f, L, M, ω)
    R, _, _ = compute_resolvent(L, M, ω)
    v = R * f
    J = norm(v) + norm(f)
    return J
end

function optimize_forcing(L, M, ω)
    f0 = zeros(size(L, 1))
    result = optimize(f -> objective_function(f, L, M, ω), f0, BFGS())
    return result.minimizer
end

# ===============================
# Helmholtz resolvent integration
# ===============================
function helmholtz_resolvent(mesh, ω; order=:lin)
    L, M = discretize(mesh, order=order)
    R, _, _ = compute_resolvent(L, M, ω)
    return R
end

end # module Resolvent
