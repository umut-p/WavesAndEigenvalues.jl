module Resolvent

include("Utils.jl")
# using LinearAlgebra, IterativeSolvers, LinearMaps


#------------------------------------------------------------------
# Resolvent Analysis at a given frequency and forcing vector
#------------------------------------------------------------------

"""
    compute_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)

Solve the resolvent problem for a given frequency ω₀ and forcing vector F̂:
    L(ω₀) * p̂ = F̂
Returns the response vector. 
p̂ = L(ω₀)^(-1) * F̂

"""
function compute_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)
    A = L(ω0)   # Resolvent operator at frequency ω₀

    p̂ = nothing   # Initialize response vector
    success = false   # Flag for successful solve
    resid = NaN   # Residual norm

    try
        p̂ = A \ F   # Direct solve ̂p = L(ω)^(-1) * F
        success = true   # Direct solve succeeded
        resid = norm(A * p̂ - F)   # Compute residual norm
    catch
        if output
            @warn "Direct solve failed; trying GMRES"
        end
        try
            Amap = LinearMap(x -> A * x, size(A),n,n)    # GMRES-compatible linear map which wraps the matrix-vector product Amap = A ⋅ x 
            p̂, _ = gmres(Amap, F; tol=tol, maxiter=maxiter, log=true)   # Solve using the GMRES method; information about the iterative solve process is ignored
            success = true    # GMRES solve succeeded
            resid = norm(A * p̂ - F)    # Compute residual norm
        catch err
            error("Resolvent solve failed: both direct and GMRES solvers failed. Error: $err")
        end
    end

    return p̂, (success=success, resid=resid, ω0=ω0)
end

# ---------------------------------------------------------------
Estimating the resolvent norm (largest singular value)
# ---------------------------------------------------------------

"""
    resolvent_norm(L, ω0)

Estimate the resolvent norm ‖(L(ω₀))⁻¹‖₂ (the maximum amplification the system can produce in response to any unit input) using the power method.

This computes the largest singular value of (L(ω₀))⁻¹, by solving an eigenvalue problem on (A' * A)⁻¹.
It measures how strongly the system amplifies input at a given frequency 𝜔₀.
A high-resolvent norm at a frequency means the system is susceptible to forcing at that frequency — a remark of potential resonance or instability.
"""

function resolvent_norm(L, ω0; tol=1e-8, maxiter=1000)
    A = L(ω0)    # Resolvent operator at frequency ω₀
    n = size(A, 1)    # Dimension of the operator
    x = randn(ComplexF64, n)   # Generates a random complex vector x of length n
    x /= norm(x)     # Normalizes x to have unit norm

    σ_prev = 0.0
    σ = 0.0

# Power iteration method
    for i in 1:maxiter
        # Solve A ⋅ z = x
        z = try
            A \ x    # Inverse solve to get z = A⁻¹ * x
        catch
            @warn "Inverse solve failed at iteration $i"
            return NaN
        end

        σ = norm(z)      # Estimate singular value σ = ||z|| = ||A⁻¹ * x||
        x = z / σ        # Normalize z to get new x

        if abs(σ - σ_prev) < tol    # Convergence check, if less than tolerance then stop
            break
        end
        σ_prev = σ
    end

    return σ        # Return estimated resolvent norm
end

# ---------------------------------------------------------------
# Compute the singular values
# ---------------------------------------------------------------
"""
    resolvent_svd(L, ω0; k=5, output=true)

Compute the leading k singular values and vectors of the resolvent operator L(ω₀)⁻¹.

Returns (U, Σ, V), where:
- U: matrix of left singular vectors (response modes)
- Σ: vector of singular values (gains)
- V: matrix of right singular vectors (forcing modes)
"""

function resolvent_svd(L, ω0; k=5, output=true)
    A = L(ω0) # Resolvent operator at frequency ω₀
    n = size(A, 1) # Dimension of the operator (number of rows in the matrix A)

    # Define matrix-free resolvent operator R ≈ A⁻¹x
    Rmap = LinearMap(x -> A \ x, n, n)

    # Generate a random input matrix for the SVD of size n × k
    X = randn(ComplexF64, n, k)

    # Apply Rmap to each column of X
    Y = similar(X)     # Allocates a matrix Y of the same size and type as X (stores the system responses to each input)
    for j in 1:k 
        Y[:, j] = Rmap * X[:, j] # For each column 𝑗, apply the resolvent operator 𝑅=A⁻¹ to the input vector X[:, j] and store the response in y_j = A⁻¹x_j
end
    # Perform SVD on the response matrix Y
    SVD = svd(Y)  # Computes the dominant response directions, "the most amplified input-output pairs".
    U = SVD.U
    S = SVD.S
    V = SVD.V

    if output
        println("→ Top $k approximate singular values of resolvent operator at $(ω0 / (2π)) Hz:")
        for i in 1:k
            println("   σ[$i] ≈ ", S[i])
        end
    end

    return U[:, 1:k], S[1:k], V[:, 1:k]
end


# ---------------------------------------------------------------
# EXAMPLE: forcing as a single node in the mesh
# ---------------------------------------------------------------
"""
    point_force(mesh, xref; amp=1.0)

Generate a forcing vector F̂ applied at the node nearest to the physical position `xref`.
Uses the `points` field of the mesh for coordinates.
"""

function point_force(mesh, xref; amp=1.0)
    coords = mesh.points  # get coordinates of all mesh nodes. Could be 3×N or N×3 

    # Handle 3×N layout (common in Helmholtz meshes)
    coords = size(coords, 1) == 3 ? permutedims(coords) : coords  # ensuring that matrix is N×3, permutedims = Transpose of the matrix coords

    # Find nearest node
    dists = [sum(abs2, coords[i, :] .- xref) for i in 1:size(coords, 1)]   # dists: Computes squared Euclidean distance from each node to xref
    idx = argmin(dists)     # idx: Index of the node with the minimum distance (nearest node to xref)

    # Create complex forcing vector
    F = zeros(ComplexF64, size(coords, 1))
    F[idx] = amp + 0im      # Sets the entry at the nearest node to the complex amplitude amp + 0im

    return F
end
end #module
