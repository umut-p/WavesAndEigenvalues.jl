module Resolvent

using LinearAlgebra
using IterativeSolvers
using LinearMaps

# Resolvent Analysis at a given frequency

"""
    resolvent_solve(L, ω0, F; tol=1e-8, maxiter=200, output=true)

Solve the resolvent problem for a given frequency ω₀ and forcing vector F̂:
    L(ω₀) * p̂ = F̂
Returns the response vector p̂.

"""
function compute_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)
    A = L(ω0)

    p̂ = nothing
    success = false
    resid = NaN

    try
        p̂ = A \ F
        success = true
        resid = norm(A * p̂ - F)
    catch
        if output
            @warn "Direct solve failed; trying GMRES"
        end
        try
            Amap = LinearMap(x -> A * x, size(A)...)
            p̂ = gmres(Amap, F; tol=tol, maxiter=maxiter, log=true)
            success = true
            resid = norm(A * p̂ - F)
        catch err
            error("Resolvent solve failed: both direct and GMRES solvers failed. Error: $err")
        end
    end

    return p̂, (success=success, resid=resid, ω0=ω0)
end
"""
  resolvent_gain(L, ω0, F; kwargs...)

Compute the resolvent gain G = ||p̂|| / ||F̂|| for the operator L(ω₀).

Returns (p̂, gain, info).
"""
function resolvent_gain(L, ω0, F; kwargs...)
    A = L(ω0)  # Directly evaluate the operator at ω₀
    p̂, info = compute_resolvent(L, ω0, F; kwargs...)
    gain = norm(p̂) / norm(F)
    return p̂, gain, info
end
"""
# ---------------------------------------------------------------
Estimating the resolvent norm (largest singular value)
# ---------------------------------------------------------------

  resolvent_norm(L, ω0)

Estimate the resolvent norm ‖(L(ω₀))⁻¹‖₂

This computes the largest singular value of (L(ω₀))⁻¹, by solving an eigenvalue problem on (A' * A)⁻¹.
"""

function resolvent_norm(L, ω0; tol=1e-8, maxiter=1000)
    A = L(ω0)
    n = size(A, 1)
    x = randn(ComplexF64, n)
    x /= norm(x)

    σ_prev = 0.0
    σ = 0.0

    for i in 1:maxiter
        # Solve A * z = x
        z = try
            A \ x
        catch
            @warn "Inverse solve failed at iteration $i"
            return NaN
        end

        σ = norm(z)
        x = z / σ

        if abs(σ - σ_prev) < tol
            break
        end
        σ_prev = σ
    end

    return σ
end

"""
    point_force(mesh, xref; amp=1.0)

Generate a forcing vector F̂ applied at the node nearest to the physical position `xref`.
Uses the `points` field of the mesh for coordinates.
"""

function point_force(mesh, xref; amp=1.0)
    coords = mesh.points  

    # Handle 3×N layout (common in Helmholtz meshes)
    coords = size(coords, 1) == 3 ? permutedims(coords) : coords  # ensure N×3

    # Find nearest node
    dists = [sum(abs2, coords[i, :] .- xref) for i in 1:size(coords, 1)]
    idx = argmin(dists)

    # Create complex forcing vector
    F = zeros(ComplexF64, size(coords, 1))
    F[idx] = amp + 0im

    return F
end
