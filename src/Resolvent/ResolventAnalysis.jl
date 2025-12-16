module Resolvent

include("Utils.jl")
# using LinearAlgebra, IterativeSolvers, ProgressMeter
#------------------------------------------------------------------
# Resolvent Analysis at a given frequency and forcing vector
#------------------------------------------------------------------
"""
    compute_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)

Solve the resolvent problem for a given frequency ω₀ and forcing vector F̂:
    L(ω₀) * p̂ = F̂
Returns the response vector.  p̂ = L(ω₀)^(-1) * F̂

"""
function compute_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)
    A = L(ω0) # Resolvent operator at frequency ω₀
    p̂ = nothing # Initialize response vector
    success = false # Flag for successful solve
    resid = NaN # Residual norm

    # First try direct solve
    try
        p̂ = A \ F # Direct solve ̂p = L(ω)^(-1) * F
        success = true # Direct solve succeeded
        resid = norm(A * p̂ - F) # Compute residual norm
    catch err_direct
        if output
            @warn "Direct solve failed. Error: $err_direct"
        end
        
    end

    return p̂, (success=success, resid=resid, ω0=ω0)
end
"""
    resolvent_norm(L, ω0)

Estimate the resolvent norm ‖(L(ω₀))⁻¹‖₂ (the maximum amplification the system can produce in response to any unit input) using the power method.

This computes the largest singular value of (L(ω₀))⁻¹, by solving an eigenvalue problem on (A' * A)⁻¹.
It measures how strongly the system amplifies the input at a given frequency 𝜔₀.
A high resolvent norm at a given frequency means the system is very sensitive to the forcing at that frequency —
a remark of potential resonance or instability.
"""

function resolvent_norm(L, ω0; tol=1e-8, maxiter=1000)
    A = L(ω0) # Resolvent operator at frequency ω₀
    n = size(A, 1) # Dimension of the operator
    x = randn(ComplexF64, n) # Generates a random complex vector 𝑥 of length n
    x /= norm(x) # Normalizes 𝑥 to have unit norm

    σ_prev = 0.0
    σ = 0.0

# Power iteration
    for i in 1:maxiter
        # Solve A ⋅ z = x where A = L(ω₀)
        z = nothing
        try
            z = A \ x # Inverse solve to get z = A⁻¹ * x
        catch
            @warn "Inverse solve failed at iteration $i"
            return NaN
        end

        σ = norm(z) # Estimate singular value σ = ||z|| = ||L(ω₀)⁻¹ * x||
        if σ < eps()
            @warn "σ too small at iteration $i"
        return NaN
        end
        x = z / σ # Normalize z to get new x

        if abs(σ - σ_prev) < tol # Convergence check, if change in σ is below tolerance then stop
            break
        end
        σ_prev = σ # Update previous σ for next iteration
    end

    return σ # Return estimated resolvent norm
end

"""
    resolvent_svd(L, ω0; k=5, output=true)
Compute the top `k` singular values and corresponding left/right singular vectors of the resolvent operator L(ω₀) at frequency ω₀.
arguments:
- `L`        : Resolvent operator function, callable as `L(ω)`
- `ω0`       : Frequency at which to evaluate the resolvent operator
- `k`        : Number of top singular values/vectors to compute (default 5)
- `output`   : Whether to print the singular values (default true)
Returns:
- `U`        : Matrix of left singular vectors (response modes)
- `S`        : Vector of singular values
- `V`        : Matrix of right singular vectors (forcing modes)
"""
function resolvent_svd(L, ω0; k=5, output=true)
    A = L(ω0)              # Resolvent operator at frequency ω₀
    n = size(A,1)          # Dimensions of the operator L(ω₀)

    # Generate random input matrix
    X = randn(ComplexF64, n, k) # Random complex forcing vectors
    
    # Apply resolvent to each column of X
    Y = zeros(ComplexF64, n, k) # Initialize response matrix
    for j in 1:k # Loop over each forcing vector
        Y[:, j] = A \ X[:, j] # Compute response for each forcing vector
    end

    # Perform SVD on response matrix
    SVD = svd(Y) # SVD decomposition of response matrix Y
    U, S, V = SVD.U, SVD.S, SVD.V # Extract U, S, V from SVD
    if output
        println("→ Top $k singular values of resolvent operator at $(ω0 / (2π)) Hz:")
        for i in 1:k 
            println("   σ[$i] ≈ ", S[i]) # Print top k singular values
        end
    end
    return U[:, 1:k], S[1:k], V[:, 1:k] # Return top k singular vectors and values
end

""" Given a node index, map it to the corresponding DOF index in the operator L.
# This is needed if there are multiple DOFs per node (e.g. pressure and velocity).
# Arguments:
# - `L`        : Resolvent operator function, callable as `L(ω)`
# - `coords`   : Node coordinates (matrix, nodes × dimensions)
# - `node_idx` : Node index to map
# - `field`    : Field type (:pressure or :velocity) to select DOF offset
# - `dof_per_node`: Number of DOFs per node (default 1)
# Returns:
# - `dof_idx`  : Corresponding DOF index in L
# Example:
# dof_idx = node_to_dof_index(L, coords, 10; field=:pressure, dof_per_node=2)
# This maps node index 10 to the pressure DOF index in L, assuming 2 DOFs per node (pressure and velocity).
"""
function node_to_dof_index(L, coords, node_idx; field::Symbol=:pressure, dof_per_node::Int=1)
    A_example = L(2π*100.0)
    nDOF = size(A_example, 1)
    nNodes = size(coords, 1)

    if nDOF == nNodes
        # 1 DOF per node → trivial mapping
        return node_idx
    elseif nDOF == nNodes * dof_per_node
        # multiple DOFs per node → choose offset based on field
        if field == :pressure
            return node_idx
        else
            error("Unknown field: $field")
        end
    else
        error("Cannot map node index to DOF index automatically. Check L and coords.")
    end
end
"""
    compute_responses(L, coords, axis, forcing_fracs, freqs; mode=:norm)

General routine to compute system responses for a set of forcing locations.
Arguments:
- `L`            : Resolvent operator function, callable as `L(ω)`
- `coords`       : Node coordinates (matrix, nodes × dimensions)
- `axis`         : Axis index (e.g. 1 for x, 2 for y, 3 for z)
- `forcing_fracs`: Array of fractions along duct length (0–1)
- `freqs`        : Array of frequencies in Hz
- 'fields'      : Field type (:pressure or :velocity) to select DOF offset
- `mode`         
        :norm` → compute max resolvent norm (global gain),
          -> Uses resolvent_norm to compute global gain at each frequency
          -> Returns array of resolvent norms per frequency
        :local` → compute local amplitude curve at forcing DOF
          -> Uses compute_resolvent to solve for each frequency and forcing location 
          -> Returns dictionary mapping forcing fraction → amplitude response array
        :forcing_norm` → computes max response norm across freqs for each forcing location
          -> Uses compute_resolvent to solve for each frequency and forcing location 
          -> Returns max response norms per forcing location         
        :svd` → Dictionary of SVD results (U, S, V) per frequency
          -> Uses resolvent_svd to compute top singular values/vectors at each frequency
          -> Returns dictionary mapping frequency → (U, S, V) tuples
"""         
function compute_responses(L, coords, axis, forcing_fracs, freqs;
                           mode=:norm, field=:pressure)
    axmin, axmax = minimum(coords[:,axis]), maximum(coords[:,axis]) # Duct length along specified axis

    if mode == :norm
        norms = Float64[]
        prog = Progress(length(freqs), desc="Resolvent norm")
        for f in freqs
            ω = 2π*f
            push!(norms, resolvent_norm(L, ω)) # Compute resolvent norm at frequency ω
            next!(prog)
        end
        return norms

    elseif mode == :forcing_norm # compute max response norm across freqs for each forcing location
        # detect DOFs per node
        ndof = size(L(2π*freqs[1]), 1) # Total DOFs in the system, e.g. pressure + velocity DOFs
        nNodes = size(coords, 1) # Number of mesh nodes
        dof_per_node = ndof ÷ nNodes # DOFs per node
        remainder = ndof % nNodes # Check for remainder
        println("ndof=$ndof, nNodes=$nNodes, dof_per_node=$dof_per_node, remainder=$remainder")
        norms = Float64[]
        outer = Progress(length(forcing_fracs), desc="Forcing locations")
        j_dofs = similar(forcing_fracs, Int) # Preallocate array for DOF indices
        for (k, frac) in enumerate(forcing_fracs) # Loop over forcing fractions 
            target = axmin + frac*(axmax-axmin) # Target position along duct
            _, j = findmin(abs.(coords[:,axis] .- target)) # Find nearest node index coordinates to target 
            j_dofs[k] = (j - 1) * dof_per_node + 1  # Map to DOF index (assuming pressure offset = 1)
        end

        for f in freqs # Loop over frequencies
            ω = 2π*f # Angular frequency
            A = L(ω) # Resolvent operator at frequency ω

            # Factorize once per frequency
            Plu = lu(A) # LU factorization for efficient solves

            # Build batched forcing matrix: columns correspond to forcing locations
            Fmat = zeros(ComplexF64, ndof, length(forcing_fracs)) # Initialize forcing matrix with size (ndof, n_forcing_fracs)
            for (k, j_dof) in enumerate(j_dofs) # Loop over forcing DOF indices
                Fmat[j_dof, k] = 1 # Unit forcing at each DOF location
            end

            # Solve all RHS in one call  
            Pmat = Plu \ Fmat # Batched solve for all forcing locations

            # Collect norms per forcing location for this frequency
            for k in 1:length(forcing_fracs)
                push!(norms, norm(Pmat[:, k])) # Compute response norm for each forcing location
            end
            next!(outer)
        end
        return norms

    elseif mode == :local # compute local amplitude curve at forcing DOF
        # detect DOFs per node
        ndof = size(L(2π*freqs[1]), 1) # Total DOFs in the system
        nNodes = size(coords, 1) # Number of mesh nodes
        dof_per_node = ndof ÷ nNodes # DOFs per node
        remainder = ndof % nNodes # Check for remainder
        println("ndof=$ndof, nNodes=$nNodes, dof_per_node=$dof_per_node, remainder=$remainder")
        responses = Dict{Float64, Vector{Float64}}()
        outer = Progress(length(forcing_fracs), desc="Forcing locations")

        for frac in forcing_fracs
            target = axmin + frac*(axmax-axmin) # Target position along duct
            _, j = findmin(abs.(coords[:,axis] .- target)) # Find nearest node index coordinates to target
            j_dof = (j - 1) * dof_per_node + 1 # Map to DOF index (assuming pressure offset = 1)

            F = zeros(ComplexF64, ndof)
            amps = Float64[]
            inner = Progress(length(freqs), desc="Frequencies")

            for f in freqs
                ω = 2π*f
                fill!(F, 0) # Reset forcing vector
                F[j_dof] = 1 # Unit forcing at selected DOF
                p̂, info = compute_resolvent(L, ω, F; output=false) # Solve resolvent for forcing at j_dof
                push!(amps, (info.success && p̂ !== nothing) ? abs(p̂[j_dof]) : NaN) # Store amplitude response at forcing DOF if successful
                next!(inner)
            end
            responses[frac] = amps
            next!(outer)
        end
        return responses

    elseif mode == :svd # Dictionary of SVD results (U, S, V) per frequency
        svd_results = Dict{Float64, Tuple{Matrix{ComplexF64}, Vector{Float64}, Matrix{ComplexF64}}}()
        prog = Progress(length(freqs), desc="SVD frequencies")
        for f in freqs
            ω = 2π*f
            U, S, V = resolvent_svd(L, ω; k=5, output=false) # Compute SVD at frequency ω for top 5 modes
            svd_results[f] = (U, S, V) # Store results in dictionary
            next!(prog)
        end
        return svd_results
    else
        error("Unknown mode: $mode")
    end
end

end # module
