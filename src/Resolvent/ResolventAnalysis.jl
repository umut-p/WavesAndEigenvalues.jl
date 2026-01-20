module Resolvent

include("Utils.jl")
# using LinearAlgebra, ProgressMeter, SparseArrays, Plots
#------------------------------------------------------------------
# Resolvent Analysis at a given frequency and forcing vector
#------------------------------------------------------------------
"""
    apply_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)

Apply the resolvent problem for a given frequency ω₀ and forcing vector F̂:
    L(ω₀) * p̂ = F̂
Returns the response vector.  p̂ = L(ω₀)^(-1) * F̂
 
"""
function apply_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)
    A = L(ω0)
    p̂ = nothing
    success = false
    resid = NaN

    # Try direct solve first
    try
        p̂ = A \ F # direct solve -> p̂ = L(ω0)⁻¹ F
        success = true
        resid = norm(A * p̂ - F) # compute residual -> ‖L(ω0) * p̂ - F‖
        return p̂, (success=success, resid=resid, ω0=ω0)
    catch err_direct
        if output
            @warn "Direct solve failed. Error: $err_direct"
        end
    end
#   # Fallback to LU factorization if direct solve fails
    try
        Plu = lu(A) # LU factorization
        p̂ = Plu \ F # solve using LU factors
        success = true
        resid = norm(A * p̂ - F) # compute residual
    catch err_lu
        if output
            @warn "LU fallback failed. Error: $err_lu"
        end
    end
    
    return p̂, (success=success, resid=resid, ω0=ω0)
end

"""
    resolvent_norm(L, ω0)

Estimate the resolvent norm ‖(L(ω₀))⁻¹‖₂ (the maximum amplification the system can produce 
in response to any unit input) using the power method.

It measures how strongly the system amplifies the input at a given frequency 𝜔₀.
A high resolvent norm at a given frequency indicates that the system is highly sensitive to forcing at that frequency, which may suggest potential resonance or instability.
"""
function resolvent_norm(L, ω0; tol=1e-8, maxiter=1000)
    A = L(ω0)
    n = size(A, 1) # number of DOFs
    x = randn(ComplexF64, n) # initial random vector
    x /= norm(x) # normalize initial vector

    # LU factorization for efficient solves
    Plu = lu(A)

    σ_prev = 0.0
    for i in 1:maxiter
        # Solve A ⋅ z = x using factorization where A = L(ω0) and Plu is its LU factorization
        z = Plu \ x # solve for z by directly solving the linear system z = L(ω0)^-1 * x
        σ = norm(z) # estimate singular value ||L(ω0)^-1||₂ -> σ = ‖L(ω0)⁻¹‖₂

        if σ < eps() # avoid division by zero
            @warn "σ too small at iteration $i"
            return NaN
        end
        x = z / σ # normalize for next iteration -> x = L(ω0)⁻¹ * x / ‖L(ω0)⁻¹‖₂
        if abs(σ - σ_prev) < tol * max(1.0, σ) # convergence check if change in σ is small enough
                                               # max(1.0, σ) gives relative tolerance when σ is large
            return σ
        end
        σ_prev = σ # update previous σ
    end
    return σ_prev # return last estimate if not converged
end

"""
    resolvent_svd(L, ω0; maxiter=50, tol=1e-6)
Computes the dominant singular value and corresponding optimal forcing mode of the resolvent 
operator L(ω₀) using the power method.
Parameters:
- `L`        : Resolvent operator function, callable as `L(ω)`
- `ω0`       : Frequency at which to evaluate the resolvent operator
- `maxiter`   : Maximum number of iterations (default 50)
- `tol`       : Convergence tolerance (default 1e-6)
Returns:
- f : Optimal forcing vector (left singular vector)
- σ : Dominant singular value (largest singular value)
- p : Optimal response vector (right singular vector)
"""
function resolvent_svd(L, ω; maxiter=50, tol=1e-6)
    A = L(ω) 
    n = size(A, 1) # number of DOFs

    # LU factorization for A and A' (transpose)
    Plu = lu(A) # factorize once
    PluT = lu(A')  # factorize adjoint once

    f = randn(ComplexF64, n) # initial random forcing
    f ./= norm(f) # normalize

    σ_prev = 0.0 
    σ = 0.0
    for iter in 1:maxiter
        # response p = A^{-1} f
        p = Plu \ f # solve for response and return p
        σ = norm(p) # estimate singular value ‖L(ω)⁻¹‖₂ -> σ = ‖L(ω)⁻¹‖₂
        if σ == 0
            @warn "zero singular value estimate"
            return f, zeros(ComplexF64, n), 0.0
        end
        # update forcing f = (A')^{-1} p
        p ./= σ # normalize response p = L(ω)⁻¹ f / ‖L(ω)⁻¹‖₂
        f = PluT \ p # solve adjoint system f = LU(L(ω)')⁻¹ p
        f ./= norm(f) # normalize forcing f = f / ‖f‖
        
        if abs(σ - σ_prev) < tol * max(1.0, σ) # convergence check if change in σ is small enough
            break
        end
        σ_prev = σ # update previous σ
    end
    # final optimal forcing and response (normalized)
    f_opt = f          # optimal forcing
    u_raw = Plu \ f_opt    # optimal response (not normalized)  u_raw = L(ω)⁻¹ f_opt
    if σ == 0 # avoid division by zero
        u_opt = zeros(ComplexF64, n)
    else
        u_opt = u_raw / σ         # normalized response (unit gain): u_opt = L(ω)⁻¹ f_opt / ‖L(ω)⁻¹‖₂ 
    end
    return f_opt, u_opt, σ # return optimal forcing, optimal response, and singular value
end
"""
    compute_responses(L, coords, axis, forcing_fracs, freqs; mode=:norm)

General routine to compute system responses for a set of forcing locations.
Arguments:
- `L`             : Resolvent operator function, callable as `L(ω)`
- `coords`       : Node coordinates array (nNodes × 3)
- `axis`         : Axis index along which to define forcing locations (1=x, 2=y, 3=z)
- `forcing_fracs` : List of target forcing locations as fractions of duct length (0 to 1)
- `freqs`        : List of frequencies (Hz) at which to compute responses
Keyword Arguments:
- `mode`         : Computation mode (:norm, :forcing_norm, :local, :svd)
- `k`            : Number of nearest nodes to consider per forcing location (for :local mode)
- `offset`       : DOF offset for pressure DOF indexing (default 1)
Returns:
- Depending on `mode`, returns:
    - :norm          → Vector of resolvent norms at each frequency
    - :forcing_norm  → Matrix of response norms for each forcing location and frequency : (freqs × total_forcing_columns): uses probe extraction at specified locations
    - :local         → Full-node pressure responses at nearest k nodes n → response matrix (freqs × (k*dof_per_node)): uses nearest nodes and dofs for each forcing fraction
    - :svd           → Dictionary mapping frequency → (f_opt, p_opt, σ) f_opt: optimal forcing, p_opt: optimal response, σ: dominant singular value
Helper Functions:
- `nearest_nodes_and_dofs` : Finds nearest nodes and corresponding DOFs for target fractions (it takes a point x, finds nearest k nodes to that point along the specified axis, and computes their global DOF indices)
- `build_sparse_Fmat`       : Builds sparse forcing matrix from DOF lists
    """         
function compute_responses(L, coords, axis, forcing_fracs, freqs; mode=:norm, k::Int=1, offset::Int=1)
     # Building Geometry / DOF info
    axmin, axmax = minimum(coords[:,axis]), maximum(coords[:,axis])
    nNodes = size(coords, 1) # number of nodes
    ndof = size(L(2π*freqs[1]), 1) # number of DOFs, uses a representative frequency to get ndof
    dof_per_node = ndof ÷ nNodes # DOFs per node, integer division
    remainder = ndof % nNodes # check divisibility
    if remainder != 0 # warn if not divisible
        @warn "ndof not divisible by nNodes: ndof=$ndof, nNodes=$nNodes, remainder=$remainder"
    end
    # Helper functions --------------------------------------------------------
    # Find k nearest nodes and corresponding global DOFs for each target fraction
    function nearest_nodes_and_dofs(coords, axis, forcing_fracs, nNodes, dof_per_node; k=1, offset=1)
        m = length(forcing_fracs) # number of target fractions
        x = coords[:,axis] # coordinates along specified axis
        axmin, axmax = minimum(x), maximum(x)
        nodes_list = Vector{Vector{Int}}(undef, m) # list of node index vectors, undef means uninitialized
        dofs_list  = Vector{Vector{Int}}(undef, m) # list of DOF index vectors, undef means uninitialized
        k = min(k, nNodes) # ensure k does not exceed number of nodes and take minimum of k and nNodes since k is number of nearest nodes to find

        for (i, frac) in enumerate(forcing_fracs) # for each target fraction
            target = axmin + frac*(axmax - axmin) # find target coordinate along axis considering forcing fraction
            d = abs.(x .- target)  # distances to target 
            idxs = partialsortperm(d, 1:k)          # distance indices of k nearest nodes. partial sortperm finds indices of the k smallest distances
            idxs = sort(idxs, by = j -> d[j])      # sort by distance, closest first, by j -> d[j] means sort indices based on their distances by looking up d[j]
            nodes_list[i] = idxs # store node indices as vector
            dofs_list[i]  = (idxs .- 1) .* dof_per_node .+ offset # compute global DOF indices (idx - 1) * dof_per_node + offset
                                                                # (idx - 1) because Julia is 1-based indexing, offset allows shifting DOF indices
        end
        return nodes_list, dofs_list # return lists of node and DOF index vectors from forcing_fracs
    end
    # build sparse forcing matrix from dofs_list (columns grouped by target)
    function build_sparse_Fmat(ndof, dofs_list) 
        colsizes = [length(v) for v in dofs_list] # number of DOFs per target, column sizes for each target in dofs_list
        totalcols = sum(colsizes) # total number of forcing columns (sum of all target sizes)
        rows = Vector{Int}(undef, totalcols) # row indices for sparse matrix 
        cols = Vector{Int}(undef, totalcols) # column indices for sparse matrix
        vals = ones(ComplexF64, totalcols) # values (all ones)
        pos = 1 # current position in rows/cols/vals
        for (j, v) in enumerate(dofs_list) # for each target's DOF vector in dofs_list
            base = sum(colsizes[1:j-1]) # base column index for this target, sum of previous targets' sizes
            for (ridx, d) in enumerate(v) # for each DOF in that vector with its local index ridx
                rows[pos] = d         # row index is the DOF, it gives the row where the forcing is applied
                cols[pos] = base + ridx # column index offset by base, it gives the column in the forcing matrix
                pos += 1 # move to next position, position in rows/cols/vals
            end
        end
        return sparse(rows, cols, vals, ndof, totalcols) # build sparse matrix, returns the sparse forcing matrix
    end
    # End of Helper functions --------------------------------------------------------------------------------
    # Compute resolvent norm curve
    if mode == :norm  
        norms = Float64[]
        prog = Progress(length(freqs), desc="Resolvent norm")
        for f in freqs
            ω = 2π*f
            push!(norms, resolvent_norm(L, ω)) # compute and store resolvent norm ||L(ω)^-1|| at this frequency 
            next!(prog)
        end
        return norms
        
    # Compute response norms at probe locations for each forcing location
    elseif mode == :forcing_norm
        # get nearest nodes and DOFs for each target fraction. returns lists of node and DOF index vectors
        nodes_list, dofs_list = nearest_nodes_and_dofs(coords, axis, forcing_fracs, nNodes, dof_per_node; k=k, offset=offset)
        
        # build Fmat (sparse)
        if use_sparse
            Fmat = build_sparse_Fmat(ndof, dofs_list)
        end
        totalcols = size(Fmat, 2) # total number of forcing columns, 2 means number of columns in Fmat, columns mean different forcing vectors
        norms = zeros(Float64, length(freqs), totalcols)  # rows: freqs, cols: forcing columns
        outer = Progress(length(freqs), desc="Frequencies")
        
        for (i_f, f) in enumerate(freqs) # for each frequency
            ω = 2π*f
            A = L(ω)
            Plu = lu(A)                     # factorize once per frequency
            Pmat = Plu \ Matrix(Fmat)       # solve all RHS at once and store responses in Pmat (dense RHS matrix)    
            # compute response norms for each forcing column
            for c in 1:totalcols  # for each forcing column
                norms[i_f, c] = norm(Pmat[:, c]) # compute norm of response for this forcing column 
            end
            next!(outer)
        end
        # return norms matrix and metadata so caller can map columns back to targets
        return (norms = norms, nodes_list = nodes_list, dofs_list = dofs_list) 
    
    # compute local full-node responses at nearest k nodes for each forcing location
    elseif mode == :local
        # return per-target full-node responses (freqs × (k*dof_per_node))
        nodes_list, dofs_list = nearest_nodes_and_dofs(coords, axis, forcing_fracs, nNodes, dof_per_node; k=k, offset=offset)

        responses = Dict{Float64, Matrix{ComplexF64}}()
        outer = Progress(length(forcing_fracs), desc="Forcing locations")
        # for each forcing fraction
        for (i_target, frac) in enumerate(forcing_fracs) # for each target fraction
            dofs = dofs_list[i_target]                 # vector of k DOFs (one per node) which was found from nearest_nodes_and_dofs
            k_here = length(dofs)                      # number of nearest nodes
            blocksize = dof_per_node                   # DOFs per node
            outcols = k_here * blocksize               # total output columns

            amps = zeros(ComplexF64, length(freqs), outcols) # output response matrix
            inner = Progress(length(freqs), desc="Frequencies") 
            F = zeros(ComplexF64, ndof)                # reused RHS vector, size ndof

            for (i_f, f) in enumerate(freqs) # for each frequency
                ω = 2π*f
                A = L(ω)
                Plu = lu(A) # factorize once per frequency
                fill!(F, 0) # reset forcing vector

                for (jnode, d) in enumerate(dofs) # for each nearest node DOF in this target
                    F[d] = 1 + 0im # unit forcing at this DOF
                    # RESOLVENT SOLUTION: p̂ = LU(L(ω))⁻¹ F
                    p̂ = Plu \ F # Solve for response with this forcing
                    F[d] = 0 # reset forcing, important for next iteration

                    # extract all DOFs of this node
                    base = (d - offset) ÷ dof_per_node   # node index (0-based)
                    start_dof = base * dof_per_node + offset # first DOF of this node with offset
                    stop_dof  = start_dof + dof_per_node - 1 # last DOF of this node

                    # write into output block
                    col_start = (jnode - 1) * blocksize + 1 # starting column index for this node
                    col_end   = jnode * blocksize # ending column index for this node
                    amps[i_f, col_start:col_end] = p̂[start_dof:stop_dof] # store response for this node
                end
                next!(inner)
            end
            responses[frac] = amps # store responses for this forcing fraction
            next!(outer)
        end
        return responses
        
    # Compute optimal forcing and response modes via SVD of the resolvent operator
    elseif mode == :svd
        results = Dict{Float64, Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}}()
        prog = Progress(length(freqs), desc="Resolvent SVD")
        for f in freqs
            ω = 2π*f
            f_opt, p_opt, σ = resolvent_svd(L, ω; maxiter=100, tol=1e-6)  # optimal forcing via SVD, σ is the dominant singular value, p_opt is the optimal response
            results[f] = (f_opt, p_opt, σ)  # store results for this frequency
            next!(prog)
        end
        return results # return dictionary mapping frequency → (f_opt, p_opt, σ)
    else
        error("Unknown mode: $mode")
    end
end
end # module
