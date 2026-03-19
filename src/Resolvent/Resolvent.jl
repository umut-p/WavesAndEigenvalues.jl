module Resolvent
include("Resolvent_analysis_exports.jl")
include("Utils.jl")
# using LinearAlgebra, ProgressMeter, SparseArrays, Plots, Statistics, Optim
#------------------------------------------------------------------
# Resolvent Analysis functions
#------------------------------------------------------------------
"""
    apply_resolvent(L, ω0, F; tol=1e-8, maxiter=200, output=true)

Apply the resolvent problem for a given frequency ω₀ and forcing vector F̂:
    L(ω₀) * p̂ = F̂
Returns to the system pressure respond to this forcing at frequency ω₀:
  p̂ = L(ω₀)^(-1) * F̂
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
A high resolvent norm at a given frequency means the system is very sensitive to the forcing at that frequency —
a remark of potential resonance or instability.
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
    local_resolvent_gain(Plu, PluH, n, in_dofs, out_dofs; tol=1e-6, maxiter=50)

"Local resolvent gain" as a *restricted* 2-norm gain:

    gain(ω) = || E * L(ω)^(-1) * B ||₂

- `B`: injects a reduced forcing vector into full DOFs at `in_dofs`
- `E`: extracts response DOFs at `out_dofs`

# Implementation: 
Power iteration on `T*T` using alternating solves with `L(ω)` and `L(ω)'`.
Pass in `Plu = lu(L(ω))` and `PluH = lu(L(ω)')` where Plu is the LU factorization of L(ω) and PluH is the LU factorization of L(ω)'.
"""
function local_resolvent_gain(Plu, PluH, n::Integer,
                              in_dofs::AbstractVector{<:Integer}, out_dofs::AbstractVector{<:Integer};
                              tol=1e-6, maxiter=50)
    nin = length(in_dofs) # number of DOFs in the forcing patch
    nout = length(out_dofs) # number of DOFs in the response patch
    if nin == 0 || nout == 0 # if there are no DOFs in the forcing patch or response patch, return 0.0
        return 0.0
    end

    # Work vectors to avoid allocations in the iteration
    f_full = zeros(ComplexF64, n)      # full forcing
    rhs_full = zeros(ComplexF64, n)    # full RHS for adjoint solve (E' y)
    g = randn(ComplexF64, nin) # initial random forcing vector, size: nin
    g ./= norm(g) # normalize the initial forcing vector
    σ_prev = 0.0 # initialize previous singular value
    for _ in 1:maxiter # loop over maximum number of iterations
        fill!(f_full, 0) # reset forcing vector
        @inbounds f_full[in_dofs] .= g # inject the forcing vector into the full DOFs at in_dofs, inbounds is used to avoid bounds checking
        p = Plu \ f_full # solve for response p = L(ω)⁻¹ f
        y = @view p[out_dofs] # extract the response at the out_dofs
        σ = norm(y) # estimate singular value ‖L(ω)⁻¹‖₂ -> σ = ‖L(ω)⁻¹‖₂
        if σ < eps()  #
            return 0.0 # avoid division by zero
        end

        # g_new ∝ T' y = B' * (A' \ (E' y))
        fill!(rhs_full, 0) # reset RHS vector with zeros
        @inbounds rhs_full[out_dofs] .= y ./ σ # inject the response at the out_dofs into the RHS vector with the scaling factor σ. 
                                               # This is the response of the system to the forcing at the out_dofs. The response is scaled by the scaling factor σ.
                                               # inbounds is used to avoid bounds checking
        w = PluH \ rhs_full # solve for w = L(ω)⁻¹ y using the LU factorization of L(ω)'. This is the response of the system to the forcing at the out_dofs.
        g_new = @view w[in_dofs] # extract the forcing vector at the in_dofs, view is used to avoid copying the data
        gnorm = norm(g_new) # estimate singular value ‖L(ω)⁻¹‖₂ -> gnorm = ‖L(ω)⁻¹‖₂, this is the norm of the forcing vector at the in_dofs.
        if gnorm < eps() # avoid division by zero
            return σ # return the singular value if the norm is too small
        end
        g .= g_new ./ gnorm # normalize the forcing vector
        if abs(σ - σ_prev) < tol * max(1.0, σ) # convergence check if change in σ is small enough
            return σ # return the singular value if the change is small enough
        end
        σ_prev = σ # update previous estimate
    end
    return σ_prev # return the last estimate if the maximum number of iterations is reached
end


"""
    resolvent_svd(L, ω; maxiter=50, tol=1e-6)
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
    n = size(A, 1)                 # number of DOFs
    Plu  = lu(A)                   # LU factorization of L(ω)
    PluT = lu(A')                  # LU factorization of L(ω)'
    f = randn(ComplexF64, n)       # initial forcing
    f ./= norm(f)
    σ_prev = 0.0
    σ = 0.0
    p = zeros(ComplexF64, n)       #

    for i in 1:maxiter
        # forward solve: p = L(ω)^(-1) f
        p = Plu \ f # solve for response p = L(ω)⁻¹ f, this is the response vector p.
        σ = norm(p) # estimate singular value ‖L(ω)⁻¹‖₂ -> σ = ‖L(ω)⁻¹‖₂, this is the singular value of the response vector p.

        # guard against zero/very small σ
        if σ ≤ eps() # avoid division by zero if σ is too small
            @warn "resolvent_svd: singular value too small at iteration $i"
            return f, zeros(ComplexF64, n), 0.0
        end
        # normalize response and update forcing via adjoint, this is the forcing vector f.
        p ./= σ # normalize response p = L(ω)⁻¹ f / ‖L(ω)⁻¹‖₂
        f = PluT \ p # solve for forcing f = L(ω)'⁻¹ p, this is the forcing vector f.
        f ./= norm(f) # normalize forcing f = f / ‖f‖  , this is the normalized forcing vector f.

        if abs(σ - σ_prev) < tol * max(1.0, σ) # convergence check if change in σ is small enough
                                               # max(1.0, σ) gives relative tolerance when σ is large
            break # break the loop if the change in σ is small enough
        end
        σ_prev = σ # update previous σ
    end
    return f, p, σ # return optimal forcing, response, and singular value
end
"""
    compute_responses(L, coords, axis, forcing_fracs, freqs; mode=:norm)

General routine to compute system responses for a set of forcing locations.
Arguments:
-  L              : Resolvent operator function, callable as `L(ω)`
-  coords         : Node coordinates array (nNodes × 3)
-  axis           : Axis index along which to define forcing locations (1=x, 2=y, 3=z)
-  forcing_fracs  : List of target forcing locations as fractions of duct length (0 to 1)
-  freqs          : List of frequencies (Hz) at which to compute responses
Keyword Arguments:
- `mode`         : Computation mode (:norm, :forcing_norm, :local, :svd)
- `radius_mm`   : Radius (in mm) around each forcing location to include nodes (default 2.0 mm)
- `offset`       : DOF offset for pressure DOF indexing (default 1)
Returns:
- Depending on `mode`, returns:
    - :norm          → Vector of resolvent norms at each frequency
    - :local         → Full-node pressure responses at nearest k nodes n → response matrix (freqs × (k*dof_per_node)): uses nearest nodes and dofs for each forcing fraction
    - :svd           → Dictionary mapping frequency → (f_opt, p_opt, σ) f_opt: optimal forcing, p_opt: optimal response, σ: dominant singular value
Helper Functions:
- `nearest_nodes_and_dofs` : Finds nearest nodes and corresponding DOFs for target fractions (it takes a point x, finds nearest k nodes to that point along the specified axis, and computes their global DOF indices)
    """ 
function compute_responses(L, coords, axis, forcing_fracs, freqs; mode=:norm,radius_mm=2.0, offset::Int=1)
    # Helper functions --------------------------------------------------------------------------------
    """
        nearest_nodes_and_dofs(coords, axis, forcing_fracs, nNodes, dof_per_node; radius_mm=2.0, offset=1)
    Finds nearest nodes and corresponding DOFs for target fractions along specified axis.
    Returns:
    - nodes_list : List of vectors of node indices for each forcing fraction
    - dofs_list  : List of vectors of DOF indices for each forcing fraction
    """
    function nearest_nodes_and_dofs(coords, axis, forcing_fracs, nNodes, dof_per_node;
                                radius_mm=1.0, offset=1)
        m = length(forcing_fracs) # number of forcing fractions
        axmin, axmax = minimum(coords[:,axis]), maximum(coords[:,axis]) # minimum and maximum coordinates along the specified axis

        nodes_list = Vector{Vector{Int}}(undef, m) # list of node indices for each forcing fraction
        dofs_list  = Vector{Vector{Int}}(undef, m) # list of DOF indices for each forcing fraction

        radius = radius_mm / 1000.0   # convert mm → m, radius in meters
        for (i, frac) in enumerate(forcing_fracs) # loop over forcing fractions
            # target coordinate along chosen axis   
            target = axmin + frac*(axmax - axmin) # target coordinate along the specified axis
            # find all nodes within radius, the distance from the target coordinate to all nodes along the specified axis
            d = abs.(coords[:,axis] .- target) # distance from the target coordinate to all nodes along the specified axis
            idxs = findall(d .≤ radius) # find all nodes within radius
            if isempty(idxs) # if no nodes are found within radius, warn and continue
                @warn "No nodes found within radius for frac=$frac (target=$target)"
            end
            nodes_list[i] = idxs # store node indices for this forcing fraction
            # convert node indices → DOF indices
            dofs_list[i] = (idxs .- 1) .* dof_per_node .+ offset # store DOF indices for this forcing fraction
        end
        return nodes_list, dofs_list # return list of node indices and DOF indices for each forcing fraction
    end

    # End of Helper functions --------------------------------------------------------------------------------

    # Building Geometry / DOF info
    x = coords[:, axis] # coordinates along specified axis
    axmin, axmax = minimum(x), maximum(x)
    nNodes = size(coords, 1) # number of nodes
    ndof = size(L(2π*freqs[1]), 1)   # captures global `freqs`
    dof_per_node = ndof ÷ nNodes # DOFs per node, integer division
    remainder = ndof % nNodes # check divisibility
    if remainder != 0 # warn if not divisible
        @warn "ndof not divisible by nNodes: ndof=$ndof, nNodes=$nNodes, remainder=$remainder"
    end

    # Compute resolvent norm curve
    if mode == :norm  # compute resolvent norm curve
        norms = Float64[]
        prog = Progress(length(freqs), desc="Resolvent norm")
        for f in freqs
            ω = 2π*f # angular frequency
            push!(norms, resolvent_norm(L, ω)) # compute and store resolvent norm at this frequency ω
            next!(prog)
        end
        return norms 
    end

    # Compute local pressure responses at forcing patch (radius-based)
    if mode == :local 
        # get nodes and DOFs inside radius for each forcing location
        nodes_list, dofs_list =
            nearest_nodes_and_dofs(coords, axis, forcing_fracs, nNodes, dof_per_node;
                                   radius_mm=radius_mm, offset=offset)

        responses = Dict{Float64, Matrix{ComplexF64}}()
        outer = Progress(length(forcing_fracs), desc="Forcing locations")

        for (i_target, frac) in enumerate(forcing_fracs)

            dofs = dofs_list[i_target]     # DOFs inside forcing patch
            k_here = length(dofs)          # number of forced nodes
            blocksize = dof_per_node       # DOFs per node
            outcols = k_here * blocksize   # total output columns = k * dof_per_node

            amps = zeros(ComplexF64, length(freqs), outcols)
            inner = Progress(length(freqs), desc="Forcing location $(round(frac; digits=2))L", barlen=20)

            F = zeros(ComplexF64, ndof)

            for (i_f, f) in enumerate(freqs) # loop over frequencies
                ω = 2π*f
                A = L(ω) 
                Plu = lu(A)  # LU factorization for efficient solves
                fill!(F, 0)  # reset forcing vector

                # loop over all DOFs in the forcing patch
                for (jnode, d) in enumerate(dofs) # for each DOF in the forcing patch
                    F[d] = 1 + 0im  # unit forcing at this DOF
                    p̂ = Plu \ F     # solve for response p̂ = L(ω)⁻¹ F
                    F[d] = 0        # reset forcing for next DOF

                    # extract all DOFs of this node
                    base = (d - offset) ÷ dof_per_node     # base node index (0-based)
                    start_dof = base * dof_per_node + offset # start DOF index for this node
                    stop_dof  = start_dof + dof_per_node - 1 # stop DOF index for this node

                    col_start = (jnode - 1) * blocksize + 1 # start column in amps, col_start is the start column in the response matrix
                    col_end   = jnode * blocksize # end column in amps, col_end is the end column in the response matrix

                    amps[i_f, col_start:col_end] = p̂[start_dof:stop_dof] # create response matrix by storing all DOFs of this node
                end

                next!(inner, showvalues = [(:freq, f)]) # update inner progress with current frequency
            end

            responses[frac] = amps # store response matrix for this forcing location
            next!(outer)
        end

        return responses
    end
    # Compute optimal forcing and response modes via SVD of the resolvent operator
    if mode == :svd # compute optimal forcing and response modes via SVD of the resolvent operator
        results = Dict{Float64, Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}}()
        prog = Progress(length(freqs), desc="Resolvent SVD")
        for f in freqs
            ω = 2π*f # angular frequency for this frequency
            f_opt, p_opt, σ = resolvent_svd(L, ω; maxiter=100, tol=1e-6) # optimal forcing via SVD, σ is the dominant singular value, p_opt is the optimal response
            results[f] = (f_opt, p_opt, σ)  # store results for this frequency
            next!(prog)
        end
        return results # return dictionary mapping frequency → (f_opt, p_opt, σ)
    end
    error("Unknown mode: $mode") # error if the mode is unknown
end

"""
    patch_pressure_dofs(coords, axis, frac, dof_per_node; radius_mm, offset=1)
Finds pressure DOFs in a small patch around a given forcing location.
Parameters:
- `coords` : Coordinates of the nodes
- `axis` : Axis along which to find the pressure DOFs
- `frac` : Fraction of the duct length along the specified axis
- `dof_per_node` : DOFs per node
- `radius_mm` : Radius of the patch in millimeters
- `offset` : Offset of the pressure DOFs
Returns:
- `idxs` : Indices of the pressure DOFs in the patch
# Implementation example:
patch_dofs = Dict{Float64, Vector{Int}}() # dictionary mapping forcing location → pressure DOFs in the patch
for xf in xf_list # loop over forcing locations
    patch_dofs[xf] = patch_pressure_dofs(coords, axis, xf, dof_per_node; radius_mm, offset=1) 
    # precompute pressure DOFs in the patch for this forcing location x_f
end
"""
function patch_pressure_dofs(coords, axis, frac, dof_per_node; radius_mm=0.1, offset=1)
    x = coords[:, axis] # coordinates along the specified axis
    axmin, axmax = minimum(x), maximum(x) # minimum and maximum coordinates along the specified axis
    target = axmin + frac * (axmax - axmin) # target coordinate along the specified axis
    radius = radius_mm / 1000.0 # radius in meters
    idxs = findall(abs.(x .- target) .≤ radius) # find all nodes within radius
    if isempty(idxs)
        # fallback: pick the nearest node so the patch is never empty
        idxs = [argmin(abs.(x .- target))]
    end
    return (idxs .- 1) .* dof_per_node .+ offset # return pressure DOFs in the patch
end

"""
    best_matching_mode(p_num, Ldom_sorted, max_modes)
Finds the best matching analytical mode index based on correlation with the numerical mode.
Parameters:
- `p_num` : Numerical pressure distribution
- `Ldom_sorted` : Sorted domain length
- `max_modes` : Maximum number of modes to consider
Returns:
- `best_k` : Best matching analytical mode index
- `best_corr` : Correlation between the optimal response and the analytical eigenmode
Helper Functions:
- `cos` : Cosine function
- `dot` : Dot product
- `norm` : Norm
"""
function best_matching_mode(p_num, Ldom_sorted, max_modes)
    best_corr = -Inf
    best_k = 1
    for n in 1:max_modes
        p_ana = @. cos((2*n - 1) * 0.5 * π * Ldom_sorted) # analytical mode shape for mode index n
        α = dot(p_ana, p_num) / dot(p_ana, p_ana) # optimal scaling factor to match numerical mode
        p_ana_scaled = α .* p_ana # scale analytical mode to best match numerical one
        corr = dot(p_ana_scaled, p_num) / (norm(p_ana_scaled) * norm(p_num)) # correlation with scaled analytical mode
        if corr > best_corr
            best_corr = corr
            best_k = n
        end
    end
    return best_k, best_corr
end

"""
    compare_popt_with_eigenmode(k::Int)
Compares the optimal response with the analytical eigenmode for a given resonance frequency.
Parameters:
- `k` : Resonance index
Returns:
- `plt` : Plot of the optimal response vs the analytical eigenmode
- `ρ` : Correlation between the optimal response and the analytical eigenmode
- `r` : Residual between the optimal response and the analytical eigenmode
Helper Functions:
- `best_matching_mode` : Finds the best matching mode index based on correlation with numerical mode
"""
function compare_popt_with_eigenmode(k::Int)
    f = freqs[k]

    # --- SVD optimal response ---
    f_opt, p_opt, σ = svd_result[f]

    # extract pressure DOFs
    p_opt_p = zeros(ComplexF64, nNodes)
    for i in 1:nNodes # loop over nodes to extract pressure DOFs
        dof = (i - 1)*dof_per_node + 1
        p_opt_p[i] = p_opt[dof]
    end

    # sort along x
    p_sorted = p_opt_p[perm]

    # phase align
    phase = angle(p_sorted[1]) # align phase to first point (x=0) to ensure consistent sign across frequencies
    p_sorted .*= exp(-1im * phase)
    p_num = real.(p_sorted)

    # --- analytical eigenmode ---
    nstar, _corr = best_matching_mode(p_num, Ldom_sorted, length(resonances))
    p_ana = @. Base.cos((2*nstar - 1) * 0.5 * π * Ldom_sorted)

    # scale analytical mode to best match numerical one
    α = LinearAlgebra.dot(p_ana, p_num) / LinearAlgebra.dot(p_ana, p_ana)
    p_ana_scaled = α .* p_ana

    # --- normalize both to unit max amplitude ---
    p_num_norm = p_num ./ maximum(abs.(p_num))
    p_ana_norm = p_ana_scaled ./ maximum(abs.(p_ana_scaled))

    # consistent sign for each graph ---
    s = Base.sign(p_num_norm[1]) # If the dot product is positive, they are mostly aligned; if negative, they are mostly opposite. This gives us a sign factor to align them.
    if s == 0 # if they are orthogonal, just set s to 1 to avoid zeroing out the modes
        s = 1.0 # this means we keep the original sign of the numerical mode if they are orthogonal
    end
    p_num_norm = p_num_norm .* s # align sign of numerical mode with analytical mode
    p_ana_norm = p_ana_norm .* s # align sign of analytical mode with numerical mode

    # --- correlation and residual (use normalized fields) ---
    ρ = LinearAlgebra.dot(p_ana_norm, p_num_norm) / (LinearAlgebra.norm(p_ana_norm) * LinearAlgebra.norm(p_num_norm))
    r = LinearAlgebra.norm(p_num_norm - p_ana_norm) / LinearAlgebra.norm(p_num_norm)

    # --- dynamic y-limits (per k) ---
    ymin = Base.min(Base.minimum(p_num_norm), Base.minimum(p_ana_norm))
    ymax = Base.max(Base.maximum(p_num_norm), Base.maximum(p_ana_norm))
    yrange = ymax - ymin
    pad = (yrange == 0.0) ? 1e-6 : 0.05 * yrange # small margin to avoid y-axis being too tight
    ylims_dynamic = (ymin - pad, ymax + pad)

    plt = Plots.plot(
        Ldom_sorted, p_num_norm,
        lw = 2,
        label = "p_opt(x) at $(Base.round(f; digits=1)) Hz",
        xlabel = "x/L",
        ylabel = "Normalized amplitude ||p||",
        title = "Optimal response vs scaled eigenmode at $(Base.round(f; digits=1)) Hz",
        size = (900, 600),
        ylims = ylims_dynamic,
    )

    Plots.plot!(
        plt,
        Ldom_sorted, p_ana_norm,
        lw = 2,
        ls = :dash,
        label = "Scaled analytical mode n=$nstar (r=$(Base.round(r; digits=3)))"
    )

    Plots.hline!([0], ls=:dash, color=:green, lw=0.8, label="")

    return plt, ρ, r
end

"""
    speaker_patch_forcing(coords, axis, wall_nodes, center_x, radius_mm, dof_per_node)
Computes the forcing vector for a speaker patch at a given center location.
Parameters:
- `coords` : Coordinates of the nodes
- `axis` : Axis along which the speaker is placed
- `wall_nodes` : List of wall node indices
- `center_x` : Center location of the speaker patch
- `radius_mm` : Radius of the speaker patch in mm
- `dof_per_node` : Number of DOFs per node
Returns:
- `f` : Forcing vector
Helper Functions:
- `π` : Pi
- `^` : Power
- `÷` : Division
- `zeros` : Zeros
- `length` : Length
"""
function speaker_patch_forcing(coords, axis, wall_nodes, center_x, radius_mm, dof_per_node)
    radius = radius_mm / 1000.0 # convert radius from mm to m
    nNodes = size(coords, 1) # number of nodes 
    ndof   = nNodes * dof_per_node # total number of DOFs
    dof_per_node = ndof ÷ nNodes # DOFs per node
    f = zeros(ndof) # initialize forcing vector

    for node in wall_nodes # for each wall node
        dof_per_node = ndof ÷ nNodes # DOFs per node
        x = coords[node, axis] # coordinate along specified axis of this node
        if abs(x - center_x) <= radius # check if within speaker patch radius
            dof = (node - 1) * dof_per_node + 1  # pressure DOF index for this node
            area_node = area / length(wall_nodes) # approximate area per wall node
            f[dof] = 1.0 * area_node # pressure forcing scaled by area
        end
    end
    return f # return forcing vector
end
"""
    speaker_scan(L_flame, coords, axis; start_x, end_x, spacing, radius_mm)
Performs a speaker scan along the wall to find the optimal speaker location.
Parameters:
- `L_flame` : Linear operator for the flame
- `coords` : Coordinates of the nodes
- `axis` : Axis along which the speaker is placed
- `start_x` : Starting location of the speaker scan
- `end_x` : Ending location of the speaker scan
- `spacing` : Spacing between speaker locations
- `radius_mm` : Radius of the speaker patch in mm
Returns:
- `speaker_centers` : Locations of the speaker centers
- `resp_norms` : Norms of the responses at each speaker location
- `responses` : Responses at each speaker location
- `dof_per_node` : Number of DOFs per node
"""
function speaker_scan(L_flame, coords, axis; start_x, end_x, spacing, radius_mm)
    # get wall nodes from mesh boundaries
    nNodes     = size(coords, 1) # number of nodes
    ndof = size(L_flame(2π*freqs[1]), 1) # number of DOFs
    dof_per_node = ndof ÷ nNodes # DOFs per node
    Nfreq    = length(freqs)
    speaker_centers = collect(start_x : spacing : end_x) # speaker center locations along axis starts from start_x to end_x with given spacing
    Nspeaker = length(speaker_centers)

    # Build operator at this frequency to infer ndof
    pbar = Progress(length(freqs), desc="Speaker scan frequencies")
    responses  = [Vector{Vector{ComplexF64}}(undef, Nfreq) for _ in 1:Nspeaker]
    resp_norms = [Vector{Float64}(undef, Nfreq) for _ in 1:Nspeaker]

    # Precompute forcing vectors
    forcing_vectors = [speaker_patch_forcing(coords, axis, wall_nodes, xc, radius_mm, dof_per_node)
        for xc in speaker_centers] # precompute forcing vectors for each speaker center location

    # Initialize response storage 
    for (i, f) in enumerate(freqs)
        next!(pbar, ; showvalues=[(:freq_Hz, f)])
        ω = 2π * f
        A = L_flame(ω)
        Afact = lu(A)

        # Solve for all speaker locations at this frequency
        for s in 1:Nspeaker
            p = Afact \ forcing_vectors[s]
            responses[s][i]  = p
            resp_norms[s][i] = norm(p)
        end
    end

    return speaker_centers, resp_norms, responses, dof_per_node
end
