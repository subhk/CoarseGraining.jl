using ..CoarseGraining: Field, SphericalGrid
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using AlgebraicMultigrid: ruge_stuben, aspreconditioner

export helmholtz_hodge_sphere_iterative, poisson_sphere_solve_iterative, 
       helmholtz_hodge_sphere_multigrid, coarsen_spherical_grid, refine_spherical_grid

"""
    helmholtz_hodge_sphere_iterative(uE, vN; max_iter=1000, tol=1e-6, method=:cg)

Spherical Helmholtz-Hodge decomposition using iterative methods for large grids.
Decomposes (uE, vN) into divergence-free and irrotational components on a sphere.

This follows FlowSieve's approach using sparse iterative solvers for the Poisson equations:
- ∇²φ = ∇·u (for irrotational part)  
- ∇²ψ = ∇×u·ẑ (for divergence-free part)

Parameters:
- `uE`, `vN`: Eastward and northward velocity components on SphericalGrid
- `max_iter`: Maximum iterations for iterative solver
- `tol`: Convergence tolerance  
- `method`: Solver method (:cg, :gmres, :bicgstabl)

Returns:
- `uE_div`, `vN_div`: Divergence-free (rotational) components
- `uE_pot`, `vN_pot`: Irrotational (potential) components  
- `φ`: Velocity potential
- `ψ`: Stream function
"""
function helmholtz_hodge_sphere_iterative(uE::Field{T,G}, vN::Field{T,G};
                                        max_iter::Int=1000, tol::Real=1e-6,
                                        method::Symbol=:cg,
                                        solver::Symbol=:direct) where {T<:Real,G<:SphericalGrid}
    grid = uE.grid
    @assert grid === vN.grid

    # Compute divergence and vorticity
    div = divergence_sphere(uE, vN)
    vort = vorticity_sphere(uE, vN)

    # Solve the Poisson equations ∇²φ = div, ∇²ψ = vort.
    #   solver = :direct (default) — FFT-in-longitude + tridiagonal-in-latitude;
    #            exact, fast, non-uniform-aware in latitude.
    #   solver = :amg — algebraic-multigrid-preconditioned CG on the symmetric
    #            stiffness; grid-size-independent iteration counts for large grids.
    # `method` is retained for API stability.
    if solver === :amg || solver === :iterative
        φ = poisson_sphere_solve_iterative(div; maxiter=max_iter, reltol=tol)
        ψ = poisson_sphere_solve_iterative(vort; maxiter=max_iter, reltol=tol)
    else
        φ = poisson_sphere_solve(div)
        ψ = poisson_sphere_solve(vort)
    end

    # Irrotational part: u_pot = ∇φ
    uE_pot, vN_pot = gradient_sphere(φ)

    # Divergence-free part as the residual u − ∇φ → exact reconstruction
    # (u = u_pot + u_div). ψ is retained as the streamfunction output.
    uE_div = Field(uE.data .- uE_pot.data, grid)
    vN_div = Field(vN.data .- vN_pot.data, grid)

    return uE_div, vN_div, uE_pot, vN_pot, φ, ψ
end

"""
    build_spherical_laplacian(grid::SphericalGrid; boundary_condition=:neumann)

Build sparse matrix representation of spherical Laplacian operator.
∇²f = (1/(a²cosφ)) [∂/∂φ(cosφ ∂f/∂φ) + (1/cosφ) ∂²f/∂λ²]

Uses finite differences with proper spherical metric factors.
"""
function build_spherical_laplacian(grid::SphericalGrid; boundary_condition::Symbol=:neumann)
    ny, nx = length(grid.lat), length(grid.lon)
    n_total = ny * nx
    a = grid.a
    
    lat = grid.lat
    lon = grid.lon
    dλ = mean(diff(lon))   # longitude assumed uniform

    # Pre-allocate sparse matrix components
    I = Int[]
    J = Int[]
    V = Float64[]
    
    # (j,i) → linear index, column-major to match vec(data)/reshape(·, ny, nx)
    idx(j, i) = (i-1)*ny + j
    
    for j in 1:ny
        φ = lat[j]
        cos_φ = cos(φ)
        # Latitude spacings (non-uniform): cell faces at j±1/2.
        Δp = j < ny ? (lat[j+1] - lat[j]) : (lat[j] - lat[j-1])
        Δm = j > 1  ? (lat[j] - lat[j-1]) : (lat[j+1] - lat[j])
        Δc = j == 1 ? Δp : (j == ny ? Δm : (Δp + Δm) / 2)   # cell width
        # Meridional flux coefficients at the FACES (j±1/2): latitude-midpoint cos,
        # not neighbour cos (neighbour-cos breaks flux telescoping → O(1) error).
        # 0 ⇒ Neumann zero-flux at the latitude boundary.
        cos_prev = j > 1 ? cos((lat[j] + lat[j-1]) / 2) : 0.0
        cos_next = j < ny ? cos((lat[j] + lat[j+1]) / 2) : 0.0
        coefm = cos_prev / (a^2 * cos_φ * Δc * Δm)   # j-1 coupling
        coefp = cos_next / (a^2 * cos_φ * Δc * Δp)   # j+1 coupling
        diag_φ = -(coefm + coefp)
        if boundary_condition == :dirichlet && (j == 1 || j == ny)
            # Fixed-value (ghost f = 0) flux through the missing boundary face.
            diag_φ -= 1.0 / (a^2 * Δc * (j == 1 ? Δp : Δm))
        end
        # Zonal term is (1/(a²cos²φ)) ∂²/∂λ² — note cos²φ.
        λ_diag = -2.0 / (a^2 * cos_φ^2 * dλ^2)
        λ_off  =  1.0 / (a^2 * cos_φ^2 * dλ^2)

        for i in 1:nx
            row = idx(j, i)
            push!(I, row); push!(J, row); push!(V, λ_diag + diag_φ)

            # Longitude derivatives (∂²/∂λ²), periodic or clamped
            if grid.periodic_lon
                i_prev = i == 1 ? nx : i-1
                i_next = i == nx ? 1 : i+1
            else
                i_prev = max(1, i-1)
                i_next = min(nx, i+1)
            end
            if i_prev != i
                push!(I, row); push!(J, idx(j, i_prev)); push!(V, λ_off)
            end
            if i_next != i
                push!(I, row); push!(J, idx(j, i_next)); push!(V, λ_off)
            end

            # Latitude derivatives (Neumann unless a coupling exists)
            if j > 1
                push!(I, row); push!(J, idx(j-1, i)); push!(V, coefm)
            end
            if j < ny
                push!(I, row); push!(J, idx(j+1, i)); push!(V, coefp)
            end
        end
    end
    
    # Create sparse matrix. L ≈ ∇² is symmetric negative semi-definite with a
    # constant nullspace; the singularity is handled in solve_poisson_iterative
    # by projecting the rhs/solution onto the zero-mean subspace (no ill-
    # conditioning hack like pinning a diagonal entry to a huge value).
    L = sparse(I, J, V, n_total, n_total)
    return L
end

"""
    _spherical_cell_weights(grid) -> Vector

Lumped cell-area weights Dⱼ = a² cosφⱼ Δcⱼ (column-major over (j,i)). The collocated
spherical Laplacian `L` is NON-symmetric (the 1/cosφ row factor), so a CG/AMG solver
cannot be applied to it directly. Multiplying by `Diagonal(D)` gives the SYMMETRIC
stiffness `S = D·L`, self-adjoint in the cosφ area inner product.
"""
function _spherical_cell_weights(grid::SphericalGrid)
    ϕ = grid.lat
    ny = length(ϕ); nx = length(grid.lon)
    a = grid.a
    Dj = Vector{Float64}(undef, ny)
    for j in 1:ny
        Δc = j == 1 ? (ϕ[2]-ϕ[1]) : (j == ny ? (ϕ[ny]-ϕ[ny-1]) : (ϕ[j+1]-ϕ[j-1])/2)
        Dj[j] = a^2 * cos(ϕ[j]) * Δc
    end
    D = Vector{Float64}(undef, ny*nx)
    @inbounds for i in 1:nx, j in 1:ny
        D[(i-1)*ny + j] = Dj[j]
    end
    return D
end

"""
    poisson_sphere_solve_iterative(rhs::Field; maxiter=500, reltol=1e-10) -> Field

Solve ∇²φ = rhs on a `SphericalGrid` with an algebraic-multigrid-preconditioned CG
iteration. Because the collocated lat-lon Laplacian is non-symmetric, the solve is
performed on the symmetric stiffness `S = D·L` (D = cell-area weights), with one
reference point pinned to remove the constant nullspace. AMG gives essentially
grid-size-independent iteration counts — the large-scale counterpart of the direct
`poisson_sphere_solve`.
"""
function poisson_sphere_solve_iterative(rhs::Field{T,G}; maxiter::Int=500,
                                        reltol::Real=1e-10) where {T<:Real,G<:SphericalGrid}
    grid = rhs.grid
    ny = length(grid.lat); nx = length(grid.lon)
    n = ny * nx
    L = build_spherical_laplacian(grid)
    D = _spherical_cell_weights(grid)
    # Symmetric positive-definite system on the zero-mean, reference-pinned subspace.
    b = vec(Float64.(rhs.data)); b .-= mean(b)
    M = -(Diagonal(D) * L)
    r = -(D .* b)
    keep = 2:n                                   # pin φ[1] = 0 ⇒ non-singular
    Mk = M[keep, keep]
    Pl = aspreconditioner(ruge_stuben(Mk))
    sol, history = cg(Mk, r[keep]; Pl=Pl, maxiter=maxiter, reltol=reltol, log=true)
    if !history.isconverged
        @warn "AMG-CG Poisson solve did not converge. Residual: $(history[:resnorm][end])"
    end
    φ = zeros(Float64, n)
    φ[keep] .= sol
    φ .-= mean(φ)
    return Field(reshape(φ, ny, nx), grid)
end

"""
    helmholtz_hodge_sphere_multigrid(uE, vN; levels=3, max_iter=100)

Multi-grid accelerated spherical Helmholtz decomposition for very large grids.
Uses coarsening strategy similar to FlowSieve's multi-resolution approach.
"""
function helmholtz_hodge_sphere_multigrid(uE::Field{T,G}, vN::Field{T,G}; 
                                        levels::Int=3, max_iter::Int=100) where {T<:Real,G<:SphericalGrid}
    grid = uE.grid
    ny, nx = length(grid.lat), length(grid.lon)
    
    if ny <= 64 && nx <= 64
        # Small enough for direct method
        return helmholtz_hodge_sphere_iterative(uE, vN; max_iter=max_iter)
    end
    
    # Coarsen to manageable size
    grids = [grid]
    uE_fields = [uE]
    vN_fields = [vN]
    
    for level in 1:levels-1
        coarse_grid = coarsen_spherical_grid(grids[end], 2)
        if length(coarse_grid.lat) <= 32 || length(coarse_grid.lon) <= 32
            break
        end
        
        # Interpolate velocities to coarser grid
        uE_coarse = interpolate_to_coarse_sphere(uE_fields[end], coarse_grid)
        vN_coarse = interpolate_to_coarse_sphere(vN_fields[end], coarse_grid)
        
        push!(grids, coarse_grid)
        push!(uE_fields, uE_coarse)
        push!(vN_fields, vN_coarse)
    end
    
    # Solve on coarsest grid
    coarse_level = length(grids)
    uE_div_c, vN_div_c, uE_pot_c, vN_pot_c, φ_c, ψ_c = 
        helmholtz_hodge_sphere_iterative(uE_fields[coarse_level], vN_fields[coarse_level]; 
                                       max_iter=max_iter)
    
    # Refine back to original resolution
    φ_result = φ_c
    ψ_result = ψ_c
    
    for level in coarse_level-1:-1:1
        φ_result = interpolate_to_fine_sphere(φ_result, grids[level])
        ψ_result = interpolate_to_fine_sphere(ψ_result, grids[level])
        
        # Optionally: perform correction iterations at this level
        # This would involve solving residual equations
    end
    
    # Compute final velocity components from refined potentials
    ∇φ_E, ∇φ_N = gradient_sphere(φ_result)
    ∇ψ_E, ∇ψ_N = gradient_sphere(ψ_result)
    
    uE_pot = ∇φ_E
    vN_pot = ∇φ_N
    uE_div = Field(∇ψ_N.data, grid)
    vN_div = Field(-∇ψ_E.data, grid)
    
    return uE_div, vN_div, uE_pot, vN_pot, φ_result, ψ_result
end

"""
    coarsen_spherical_grid(grid::SphericalGrid, factor::Int)

Create a coarsened spherical grid by subsampling every `factor` points.
"""
function coarsen_spherical_grid(grid::SphericalGrid, factor::Int)
    lon_coarse = grid.lon[1:factor:end]
    lat_coarse = grid.lat[1:factor:end]
    return SphericalGrid(lon_coarse, lat_coarse, grid.a, grid.periodic_lon)
end

"""
    interpolate_to_coarse_sphere(field::Field, target_grid::SphericalGrid)

Interpolate field from fine to coarse spherical grid using area-weighted averaging.
"""
function interpolate_to_coarse_sphere(field::Field{T,G}, target_grid::SphericalGrid) where {T,G}
    # Simple subsampling for now - could implement proper area-weighted averaging
    source_grid = field.grid
    ny_target, nx_target = length(target_grid.lat), length(target_grid.lon)
    ny_source, nx_source = length(source_grid.lat), length(source_grid.lon)
    
    factor_y = ny_source ÷ ny_target
    factor_x = nx_source ÷ nx_target
    
    coarse_data = field.data[1:factor_y:end, 1:factor_x:end]
    return Field(coarse_data, target_grid)
end

"""
    interpolate_to_fine_sphere(field::Field, target_grid::SphericalGrid)

Interpolate field from coarse to fine spherical grid using bilinear interpolation.
"""
function interpolate_to_fine_sphere(field::Field{T,G}, target_grid::SphericalGrid) where {T,G}
    # Simple expansion for now - could implement proper interpolation
    source_grid = field.grid
    ny_target, nx_target = length(target_grid.lat), length(target_grid.lon)
    ny_source, nx_source = size(field.data)
    
    # Use simple nearest-neighbor upsampling
    fine_data = zeros(T, ny_target, nx_target)
    factor_y = ny_target ÷ ny_source
    factor_x = nx_target ÷ nx_source
    
    for j in 1:ny_target
        for i in 1:nx_target
            js = min(ceil(Int, j/factor_y), ny_source)
            is = min(ceil(Int, i/factor_x), nx_source)
            fine_data[j, i] = field.data[js, is]
        end
    end
    
    return Field(fine_data, target_grid)
end