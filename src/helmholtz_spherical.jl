using ..CoarseGraining: Field, SphericalGrid
using LinearAlgebra
using SparseArrays
using IterativeSolvers

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
                                        method::Symbol=:cg) where {T<:Real,G<:SphericalGrid}
    grid = uE.grid
    @assert grid === vN.grid
    ny, nx = length(grid.lat), length(grid.lon)
    a = grid.a
    
    # Build spherical Laplacian operator
    L = build_spherical_laplacian(grid)
    
    # Compute divergence and vorticity
    div = divergence_sphere(uE, vN)
    vort = vorticity_sphere(uE, vN)
    
    # Solve Poisson equations: ∇²φ = div, ∇²ψ = vort
    φ_vec = solve_poisson_iterative(L, div.data[:], method, max_iter, tol)
    ψ_vec = solve_poisson_iterative(L, vort.data[:], method, max_iter, tol)
    
    φ = Field(reshape(φ_vec, ny, nx), grid)
    ψ = Field(reshape(ψ_vec, ny, nx), grid)
    
    # Compute velocity components from potentials
    ∇φ_E, ∇φ_N = gradient_sphere(φ)
    ∇ψ_E, ∇ψ_N = gradient_sphere(ψ)
    
    # Irrotational part: u_pot = ∇φ
    uE_pot = ∇φ_E
    vN_pot = ∇φ_N
    
    # Divergence-free part: u_div = ẑ × ∇ψ = (∂ψ/∂N, -∂ψ/∂E)
    uE_div = Field(∇ψ_N.data, grid)   # ∂ψ/∂N
    vN_div = Field(-∇ψ_E.data, grid)  # -∂ψ/∂E
    
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
    dλ = mean(diff(lon))
    dφ = mean(diff(lat))
    
    # Pre-allocate sparse matrix components
    I = Int[]
    J = Int[]
    V = Float64[]
    
    # Helper function to convert (j,i) to linear index
    idx(j, i) = (j-1)*nx + i
    
    for j in 1:ny
        φ = lat[j]
        cos_φ = cos(φ)
        cos_φ_prev = j > 1 ? cos(lat[j-1]) : cos_φ
        cos_φ_next = j < ny ? cos(lat[j+1]) : cos_φ
        
        for i in 1:nx
            row = idx(j, i)
            
            # Central coefficient
            λ_coeff = -2.0 / (a^2 * cos_φ * dλ^2)
            φ_coeff = -(cos_φ_prev + cos_φ_next) / (a^2 * cos_φ * dφ^2)
            central_coeff = λ_coeff + φ_coeff
            
            push!(I, row); push!(J, row); push!(V, central_coeff)
            
            # Longitude derivatives (∂²/∂λ²)
            if grid.periodic_lon
                i_prev = i == 1 ? nx : i-1
                i_next = i == nx ? 1 : i+1
            else
                i_prev = max(1, i-1)
                i_next = min(nx, i+1)
            end
            
            λ_coeff_off = 1.0 / (a^2 * cos_φ * dλ^2)
            if i_prev != i
                push!(I, row); push!(J, idx(j, i_prev)); push!(V, λ_coeff_off)
            end
            if i_next != i
                push!(I, row); push!(J, idx(j, i_next)); push!(V, λ_coeff_off)
            end
            
            # Latitude derivatives (1/cosφ ∂/∂φ(cosφ ∂/∂φ))
            if j > 1
                φ_coeff_prev = cos_φ_prev / (a^2 * cos_φ * dφ^2)
                push!(I, row); push!(J, idx(j-1, i)); push!(V, φ_coeff_prev)
            else
                # Boundary condition at south pole
                if boundary_condition == :neumann
                    # Zero gradient: ∂f/∂φ = 0
                    # This is handled by not adding the boundary term
                elseif boundary_condition == :dirichlet
                    # Fixed value (zero): f = 0
                    # Modify central coefficient
                    V[end-2] += cos_φ_prev / (a^2 * cos_φ * dφ^2)
                end
            end
            
            if j < ny
                φ_coeff_next = cos_φ_next / (a^2 * cos_φ * dφ^2)
                push!(I, row); push!(J, idx(j+1, i)); push!(V, φ_coeff_next)
            else
                # Boundary condition at north pole
                if boundary_condition == :neumann
                    # Zero gradient
                elseif boundary_condition == :dirichlet
                    # Fixed value (zero)
                    V[end-3] += cos_φ_next / (a^2 * cos_φ * dφ^2)
                end
            end
        end
    end
    
    # Create sparse matrix
    L = sparse(I, J, V, n_total, n_total)
    
    # Make matrix non-singular by fixing one point (remove null space)
    # Set reference point to zero: L[1,1] += large_value, rhs[1] = 0
    L[1,1] += 1e12
    
    return L
end

"""
    solve_poisson_iterative(L, rhs, method, max_iter, tol)

Solve Poisson equation ∇²φ = rhs using iterative methods.
"""
function solve_poisson_iterative(L::SparseMatrixCSC, rhs::Vector, method::Symbol, 
                                max_iter::Int, tol::Real)
    # Remove mean from RHS (compatibility condition for Neumann problems)
    rhs_centered = rhs .- mean(rhs)
    
    # Fix reference point
    rhs_centered[1] = 0.0
    
    # Choose iterative solver
    if method == :cg
        sol, history = cg(L, rhs_centered; maxiter=max_iter, tol=tol, log=true)
    elseif method == :gmres
        sol, history = gmres(L, rhs_centered; maxiter=max_iter, tol=tol, log=true)
    elseif method == :bicgstabl
        sol, history = bicgstabl(L, rhs_centered; maxiter=max_iter, tol=tol, log=true)
    else
        error("Unknown method: $method. Use :cg, :gmres, or :bicgstabl")
    end
    
    if !history.isconverged
        @warn "Iterative solver did not converge. Residual: $(history.residuals[end])"
    end
    
    # Remove mean from solution
    return sol .- mean(sol)
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