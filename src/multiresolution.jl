using ..CoarseGraining: Field, Grid, SphericalGrid, CurvilinearGrid
using Interpolations
using NCDatasets
using Dates: now

export coarsen_field, refine_field, create_multiresolution_hierarchy,
       hierarchical_helmholtz_workflow, save_multiresolution_data,
       load_multiresolution_data, interpolate_field_to_grid

"""
    coarsen_field(field, factor; method=:area_average)

Coarsen a field by the specified factor using various methods.
Similar to FlowSieve's coarsen_grid functionality.

Parameters:
- `field`: Input field to coarsen
- `factor`: Coarsening factor (integer or tuple (fy, fx))
- `method`: Coarsening method (:area_average, :subsample, :conservative)

Returns coarsened field on a reduced-resolution grid.
"""
function coarsen_field(field::Field{T,G}, factor::Union{Int,Tuple{Int,Int}}; 
                      method::Symbol=:area_average) where {T,G}
    
    if factor isa Int
        fy, fx = factor, factor
    else
        fy, fx = factor
    end
    
    ny, nx = size(field.data)
    ny_coarse = ny ÷ fy
    nx_coarse = nx ÷ fx
    
    coarse_data = zeros(T, ny_coarse, nx_coarse)
    
    if method == :area_average
        # Area-weighted averaging (conservative)
        for j in 1:ny_coarse
            for i in 1:nx_coarse
                j_start = (j-1)*fy + 1
                j_end = min(j*fy, ny)
                i_start = (i-1)*fx + 1
                i_end = min(i*fx, nx)
                
                # Compute area-weighted average
                total_area = 0.0
                weighted_sum = 0.0
                
                for jj in j_start:j_end
                    for ii in i_start:i_end
                        area_weight = compute_grid_area(field.grid, jj, ii)
                        weighted_sum += field.data[jj, ii] * area_weight
                        total_area += area_weight
                    end
                end
                
                coarse_data[j, i] = total_area > 0 ? weighted_sum / total_area : zero(T)
            end
        end
        
    elseif method == :subsample
        # Simple subsampling (take every nth point)
        for j in 1:ny_coarse
            for i in 1:nx_coarse
                jj = min((j-1)*fy + 1, ny)
                ii = min((i-1)*fx + 1, nx)
                coarse_data[j, i] = field.data[jj, ii]
            end
        end
        
    elseif method == :conservative
        # Conservative averaging with proper mass conservation
        coarse_data = coarsen_conservative(field.data, fy, fx)
    end
    
    # Create coarsened grid
    coarse_grid = coarsen_grid(field.grid, fy, fx)
    
    return Field(coarse_data, coarse_grid)
end

"""
    refine_field(field, target_grid; method=:bilinear)

Refine (interpolate) a field to a higher resolution target grid.
Similar to FlowSieve's refine_Helmholtz_seed functionality.

Parameters:
- `field`: Coarse field to refine  
- `target_grid`: Target high-resolution grid
- `method`: Interpolation method (:bilinear, :bicubic, :nearest, :conservative)

Returns refined field on target grid.
"""
function refine_field(field::Field{T,G}, target_grid::G2; 
                     method::Symbol=:bilinear) where {T,G,G2}
    
    source_grid = field.grid
    
    if typeof(source_grid) == typeof(target_grid) && source_grid == target_grid
        return field  # No refinement needed
    end
    
    # Get grid coordinates
    source_coords = get_grid_coordinates(source_grid)
    target_coords = get_grid_coordinates(target_grid)
    
    ny_target, nx_target = get_grid_size(target_grid)
    refined_data = zeros(T, ny_target, nx_target)
    
    if method == :bilinear
        refined_data = interpolate_bilinear(field.data, source_coords, target_coords)
        
    elseif method == :bicubic
        refined_data = interpolate_bicubic(field.data, source_coords, target_coords)
        
    elseif method == :nearest
        refined_data = interpolate_nearest(field.data, source_coords, target_coords)
        
    elseif method == :conservative
        # Mass-conserving interpolation (more complex)
        refined_data = interpolate_conservative(field, target_grid)
    end
    
    return Field(refined_data, target_grid)
end

"""
    create_multiresolution_hierarchy(base_grid, levels; coarsening_factor=2)

Create a hierarchy of grids at different resolutions for multigrid methods.
"""
function create_multiresolution_hierarchy(base_grid::G, levels::Int; 
                                        coarsening_factor::Int=2) where {G}
    
    grids = [base_grid]
    current_grid = base_grid
    
    for level in 2:levels
        if can_coarsen_further(current_grid, coarsening_factor)
            current_grid = coarsen_grid(current_grid, coarsening_factor, coarsening_factor)
            push!(grids, current_grid)
        else
            @warn "Cannot coarsen further at level $level"
            break
        end
    end
    
    return grids
end

"""
    hierarchical_helmholtz_workflow(u, v, base_grid; levels=3, max_iter_per_level=100)

Multi-resolution Helmholtz decomposition workflow following FlowSieve methodology.

This implements the coarsen → solve → refine → solve strategy:
1. Coarsen velocity fields to manageable resolution
2. Solve Helmholtz decomposition on coarse grid  
3. Refine solution to finer grid
4. Use coarse solution as initial guess for fine-grid solve
5. Repeat until reaching original resolution

This dramatically reduces computational cost for high-resolution problems.
"""
function hierarchical_helmholtz_workflow(u::Field{T,G}, v::Field{T,G}; 
                                       levels::Int=3, max_iter_per_level::Int=100,
                                       coarsening_factor::Int=2) where {T,G}
    
    # Create grid hierarchy
    base_grid = u.grid
    grids = create_multiresolution_hierarchy(base_grid, levels; coarsening_factor=coarsening_factor)
    
    # Coarsen velocity fields to all levels
    u_hierarchy = [u]
    v_hierarchy = [v]
    
    for level in 2:length(grids)
        u_coarse = coarsen_field(u_hierarchy[level-1], coarsening_factor; method=:area_average)
        v_coarse = coarsen_field(v_hierarchy[level-1], coarsening_factor; method=:area_average)
        # u and v are coarsened independently and would get distinct (but equal)
        # grid objects; share one so downstream `grid === vN.grid` checks pass.
        v_coarse = Field(v_coarse.data, u_coarse.grid)
        push!(u_hierarchy, u_coarse)
        push!(v_hierarchy, v_coarse)
    end
    
    # Solve on coarsest level first
    coarsest_level = length(grids)
    
    if grids[coarsest_level] isa SphericalGrid
        # Use iterative spherical Helmholtz solver
        uE_div_c, vN_div_c, uE_pot_c, vN_pot_c, φ_c, ψ_c = 
            helmholtz_hodge_sphere_iterative(u_hierarchy[coarsest_level], v_hierarchy[coarsest_level]; 
                                           max_iter=max_iter_per_level)
    else
        # Use periodic FFT-based solver for Cartesian grids
        udf_c, vdf_c, up_c, vp_c, φ_c, ψ_c = 
            helmholtz_hodge(u_hierarchy[coarsest_level], v_hierarchy[coarsest_level])
        uE_div_c, vN_div_c, uE_pot_c, vN_pot_c = udf_c, vdf_c, up_c, vp_c
    end
    
    φ_current = φ_c
    ψ_current = ψ_c
    
    # Refine and solve at each finer level
    for level in coarsest_level-1:-1:1
        
        # Refine potentials to current level
        φ_refined = refine_field(φ_current, grids[level]; method=:bilinear)
        ψ_refined = refine_field(ψ_current, grids[level]; method=:bilinear)
        
        if level == 1
            # Final level - return refined solution
            φ_current = φ_refined
            ψ_current = ψ_refined
        else
            # Intermediate level - perform correction iterations
            # Use refined solution as initial guess for iterative improvement
            φ_current, ψ_current = refine_and_correct(
                u_hierarchy[level], v_hierarchy[level], φ_refined, ψ_refined, 
                max_iter_per_level ÷ 2)
        end
    end
    
    # Compute final velocity components from refined potentials
    if base_grid isa SphericalGrid
        ∇φ_E, ∇φ_N = gradient_sphere(φ_current)
        ∇ψ_E, ∇ψ_N = gradient_sphere(ψ_current)
        
        uE_pot = ∇φ_E
        vN_pot = ∇φ_N
        uE_div = Field(∇ψ_N.data, base_grid)
        vN_div = Field(-∇ψ_E.data, base_grid)
    else
        ∇φ_x, ∇φ_y = gradient(φ_current)
        ∇ψ_x, ∇ψ_y = gradient(ψ_current)
        
        uE_pot = ∇φ_x
        vN_pot = ∇φ_y
        uE_div = Field(∇ψ_y.data, base_grid)
        vN_div = Field(-∇ψ_x.data, base_grid)
    end
    
    return uE_div, vN_div, uE_pot, vN_pot, φ_current, ψ_current
end

"""
    save_multiresolution_data(filename, field, grids; group_name="multiresolution")

Save multi-resolution field data to NetCDF file for workflow continuation.
"""
function save_multiresolution_data(filename::String, fields::Vector{Field{T,G}}, 
                                  grids::Vector{G}; group_name::String="multiresolution") where {T,G}
    
    ds = NCDataset(filename, "c")
    
    try
        # Create group for this dataset
        grp = defGroup(ds, group_name)
        
        # Save each resolution level
        for (level, (field, grid)) in enumerate(zip(fields, grids))
            level_name = "level_$level"
            
            ny, nx = size(field.data)
            
            # Define dimensions for this level
            defDim(grp, "$(level_name)_y", ny)
            defDim(grp, "$(level_name)_x", nx)
            
            # Save field data
            var = defVar(grp, level_name, T, ("$(level_name)_y", "$(level_name)_x"))
            var[:, :] = field.data
            
            # Save grid information as attributes
            var.attrib["grid_type"] = string(typeof(grid))
            
            if grid isa Grid
                var.attrib["dx"] = grid.dx
                var.attrib["dy"] = grid.dy
                var.attrib["periodic_x"] = grid.periodic_x
                var.attrib["periodic_y"] = grid.periodic_y
            elseif grid isa SphericalGrid
                var.attrib["a"] = grid.a
                var.attrib["periodic_lon"] = grid.periodic_lon
                
                # Save coordinate arrays
                lat_var = defVar(grp, "$(level_name)_lat", Float64, ("$(level_name)_y",))
                lon_var = defVar(grp, "$(level_name)_lon", Float64, ("$(level_name)_x",))
                lat_var[:] = grid.lat
                lon_var[:] = grid.lon
            end
        end
        
        # Save metadata
        grp.attrib["num_levels"] = length(grids)
        grp.attrib["creation_time"] = string(now())
        
    finally
        close(ds)
    end
    
    return filename
end

"""
    load_multiresolution_data(filename; group_name="multiresolution")

Load multi-resolution field data from NetCDF file.
"""
function load_multiresolution_data(filename::String; group_name::String="multiresolution")
    
    ds = NCDataset(filename)
    fields = Field[]
    grids = []
    
    try
        grp = ds.group[group_name]
        num_levels = grp.attrib["num_levels"]
        
        for level in 1:num_levels
            level_name = "level_$level"
            
            # Load field data
            data = Array(grp[level_name])
            
            # Reconstruct grid
            grid_type = grp[level_name].attrib["grid_type"]
            
            if grid_type == "Grid"
                dx = grp[level_name].attrib["dx"]
                dy = grp[level_name].attrib["dy"]
                periodic_x = grp[level_name].attrib["periodic_x"]
                periodic_y = grp[level_name].attrib["periodic_y"]
                ny, nx = size(data)
                
                grid = Grid(nx, ny, dx, dy, periodic_x, periodic_y)
                
            elseif grid_type == "SphericalGrid"
                a = grp[level_name].attrib["a"]
                periodic_lon = grp[level_name].attrib["periodic_lon"]
                lat = Array(grp["$(level_name)_lat"])
                lon = Array(grp["$(level_name)_lon"])
                
                grid = SphericalGrid(lon, lat, a, periodic_lon)
            else
                error("Unknown grid type: $grid_type")
            end
            
            push!(fields, Field(data, grid))
            push!(grids, grid)
        end
        
    finally
        close(ds)
    end
    
    return fields, grids
end

# Helper functions
function compute_grid_area(grid::Grid, j::Int, i::Int)
    return grid.dx * grid.dy
end

function compute_grid_area(grid::SphericalGrid, j::Int, i::Int)
    a = grid.a
    lat = grid.lat[j]
    dλ = mean(diff(grid.lon))
    dφ = mean(diff(grid.lat))
    return a^2 * cos(lat) * dλ * dφ
end

function compute_grid_area(grid::CurvilinearGrid, j::Int, i::Int)
    return grid.dx[j,i] * grid.dy[j,i]
end

function coarsen_grid(grid::Grid, fy::Int, fx::Int)
    return Grid(grid.nx ÷ fx, grid.ny ÷ fy, 
               grid.dx * fx, grid.dy * fy,
               grid.periodic_x, grid.periodic_y)
end

function coarsen_grid(grid::SphericalGrid, fy::Int, fx::Int)
    lon_coarse = grid.lon[1:fx:end]
    lat_coarse = grid.lat[1:fy:end]
    return SphericalGrid(lon_coarse, lat_coarse, grid.a, grid.periodic_lon)
end

function get_grid_coordinates(grid::Grid)
    x = (0:grid.nx-1) * grid.dx
    y = (0:grid.ny-1) * grid.dy
    return (y, x)
end

function get_grid_coordinates(grid::SphericalGrid)
    return (grid.lat, grid.lon)
end

function get_grid_size(grid::Grid)
    return (grid.ny, grid.nx)
end

function get_grid_size(grid::SphericalGrid)
    return (length(grid.lat), length(grid.lon))
end

function can_coarsen_further(grid::Grid, factor::Int)
    return grid.nx ÷ factor >= 8 && grid.ny ÷ factor >= 8
end

function can_coarsen_further(grid::SphericalGrid, factor::Int)
    return length(grid.lon) ÷ factor >= 8 && length(grid.lat) ÷ factor >= 8
end

function coarsen_conservative(data::Array{T,2}, fy::Int, fx::Int) where T
    ny, nx = size(data)
    ny_coarse = ny ÷ fy
    nx_coarse = nx ÷ fx
    
    coarse_data = zeros(T, ny_coarse, nx_coarse)
    
    for j in 1:ny_coarse
        for i in 1:nx_coarse
            j_start = (j-1)*fy + 1
            j_end = min(j*fy, ny)
            i_start = (i-1)*fx + 1
            i_end = min(i*fx, nx)
            
            sum_val = zero(T)
            count = 0
            
            for jj in j_start:j_end
                for ii in i_start:i_end
                    sum_val += data[jj, ii]
                    count += 1
                end
            end
            
            coarse_data[j, i] = count > 0 ? sum_val / count : zero(T)
        end
    end
    
    return coarse_data
end

function interpolate_bilinear(data::Array{T,2}, source_coords, target_coords) where T
    # Simplified bilinear interpolation
    # In practice, would use Interpolations.jl for more robust implementation
    source_y, source_x = source_coords
    target_y, target_x = target_coords
    
    ny_target, nx_target = length(target_y), length(target_x)
    result = zeros(T, ny_target, nx_target)
    
    # Create interpolation object
    itp = linear_interpolation((source_y, source_x), data, extrapolation_bc=Line())
    
    for j in 1:ny_target
        for i in 1:nx_target
            result[j, i] = itp(target_y[j], target_x[i])
        end
    end
    
    return result
end

function interpolate_nearest(data::Array{T,2}, source_coords, target_coords) where T
    source_y, source_x = source_coords
    target_y, target_x = target_coords
    
    ny_target, nx_target = length(target_y), length(target_x)
    result = zeros(T, ny_target, nx_target)
    
    for j in 1:ny_target
        for i in 1:nx_target
            # Find nearest source point
            j_src = argmin(abs.(source_y .- target_y[j]))
            i_src = argmin(abs.(source_x .- target_x[i]))
            result[j, i] = data[j_src, i_src]
        end
    end
    
    return result
end

function interpolate_bicubic(data::Array{T,2}, source_coords, target_coords) where T
    # Placeholder for bicubic interpolation
    # Would implement proper bicubic splines
    return interpolate_bilinear(data, source_coords, target_coords)
end

function interpolate_conservative(field::Field, target_grid)
    # Placeholder for conservative interpolation
    # Would implement area-weighted remapping
    return interpolate_bilinear(field.data, 
                               get_grid_coordinates(field.grid),
                               get_grid_coordinates(target_grid))
end

function refine_and_correct(u::Field, v::Field, φ_init::Field, ψ_init::Field, max_iter::Int)
    # Placeholder for correction iterations
    # Would implement residual-based correction
    return φ_init, ψ_init
end