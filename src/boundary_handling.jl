using ..CoarseGraining: Field, Grid, SphericalGrid, CurvilinearGrid, Kernel

export coarse_grain_adaptive, compute_adaptive_bounds, land_avoiding_stencil,
       coarse_grain_masked_adaptive, extend_field_to_boundaries,
       apply_coastal_boundary_conditions

"""
    coarse_grain_adaptive(field, kernel; mask=nothing, boundary_mode=:adaptive)

Advanced coarse-graining with sophisticated boundary handling following FlowSieve methodology.

Features:
- Adaptive integration bounds based on kernel size and physical distance
- Land-avoiding stencils for coastal regions
- Non-uniform grid support
- Multiple boundary condition options

Parameters:
- `field`: Field to filter (Grid, SphericalGrid, or CurvilinearGrid)
- `kernel`: Filtering kernel
- `mask`: Optional land/water mask (true = water, false = land)
- `boundary_mode`: :adaptive, :fixed, :periodic, :extend

Returns filtered field with proper boundary treatment.
"""
function coarse_grain_adaptive(field::Field{T,G}, kernel::Kernel;
                              mask::Union{BitArray{2},Nothing}=nothing,
                              boundary_mode::Symbol=:adaptive,
                              deform_around_land::Bool=true,
                              kernel_padding::Real=2.0) where {T<:Real,G}

    A = field.data
    ny, nx = size(A)
    K = kernel.weights
    ry, rx = kernel.radius_y, kernel.radius_x
    out = similar(A)

    # Precompute mean grid spacing once for spherical grids (constant across the
    # field) instead of recomputing `mean(diff(...))` per point/neighbor.
    if field.grid isa SphericalGrid
        dλ_mean = mean(diff(field.grid.lon))
        dφ_mean = mean(diff(field.grid.lat))
    else
        dλ_mean = 0.0
        dφ_mean = 0.0
    end

    # Process each point with adaptive boundaries
    for j in 1:ny
        for i in 1:nx

            # Skip land points if mask provided
            if mask !== nothing && !mask[j, i]
                out[j, i] = zero(T)
                continue
            end

            # Compute adaptive integration bounds
            if boundary_mode == :adaptive
                j_bounds, i_bounds = compute_adaptive_bounds(
                    field.grid, j, i, kernel, kernel_padding, dλ_mean, dφ_mean)
            else
                # Use fixed kernel bounds
                j_bounds = max(1, j-ry):min(ny, j+ry)
                i_bounds = max(1, i-rx):min(nx, i+rx)
            end

            # Apply filtering with land-avoiding stencil
            filtered_val, norm_factor = apply_land_avoiding_filter(
                A, K, field.grid, j, i, j_bounds, i_bounds,
                ry, rx, mask, deform_around_land, dλ_mean, dφ_mean)

            # Normalize by effective kernel weight
            if norm_factor > 0
                out[j, i] = T(filtered_val / norm_factor)
            else
                out[j, i] = A[j, i]  # Fallback to original value
            end
        end
    end

    return Field(out, field.grid)
end

"""
    grid_distance(grid, lat1, lon1, lat2, lon2)

Physical distance between two grid points. Dispatched on grid type so the call
site is type-stable (no abstractly-typed closure / dynamic dispatch in hot loops).
"""
@inline function grid_distance(grid::SphericalGrid, lat1, lon1, lat2, lon2)
    # Haversine great-circle distance
    dφ = lat2 - lat1
    dλ = lon2 - lon1
    α = sin(dφ/2)^2 + cos(lat1) * cos(lat2) * sin(dλ/2)^2
    return 2 * grid.a * atan(sqrt(α), sqrt(1-α))
end

@inline function grid_distance(grid::CurvilinearGrid, lat1, lon1, lat2, lon2)
    # Simple Euclidean distance on projected coordinates
    dx = (lon2 - lon1) * cos((lat1 + lat2)/2)
    dy = lat2 - lat1
    return grid.a * sqrt(dx^2 + dy^2)
end

"""
    compute_adaptive_bounds(grid, j, i, kernel, padding, dλ_mean, dφ_mean)

Compute adaptive integration bounds based on physical distance and kernel scale.
Implements FlowSieve's get_lat_bounds and get_lon_bounds methodology.
`dλ_mean`/`dφ_mean` are the precomputed mean grid spacings (used for SphericalGrid).
"""
function compute_adaptive_bounds(grid::G, j::Int, i::Int, kernel::Kernel,
                                padding::Real, dλ_mean::Float64, dφ_mean::Float64) where {G}

    ry, rx = kernel.radius_y, kernel.radius_x
    ny, nx = size_from_grid(grid)

    if grid isa Grid
        # Cartesian grid - use simple padding
        dx, dy = grid.dx, grid.dy
        scale_x = rx * dx * padding
        scale_y = ry * dy * padding

        j_extent = ceil(Int, scale_y / dy)
        i_extent = ceil(Int, scale_x / dx)

        j_bounds = max(1, j-j_extent):min(ny, j+j_extent)
        i_bounds = max(1, i-i_extent):min(nx, i+i_extent)

    elseif grid isa SphericalGrid
        # Spherical grid - use great circle distances
        lat = grid.lat
        lon = grid.lon
        a = grid.a

        ref_lat = lat[j]
        ref_lon = lon[i]

        # Estimate kernel scale in meters
        if grid.a > 1e6  # Assume Earth-like radius
            scale_m = padding * max(rx * a * dλ_mean * cos(ref_lat), ry * a * dφ_mean)
        else
            scale_m = padding * max(rx, ry)
        end

        # Find latitude bounds
        j_min = j; j_max = j
        for jj in j-1:-1:1
            if grid_distance(grid, ref_lat, ref_lon, lat[jj], ref_lon) > scale_m
                break
            end
            j_min = jj
        end
        for jj in j+1:ny
            if grid_distance(grid, ref_lat, ref_lon, lat[jj], ref_lon) > scale_m
                break
            end
            j_max = jj
        end

        # Find longitude bounds (accounting for periodicity)
        i_min = i; i_max = i
        for ii in i-1:-1:1
            if grid_distance(grid, ref_lat, ref_lon, ref_lat, lon[ii]) > scale_m
                break
            end
            i_min = ii
        end
        for ii in i+1:nx
            if grid_distance(grid, ref_lat, ref_lon, ref_lat, lon[ii]) > scale_m
                break
            end
            i_max = ii
        end

        # Handle longitude periodicity
        if grid.periodic_lon
            if i_min == 1
                for ii in nx:-1:nx-rx
                    if grid_distance(grid, ref_lat, ref_lon, ref_lat, lon[ii]) <= scale_m
                        i_min = ii
                    else
                        break
                    end
                end
            end
            if i_max == nx
                for ii in 1:rx
                    if grid_distance(grid, ref_lat, ref_lon, ref_lat, lon[ii]) <= scale_m
                        i_max = ii
                    else
                        break
                    end
                end
            end
        end

        j_bounds = j_min:j_max
        i_bounds = i_min:i_max

    else  # CurvilinearGrid
        # Use precomputed distance function for irregular grids
        scale_m = padding * max(rx, ry) * estimate_grid_scale(grid, j, i)

        j_min, j_max = find_distance_bounds(grid, j, i, scale_m, :lat)
        i_min, i_max = find_distance_bounds(grid, j, i, scale_m, :lon)

        j_bounds = max(1, j_min):min(ny, j_max)
        i_bounds = max(1, i_min):min(nx, i_max)
    end

    return j_bounds, i_bounds
end

"""
    apply_land_avoiding_filter(A, K, grid, j, i, j_bounds, i_bounds, ry, rx, mask, deform, dλ_mean, dφ_mean)

Apply filtering with land-avoiding stencil and proper normalization.
"""
function apply_land_avoiding_filter(A::Array{T,2}, K::Array{Float64,2}, grid::G,
                                   j::Int, i::Int, j_bounds, i_bounds,
                                   ry::Int, rx::Int, mask, deform_around_land::Bool,
                                   dλ_mean::Float64, dφ_mean::Float64) where {T,G}

    filtered_val = 0.0
    norm_factor = 0.0
    ny, nx = size(A)

    for jj in j_bounds
        for ii in i_bounds
            # Handle periodicity
            jj_idx, ii_idx = handle_periodicity(grid, jj, ii, ny, nx)

            # Skip if out of bounds and not periodic
            if jj_idx < 1 || jj_idx > ny || ii_idx < 1 || ii_idx > nx
                continue
            end

            # Check land mask
            is_water = mask === nothing || mask[jj_idx, ii_idx]

            # Compute kernel weight
            if grid isa Grid
                # Regular Cartesian kernel indexing
                ky = jj - j + ry + 1
                kx = ii - i + rx + 1
                if 1 ≤ ky ≤ size(K,1) && 1 ≤ kx ≤ size(K,2)
                    kernel_weight = K[ky, kx]
                else
                    continue
                end
            else
                # Distance-based kernel for spherical/curvilinear grids
                kernel_weight = compute_distance_kernel(grid, j, i, jj_idx, ii_idx, ry, rx, dλ_mean, dφ_mean)
            end

            # Apply land-avoiding logic
            if deform_around_land
                # Include kernel weight in normalization only for water points
                if is_water
                    norm_factor += kernel_weight
                    filtered_val += kernel_weight * A[jj_idx, ii_idx]
                end
            else
                # Always include in normalization, but zero out land values
                norm_factor += kernel_weight
                if is_water
                    filtered_val += kernel_weight * A[jj_idx, ii_idx]
                end
            end
        end
    end

    return filtered_val, norm_factor
end

"""
    compute_distance_kernel(grid, j, i, jj, ii, ry, rx, dλ_mean, dφ_mean)

Compute kernel weight based on physical distance for non-Cartesian grids.
`dλ_mean`/`dφ_mean` are precomputed mean grid spacings (SphericalGrid only).
"""
function compute_distance_kernel(grid::G, j::Int, i::Int,
                                jj::Int, ii::Int, ry::Int, rx::Int,
                                dλ_mean::Float64, dφ_mean::Float64) where {G}

    if grid isa SphericalGrid
        lat1, lon1 = grid.lat[j], grid.lon[i]
        lat2, lon2 = grid.lat[jj], grid.lon[ii]
        distance = grid_distance(grid, lat1, lon1, lat2, lon2)

        # Convert kernel radius to physical scale
        a = grid.a
        σx = rx * a * dλ_mean * cos(lat1)
        σy = ry * a * dφ_mean

        # Gaussian kernel based on distance
        return exp(-0.5 * distance^2 / (σx^2 + σy^2))

    elseif grid isa CurvilinearGrid
        lat1, lon1 = grid.lat[j,i], grid.lon[j,i]
        lat2, lon2 = grid.lat[jj,ii], grid.lon[jj,ii]
        distance = grid_distance(grid, lat1, lon1, lat2, lon2)

        # Estimate local grid scale
        dx_local = grid.dx[j,i]
        dy_local = grid.dy[j,i]
        σ = sqrt((rx * dx_local)^2 + (ry * dy_local)^2)

        return exp(-0.5 * (distance/σ)^2)
    else
        return 1.0  # Fallback
    end
end

"""
    handle_periodicity(grid, j, i, ny, nx)

Handle periodic boundary conditions appropriately for each grid type.
"""
function handle_periodicity(grid::G, j::Int, i::Int, ny::Int, nx::Int) where {G}
    jj = j
    ii = i

    if grid isa Grid
        if grid.periodic_y && (j < 1 || j > ny)
            jj = mod(j-1, ny) + 1
        else
            jj = clamp(j, 1, ny)
        end

        if grid.periodic_x && (i < 1 || i > nx)
            ii = mod(i-1, nx) + 1
        else
            ii = clamp(i, 1, nx)
        end

    elseif grid isa SphericalGrid
        # Latitude is never periodic
        jj = clamp(j, 1, ny)

        # Longitude may be periodic
        if grid.periodic_lon && (i < 1 || i > nx)
            ii = mod(i-1, nx) + 1
        else
            ii = clamp(i, 1, nx)
        end

    elseif grid isa CurvilinearGrid
        # Handle based on grid flags
        if grid.periodic_y && (j < 1 || j > ny)
            jj = mod(j-1, ny) + 1
        else
            jj = clamp(j, 1, ny)
        end

        if grid.periodic_x && (i < 1 || i > nx)
            ii = mod(i-1, nx) + 1
        else
            ii = clamp(i, 1, nx)
        end
    end

    return jj, ii
end

"""
    extend_field_to_boundaries(field, extension_width; method=:mirror)

Extend field values to boundary regions to reduce edge effects.
Similar to FlowSieve's boundary extension routines.

Methods:
- :mirror - Mirror boundary values
- :extrapolate - Linear extrapolation
- :zero - Zero padding
- :periodic - Periodic extension (if appropriate)
"""
function extend_field_to_boundaries(field::Field{T,G}, extension_width::Int;
                                   method::Symbol=:mirror) where {T,G}

    A = field.data
    ny, nx = size(A)
    w = extension_width

    # Create extended array
    A_ext = zeros(T, ny + 2*w, nx + 2*w)

    # Copy original data to center
    A_ext[w+1:end-w, w+1:end-w] = A

    if method == :mirror
        # Mirror boundaries
        A_ext[1:w, w+1:end-w] = A[w:-1:1, :]  # Top
        A_ext[end-w+1:end, w+1:end-w] = A[end:-1:end-w+1, :]  # Bottom
        A_ext[w+1:end-w, 1:w] = A[:, w:-1:1]  # Left
        A_ext[w+1:end-w, end-w+1:end] = A[:, end:-1:end-w+1]  # Right

        # Corners
        A_ext[1:w, 1:w] = A[w:-1:1, w:-1:1]
        A_ext[1:w, end-w+1:end] = A[w:-1:1, end:-1:end-w+1]
        A_ext[end-w+1:end, 1:w] = A[end:-1:end-w+1, w:-1:1]
        A_ext[end-w+1:end, end-w+1:end] = A[end:-1:end-w+1, end:-1:end-w+1]

    elseif method == :extrapolate
        # Linear extrapolation
        for i in 1:w
            A_ext[i, w+1:end-w] = 2*A[1, :] - A[i+1, :]  # Top
            A_ext[end-i+1, w+1:end-w] = 2*A[end, :] - A[end-i, :]  # Bottom
            A_ext[w+1:end-w, i] = 2*A[:, 1] - A[:, i+1]  # Left
            A_ext[w+1:end-w, end-i+1] = 2*A[:, end] - A[:, end-i]  # Right
        end

    elseif method == :zero
        # Zero padding (already initialized to zero)

    elseif method == :periodic
        # Periodic extension
        A_ext[1:w, w+1:end-w] = A[end-w+1:end, :]  # Top
        A_ext[end-w+1:end, w+1:end-w] = A[1:w, :]  # Bottom
        A_ext[w+1:end-w, 1:w] = A[:, end-w+1:end]  # Left
        A_ext[w+1:end-w, end-w+1:end] = A[:, 1:w]  # Right
    end

    # Create extended grid
    if field.grid isa Grid
        dx, dy = field.grid.dx, field.grid.dy
        extended_grid = Grid(nx + 2*w, ny + 2*w, dx, dy,
                           field.grid.periodic_x, field.grid.periodic_y)
    else
        # For spherical/curvilinear grids, this would require more sophisticated handling
        extended_grid = field.grid  # Return original grid for now
        A_ext = A  # Don't extend for non-Cartesian grids
    end

    return Field(A_ext, extended_grid)
end

# Helper functions
function size_from_grid(grid::Grid)
    return (grid.ny, grid.nx)
end

function size_from_grid(grid::SphericalGrid)
    return (length(grid.lat), length(grid.lon))
end

function size_from_grid(grid::CurvilinearGrid)
    return size(grid.lat)
end

function estimate_grid_scale(grid::CurvilinearGrid, j::Int, i::Int)
    return sqrt(grid.dx[j,i]^2 + grid.dy[j,i]^2)
end

function find_distance_bounds(grid::CurvilinearGrid, j::Int, i::Int,
                             scale_m::Real, direction::Symbol)
    lat = grid.lat
    lon = grid.lon
    ny, nx = size(lat)

    if direction == :lat
        min_idx, max_idx = j, j
        for jj in j-1:-1:1
            if grid_distance(grid, lat[j,i], lon[j,i], lat[jj,i], lon[j,i]) > scale_m
                break
            end
            min_idx = jj
        end
        for jj in j+1:ny
            if grid_distance(grid, lat[j,i], lon[j,i], lat[jj,i], lon[j,i]) > scale_m
                break
            end
            max_idx = jj
        end
    else  # :lon
        min_idx, max_idx = i, i
        for ii in i-1:-1:1
            if grid_distance(grid, lat[j,i], lon[j,i], lat[j,i], lon[j,ii]) > scale_m
                break
            end
            min_idx = ii
        end
        for ii in i+1:nx
            if grid_distance(grid, lat[j,i], lon[j,i], lat[j,i], lon[j,ii]) > scale_m
                break
            end
            max_idx = ii
        end
    end

    return min_idx, max_idx
end
