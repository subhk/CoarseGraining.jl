module ModelIO

using NCDatasets
using ..CoarseGraining: Field, Grid

export detect_model, load_model_var, average_to_rho, average_to_tracer_mitgcm

const EARTH_RADIUS = 6.371e6

function detect_model(ds::NCDataset)
    haskey(ds, "lon_rho") && return :ROMS
    haskey(ds, "lon_u") && return :ROMS
    haskey(ds, "XC") && return :MITgcm
    haskey(ds, "Longitude_t") && return :MITgcm
    # CROCO uses ROMS convention
    haskey(ds, "lon_rho") && return :CROCO
    return :unknown
end

function _read2d(ds, name; t=1, z=1)
    v = ds[name]
    dims = dimnames(v)
    A = Array(v)
    nd = ndims(A)
    if nd == 2
        return permutedims(A, (2,1))  # (y,x)
    elseif nd == 3
        # assume time or depth present
        if "time" in dims || first(dims) in ("ocean_time", "time")
            return permutedims(Array(view(A, t, :, :)), (2,1))
        else
            return permutedims(Array(view(A, :, :, z)), (2,1))
        end
    elseif nd == 4
        # likely (time, z, y, x)
        return permutedims(Array(view(A, t, z, :, :)), (2,1))
    else
        error("Unsupported variable dims for $name: ndims=$(nd)")
    end
end

"""
    average_to_rho(u, v)

Average ROMS/CROCO C-grid u and v components to the rho (cell center) points.
`u` is size (ny, nxu) and `v` is size (nyv, nx). Returns arrays at (ny, nx).
"""
function average_to_rho(u::AbstractArray{<:Real,2}, v::AbstractArray{<:Real,2})
    ny_u, nx_u = size(u)
    ny_v, nx_v = size(v)
    ny = min(ny_u, ny_v)
    nx = min(nx_u, nx_v)
    ur = 0.5 .* (u[1:ny, 1:nx] .+ u[1:ny, 2:nx+1])
    vr = 0.5 .* (v[1:ny, 1:nx] .+ v[2:ny+1, 1:nx])
    return ur, vr
end

"""
    average_to_tracer_mitgcm(U, V)

Average MITgcm staggered U (x-face) and V (y-face) onto tracer (cell center) points.
Supports U sized (ny, nx+1) and V sized (ny+1, nx) or both sized (ny, nx) if already centered.
"""
function average_to_tracer_mitgcm(U::AbstractArray{<:Real,2}, V::AbstractArray{<:Real,2})
    nyU, nxU = size(U)
    nyV, nxV = size(V)
    if nxU == nxV+1 && nyV == nyU+1
        ur = 0.5 .* (U[:, 1:nxV] .+ U[:, 2:nxV+1])
        vr = 0.5 .* (V[1:nyU, :] .+ V[2:nyU+1, :])
    elseif nxU == nxV+1
        ur = 0.5 .* (U[:, 1:nxV] .+ U[:, 2:nxV+1])
        vr = V
    elseif nyV == nyU+1
        ur = U
        vr = 0.5 .* (V[1:nyU, :] .+ V[2:nyU+1, :])
    else
        ur = U; vr = V
    end
    ny = min(size(ur,1), size(vr,1))
    nx = min(size(ur,2), size(vr,2))
    return ur[1:ny,1:nx], vr[1:ny,1:nx]
end

function _mean_dxdy(lon::AbstractArray, lat::AbstractArray; a::Real=EARTH_RADIUS)
    # Estimate typical dx, dy using central finite differences in degrees -> meters
    ny, nx = size(lon)
    ic = clamp(nx ÷ 2, 2, nx-1)
    jc = clamp(ny ÷ 2, 2, ny-1)
    deg2rad = π/180
    ϕ = lat .* deg2rad
    λ = lon .* deg2rad
    cϕ = cos(ϕ[jc,ic])
    dx = a * cϕ * abs(λ[jc,ic+1] - λ[jc,ic-1]) / 2
    dy = a * abs(ϕ[jc+1,ic] - ϕ[jc-1,ic]) / 2
    return dx, dy
end

function _compute_dxdy(lon::AbstractArray, lat::AbstractArray; a::Real=EARTH_RADIUS)
    ny, nx = size(lon)
    deg2rad = π/180
    ϕ = lat .* deg2rad
    λ = lon .* deg2rad
    dx = similar(λ)
    dy = similar(ϕ)
    for j in 1:ny, i in 1:nx
        il = max(i-1, 1); ir = min(i+1, nx)
        jl = max(j-1, 1); jr = min(j+1, ny)
        cϕ = cos(ϕ[j,i])
        dx[j,i] = a * cϕ * abs(λ[j,ir] - λ[j,il]) / max(ir - il, 1)
        dy[j,i] = a * abs(ϕ[jr,i] - ϕ[jl,i]) / max(jr - jl, 1)
    end
    return dx, dy
end

function _detect_periodicity(lon::AbstractArray, lat::AbstractArray)
    lonmin = minimum(lon); lonmax = maximum(lon)
    periodic_x = (lonmax - lonmin) > 350
    periodic_y = false
    return periodic_x, periodic_y
end

"""
    load_model_var(path; model=:auto, varname, at=:rho, t=1, z=1, earth_radius=6.371e6)

Load a 2D field from CROCO/ROMS/MITgcm NetCDF into a CoarseGraining Field with an approximate Cartesian Grid.
For C-grid velocity, use `at=:rho` to average to rho points.
Returns (Field, metadata::Dict).
"""
function load_model_var(path::AbstractString; model::Symbol=:auto, varname::AbstractString, at::Symbol=:rho, t::Int=1, z::Int=1, earth_radius::Real=EARTH_RADIUS, curvilinear::Bool=false, include_mask::Bool=true)
    ds = NCDataset(path)
    m = model == :auto ? detect_model(ds) : model
    meta = Dict{Symbol,Any}(:model => m)
    if m in (:ROMS, :CROCO)
        lon = permutedims(Array(ds["lon_rho"]), (2,1))
        lat = permutedims(Array(ds["lat_rho"]), (2,1))
        dx, dy = _mean_dxdy(lon, lat; a=earth_radius)
        meta[:lon] = lon; meta[:lat] = lat
        if include_mask && haskey(ds, "mask_rho")
            meta[:mask] = permutedims(Bool.(Array(ds["mask_rho"]) .> 0), (2,1))
        end
        if varname in ("u", "v")
            if at == :rho
                u = _read2d(ds, "u"; t=t, z=z)
                v = _read2d(ds, "v"; t=t, z=z)
                ur, vr = average_to_rho(u, v)
                if varname == "u"
                    data = ur
                else
                    data = vr
                end
            elseif varname == "u"
                data = _read2d(ds, "u"; t=t, z=z)
            else
                data = _read2d(ds, "v"; t=t, z=z)
            end
        else
            data = _read2d(ds, varname; t=t, z=z)
        end
        close(ds)
        if curvilinear
            dx2, dy2 = _compute_dxdy(lon, lat; a=earth_radius)
            px, py = _detect_periodicity(lon, lat)
            g = CurvilinearGrid(lon, lat, dx2, dy2, px, py, earth_radius)
        else
            g = Grid(size(data,2), size(data,1), dx, dy, false, false)
        end
        return Field(data, g), meta
    elseif m == :MITgcm
        # Tracer grid
        lon = haskey(ds, "XC") ? permutedims(Array(ds["XC"]), (2,1)) : permutedims(Array(ds["Longitude_t"]), (2,1))
        lat = haskey(ds, "YC") ? permutedims(Array(ds["YC"]), (2,1)) : permutedims(Array(ds["Latitude_t"]), (2,1))
        dx, dy = _mean_dxdy(lon, lat; a=earth_radius)
        meta[:lon] = lon; meta[:lat] = lat
        if varname in ("U", "V") && at == :rho
            U = _read2d(ds, "U"; t=t, z=z)
            V = _read2d(ds, "V"; t=t, z=z)
            ur, vr = average_to_tracer_mitgcm(U, V)
            data = varname == "U" ? ur : vr
        else
        data = _read2d(ds, varname; t=t, z=z)
        if include_mask
            if haskey(ds, "maskC")
                meta[:mask] = permutedims(Bool.(Array(ds["maskC"]) .> 0), (2,1))
            elseif haskey(ds, "HFacC")
                # Use surface layer occupancy > 0 as mask
                H = Array(ds["HFacC"])  # (x,y,z) or (z,y,x)
                nd = ndims(H)
                M = nothing
                if nd == 3
                    # Assume (z, y, x) or (x,y,z). Try both; prefer last dim as x
                    if size(H,3) == size(lon,2)
                        M = permutedims(H[1, :, :], (2,1)) .> 0
                    else
                        M = permutedims(H[:, :, 1], (2,1)) .> 0
                    end
                end
                if M !== nothing
                    meta[:mask] = Bool.(M)
                end
            end
        end
        end
        close(ds)
        if curvilinear
            dx2, dy2 = _compute_dxdy(lon, lat; a=earth_radius)
            px, py = _detect_periodicity(lon, lat)
            g = CurvilinearGrid(lon, lat, dx2, dy2, px, py, earth_radius)
        else
            g = Grid(size(data,2), size(data,1), dx, dy, false, false)
        end
        return Field(data, g), meta
    else
        close(ds)
        error("Unknown or unsupported model format; provide model=:ROMS|:CROCO|:MITgcm")
    end
end

end # module

using .ModelIO: detect_model, load_model_var, average_to_rho, average_to_tracer_mitgcm
