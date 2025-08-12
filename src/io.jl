module IO

using NCDatasets
using ..CoarseGraining: Field, Grid

export load_netcdf_var, write_netcdf_field, read_attr, write_attr, load_region_masks, load_vector_vars, write_vector_vars, write_region_stats, write_regions_file, write_region_stats_with_attrs, write_region_stats_and_masks_with_attrs

function load_netcdf_var(path::AbstractString, varname::AbstractString;
                         xdim::AbstractString="x", ydim::AbstractString="y",
                         dx::Real=1.0, dy::Real=1.0,
                         periodic_x::Bool=true, periodic_y::Bool=true)
    ds = NCDataset(path)
    A = permutedims(Array(ds[varname]), (2,1)) # assume vars ordered (y,x)
    close(ds)
    grid = Grid(size(A,2), size(A,1), float(dx), float(dy), periodic_x, periodic_y)
    return Field(A, grid)
end

function write_netcdf_field(path::AbstractString, varname::AbstractString, field::Field)
    ny, nx = size(field.data)
    ds = NCDataset(path, "c")
    defDim(ds, "y", ny)
    defDim(ds, "x", nx)
    v = defVar(ds, varname, eltype(field.data), ("y", "x"))
    v[:] = permutedims(field.data, (1,2))
    close(ds)
    return path
end

function read_attr(path::AbstractString, name::AbstractString; varname::Union{String,Nothing}=nothing)
    ds = NCDataset(path)
    val = if varname === nothing
        attget(ds, name)
    else
        attget(ds[varname], name)
    end
    close(ds)
    return val
end

function write_attr(path::AbstractString, name::AbstractString, value; varname::Union{String,Nothing}=nothing)
    ds = NCDataset(path, "a")
    if varname === nothing
        attput(ds, name, value)
    else
        attput(ds[varname], name, value)
    end
    close(ds)
    return nothing
end

end # module IO

function load_region_masks(path::AbstractString, names::Vector{String}; threshold=0)
    ds = NCDataset(path)
    masks = Dict{String,BitArray{2}}()
    for name in names
        A = permutedims(Array(ds[name]), (2,1))
        masks[name] = BitArray(A .> threshold)
    end
    close(ds)
    return masks
end

function load_vector_vars(path::AbstractString, uname::String, vname::String; dx=1.0, dy=1.0, periodic_x=true, periodic_y=true)
    u = load_netcdf_var(path, uname; dx=dx, dy=dy, periodic_x=periodic_x, periodic_y=periodic_y)
    v = load_netcdf_var(path, vname; dx=dx, dy=dy, periodic_x=periodic_x, periodic_y=periodic_y)
    return u, v
end

function write_vector_vars(path::AbstractString, uname::String, vname::String, u::Field, v::Field)
    ny, nx = size(u.data)
    ds = NCDataset(path, "c")
    defDim(ds, "y", ny)
    defDim(ds, "x", nx)
    vu = defVar(ds, uname, eltype(u.data), ("y", "x"))
    vv = defVar(ds, vname, eltype(v.data), ("y", "x"))
    vu[:] = permutedims(u.data, (1,2))
    vv[:] = permutedims(v.data, (1,2))
    close(ds)
    return path
end

"""
    write_region_stats(path, stats::Dict{String,Tuple{Float64,Float64}})

Create or overwrite a NetCDF file with region names and their mean/std values.
Writes:
- dim `region = N`
- var `region_names` (String), `region_mean` (Float64), `region_std` (Float64)
"""
function write_region_stats(path::AbstractString, stats::Dict{String,Tuple{Float64,Float64}})
    names = collect(keys(stats))
    N = length(names)
    means = [stats[n][1] for n in names]
    stds  = [stats[n][2] for n in names]
    ds = NCDataset(path, "c")
    defDim(ds, "region", N)
    vnames = defVar(ds, "region_names", String, ("region",))
    vmean  = defVar(ds, "region_mean", Float64, ("region",))
    vstd   = defVar(ds, "region_std", Float64, ("region",))
    vnames[:] = names
    vmean[:]  = means
    vstd[:]   = stds
    close(ds)
    return path
end

end # module IO

using .IO: load_netcdf_var, write_netcdf_field, read_attr, write_attr, load_region_masks, load_vector_vars, write_vector_vars, write_region_stats
using .IO: write_regions_file, write_region_stats_with_attrs, write_region_stats_and_masks_with_attrs

function write_regions_file(path::AbstractString, regions::Dict{String,BitArray{2}}, lon::AbstractVector, lat::AbstractVector)
    ny = length(lat)
    nx = length(lon)
    names = collect(keys(regions))
    ds = NCDataset(path, "c")
    defDim(ds, "y", ny)
    defDim(ds, "x", nx)
    defDim(ds, "region", length(names))
    vlon = defVar(ds, "lon", eltype(lon), ("x",))
    vlat = defVar(ds, "lat", eltype(lat), ("y",))
    vlon[:] = lon
    vlat[:] = lat
    vnames = defVar(ds, "region_names", String, ("region",))
    vnames[:] = names
    for name in names
        mask = regions[name]
        @assert size(mask) == (ny, nx)
        v = defVar(ds, name, Int8, ("y", "x"))
        v[:] = Int8.(mask)
    end
    close(ds)
    return path
end

function write_region_stats_with_attrs(path::AbstractString, stats::Dict{String,Tuple{Float64,Float64}}; global_attrs=Dict{String,Any}(), var_attrs=Dict{String,Dict{String,Any}}())
    names = collect(keys(stats))
    N = length(names)
    means = [stats[n][1] for n in names]
    stds  = [stats[n][2] for n in names]
    ds = NCDataset(path, "c")
    defDim(ds, "region", N)
    vnames = defVar(ds, "region_names", String, ("region",))
    vmean  = defVar(ds, "region_mean", Float64, ("region",))
    vstd   = defVar(ds, "region_std", Float64, ("region",))
    vnames[:] = names
    vmean[:]  = means
    vstd[:]   = stds
    for (k, v) in global_attrs
        attput(ds, k, v)
    end
    for (var, attrs) in var_attrs
        if haskey(ds, var)
            for (k, v) in attrs
                attput(ds[var], k, v)
            end
        end
    end
    close(ds)
    return path
end

function write_region_stats_and_masks_with_attrs(f::Field, regions::Dict{String,BitArray{2}}, stats_path::AbstractString; masks_path::Union{Nothing,AbstractString}=nothing, lon=nothing, lat=nothing, stats_attrs=Dict{String,Any}(), masks_attrs=Dict{String,Any}())
    stats = CoarseGraining.region_mean_std(f, regions)
    write_region_stats_with_attrs(stats_path, stats; global_attrs=stats_attrs)
    if masks_path !== nothing && lon !== nothing && lat !== nothing
        write_regions_file(masks_path, regions, lon, lat)
        if !isempty(masks_attrs)
            ds = NCDataset(masks_path, "a")
            for (k, v) in masks_attrs
                attput(ds, k, v)
            end
            close(ds)
        end
        return (stats_path, masks_path)
    else
        return (stats_path, nothing)
    end
end
