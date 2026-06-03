using ..CoarseGraining: Field, Grid, SphericalGrid, coarse_grain, coarse_grain_sphere
using NCDatasets
import Base.Threads

export map_horizontal, coarse_grain_4d, coarse_grain_netcdf, coarse_grain_scales

"""
    map_horizontal(f, data; threaded=true)

Apply a 2D-slice function `f(::AbstractMatrix)::AbstractMatrix` to every horizontal
slice of an N-D array whose leading two dimensions are `(nlat, nlon)`. The remaining
dimensions (depth, time, …) are looped over — threaded across slices — and the result
keeps the original shape. This is the workhorse for applying any 2D filter/diagnostic
to 3D/4D `(…, lat, lon)` fields.
"""
function map_horizontal(f, data::AbstractArray; threaded::Bool=true)
    ndims(data) >= 2 || error("data needs ≥2 dims with (lat, lon) leading")
    ny, nx = size(data, 1), size(data, 2)
    trailing = size(data)[3:end]
    nslices = prod(trailing; init=1)
    rd = reshape(data, ny, nx, nslices)
    first_out = f(Array(@view rd[:, :, 1]))
    size(first_out) == (ny, nx) || error("slice function must return a ($ny,$nx) matrix")
    out = Array{eltype(first_out)}(undef, ny, nx, nslices)
    out[:, :, 1] .= first_out
    if threaded && nslices > 1
        Threads.@threads for k in 2:nslices
            out[:, :, k] .= f(Array(@view rd[:, :, k]))
        end
    else
        for k in 2:nslices
            out[:, :, k] .= f(Array(@view rd[:, :, k]))
        end
    end
    return reshape(out, ny, nx, trailing...)
end

"""
    coarse_grain_4d(data, grid, ℓ; mask=nothing, smooth=false, fill_value=NaN, threaded=true)

Apply the spherical area-weighted top-hat filter `coarse_grain_sphere` (scale `ℓ`,
great-circle diameter) to every horizontal slice of `data`, whose leading two dims
must be `(nlat, nlon)` matching `grid`. Trailing dims (depth, time, …) are looped.
"""
function coarse_grain_4d(data::AbstractArray, grid::SphericalGrid, ℓ::Real;
                         mask::Union{BitArray{2},Nothing}=nothing, smooth::Bool=false,
                         fill_value=NaN, threaded::Bool=true)
    ny, nx = length(grid.lat), length(grid.lon)
    (size(data, 1), size(data, 2)) == (ny, nx) ||
        error("leading dims $(size(data)[1:2]) must equal (nlat, nlon) = ($ny, $nx)")
    f = slice -> coarse_grain_sphere(Field(slice, grid), ℓ;
                                     smooth=smooth, mask=mask, fill_value=fill_value).data
    return map_horizontal(f, data; threaded=threaded)
end

"""
    coarse_grain_netcdf(infile, varname, ℓ; lon_name="lon", lat_name="lat",
                        outfile=nothing, a=6.371e6, degrees=true, periodic_lon=true,
                        smooth=false, threaded=true) -> (filtered, grid)

Read variable `varname` from CF-convention NetCDF `infile`, spherical-filter every
`(lat, lon)` slice at scale `ℓ` (great-circle diameter, metres), and return the
filtered array in the file's native dimension order plus the `SphericalGrid` used.
The variable may store its dimensions in any order as long as it has dims named
`lat_name`/`lon_name`. If `outfile` is given, the filtered variable (and lon/lat
coordinates) are written to a new NetCDF file.
"""
function coarse_grain_netcdf(infile::String, varname::String, ℓ::Real;
                             lon_name::String="lon", lat_name::String="lat",
                             outfile::Union{String,Nothing}=nothing, a::Real=6.371e6,
                             degrees::Bool=true, periodic_lon::Bool=true,
                             smooth::Bool=false, threaded::Bool=true)
    ds = NCDataset(infile)
    local out_native, grid, dn, lonv, latv
    try
        var = ds[varname]
        dn = collect(NCDatasets.dimnames(var))
        londim = findfirst(==(lon_name), dn)
        latdim = findfirst(==(lat_name), dn)
        (londim === nothing || latdim === nothing) &&
            error("variable $varname lacks $lat_name/$lon_name dims (has $dn)")
        lonv = Array(ds[lon_name]); latv = Array(ds[lat_name])
        vdata = Float64.(Array(var))
        # Permute so dims become (lat, lon, others…)
        others = [d for d in 1:ndims(vdata) if d != latdim && d != londim]
        perm = (latdim, londim, others...)
        pdata = permutedims(vdata, perm)
        latr = degrees ? latv .* (π/180) : float.(latv)
        lonr = degrees ? lonv .* (π/180) : float.(lonv)
        grid = SphericalGrid(collect(lonr), collect(latr), float(a), periodic_lon)
        filt = coarse_grain_4d(pdata, grid, ℓ; smooth=smooth, threaded=threaded)
        out_native = permutedims(filt, invperm(collect(perm)))
    finally
        close(ds)
    end
    if outfile !== nothing
        _write_filtered_netcdf(outfile, varname, dn, lonv, latv, lon_name, lat_name, out_native)
    end
    return out_native, grid
end

"""
    coarse_grain_scales(field, scales; smooth=false, mask=nothing, fill_value=NaN,
                        threaded=true, as_fields=false) -> (data, scales)

Coarse-grain `field` at every scale in `scales` in one call (FlowSieve-style scale
sweep). On a `SphericalGrid`, `scales` are great-circle diameters `ℓ` (metres) passed
to `coarse_grain_sphere` (honouring `smooth`/`mask`/`fill_value`); on a Cartesian
`Grid`, `scales` are filter lengths `L` (cells) passed to the top-hat
`coarse_grain(field, L)`. Returns `(data, scales)` with `data` of shape
`(ny, nx, nscales)` (or a `Vector{Field}` if `as_fields=true`); threaded over scales.
Scale-band content between successive scales is `data[:,:,k] .- data[:,:,k+1]`.
"""
function coarse_grain_scales(field::Field{T,G}, scales::AbstractVector;
                             smooth::Bool=false, mask::Union{BitArray{2},Nothing}=nothing,
                             fill_value=NaN, threaded::Bool=true, as_fields::Bool=false) where {T,G}
    ny, nx = size(field.data)
    ns = length(scales)
    out = Array{T,3}(undef, ny, nx, ns)
    if threaded && ns > 1
        Threads.@threads for k in 1:ns
            out[:, :, k] .= _filter_one(field, scales[k], smooth, mask, fill_value)
        end
    else
        for k in 1:ns
            out[:, :, k] .= _filter_one(field, scales[k], smooth, mask, fill_value)
        end
    end
    if as_fields
        return [Field(out[:, :, k], field.grid) for k in 1:ns], collect(scales)
    end
    return out, collect(scales)
end

# Single-scale filter dispatched by grid type.
_filter_one(field::Field{T,G}, scale, smooth, mask, fill_value) where {T,G<:SphericalGrid} =
    coarse_grain_sphere(field, scale; smooth=smooth, mask=mask, fill_value=fill_value).data
# Cartesian top-hat (index-space); `mask`/`smooth` apply to the spherical path only.
_filter_one(field::Field{T,G}, scale, smooth, mask, fill_value) where {T,G<:Grid} =
    coarse_grain(field, float(scale); kernel=:tophat).data

function _write_filtered_netcdf(outfile, varname, dn, lonv, latv, lon_name, lat_name, data_native)
    ds = NCDataset(outfile, "c")
    try
        for (k, d) in enumerate(dn)
            defDim(ds, d, size(data_native, k))
        end
        if !haskey(ds, lon_name)
            v = defVar(ds, lon_name, eltype(lonv), (lon_name,)); v[:] = lonv
        end
        if !haskey(ds, lat_name)
            v = defVar(ds, lat_name, eltype(latv), (lat_name,)); v[:] = latv
        end
        dv = defVar(ds, varname, eltype(data_native), Tuple(dn))
        dv[ntuple(_ -> Colon(), ndims(data_native))...] = data_native
    finally
        close(ds)
    end
    return outfile
end
