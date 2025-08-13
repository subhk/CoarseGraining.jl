using ..CoarseGraining: Field, Grid

export regrid_index_bilinear, regrid_lonlat_nearest

"""
    regrid_index_bilinear(field, nx, ny)

Resample a 2D field to new integer dimensions `(ny, nx)` using bilinear interpolation in index space.
This assumes source grid is quasi-uniform and monotonic; it does not use lon/lat coordinates.
Returns a new Field with a uniform Grid using scaled mean dx, dy.
"""
function regrid_index_bilinear(field::Field{T,G}, nx::Int, ny::Int) where {T,G}
    A = Float64.(field.data)
    sy, sx = size(A)
    # Coordinates in source index space and target normalized coords
    y = collect(1:sy)
    x = collect(1:sx)
    yt = range(1, sy, length=ny)
    xt = range(1, sx, length=nx)
    out = Array{Float64}(undef, ny, nx)
    for (jj, yy) in enumerate(yt)
        y0 = clamp(floor(Int, yy), 1, sy-1)
        y1 = y0 + 1
        wy = yy - y0
        for (ii, xx) in enumerate(xt)
            x0 = clamp(floor(Int, xx), 1, sx-1)
            x1 = x0 + 1
            wx = xx - x0
            v00 = A[y0, x0]
            v10 = A[y1, x0]
            v01 = A[y0, x1]
            v11 = A[y1, x1]
            v0 = (1-wx)*v00 + wx*v01
            v1 = (1-wx)*v10 + wx*v11
            out[jj, ii] = (1-wy)*v0 + wy*v1
        end
    end
    # New grid: scale mean dx, dy
    dx = getfield(field.grid, :dx)
    dy = getfield(field.grid, :dy)
    mdx = dx isa Number ? dx * (sx/nx) : mean(dx) * (sx/nx)
    mdy = dy isa Number ? dy * (sy/ny) : mean(dy) * (sy/ny)
    g = Grid(nx, ny, mdx, mdy, getfield(field.grid, :periodic_x), getfield(field.grid, :periodic_y))
    return Field(T.(out), g)
end

"""
    regrid_lonlat_nearest(field, lon_src, lat_src, lon_tgt, lat_tgt; degrees=true, radius=6.371e6)

Nearest-neighbor regrid using lon/lat without external dependencies.
- `lon_src, lat_src`: source 2D arrays matching `field.data`.
- `lon_tgt, lat_tgt`: target lon/lat as 2D arrays of desired shape.
Returns a new array of size `size(lon_tgt)` with nearest source values.
Note: O(N_src * N_tgt), suitable for moderate sizes.
"""
function regrid_lonlat_nearest(field::Field{T,G}, lon_src::AbstractArray, lat_src::AbstractArray, lon_tgt::AbstractArray, lat_tgt::AbstractArray; degrees::Bool=true, radius::Real=6.371e6) where {T,G}
    Asrc = field.data
    ny, nx = size(Asrc)
    @assert size(lon_src) == (ny, nx) && size(lat_src) == (ny, nx)
    nty, ntx = size(lon_tgt)
    @assert size(lat_tgt) == (nty, ntx)
    # Flatten source coords
    if degrees
        deg2rad = π/180
        ϕs = lat_src .* deg2rad
        λs = lon_src .* deg2rad
        ϕt = lat_tgt .* deg2rad
        λt = lon_tgt .* deg2rad
    else
        ϕs = lat_src; λs = lon_src
        ϕt = lat_tgt; λt = lon_tgt
    end
    src_vals = vec(Asrc)
    ϕs_v = vec(ϕs)
    λs_v = vec(λs)
    out = Array{eltype(Asrc)}(undef, nty, ntx)
    # Haversine distance
    function hav2(φ1, λ1, φ2, λ2)
        dφ = φ2 - φ1
        dλ = λ2 - λ1
        a = sin(dφ/2)^2 + cos(φ1)*cos(φ2)*sin(dλ/2)^2
        return a
    end
    for j in 1:nty
        for i in 1:ntx
            φ = ϕt[j,i]; λ = λt[j,i]
            best = 1.0
            bestk = 1
            for k in eachindex(src_vals)
                a = hav2(ϕs_v[k], λs_v[k], φ, λ)
                if a < best
                    best = a; bestk = k
                end
            end
            out[j,i] = src_vals[bestk]
        end
    end
    return out
end

