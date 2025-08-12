using ..CoarseGraining: Field, Grid

export regrid_index_bilinear

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

