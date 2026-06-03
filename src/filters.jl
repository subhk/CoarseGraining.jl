using ..CoarseGraining: Grid, Field, Kernel, SphericalGrid
using DSP
import Base.Threads

export coarse_grain, coarse_grain_sphere, coarse_grain_fft, coarse_grain_gaussian_separable, select_tile,
       coarse_grain_butterworth, coarse_grain_butterworth_length,
       coarse_grain_butterworth_cycles, coarse_grain_butterworth_cells, coarse_grain_butterworth_nyquist

function padidx(i, n, periodic)
    if 1 <= i <= n
        return i
    elseif periodic
        j = mod(i-1, n) + 1
        return j
    else
        return clamp(i, 1, n)
    end
end

"""
    coarse_grain_butterworth_dsp(field, kc; order=2, zero_phase=true)

Butterworth low-pass using DSP.jl digital IIR filtering applied separably along x and y.

⚠️  **IMPORTANT DIFFERENCES FROM FFT VERSION**:
- Applies **separable 1D filters** (x then y), not true 2D isotropic filtering
- May introduce **edge artifacts** due to IIR nature (unlike periodic FFT version)  
- Different frequency response: H(kx)×H(ky) ≠ H(√(kx²+ky²))

Parameters:
- `kc`: cutoff wavenumber (scalar/tuple) in radians per length
- `order`: Butterworth filter order (steepness)
- `zero_phase`: if true, uses filtfilt (forward-backward) to eliminate phase shift

For exact equivalence to the FFT version, use `coarse_grain_butterworth()` instead.
"""
function coarse_grain_butterworth_dsp(field::Field{T,G}, kc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    @assert order >= 1
    A = Float64.(field.data)
    ny, nx = size(A)
    dx = field.grid.dx; dy = field.grid.dy
    if kc isa Tuple
        kcx, kcy = kc
    else
        kcx = kcy = Float64(kc)
    end
    wx = clamp(kcx*dx/π, 0.0, 1.0)
    wy = clamp(kcy*dy/π, 0.0, 1.0)
    fx = digitalfilter(Lowpass(wx), Butterworth(order))
    fy = digitalfilter(Lowpass(wy), Butterworth(order))
    # Apply along x (dim=2) then y (dim=1). Some DSP versions don't support `dims`,
    # so we manually loop over vectors.
    B = _apply_iir_along(A, fx; dim=2, zero_phase=zero_phase)
    C = _apply_iir_along(B, fy; dim=1, zero_phase=zero_phase)
    return Field(T.(C), field.grid)
end

# Helper: apply IIR filter along a specific dimension by looping vectors
function _apply_iir_along(A::AbstractMatrix{<:Real}, filt, ; dim::Int, zero_phase::Bool)
    ny, nx = size(A)
    out = similar(A, Float64)
    if dim == 2
        @inbounds for j in 1:ny
            row = @view A[j, :]
            x = collect(row)
            # Use periodic extension to avoid edge transients boosting variance
            xr = vcat(x, x, x)
            ylong = zero_phase ? DSP.filtfilt(filt, xr) : DSP.filt(filt, xr)
            y = @view ylong[length(x)+1:2*length(x)]
            @views out[j, :] .= y
        end
    elseif dim == 1
        @inbounds for i in 1:nx
            col = @view A[:, i]
            x = collect(col)
            xr = vcat(x, x, x)
            ylong = zero_phase ? DSP.filtfilt(filt, xr) : DSP.filt(filt, xr)
            y = @view ylong[length(x)+1:2*length(x)]
            @views out[:, i] .= y
        end
    else
        error("dim must be 1 or 2, got $dim")
    end
    return out
end

"""
    coarse_grain_butterworth_length_dsp(field, ℓc; order=2, zero_phase=true)

DSP-based Butterworth specifying cutoff length(s) ℓ.
"""
function coarse_grain_butterworth_length_dsp(field::Field{T,G}, ℓc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    if ℓc isa Tuple
        ℓx, ℓy = ℓc
        @assert ℓx > 0 && ℓy > 0
        return coarse_grain_butterworth_dsp(field, (2π/ℓx, 2π/ℓy); order=order, zero_phase=zero_phase)
    else
        @assert ℓc > 0
        return coarse_grain_butterworth_dsp(field, 2π/ℓc; order=order, zero_phase=zero_phase)
    end
end

function coarse_grain_butterworth_cycles_dsp(field::Field{T,G}, cycles::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    nx = field.grid.nx; ny = field.grid.ny
    dx = field.grid.dx; dy = field.grid.dy
    if cycles isa Tuple
        cx, cy = cycles
        @assert cx >= 0 && cy >= 0
        kcx = 2π*cx/(nx*dx)
        kcy = 2π*cy/(ny*dy)
        return coarse_grain_butterworth_dsp(field, (kcx, kcy); order=order, zero_phase=zero_phase)
    else
        c = cycles
        @assert c >= 0
        kcx = 2π*c/(nx*dx)
        kcy = 2π*c/(ny*dy)
        return coarse_grain_butterworth_dsp(field, (kcx, kcy); order=order, zero_phase=zero_phase)
    end
end

function coarse_grain_butterworth_cells_dsp(field::Field{T,G}, cells::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    dx = field.grid.dx; dy = field.grid.dy
    if cells isa Tuple
        cx, cy = cells
        @assert cx > 0 && cy > 0
        return coarse_grain_butterworth_length_dsp(field, (cx*dx, cy*dy); order=order, zero_phase=zero_phase)
    else
        c = cells
        @assert c > 0
        return coarse_grain_butterworth_length_dsp(field, c*dx; order=order, zero_phase=zero_phase)
    end
end

function coarse_grain_butterworth_nyquist_dsp(field::Field{T,G}, frac::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    dx = field.grid.dx; dy = field.grid.dy
    if frac isa Tuple
        fx, fy = frac
        @assert 0 < fx <= 1 && 0 < fy <= 1
        return coarse_grain_butterworth_dsp(field, (fx*(π/dx), fy*(π/dy)); order=order, zero_phase=zero_phase)
    else
        f = frac
        @assert 0 < f <= 1
        return coarse_grain_butterworth_dsp(field, (f*(π/dx), f*(π/dy)); order=order, zero_phase=zero_phase)
    end
end

"""
    _auto_tile(ny, nx, ry, rx)

Heuristic tile size selection to improve cache locality for real-space filters.
Targets roughly 256 KiB working set per tile (double precision) considering input + output.
"""
function _auto_tile(ny::Int, nx::Int, ry::Int, rx::Int; threads::Int=Threads.nthreads())
    # Target bytes per tile (approx): 256 KiB
    target = 256 * 1024
    bytes_per_point = 8 * 2 # input + output
    # Assume halo increases needed reads; keep a margin factor
    margin = 1.5
    # Assume each thread should fit a tile in (approx) per-core cache budget
    area = Int(floor((target) / (bytes_per_point * margin)))
    side = Int(floor(sqrt(area)))
    bj = clamp(side, 32, ny)
    bi = clamp(side, 32, nx)
    return (bj, bi)
end

"""
    select_tile(ny, nx, ry, rx; large_thresh=1_000_000)

Public helper to pick a reasonable tile `(bj, bi)` for cache-friendly real-space filtering.
For arrays smaller than `large_thresh`, returns `(64,64)`; otherwise uses a 256 KiB heuristic.
"""
function select_tile(ny::Int, nx::Int, ry::Int, rx::Int; large_thresh::Int=1_000_000, threads::Int=Threads.nthreads())
    if ny*nx <= large_thresh
        return (min(64, ny), min(64, nx))
    else
        return _auto_tile(ny, nx, ry, rx; threads=threads)
    end
end

"""
    select_tile(field::Field, kernel::Kernel; large_thresh=1_000_000, threads=Threads.nthreads())

Select a tile for a concrete field and kernel.
"""
function select_tile(field::Field, kernel::Kernel; large_thresh::Int=1_000_000, threads::Int=Threads.nthreads())
    ny, nx = size(field.data)
    ry, rx = kernel.radius_y, kernel.radius_x
    return select_tile(ny, nx, ry, rx; large_thresh=large_thresh, threads=threads)
end

function coarse_grain(field::Field{T,G}, kernel::Kernel; threaded::Bool=false, tile::Tuple{Int,Int}=(0,0)) where {T<:Real,G}
    A = field.data
    ny, nx = size(A)
    K = kernel.weights
    ry, rx = kernel.radius_y, kernel.radius_x
    out = similar(A)
    bj, bi = tile
    # Auto-select tile if requested
    if threaded && bj == 0 && bi == 0
        bj, bi = select_tile(ny, nx, ry, rx)
    end
    if bj > 0 && bi > 0
        if threaded
            Threads.@threads for j0 in 1:bj:ny
                j1 = min(j0 + bj - 1, ny)
                for i0 in 1:bi:nx
                    i1 = min(i0 + bi - 1, nx)
                    @inbounds for j in j0:j1, i in i0:i1
                        acc = 0.0
                        @inbounds for ky in -ry:ry
                            jj = padidx(j+ky, ny, _periodic_y(field.grid))
                            @inbounds for kx in -rx:rx
                                ii = padidx(i+kx, nx, _periodic_x(field.grid))
                                acc += K[ky+ry+1, kx+rx+1] * float(A[jj, ii])
                            end
                        end
                        out[j,i] = T(acc)
                    end
                end
            end
        else
            for j0 in 1:bj:ny
                j1 = min(j0 + bj - 1, ny)
                for i0 in 1:bi:nx
                    i1 = min(i0 + bi - 1, nx)
                    @inbounds for j in j0:j1, i in i0:i1
                        acc = 0.0
                        @inbounds for ky in -ry:ry
                            jj = padidx(j+ky, ny, _periodic_y(field.grid))
                            @inbounds for kx in -rx:rx
                                ii = padidx(i+kx, nx, _periodic_x(field.grid))
                                acc += K[ky+ry+1, kx+rx+1] * float(A[jj, ii])
                            end
                        end
                        out[j,i] = T(acc)
                    end
                end
            end
        end
    else
        if threaded
            Threads.@threads for j in 1:ny
                @inbounds for i in 1:nx
                    acc = 0.0
                    @inbounds for ky in -ry:ry
                        jj = padidx(j+ky, ny, _periodic_y(field.grid))
                        @inbounds for kx in -rx:rx
                            ii = padidx(i+kx, nx, _periodic_x(field.grid))
                            acc += K[ky+ry+1, kx+rx+1] * float(A[jj, ii])
                        end
                    end
                    out[j,i] = T(acc)
                end
            end
        else
            @inbounds for j in 1:ny
                @inbounds for i in 1:nx
                    acc = 0.0
                    @inbounds for ky in -ry:ry
                        jj = padidx(j+ky, ny, _periodic_y(field.grid))
                        @inbounds for kx in -rx:rx
                            ii = padidx(i+kx, nx, _periodic_x(field.grid))
                            acc += K[ky+ry+1, kx+rx+1] * float(A[jj, ii])
                        end
                    end
                    out[j,i] = T(acc)
                end
            end
        end
    end
    return Field(out, field.grid)
end

"""
    coarse_grain(field, L; kernel=:tophat, truncate=3.0, threaded=false, tile=(0,0))
    coarse_grain(field, Lx, Ly; kernel=:tophat, truncate=3.0, threaded=false, tile=(0,0))

Convenience real-space coarse-graining by filter length `L` (in grid cells),
selecting the kernel type with `kernel`:

- `:tophat` / `:boxcar` — uniform box average, radius `max(1, round(Int, L/2))`.
- `:gaussian` — Gaussian with standard deviation `σ = L/2`.

Both use half-width `L/2`, so switching the kernel at the same `L` gives comparable
smoothing. Defaults to a **top-hat** filter. Builds the kernel and applies the
real-space `coarse_grain(field, kernel)`; existing methods are unchanged.
"""
function coarse_grain(field::Field, Lx::Real, Ly::Real; kernel::Symbol=:tophat,
                      truncate::Real=3.0, threaded::Bool=false, tile::Tuple{Int,Int}=(0,0))
    K = _build_kernel(kernel, Lx, Ly, truncate)
    return coarse_grain(field, K; threaded=threaded, tile=tile)
end

coarse_grain(field::Field, L::Real; kwargs...) = coarse_grain(field, L, L; kwargs...)

function _build_kernel(kernel::Symbol, Lx::Real, Ly::Real, truncate::Real)
    if kernel === :tophat || kernel === :boxcar
        return boxcar_kernel(max(1, round(Int, Lx/2)), max(1, round(Int, Ly/2)))
    elseif kernel === :gaussian
        return gaussian_kernel(Lx/2, Ly/2; truncate=truncate)
    else
        error("Unknown kernel $kernel; use :tophat, :boxcar, or :gaussian")
    end
end

# FlowSieve coarse-graining kernels G(dist; scale), normalised distance D = dist/(scale/2).
# Matches husseinaluie/FlowSieve Functions/kernel.cpp exactly.
@inline function _sphere_kernel(D::Real, kerneltype::Symbol)
    if kerneltype === :tophat || kerneltype === :boxcar
        return D < 1 ? 1.0 : 0.0
    elseif kerneltype === :gaussian
        return exp(-D^2)
    elseif kerneltype === :hyper_gaussian
        return exp(-D^4)
    elseif kerneltype === :smooth_hat
        return 0.5 * (1 - tanh((D - 1) / 0.1))
    else
        error("Unknown sphere kernel :$kerneltype; use :tophat, :gaussian, :hyper_gaussian, or :smooth_hat")
    end
end
# Search radius as a multiple of scale/2 (FlowSieve KernPad): far enough to capture the tail.
_sphere_kernel_pad(k::Symbol) = (k === :tophat || k === :boxcar) ? 1.1 :
                                k === :hyper_gaussian ? 2.5 :
                                k === :smooth_hat ? 2.5 : 5.0

"""
    coarse_grain_sphere(field::Field{T,SphericalGrid}, ℓ; kernel=:tophat, smooth=false, mask=nothing, fill_value=NaN)

Geometrically-correct coarse-graining on a sphere with FlowSieve's kernels. `ℓ` is the
filter scale (great-circle **diameter**, units of `grid.a`); the normalised distance is
`D = dist/(ℓ/2)`. `kernel` selects the shape (matching FlowSieve `kernel.cpp`):
`:tophat` (`D<1`), `:gaussian` (`exp(−D²)`), `:hyper_gaussian` (`exp(−D⁴)`), `:smooth_hat`
(`½(1−tanh((D−1)/0.1))`). Each neighbour is weighted by its cell area (∝ cosφ·Δφ_cell)
— the cosφ pole-convergence correction. The neighbour search radius scales with the
kernel (FlowSieve `KernPad`). `mask` (true=valid) skips land; `fill_value` where none
contributes. `smooth=true` is a back-compat alias for `kernel=:smooth_hat`.
"""
function coarse_grain_sphere(field::Field{T,G}, ℓ::Real; kernel::Symbol=:tophat, smooth::Bool=false,
                             mask::Union{BitArray{2},Nothing}=nothing,
                             fill_value=NaN) where {T<:Real,G<:SphericalGrid}
    kerneltype = (smooth && kernel === :tophat) ? :smooth_hat : kernel
    grid = field.grid
    A = field.data
    ny = length(grid.lat); nx = length(grid.lon)
    @assert size(A) == (ny, nx)
    mask === nothing || @assert size(mask) == (ny, nx)
    a = grid.a
    ϕ = grid.lat; λ = grid.lon
    cosϕ = cos.(ϕ)
    # Local latitude cell widths (non-uniform aware) for the area weight.
    Δϕ = Vector{Float64}(undef, ny)
    @inbounds for j in 1:ny
        Δϕ[j] = j == 1 ? (ϕ[2]-ϕ[1]) : (j == ny ? (ϕ[ny]-ϕ[ny-1]) : (ϕ[j+1]-ϕ[j-1])/2)
    end
    dλ = nx > 1 ? (λ[end]-λ[1])/(nx-1) : 1.0
    rad = ℓ / 2                      # great-circle radius (kernel half-width)
    pad = _sphere_kernel_pad(kerneltype)
    αr  = (rad * pad) / a            # search angular radius (FlowSieve KernPad)
    periodic = grid.periodic_lon
    out = similar(A)
    @inbounds for j in 1:ny
        ϕj = ϕ[j]
        # Latitude index window: |ϕ[jj]-ϕj| ≤ αr
        jmin = j; while jmin > 1 && (ϕj - ϕ[jmin-1]) <= αr; jmin -= 1; end
        jmax = j; while jmax < ny && (ϕ[jmax+1] - ϕj) <= αr; jmax += 1; end
        # Longitude half-extent in cells (wider near poles), capped to half the circle.
        cj = max(cosϕ[j], 1e-6)
        iext = min(nx ÷ 2, ceil(Int, αr / (cj * dλ)) + 1)
        for i in 1:nx
            if mask !== nothing && !mask[j, i]
                out[j, i] = T(fill_value); continue
            end
            λi = λ[i]
            acc = 0.0; wsum = 0.0
            for jj in jmin:jmax
                area = cosϕ[jj] * Δϕ[jj]
                for dii in -iext:iext
                    ii0 = i + dii
                    if 1 <= ii0 <= nx
                        ii = ii0
                    elseif periodic
                        ii = mod(ii0 - 1, nx) + 1
                    else
                        continue
                    end
                    (mask !== nothing && !mask[jj, ii]) && continue
                    dist = grid_distance(grid, ϕj, λi, ϕ[jj], λ[ii])
                    w = _sphere_kernel(dist / rad, kerneltype)
                    w == 0.0 && continue
                    wgt = w * area
                    acc += wgt * float(A[jj, ii])
                    wsum += wgt
                end
            end
            out[j, i] = wsum > 0 ? T(acc / wsum) : T(fill_value)
        end
    end
    return Field(out, grid)
end

function coarse_grain_fft(field::Field{T,G}, σx::Real, σy::Real) where {T<:Real,G}
    # Gaussian filter via FFT assuming periodic boundaries.
    A = Float64.(field.data)
    ny, nx = size(A)
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*field.grid.dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*field.grid.dy))
    KX = reshape(kx, 1, :)
    KY = reshape(ky, :, 1)
    # Transfer function in FFT ordering
    transfer_func = @. exp(-0.5*((σx^2)*(KX^2) + (σy^2)*(KY^2)))
    F = fft(A)
    out = real(ifft(F .* transfer_func))
    return Field(T.(out), field.grid)
end

"""
    coarse_grain_butterworth(field, kc; order=2)

Apply an FFT-based Butterworth low-pass filter of given `order`.

Parameters:
- `kc`: cutoff wavenumber (scalar, radians per length) for isotropic filter,
        or a tuple `(kcx, kcy)` for anisotropic cutoff along x and y.
- `order`: positive integer Butterworth order (steepness).

Assumes periodic boundaries. For convenience, pass `kc = 2π/ℓc` to specify a cutoff length ℓc.
"""
function coarse_grain_butterworth(field::Field{T,G}, kc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2) where {T<:Real,G}
    @assert order >= 1 "Butterworth order must be >= 1"
    A = Float64.(field.data)
    ny, nx = size(A)
    dx = field.grid.dx; dy = field.grid.dy
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*dy))
    KX = reshape(kx, 1, :)
    KY = reshape(ky, :, 1)
    if kc isa Tuple
        kcx, kcy = kc
        @assert kcx > 0 && kcy > 0
        R = @. sqrt((KX/kcx)^2 + (KY/kcy)^2)
    else
        @assert kc > 0
        R = @. sqrt((KX)^2 + (KY)^2) / kc
    end
    n = order
    H = @. 1.0 / (1.0 + R^(2n))
    F = fft(A)
    out = real(ifft(F .* H))
    return Field(T.(out), field.grid)
end

"""
    coarse_grain_butterworth_length(field, ℓc; order=2)

Convenience wrapper for Butterworth filtering using cutoff length scale(s) ℓc.
- If `ℓc` is scalar, uses isotropic cutoff with `kc = 2π/ℓc`.
- If `ℓc` is a tuple `(ℓx, ℓy)`, uses anisotropic cutoff `(kcx, kcy) = (2π/ℓx, 2π/ℓy)`.
"""
function coarse_grain_butterworth_length(field::Field{T,G}, ℓc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2) where {T<:Real,G}
    if ℓc isa Tuple
        ℓx, ℓy = ℓc
        @assert ℓx > 0 && ℓy > 0
        return coarse_grain_butterworth(field, (2π/ℓx, 2π/ℓy); order=order)
    else
        @assert ℓc > 0
        return coarse_grain_butterworth(field, 2π/ℓc; order=order)
    end
end

"""
    coarse_grain_butterworth_cycles(field, cycles; order=2)

Butterworth using cutoff specified as cycles per domain along each axis.
`cycles` can be a scalar or a tuple `(cx, cy)`; converted to wavenumbers
`kcx = 2π*cx/(nx*dx)`, `kcy = 2π*cy/(ny*dy)`.
"""
function coarse_grain_butterworth_cycles(field::Field{T,G}, cycles::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2) where {T<:Real,G}
    nx = field.grid.nx; ny = field.grid.ny
    dx = field.grid.dx; dy = field.grid.dy
    if cycles isa Tuple
        cx, cy = cycles
        @assert cx >= 0 && cy >= 0
        kcx = 2π*cx/(nx*dx)
        kcy = 2π*cy/(ny*dy)
        return coarse_grain_butterworth(field, (kcx, kcy); order=order)
    else
        c = cycles
        @assert c >= 0
        kcx = 2π*c/(nx*dx)
        kcy = 2π*c/(ny*dy)
        return coarse_grain_butterworth(field, (kcx, kcy); order=order)
    end
end

"""
    coarse_grain_butterworth_cells(field, cells; order=2)

Butterworth using cutoff specified in grid cells.
`cells` can be a scalar or a tuple `(cx, cy)` representing cutoff lengths in cells,
converted to physical lengths ℓx=cx*dx, ℓy=cy*dy.
"""
function coarse_grain_butterworth_cells(field::Field{T,G}, cells::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2) where {T<:Real,G}
    dx = field.grid.dx; dy = field.grid.dy
    if cells isa Tuple
        cx, cy = cells
        @assert cx > 0 && cy > 0
        return coarse_grain_butterworth_length(field, (cx*dx, cy*dy); order=order)
    else
        c = cells
        @assert c > 0
        return coarse_grain_butterworth_length(field, c*dx; order=order)
    end
end

"""
    coarse_grain_butterworth_nyquist(field, frac; order=2)

Butterworth using cutoff specified as a fraction of the Nyquist wavenumber along each axis.
`frac` can be a scalar or a tuple `(fx, fy)` with 0 < f ≤ 1.
`kcx = fx * (π/dx)`, `kcy = fy * (π/dy)`.
"""
function coarse_grain_butterworth_nyquist(field::Field{T,G}, frac::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2) where {T<:Real,G}
    dx = field.grid.dx; dy = field.grid.dy
    if frac isa Tuple
        fx, fy = frac
        @assert 0 < fx <= 1 && 0 < fy <= 1
        return coarse_grain_butterworth(field, (fx*(π/dx), fy*(π/dy)); order=order)
    else
        f = frac
        @assert 0 < f <= 1
        return coarse_grain_butterworth(field, (f*(π/dx), f*(π/dy)); order=order)
    end
end

"""
    coarse_grain_gaussian_separable(field, σx, σy; truncate=3.0)

Separable Gaussian filtering using 1D convolutions along x then y. Supports periodic/clamped boundaries.
"""
function coarse_grain_gaussian_separable(field::Field{T,G}, σx::Real, σy::Real; truncate::Real=3.0, threaded::Bool=true) where {T<:Real,G}
    A = Float64.(field.data)
    ny, nx = size(A)
    rx = max(1, Int(ceil(truncate*σx)))
    ry = max(1, Int(ceil(truncate*σy)))
    kx = [exp(-0.5*(i/σx)^2) for i in -rx:rx]; kx ./= sum(kx)
    ky = [exp(-0.5*(j/σy)^2) for j in -ry:ry]; ky ./= sum(ky)
    tmp = similar(A)
    out = similar(A)
    # Convolve along x (row-wise). This has strided reads in column-major layout,
    # but is still efficient and benefits from threading.
    if threaded
        Threads.@threads for j in 1:ny
            @inbounds for i in 1:nx
                acc = 0.0
                @inbounds @simd for s in -rx:rx
                    ii = padidx(i+s, nx, _periodic_x(field.grid))
                    acc += kx[s+rx+1] * A[j, ii]
                end
                tmp[j,i] = acc
            end
        end
    else
        @inbounds for j in 1:ny
            @inbounds for i in 1:nx
                acc = 0.0
                @inbounds @simd for s in -rx:rx
                    ii = padidx(i+s, nx, _periodic_x(field.grid))
                    acc += kx[s+rx+1] * A[j, ii]
                end
                tmp[j,i] = acc
            end
        end
    end
    # Convolve along y. For column-major arrays, iterating over i outer and j inner
    # yields contiguous memory access on tmp[:, i].
    if threaded
        Threads.@threads for i in 1:nx
            @inbounds for j in 1:ny
                acc = 0.0
                @inbounds @simd for t in -ry:ry
                    jj = padidx(j+t, ny, _periodic_y(field.grid))
                    acc += ky[t+ry+1] * tmp[jj, i]
                end
                out[j,i] = acc
            end
        end
    else
        @inbounds for i in 1:nx
            @inbounds for j in 1:ny
                acc = 0.0
                @inbounds @simd for t in -ry:ry
                    jj = padidx(j+t, ny, _periodic_y(field.grid))
                    acc += ky[t+ry+1] * tmp[jj, i]
                end
                out[j,i] = acc
            end
        end
    end
    return Field(T.(out), field.grid)
end
