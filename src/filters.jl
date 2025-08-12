using ..CoarseGraining: Grid, Field, Kernel

export coarse_grain, coarse_grain_fft, coarse_grain_gaussian_separable

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

function coarse_grain(field::Field{T,G}, kernel::Kernel) where {T<:Real,G}
    A = field.data
    ny, nx = size(A)
    K = kernel.weights
    ry, rx = kernel.radius_y, kernel.radius_x
    out = similar(A)
    for j in 1:ny
        for i in 1:nx
            acc = zero(Float64)
            for ky in -ry:ry
                jj = padidx(j+ky, ny, field.grid.periodic_y)
                for kx in -rx:rx
                    ii = padidx(i+kx, nx, field.grid.periodic_x)
                    acc += K[ky+ry+1, kx+rx+1] * float(A[jj, ii])
                end
            end
            out[j,i] = T(acc)
        end
    end
    return Field(out, field.grid)
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
    G = @. exp(-0.5*((σx^2)*(KX^2) + (σy^2)*(KY^2)))
    F = fft(A)
    out = real(ifft(F .* G))
    return Field(T.(out), field.grid)
end

"""
    coarse_grain_gaussian_separable(field, σx, σy; truncate=3.0)

Separable Gaussian filtering using 1D convolutions along x then y. Supports periodic/clamped boundaries.
"""
function coarse_grain_gaussian_separable(field::Field{T,G}, σx::Real, σy::Real; truncate::Real=3.0) where {T<:Real,G}
    A = Float64.(field.data)
    ny, nx = size(A)
    rx = max(1, Int(ceil(truncate*σx)))
    ry = max(1, Int(ceil(truncate*σy)))
    kx = [exp(-0.5*(i/σx)^2) for i in -rx:rx]; kx ./= sum(kx)
    ky = [exp(-0.5*(j/σy)^2) for j in -ry:ry]; ky ./= sum(ky)
    tmp = similar(A)
    out = similar(A)
    # Convolve along x
    for j in 1:ny
        for i in 1:nx
            acc = 0.0
            for s in -rx:rx
                ii = padidx(i+s, nx, field.grid.periodic_x)
                acc += kx[s+rx+1] * A[j, ii]
            end
            tmp[j,i] = acc
        end
    end
    # Convolve along y
    for j in 1:ny
        for i in 1:nx
            acc = 0.0
            for t in -ry:ry
                jj = padidx(j+t, ny, field.grid.periodic_y)
                acc += ky[t+ry+1] * tmp[jj, i]
            end
            out[j,i] = acc
        end
    end
    return Field(T.(out), field.grid)
end
