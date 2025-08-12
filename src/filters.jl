using ..CoarseGraining: Grid, Field, Kernel

export coarse_grain, coarse_grain_fft

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

function coarse_grain(field::Field{T}, kernel::Kernel) where {T<:Real}
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

function coarse_grain_fft(field::Field{T}, σx::Real, σy::Real) where {T<:Real}
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
