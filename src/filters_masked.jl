using ..CoarseGraining: Field, Grid, Kernel

export coarse_grain_masked

"""
    coarse_grain_masked(field, kernel, mask; normalize=true, fill_value=NaN)

Masked real-space convolution: applies `kernel` only over valid (true) entries of `mask`.
If `normalize=true`, renormalizes by the sum of weights over valid neighbors.
Boundary handling follows grid periodic flags; invalid or out-of-bound samples are skipped.
If no valid neighbors contribute, writes `fill_value`.
"""
function coarse_grain_masked(field::Field{T,G}, kernel::Kernel, mask::BitArray{2}; normalize::Bool=true, fill_value=NaN) where {T<:Real,G}
    A = field.data
    ny, nx = size(A)
    @assert size(mask) == (ny, nx)
    K = kernel.weights
    ry, rx = kernel.radius_y, kernel.radius_x
    out = similar(A)
    periodic_x = getfield(field.grid, :periodic_x)
    periodic_y = getfield(field.grid, :periodic_y)
    @inbounds for j in 1:ny
        for i in 1:nx
            acc = 0.0
            wsum = 0.0
            for ky in -ry:ry
                jj = j+ky
                if !(1 <= jj <= ny)
                    if periodic_y
                        jj = mod(jj-1, ny) + 1
                    else
                        continue
                    end
                end
                for kx in -rx:rx
                    ii = i+kx
                    if !(1 <= ii <= nx)
                        if periodic_x
                            ii = mod(ii-1, nx) + 1
                        else
                            continue
                        end
                    end
                    if mask[jj, ii]
                        w = K[ky+ry+1, kx+rx+1]
                        acc += w * float(A[jj, ii])
                        wsum += w
                    end
                end
            end
            if wsum > 0 && normalize
                out[j,i] = T(acc / wsum)
            elseif wsum > 0
                out[j,i] = T(acc)
            else
                out[j,i] = T(fill_value)
            end
        end
    end
    return Field(out, field.grid)
end

