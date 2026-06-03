using ..CoarseGraining: Field, Grid, Kernel
import Base.Threads

export coarse_grain_masked, coarse_grain_gaussian_separable_masked

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
    periodic_x = _periodic_x(field.grid)
    periodic_y = _periodic_y(field.grid)
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

"""
    coarse_grain_gaussian_separable_masked(field, σx, σy, mask;
                                           truncate=3.0, threaded=true, normalize=true, fill_value=NaN)

Separable Gaussian filtering with a land/sea mask for speed.
- `mask` is a `BitArray{2}` with `true` for valid points.
- Each 1D pass renormalizes by the sum of weights over valid points when `normalize=true`.
- Writes `fill_value` where no valid neighbors contribute.
"""
function coarse_grain_gaussian_separable_masked(field::Field{T,G}, σx::Real, σy::Real, mask::BitArray{2};
                                                truncate::Real=3.0, threaded::Bool=true, normalize::Bool=true, fill_value=NaN) where {T<:Real,G}
    A = Float64.(field.data)
    ny, nx = size(A)
    @assert size(mask) == (ny, nx)
    rx = max(1, Int(ceil(truncate*σx)))
    ry = max(1, Int(ceil(truncate*σy)))
    kx = [exp(-0.5*(i/σx)^2) for i in -rx:rx]; kx ./= sum(kx)
    ky = [exp(-0.5*(j/σy)^2) for j in -ry:ry]; ky ./= sum(ky)
    tmp = similar(A)
    tmpmask = similar(mask)
    out = similar(A)
    periodic_x = _periodic_x(field.grid)
    periodic_y = _periodic_y(field.grid)
    # X pass
    if threaded
        Threads.@threads for j in 1:ny
            acc::Float64 = 0.0
            for i in 1:nx
                acc = 0.0; wsum = 0.0
                for s in -rx:rx
                    ii = i+s
                    if !(1 <= ii <= nx)
                        if periodic_x
                            ii = mod(ii-1, nx) + 1
                        else
                            continue
                        end
                    end
                    if mask[j, ii]
                        w = kx[s+rx+1]
                        acc += w * A[j, ii]
                        wsum += w
                    end
                end
                if wsum > 0
                    tmp[j,i] = normalize ? acc/wsum : acc
                    tmpmask[j,i] = true
                else
                    tmp[j,i] = fill_value
                    tmpmask[j,i] = false
                end
            end
        end
    else
        @inbounds for j in 1:ny
            for i in 1:nx
                acc = 0.0; wsum = 0.0
                @inbounds for s in -rx:rx
                    ii = i+s
                    if !(1 <= ii <= nx)
                        if periodic_x
                            ii = mod(ii-1, nx) + 1
                        else
                            continue
                        end
                    end
                    if mask[j, ii]
                        w = kx[s+rx+1]
                        acc += w * A[j, ii]
                        wsum += w
                    end
                end
                if wsum > 0
                    tmp[j,i] = normalize ? acc/wsum : acc
                    tmpmask[j,i] = true
                else
                    tmp[j,i] = fill_value
                    tmpmask[j,i] = false
                end
            end
        end
    end
    # Y pass
    if threaded
        Threads.@threads for i in 1:nx
            for j in 1:ny
                acc = 0.0; wsum = 0.0
                for t in -ry:ry
                    jj = j+t
                    if !(1 <= jj <= ny)
                        if periodic_y
                            jj = mod(jj-1, ny) + 1
                        else
                            continue
                        end
                    end
                    if tmpmask[jj, i]
                        w = ky[t+ry+1]
                        acc += w * tmp[jj, i]
                        wsum += w
                    end
                end
                if wsum > 0
                    out[j,i] = normalize ? acc/wsum : acc
                else
                    out[j,i] = fill_value
                end
            end
        end
    else
        @inbounds for i in 1:nx
            for j in 1:ny
                acc = 0.0; wsum = 0.0
                @inbounds for t in -ry:ry
                    jj = j+t
                    if !(1 <= jj <= ny)
                        if periodic_y
                            jj = mod(jj-1, ny) + 1
                        else
                            continue
                        end
                    end
                    if tmpmask[jj, i]
                        w = ky[t+ry+1]
                        acc += w * tmp[jj, i]
                        wsum += w
                    end
                end
                if wsum > 0
                    out[j,i] = normalize ? acc/wsum : acc
                else
                    out[j,i] = fill_value
                end
            end
        end
    end
    return Field(T.(out), field.grid)
end

