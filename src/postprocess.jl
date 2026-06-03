using ..CoarseGraining: Field, SphericalGrid

export RegionAccumulator, HistogramAccumulator, RunningMean,
       update!, results, cell_area_weights

"""
    cell_area_weights(grid::SphericalGrid) -> Matrix

Per-cell area weights ∝ cosφ·Δφ_cell (the longitude-uniform factor is omitted; it
cancels in any normalized average). Pass as the `area` argument to the streaming
accumulators so means are area-weighted on the sphere.
"""
function cell_area_weights(grid::SphericalGrid)
    ϕ = grid.lat; ny = length(ϕ); nx = length(grid.lon)
    w = Matrix{Float64}(undef, ny, nx)
    @inbounds for j in 1:ny
        Δϕ = j == 1 ? (ϕ[2]-ϕ[1]) : (j == ny ? (ϕ[ny]-ϕ[ny-1]) : (ϕ[j+1]-ϕ[j-1])/2)
        aj = cos(ϕ[j]) * Δϕ
        for i in 1:nx
            w[j, i] = aj
        end
    end
    return w
end

# ─── Region accumulator ──────────────────────────────────────────────────────
"""
    RegionAccumulator(nregions)

Streaming per-region area-weighted mean and standard deviation. Feed 2D slices with
`update!`, then call `results`. No full-dataset storage is required.
"""
struct RegionAccumulator
    nregions::Int
    sumw::Vector{Float64}
    sumwx::Vector{Float64}
    sumwx2::Vector{Float64}
end
RegionAccumulator(nregions::Integer) =
    RegionAccumulator(Int(nregions), zeros(nregions), zeros(nregions), zeros(nregions))

"""
    update!(acc::RegionAccumulator, field, region_ids; area=nothing)

Add one 2D field slice. `region_ids[j,i]` is the integer region index (values outside
`1:nregions`, e.g. 0, are skipped). `area` is an optional per-cell weight matrix
(e.g. `cell_area_weights(grid)`); `nothing` ⇒ unit weights. Non-finite values are skipped.
"""
function update!(acc::RegionAccumulator, field::Field, region_ids::AbstractMatrix{<:Integer};
                 area::Union{AbstractMatrix,Nothing}=nothing)
    A = field.data
    ny, nx = size(A)
    size(region_ids) == (ny, nx) || error("region_ids size must match field")
    @inbounds for j in 1:ny, i in 1:nx
        r = region_ids[j, i]
        (r < 1 || r > acc.nregions) && continue
        x = float(A[j, i])
        isfinite(x) || continue
        w = area === nothing ? 1.0 : float(area[j, i])
        acc.sumw[r]   += w
        acc.sumwx[r]  += w * x
        acc.sumwx2[r] += w * x * x
    end
    return acc
end

"""
    results(acc::RegionAccumulator) -> (mean, std, weight)

Per-region area-weighted mean, standard deviation, and total weight. Empty regions
return `NaN`.
"""
function results(acc::RegionAccumulator)
    mean = similar(acc.sumw); std = similar(acc.sumw)
    @inbounds for r in 1:acc.nregions
        if acc.sumw[r] > 0
            m = acc.sumwx[r] / acc.sumw[r]
            v = acc.sumwx2[r] / acc.sumw[r] - m * m
            mean[r] = m; std[r] = sqrt(max(v, 0.0))
        else
            mean[r] = NaN; std[r] = NaN
        end
    end
    return (mean=mean, std=std, weight=copy(acc.sumw))
end

# ─── Histogram accumulator ───────────────────────────────────────────────────
"""
    HistogramAccumulator(edges)

Streaming (optionally weighted) histogram over fixed bin `edges` — e.g. for Okubo–Weiss
distributions. Feed values with `update!`, read with `results`.
"""
struct HistogramAccumulator
    edges::Vector{Float64}
    counts::Vector{Float64}
end
HistogramAccumulator(edges::AbstractVector) =
    HistogramAccumulator(collect(float.(edges)), zeros(length(edges) - 1))

"""
    update!(h::HistogramAccumulator, values; weights=nothing)

Bin `values` (any array) into `h`. `weights` optionally weights each value (e.g. area).
Non-finite values and values outside `[edges[1], edges[end])` are skipped.
"""
function update!(h::HistogramAccumulator, values::AbstractArray; weights=nothing)
    nb = length(h.edges) - 1
    @inbounds for idx in eachindex(values)
        v = float(values[idx])
        isfinite(v) || continue
        b = searchsortedlast(h.edges, v)
        (b < 1 || b > nb) && continue
        h.counts[b] += weights === nothing ? 1.0 : float(weights[idx])
    end
    return h
end

results(h::HistogramAccumulator) =
    (edges=copy(h.edges), centers=0.5 .* (h.edges[1:end-1] .+ h.edges[2:end]), counts=copy(h.counts))

# ─── Running (time) mean via Welford ─────────────────────────────────────────
"""
    RunningMean(ny, nx)

Streaming per-cell mean and variance over slices (Welford's algorithm). Feed slices
with `update!`, read with `results`.
"""
mutable struct RunningMean
    n::Int
    mean::Matrix{Float64}
    M2::Matrix{Float64}
end
RunningMean(ny::Integer, nx::Integer) = RunningMean(0, zeros(ny, nx), zeros(ny, nx))

"""
    update!(rm::RunningMean, slice)

Add one 2D slice to the running per-cell mean/variance.
"""
function update!(rm::RunningMean, slice::AbstractMatrix)
    size(slice) == size(rm.mean) || error("slice size must match RunningMean")
    rm.n += 1
    n = rm.n
    @inbounds for I in eachindex(rm.mean)
        x = float(slice[I])
        δ = x - rm.mean[I]
        rm.mean[I] += δ / n
        rm.M2[I] += δ * (x - rm.mean[I])
    end
    return rm
end

"""
    results(rm::RunningMean) -> (mean, var, n)

Per-cell mean, sample variance (divided by `n-1`), and slice count.
"""
results(rm::RunningMean) =
    (mean=copy(rm.mean), var = rm.n > 1 ? rm.M2 ./ (rm.n - 1) : zero(rm.M2), n=rm.n)
