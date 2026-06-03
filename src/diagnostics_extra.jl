using ..CoarseGraining: Field, Grid
using FFTW
using Statistics

export ke_spectrum_isotropic, zonal_mean_std, region_mean_std, write_region_stats_and_masks

"""
    ke_spectrum_isotropic(u, v; nbins=Int(floor(min(nx,ny)/2)))

Compute isotropic KE spectrum E(k) for periodic Cartesian grid via 2D FFT and radial binning.
Returns (k_centers, E_k).
"""
function ke_spectrum_isotropic(u::Field{T,G}, v::Field{T,G}; nbins::Union{Int,Nothing}=nothing, normalize::Symbol=:counts, return_edges::Bool=false, bins::Symbol=:linear) where {T<:Real,G}
    @assert u.grid isa Grid "ke_spectrum_isotropic supports Cartesian Grid"
    @assert u.grid === v.grid
    ny, nx = size(u.data)
    nb = isnothing(nbins) ? Int(floor(min(nx, ny)/2)) : nbins
    dx, dy = u.grid.dx, u.grid.dy
    Û = fft(Float64.(u.data))
    V̂ = fft(Float64.(v.data))
    KÊ = 0.5 .* (abs2.(Û) .+ abs2.(V̂)) ./ (nx*ny)^2
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*dy))
    KX = reshape(kx, 1, :)
    KY = reshape(ky, :, 1)
    kr = sqrt.(KX.^2 .+ KY.^2)
    kmax = maximum(kr)
    if bins == :linear
        edges = range(0, stop=kmax, length=nb+1)
    elseif bins == :log
        # exclude zero; pick smallest positive kr as kmin
        kmin = minimum(kr[kr .> 0])
        edges = exp.(range(log(kmin), log(kmax + eps()), length=nb+1))
    else
        error("Unknown bins=:$(bins). Use :linear or :log")
    end
    Ek = zeros(Float64, nb)
    counts = zeros(Int, nb)
    for j in 1:ny, i in 1:nx
        k = kr[j,i]
        if bins == :log && k <= 0
            continue
        end
        b = clamp(searchsortedlast(edges, k), 1, nb)
        Ek[b] += KÊ[j,i]
        counts[b] += 1
    end
    kc = [(edges[i]+edges[i+1])/2 for i in 1:nb]
    if normalize == :counts
        for i in 1:nb
            if counts[i] > 0
                Ek[i] /= counts[i]
            end
        end
    elseif normalize == :density
        for i in 1:nb
            width = edges[i+1] - edges[i]
            if width > 0
                Ek[i] /= width
            end
        end
    elseif normalize == :shellarea
        for i in 1:nb
            kc_i = kc[i]
            width = edges[i+1] - edges[i]
            area = 2π*kc_i*width
            if area > 0
                Ek[i] /= area
            end
        end
    elseif normalize == :energy
        # Scale so that sum(Ek) ≈ mean KE in real space
        KE_real = 0.5 * mean(u.data.^2 .+ v.data.^2)
        s = sum(Ek)
        if s > 0
            Ek .*= (KE_real / s)
        end
    end
    return return_edges ? (kc, Ek, collect(edges)) : (kc, Ek)
end

"""
    zonal_mean_std(field) -> (mean::Vector, std::Vector)

Compute zonal mean and std per row for Cartesian field.
"""
function zonal_mean_std(f::Field{T,G}) where {T<:Real,G}
    @assert f.grid isa Grid
    ny, nx = size(f.data)
    m = [mean(view(f.data, j, :)) for j in 1:ny]
    s = [std(view(f.data, j, :)) for j in 1:ny]
    return m, s
end

"""
    region_mean_std(field, regions) -> Dict{String,(mean,std)}

Compute mean/std in named boolean masks. `regions` maps name => BitArray mask matching field size.
"""
function region_mean_std(f::Field{T,G}, regions::Dict{String,BitArray{2}}) where {T<:Real,G}
    ny, nx = size(f.data)
    out = Dict{String,Tuple{Float64,Float64}}()
    for (name, mask) in regions
        @assert size(mask) == (ny, nx)
        vals = f.data[mask]
        out[name] = (mean(vals), std(vals))
    end
    return out
end

"""
    write_region_stats_and_masks(field, regions, stats_path; masks_path=nothing, lon=nothing, lat=nothing)

Compute region stats and write them to `stats_path` via `write_region_stats`. If `masks_path`, `lon`, and `lat` are provided, also write a masks file via `write_regions_file`.
Returns a tuple of written paths (stats_path, masks_path_or_nothing).
"""
function write_region_stats_and_masks(f::Field{T,G}, regions::Dict{String,BitArray{2}}, stats_path::AbstractString; masks_path::Union{Nothing,AbstractString}=nothing, lon=nothing, lat=nothing) where {T<:Real,G}
    stats = region_mean_std(f, regions)
    write_region_stats(stats_path, stats)
    if masks_path !== nothing && lon !== nothing && lat !== nothing
        write_regions_file(masks_path, regions, lon, lat)
        return (stats_path, masks_path)
    else
        return (stats_path, nothing)
    end
end
