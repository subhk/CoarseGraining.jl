using ..CoarseGraining: Field, Grid
using FFTW
using Statistics

export ke_spectrum_isotropic, zonal_mean_std, region_mean_std

"""
    ke_spectrum_isotropic(u, v; nbins=Int(floor(min(nx,ny)/2)))

Compute isotropic KE spectrum E(k) for periodic Cartesian grid via 2D FFT and radial binning.
Returns (k_centers, E_k).
"""
function ke_spectrum_isotropic(u::Field{T,G}, v::Field{T,G}; nbins::Int=nothing) where {T<:Real,G}
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
    edges = range(0, stop=kmax, length=nb+1)
    Ek = zeros(Float64, nb)
    counts = zeros(Int, nb)
    for j in 1:ny, i in 1:nx
        k = kr[j,i]
        b = clamp(searchsortedlast(edges, k), 1, nb)
        Ek[b] += KÊ[j,i]
        counts[b] += 1
    end
    kc = [(edges[i]+edges[i+1])/2 for i in 1:nb]
    return kc, Ek
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

