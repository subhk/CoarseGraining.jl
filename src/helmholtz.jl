using ..CoarseGraining: Field, Grid
using FFTW

export helmholtz_hodge

"""
    helmholtz_hodge(u::Field{T}, v::Field{T}) -> (udf, vdf, upot, vpot, ϕ, ψ)

2D Helmholtz-Hodge decomposition on a periodic Cartesian grid using FFTs.
Returns divergence-free part (udf, vdf), potential part (upot, vpot), and
scalar potentials ϕ (potential) and ψ (streamfunction).
"""
function helmholtz_hodge(u::Field{T,G}, v::Field{T,G}) where {T<:Real,G}
    grid = u.grid
    @assert grid isa Grid "helmholtz_hodge currently supports Cartesian Grid"
    @assert size(u.data) == size(v.data)
    ny, nx = size(u.data)
    dx, dy = grid.dx, grid.dy
    # Wavenumbers (radians) matching FFT output ordering
    # For even n: [0, 1, ..., n/2-1, -n/2, ..., -1]
    # For odd  n: [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1]
    local function fftfreq_rad(n::Int, d::Float64)
        h = n ÷ 2
        idx = iseven(n) ? vcat(0:h-1, -h:-1) : vcat(0:h, -h:-1)
        return idx .* (2π/(n*d))
    end
    kx = fftfreq_rad(nx, dx)
    ky = fftfreq_rad(ny, dy)
    KX = reshape(kx, 1, :)
    KY = reshape(ky, :, 1)
    k2 = KX.^2 .+ KY.^2
    k2[1,1] = 1.0  # avoid divide by zero at k=0 (mean component)

    Û = fft(Float64.(u.data))
    V̂ = fft(Float64.(v.data))

    # Build spectral projectors for potential and divergence-free parts
    # Longitudinal (potential) projection: F̂_pot = (k k^T / k^2) F̂
    dotkF = KX .* Û .+ KY .* V̂
    Û_pot = (KX ./ k2) .* dotkF
    V̂_pot = (KY ./ k2) .* dotkF
    # Handle zero mode explicitly (already yields zero because dotkF[1,1]=0)
    Û_pot[1,1] = 0.0
    V̂_pot[1,1] = 0.0
    # Divergence-free remainder
    Û_df = Û .- Û_pot
    V̂_df = V̂ .- V̂_pot

    # Inverse FFT to real space components
    upot = real(ifft(Û_pot))
    vpot = real(ifft(V̂_pot))
    udf  = real(ifft(Û_df))
    vdf  = real(ifft(V̂_df))

    # Also compute scalar potentials for users that need them
    # Divergence and vorticity in Fourier space
    div̂ = (im .* KX) .* Û .+ (im .* KY) .* V̂
    vort̂ = (im .* KX) .* V̂ .- (im .* KY) .* Û
    # Solve Poisson ∇²ϕ = ∇·F and ∇²ψ = ζ in spectral space
    ϕ̂ = -div̂ ./ k2; ϕ̂[1,1] = 0.0
    ψ̂ = -vort̂ ./ k2; ψ̂[1,1] = 0.0
    ϕ = real(ifft(ϕ̂))
    ψ = real(ifft(ψ̂))

    return Field(T.(udf), grid), Field(T.(vdf), grid), Field(T.(upot), grid), Field(T.(vpot), grid), Field(T.(ϕ), grid), Field(T.(ψ), grid)
end
