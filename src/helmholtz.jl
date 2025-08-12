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
    # Wavenumbers
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*dy))
    KX = reshape(kx, 1, :)
    KY = reshape(ky, :, 1)
    k2 = @. (KX^2 + KY^2)
    k2[1,1] = 1.0  # avoid divide by zero at k=0 (mean component)

    Û = fft(Float64.(u.data))
    V̂ = fft(Float64.(v.data))

    # Divergence and vorticity in Fourier space
    div̂ = @. (KX .* Û + KY .* V̂) * im
    vort̂ = @. (KX .* V̂ - KY .* Û) * im

    ϕ̂ = div̂ ./ k2
    ψ̂ = vort̂ ./ k2
    ϕ̂[1,1] = 0.0
    ψ̂[1,1] = 0.0

    # Grad φ in spectral space
    ∂xϕ̂ = @. (im*KX) * ϕ̂
    ∂yϕ̂ = @. (im*KY) * ϕ̂
    # Rot ψ = (∂yψ, -∂xψ)
    ∂yψ̂ = @. (im*KY) * ψ̂
    ∂xψ̂ = @. (im*KX) * ψ̂

    upot = real(ifft(∂xϕ̂))
    vpot = real(ifft(∂yϕ̂))
    udf = real(ifft(∂yψ̂))
    vdf = real(ifft(-∂xψ̂))

    ϕ = real(ifft(ϕ̂))
    ψ = real(ifft(ψ̂))

    return Field(T.(udf), grid), Field(T.(vdf), grid), Field(T.(upot), grid), Field(T.(vpot), grid), Field(T.(ϕ), grid), Field(T.(ψ), grid)
end
