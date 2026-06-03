using ..CoarseGraining: Field, SphericalGrid
using FFTW
using LinearAlgebra

export gradient_sphere, vorticity_sphere, divergence_sphere, vel_spher_to_cart, vel_cart_to_spher,
       helmholtz_hodge_sphere, poisson_sphere_solve

# ─── Non-uniform finite-difference helpers ──────────────────────────────────
# Latitude (ϕ) may be non-uniformly spaced (stretched grids); longitude (λ) is
# assumed uniform — that matches lat-lon grids and is required by the FFT solver.

# 2nd-order central first-derivative weights for a node with neighbour spacings
# hL = x[j]-x[j-1], hR = x[j+1]-x[j]:  f'(x[j]) ≈ wL f[j-1] + w0 f[j] + wR f[j+1].
# Reduces to (−1, 0, 1)/(2h) when hL = hR = h.
@inline function _nu_weights(hL::Real, hR::Real)
    s = hL + hR
    return (-hR / (hL * s), (hR - hL) / (hL * hR), hL / (hR * s))
end

# ∂B/∂ϕ along latitude (dim 1, non-periodic, possibly non-uniform), one-sided at
# the latitude boundaries. Returns Float64 matrix.
function _ddϕ(B::AbstractMatrix, ϕ::AbstractVector)
    ny, nx = size(B)
    out = Matrix{Float64}(undef, ny, nx)
    @inbounds for j in 1:ny
        if j == 1
            h = ϕ[2] - ϕ[1]
            for i in 1:nx
                out[j, i] = (B[2, i] - B[1, i]) / h
            end
        elseif j == ny
            h = ϕ[ny] - ϕ[ny-1]
            for i in 1:nx
                out[j, i] = (B[ny, i] - B[ny-1, i]) / h
            end
        else
            wL, w0, wR = _nu_weights(ϕ[j] - ϕ[j-1], ϕ[j+1] - ϕ[j])
            for i in 1:nx
                out[j, i] = wL * B[j-1, i] + w0 * B[j, i] + wR * B[j+1, i]
            end
        end
    end
    return out
end

function gradient_sphere(f::Field{T,G}) where {T<:Real,G}
    grid = f.grid
    @assert grid isa SphericalGrid "gradient_sphere requires SphericalGrid"
    ny = length(grid.lat)
    nx = length(grid.lon)
    a = grid.a
    ϕ = grid.lat
    λ = grid.lon
    A = f.data
    dλ = mean(diff(λ))           # longitude assumed uniform
    cosϕ = cos.(ϕ)
    # ∂f/∂λ (periodic, uniform) and ∂f/∂ϕ (latitude, non-uniform)
    ∂λ = similar(A)
    @inbounds for i in 1:nx
        il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
        ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
        for j in 1:ny
            ∂λ[j,i] = (A[j,ir] - A[j,il]) / (2dλ)
        end
    end
    ∂ϕ = _ddϕ(A, ϕ)
    # Physical gradients (east/north components)
    gx = similar(A)
    gy = similar(A)
    inva = 1.0 / a
    @inbounds for j in 1:ny
        invc = 1.0 / (a * cosϕ[j])
        @views gx[j, :] .= ∂λ[j, :] .* invc
        @views gy[j, :] .= ∂ϕ[j, :] .* inva
    end
    return Field(gx, grid), Field(gy, grid)
end

"""
    vorticity_sphere(uE, vN, grid)

Relative vorticity on sphere (vertical component), with `uE` eastward and `vN` northward.
"""
function vorticity_sphere(uE::Field{T,G}, vN::Field{T,G}) where {T<:Real,G}
    grid = uE.grid
    @assert grid === vN.grid
    @assert grid isa SphericalGrid "vorticity_sphere requires SphericalGrid"
    ny = length(grid.lat)
    nx = length(grid.lon)
    a = grid.a
    ϕ = grid.lat
    λ = grid.lon
    dλ = mean(diff(λ))           # longitude assumed uniform
    cosϕ = cos.(ϕ)
    u = uE.data
    v = vN.data
    ∂v∂λ = similar(u)
    @inbounds for i in 1:nx
        il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
        ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
        for j in 1:ny
            ∂v∂λ[j,i] = (v[j,ir] - v[j,il])/(2dλ)
        end
    end
    ∂ucos∂ϕ = _ddϕ(u .* cosϕ, ϕ)   # ∂(uE cosϕ)/∂ϕ, latitude non-uniform
    ζ = similar(u)
    @inbounds for j in 1:ny
        invc = 1.0 / (a * cosϕ[j])
        @views ζ[j, :] .= (∂v∂λ[j, :] .- ∂ucos∂ϕ[j, :]) .* invc
    end
    return Field(ζ, grid)
end

"""
    divergence_sphere(uE, vN)

Horizontal divergence on sphere for eastward (`uE`) and northward (`vN`).
div = 1/(a cosϕ)[∂uE/∂λ + ∂(vN cosϕ)/∂ϕ].
"""
function divergence_sphere(uE::Field{T,G}, vN::Field{T,G}) where {T<:Real,G}
    grid = uE.grid
    @assert grid === vN.grid
    @assert grid isa SphericalGrid "divergence_sphere requires SphericalGrid"
    ny = length(grid.lat)
    nx = length(grid.lon)
    a = grid.a
    ϕ = grid.lat
    λ = grid.lon
    dλ = mean(diff(λ))           # longitude assumed uniform
    cosϕ = cos.(ϕ)
    u = uE.data
    v = vN.data
    ∂u∂λ = similar(u)
    @inbounds for i in 1:nx
        il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
        ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
        for j in 1:ny
            ∂u∂λ[j,i] = (u[j,ir] - u[j,il])/(2dλ)
        end
    end
    ∂vcos∂ϕ = _ddϕ(v .* cosϕ, ϕ)   # ∂(vN cosϕ)/∂ϕ, latitude non-uniform
    div = similar(u)
    @inbounds for j in 1:ny
        invc = 1.0 / (a * cosϕ[j])
        @views div[j, :] .= (∂u∂λ[j, :] .+ ∂vcos∂ϕ[j, :]) .* invc
    end
    return Field(div, grid)
end

"""
    helmholtz_hodge_sphere(uE, vN) -> (uE_df, vN_df, uE_pot, vN_pot, ϕ, ψ)

Baseline spherical Helmholtz–Hodge using FFT in longitude and finite differences in latitude.
Assumes periodic in longitude and avoids exact poles in latitude.
"""
function helmholtz_hodge_sphere(uE::Field{T,G}, vN::Field{T,G}) where {T<:Real,G}
    grid = uE.grid
    @assert grid === vN.grid
    @assert grid isa SphericalGrid "helmholtz_hodge_sphere requires SphericalGrid"
    # Compute divergence and vorticity
    div = divergence_sphere(uE, vN)
    vort = vorticity_sphere(uE, vN)
    # Solve Poisson on sphere
    ϕ = poisson_sphere_solve(div)
    ψ = poisson_sphere_solve(vort)
    # Potential (irrotational) part = ∇ϕ.
    uE_pot, vN_pot = gradient_sphere(ϕ)
    # Divergence-free part as the residual u − ∇ϕ. This reconstructs exactly
    # (u = u_pot + u_df) on bounded/collocated grids, where a separately solved
    # rot(ψ) leaves a harmonic remainder. ψ is still returned as the streamfunction.
    uE_df = Field(uE.data .- uE_pot.data, grid)
    vN_df = Field(vN.data .- vN_pot.data, grid)
    return uE_df, vN_df, uE_pot, vN_pot, ϕ, ψ
end

function rot_sphere(ψ::Field{T,G}) where {T<:Real,G}
    grid = ψ.grid
    @assert grid isa SphericalGrid
    ny = length(grid.lat)
    nx = length(grid.lon)
    a = grid.a
    ϕ = grid.lat
    λ = grid.lon
    dλ = mean(diff(λ))           # longitude assumed uniform
    A = ψ.data
    # ∂ψ/∂λ (periodic, uniform) and ∂ψ/∂ϕ (latitude, non-uniform)
    ∂λ = similar(A)
    @inbounds for i in 1:nx
        il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
        ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
        for j in 1:ny
            ∂λ[j,i] = (A[j,ir] - A[j,il])/(2dλ)
        end
    end
    ∂ϕ = _ddϕ(A, ϕ)
    cosϕ = cos.(ϕ)
    uE = similar(A)
    vN = similar(A)
    inva = 1.0 / a
    @inbounds for j in 1:ny
        # Divergence-free field from streamfunction on a sphere:
        # uE =  (1/(a cosϕ)) ∂ψ/∂ϕ,   vN = -(1/a) ∂ψ/∂λ
        invc = 1.0 / (a * cosϕ[j])
        @views uE[j, :] .=  ∂ϕ[j, :] .* invc
        @views vN[j, :] .= ∂λ[j, :] .* (-inva)
    end
    return Field(uE, grid), Field(vN, grid)
end

function poisson_sphere_solve(rhs::Field{T,G}) where {T<:Real,G}
    grid = rhs.grid
    @assert grid isa SphericalGrid "poisson_sphere_solve requires SphericalGrid"
    ny = length(grid.lat)
    nx = length(grid.lon)
    a = grid.a
    ϕ = grid.lat
    λ = grid.lon
    cosϕ = cos.(ϕ)
    # Fourier in longitude (requires uniform longitude)
    R̂ = fft(Float64.(rhs.data), 2)
    Φ̂ = similar(R̂)
    # Map index to zonal wavenumber m
    idx_to_m(k, nx) = k <= nx ÷ 2 ? k-1 : k-1 - nx
    # Meridional flux operator (1/(a²cosϕ)) ∂ϕ(cosϕ ∂ϕ·), non-uniform in latitude.
    # Off-diagonal bands and the m-independent diagonal depend only on latitude, so
    # build them once instead of rebuilding the tridiagonal per column.
    dl0 = zeros(Float64, ny-1)
    du0 = zeros(Float64, ny-1)
    d0  = zeros(Float64, ny)     # m-independent diagonal part
    gdiag = zeros(Float64, ny)   # per-row factor multiplying m² on the diagonal
    for j in 1:ny
        Δp = j < ny ? (ϕ[j+1] - ϕ[j]) : (ϕ[j] - ϕ[j-1])
        Δm = j > 1  ? (ϕ[j] - ϕ[j-1]) : (ϕ[j+1] - ϕ[j])
        Δc = j == 1 ? Δp : (j == ny ? Δm : (Δp + Δm) / 2)   # cell width
        cjp = j < ny ? cos((ϕ[j]+ϕ[j+1])/2) : 0.0
        cjm = j > 1  ? cos((ϕ[j]+ϕ[j-1])/2) : 0.0
        invc = 1.0 / (a*a*cosϕ[j]*Δc)
        coefp = cjp * invc / Δp
        coefm = cjm * invc / Δm
        if j > 1
            dl0[j-1] = coefm
        end
        if j < ny
            du0[j]   = coefp
        end
        d0[j]    = -(coefp + coefm)
        gdiag[j] = 1.0 / (a*a*cosϕ[j]*cosϕ[j])
    end
    # Mild regularization to avoid nullspace at m=0 (Neumann-like at boundaries)
    d0[1]   -= 1e-12
    d0[end] -= 1e-12
    for k in 1:nx
        m = idx_to_m(k, nx)
        m2 = m*m
        d = d0 .- m2 .* gdiag
        A = Tridiagonal(dl0, d, du0)
        b = @view R̂[:, k]
        Φ̂[:, k] = A \ b
    end
    ϕfield = real(ifft(Φ̂, 2))
    return Field(ϕfield, grid)
end

"""
    vel_spher_to_cart(uE, vN, wR, lon, lat) -> (ux, uy, uz)

Convert east/north/radial components at a point to Cartesian in ECEF frame.
"""
function vel_spher_to_cart(uE::Real, vN::Real, wR::Real, lon::Real, lat::Real)
    # local basis: eE = (-sin λ, cos λ, 0), eN = (-sin ϕ cos λ, -sin ϕ sin λ, cos ϕ), eR = (cos ϕ cos λ, cos ϕ sin λ, sin ϕ)
    sλ, cλ = sin(lon), cos(lon)
    sϕ, cϕ = sin(lat), cos(lat)
    eE = (-sλ,  cλ,  0.0)
    eN = (-sϕ*cλ, -sϕ*sλ, cϕ)
    eR = ( cϕ*cλ,  cϕ*sλ, sϕ)
    ux = uE*eE[1] + vN*eN[1] + wR*eR[1]
    uy = uE*eE[2] + vN*eN[2] + wR*eR[2]
    uz = uE*eE[3] + vN*eN[3] + wR*eR[3]
    return (ux, uy, uz)
end

"""
    vel_cart_to_spher(ux, uy, uz, lon, lat) -> (uE, vN, wR)

Convert Cartesian velocity to east/north/radial at a point.
"""
function vel_cart_to_spher(ux::Real, uy::Real, uz::Real, lon::Real, lat::Real)
    sλ, cλ = sin(lon), cos(lon)
    sϕ, cϕ = sin(lat), cos(lat)
    eE = (-sλ,  cλ,  0.0)
    eN = (-sϕ*cλ, -sϕ*sλ, cϕ)
    eR = ( cϕ*cλ,  cϕ*sλ, sϕ)
    uE = ux*eE[1] + uy*eE[2] + uz*eE[3]
    vN = ux*eN[1] + uy*eN[2] + uz*eN[3]
    wR = ux*eR[1] + uy*eR[2] + uz*eR[3]
    return (uE, vN, wR)
end
