using ..CoarseGraining: Field, SphericalGrid
using FFTW
using LinearAlgebra

export gradient_sphere, vorticity_sphere, divergence_sphere, vel_spher_to_cart, vel_cart_to_spher,
       helmholtz_hodge_sphere, poisson_sphere_solve

function gradient_sphere(f::Field{T,G}) where {T<:Real,G}
    grid = f.grid
    @assert grid isa SphericalGrid "gradient_sphere requires SphericalGrid"
    ny = length(grid.lat)
    nx = length(grid.lon)
    a = grid.a
    ϕ = grid.lat
    λ = grid.lon
    A = f.data
    dλ = mean(diff(λ))
    dϕ = mean(diff(ϕ))
    # metric factors
    cosϕ = cos.(ϕ)
    # ∂f/∂λ and ∂f/∂ϕ with central differences and periodic in λ
    ∂λ = similar(A)
    ∂ϕ = similar(A)
    for j in 1:ny
        for i in 1:nx
            il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
            ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
            jl = j == 1 ? 1 : j-1
            jr = j == ny ? ny : j+1
            ∂λ[j,i] = (A[j,ir] - A[j,il])/(2dλ)
            ∂ϕ[j,i] = (A[jr,i] - A[jl,i])/(2dϕ)
        end
    end
    # Physical gradients (east/north components)
    gx = similar(A)
    gy = similar(A)
    for j in 1:ny
        @inbounds gx[j,:] = ∂λ[j,:] ./ (a*cosϕ[j])
        @inbounds gy[j,:] = ∂ϕ[j,:] ./ a
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
    dλ = mean(diff(λ))
    dϕ = mean(diff(ϕ))
    cosϕ = cos.(ϕ)
    u = uE.data
    v = vN.data
    ∂v∂λ = similar(u)
    ∂(ucos)/∂ϕ = similar(u)
    for j in 1:ny
        for i in 1:nx
            il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
            ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
            jl = j == 1 ? 1 : j-1
            jr = j == ny ? ny : j+1
            ∂v∂λ[j,i] = (v[j,ir] - v[j,il])/(2dλ)
            ucos_l = u[jl,i]*cosϕ[jl]
            ucos_r = u[jr,i]*cosϕ[jr]
            ∂(ucos)/∂ϕ[j,i] = (ucos_r - ucos_l)/(2dϕ)
        end
    end
    ζ = similar(u)
    for j in 1:ny
        @inbounds ζ[j,:] = (∂v∂λ[j,:] .- ∂(ucos)/∂ϕ[j,:]) ./ (a*cosϕ[j])
    end
    return Field(ζ, grid)
end

"""
    divergence_sphere(uE, vN)

Horizontal divergence on sphere for eastward (`uE`) and northward (`vN`).
div = 1/(a cosϕ)[∂(uE cosϕ)/∂λ + ∂vN/∂ϕ].
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
    dλ = mean(diff(λ))
    dϕ = mean(diff(ϕ))
    cosϕ = cos.(ϕ)
    u = uE.data
    v = vN.data
    ∂(u cos)/∂λ = similar(u)
    ∂v/∂ϕ = similar(u)
    for j in 1:ny
        for i in 1:nx
            il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
            ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
            jl = j == 1 ? 1 : j-1
            jr = j == ny ? ny : j+1
            ul = u[j,il]*cosϕ[j]
            ur = u[j,ir]*cosϕ[j]
            ∂(u cos)/∂λ[j,i] = (ur - ul)/(2dλ)
            ∂v/∂ϕ[j,i] = (v[jr,i] - v[jl,i])/(2dϕ)
        end
    end
    div = similar(u)
    for j in 1:ny
        @inbounds div[j,:] = (∂(u cos)/∂λ[j,:] .+ ∂v/∂ϕ[j,:]) ./ (a*cosϕ[j])
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
    # Potential velocity = grad(ϕ); Divergence-free = rot(ψ)
    gϕλ, gϕϕ = gradient_sphere(ϕ)
    uE_pot = gϕλ
    vN_pot = gϕϕ
    uE_df, vN_df = rot_sphere(ψ)
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
    dλ = mean(diff(λ))
    dϕ = mean(diff(ϕ))
    A = ψ.data
    # ∂ψ/∂ϕ and ∂ψ/∂λ
    ∂λ = similar(A)
    ∂ϕ = similar(A)
    for j in 1:ny
        for i in 1:nx
            il = i == 1 ? (grid.periodic_lon ? nx : 1) : i-1
            ir = i == nx ? (grid.periodic_lon ? 1 : nx) : i+1
            jl = j == 1 ? 1 : j-1
            jr = j == ny ? ny : j+1
            ∂λ[j,i] = (A[j,ir] - A[j,il])/(2dλ)
            ∂ϕ[j,i] = (A[jr,i] - A[jl,i])/(2dϕ)
        end
    end
    cosϕ = cos.(ϕ)
    uE = similar(A)
    vN = similar(A)
    for j in 1:ny
        @inbounds uE[j,:] = -∂ϕ[j,:] ./ a
        @inbounds vN[j,:] =  ∂λ[j,:] ./ (a*cosϕ[j])
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
    dϕ = mean(diff(ϕ))
    cosϕ = cos.(ϕ)
    # Fourier in longitude
    R̂ = fft(Float64.(rhs.data), 2)
    Φ̂ = similar(R̂)
    # Map index to zonal wavenumber m
    function idx_to_m(k, nx)
        k <= nx ÷ 2 ? k-1 : k-1 - nx
    end
    for k in 1:nx
        m = idx_to_m(k, nx)
        m2 = (m == 0 ? 0 : m*m)
        # Build tridiagonal A f = b for this m
        dl = zeros(Float64, ny-1)
        d  = zeros(Float64, ny)
        du = zeros(Float64, ny-1)
        for j in 1:ny
            cjp = j < ny ? cos((ϕ[j]+ϕ[j+1])/2) : 0.0
            cjm = j > 1  ? cos((ϕ[j]+ϕ[j-1])/2) : 0.0
            coef = 1.0 / (a*a*cosϕ[j])
            if j > 1
                dl[j-1] =  coef * (cjm/(dϕ*dϕ))
            end
            d[j]    = -coef * ((cjp + cjm)/(dϕ*dϕ)) - (m2)/(a*a*cosϕ[j]*cosϕ[j])
            if j < ny
                du[j]   =  coef * (cjp/(dϕ*dϕ))
            end
        end
        # Mild regularization to avoid nullspace at m=0 (Neumann-like at boundaries)
        d[1]  -= 1e-12
        d[end]-= 1e-12
        A = Tridiagonal(dl, d, du)
        b = R̂[:, k]
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
