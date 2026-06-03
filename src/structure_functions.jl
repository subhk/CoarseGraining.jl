using ..CoarseGraining: Field, Grid, SphericalGrid
using LinearAlgebra

export structure_function, velocity_structure_function, spectrum_from_sf2, sf2_from_spectrum,
       advective_structure_function, spectral_flux,
       structure_function_sphere, velocity_structure_function_sphere

# Bessel J0 via Abramowitz & Stegun rational approximations (≈1e-7), so we do not
# pull in SpecialFunctions just for the Hankel/Bessel spectral transform.
@inline function _besselj0(x::Real)
    ax = abs(x)
    if ax < 3.0
        y = (x/3)^2
        return 1 + y*(-2.2499997 + y*(1.2656208 + y*(-0.3163866 +
               y*(0.0444479 + y*(-0.0039444 + y*0.0002100)))))
    else
        z = 3.0/ax
        f0 = 0.79788456 + z*(-0.00000077 + z*(-0.00552740 + z*(-0.00009512 +
              z*(0.00137237 + z*(-0.00072805 + z*0.00014476)))))
        θ0 = ax - 0.78539816 + z*(-0.04166397 + z*(-0.00003954 + z*(0.00262573 +
              z*(-0.00054125 + z*(-0.00029333 + z*0.00013558)))))
        return f0/sqrt(ax) * cos(θ0)
    end
end

# Bessel J1 (A&S); J2/J3 by upward recurrence J_{n+1} = (2n/x)J_n − J_{n-1}.
@inline function _besselj1(x::Real)
    s = sign(x); ax = abs(x)
    if ax < 3.0
        y = (ax/3)^2
        r = ax*(0.5 + y*(-0.56249985 + y*(0.21093573 + y*(-0.03954289 +
            y*(0.00443319 + y*(-0.00031761 + y*0.00001109))))))
        return s*r
    else
        z = 3.0/ax
        f1 = 0.79788456 + z*(0.00000156 + z*(0.01659667 + z*(0.00017105 +
              z*(-0.00249511 + z*(0.00113653 + z*(-0.00020033))))))
        θ1 = ax - 2.35619449 + z*(0.12499612 + z*(0.00005650 + z*(-0.00637879 +
              z*(0.00074348 + z*(0.00079824 + z*(-0.00029166))))))
        return s*f1/sqrt(ax)*cos(θ1)
    end
end
@inline _besselj2(x::Real) = abs(x) < 1e-4 ? x*x/8 : 2.0/x*_besselj1(x) - _besselj0(x)
@inline _besselj3(x::Real) = abs(x) < 1e-4 ? x*x*x/48 : 4.0/x*_besselj2(x) - _besselj1(x)

@inline function _shift_idx(i::Int, s::Int, n::Int, periodic::Bool)
    j = i + s
    if 1 <= j <= n
        return j, true
    elseif periodic
        return mod(j-1, n)+1, true
    else
        return 0, false
    end
end

"""
    structure_function(field, seps; order=2) -> (r, Sp)

Isotropic order-`p` structure function `Sp(r) = ⟨|f(x+r) − f(x)|^p⟩`, averaged over
positions and the x/y directions, for a scalar `field` on a Cartesian `Grid`.
`seps` are separations in grid cells; `r` is returned in physical units `s·(dx+dy)/2`
(assumes `dx ≈ dy`). Periodic axes wrap; non-periodic axes drop out-of-range pairs.
"""
function structure_function(field::Field{T,G}, seps::AbstractVector{<:Integer}; order::Integer=2) where {T,G}
    @assert field.grid isa Grid "structure_function supports Cartesian Grid"
    A = field.data
    ny, nx = size(A)
    dx, dy = field.grid.dx, field.grid.dy
    px, py = field.grid.periodic_x, field.grid.periodic_y
    d = 0.5*(dx + dy)
    ns = length(seps)
    Sp = zeros(Float64, ns); r = zeros(Float64, ns)
    for (k, s) in enumerate(seps)
        if s == 0
            Sp[k] = 0.0; r[k] = 0.0; continue
        end
        acc = 0.0; cnt = 0
        @inbounds for j in 1:ny, i in 1:nx
            iiw, okx = _shift_idx(i, s, nx, px)
            if okx
                δ = float(A[j, iiw]) - float(A[j, i]); acc += abs(δ)^order; cnt += 1
            end
            jjw, oky = _shift_idx(j, s, ny, py)
            if oky
                δ = float(A[jjw, i]) - float(A[j, i]); acc += abs(δ)^order; cnt += 1
            end
        end
        Sp[k] = cnt > 0 ? acc/cnt : NaN
        r[k] = s * d
    end
    return r, Sp
end

"""
    velocity_structure_function(u, v, seps; order=2) -> (r, SF_long, SF_trans)

Longitudinal and transverse velocity structure functions on a Cartesian `Grid`.
For each separation, the longitudinal increment is the velocity component *along* the
separation and the transverse one is *perpendicular*; both are averaged as signed
`⟨δ^order⟩` (so odd orders retain sign, as used for energy-flux estimates).
"""
function velocity_structure_function(u::Field{T,G}, v::Field{T,G}, seps::AbstractVector{<:Integer};
                                     order::Integer=2) where {T,G}
    @assert u.grid isa Grid && v.grid isa Grid "velocity_structure_function supports Cartesian Grid"
    U = u.data; V = v.data
    ny, nx = size(U)
    dx, dy = u.grid.dx, u.grid.dy
    px, py = u.grid.periodic_x, u.grid.periodic_y
    d = 0.5*(dx + dy)
    ns = length(seps)
    SL = zeros(Float64, ns); ST = zeros(Float64, ns); r = zeros(Float64, ns)
    for (k, s) in enumerate(seps)
        if s == 0
            r[k] = 0.0; continue
        end
        accL = 0.0; accT = 0.0; cnt = 0
        @inbounds for j in 1:ny, i in 1:nx
            iiw, okx = _shift_idx(i, s, nx, px)   # separation along x: long = u, trans = v
            if okx
                dL = float(U[j, iiw]) - float(U[j, i]); dT = float(V[j, iiw]) - float(V[j, i])
                accL += dL^order; accT += dT^order; cnt += 1
            end
            jjw, oky = _shift_idx(j, s, ny, py)   # separation along y: long = v, trans = u
            if oky
                dL = float(V[jjw, i]) - float(V[j, i]); dT = float(U[jjw, i]) - float(U[j, i])
                accL += dL^order; accT += dT^order; cnt += 1
            end
        end
        SL[k] = cnt > 0 ? accL/cnt : NaN
        ST[k] = cnt > 0 ? accT/cnt : NaN
        r[k] = s * d
    end
    return r, SL, ST
end

# Trapezoidal cell widths for an (irregular) wavenumber grid.
function _trap_widths(k::AbstractVector)
    nk = length(k)
    Δ = Vector{Float64}(undef, nk)
    @inbounds for j in 1:nk
        Δ[j] = j == 1 ? (k[2]-k[1]) : (j == nk ? (k[nk]-k[nk-1]) : (k[j+1]-k[j-1])/2)
    end
    return Δ
end

"""
    sf2_from_spectrum(k, E, r) -> S2

Forward isotropic relation `S2(r) = 4 ∫ E(k)[1 − J₀(kr)] dk` (trapezoidal in `k`).
Useful as the model that [`spectrum_from_sf2`](@ref) inverts.
"""
function sf2_from_spectrum(k::AbstractVector, E::AbstractVector, r::AbstractVector)
    Δk = _trap_widths(k)
    return [4*sum(E[j]*(1 - _besselj0(k[j]*ri))*Δk[j] for j in eachindex(k)) for ri in r]
end

"""
    spectrum_from_sf2(r, S2, k) -> E

Estimate the isotropic energy spectrum `E(k)` from a 2nd-order structure function
`S2(r)` by least-squares inversion of `S2(r) = 4 Σ E(k)[1 − J₀(kr)] Δk` (Bessel/Hankel
relation). Use `length(r) ≥ length(k)` for a well-posed solve. This is an estimate —
the inverse problem is mildly ill-conditioned at the largest scales.
"""
function spectrum_from_sf2(r::AbstractVector, S2::AbstractVector, k::AbstractVector)
    Δk = _trap_widths(k)
    nr = length(r); nk = length(k)
    M = Matrix{Float64}(undef, nr, nk)
    @inbounds for i in 1:nr, j in 1:nk
        M[i, j] = 4*(1 - _besselj0(k[j]*r[i]))*Δk[j]
    end
    return M \ collect(float.(S2))
end

"""
    advective_structure_function(u, v, seps) -> (r, V, SL)

Third-order / "advective" structure functions on a Cartesian `Grid` (Xie & Bühler
2018): `V(r) = ⟨|δu|² δu_L⟩` and the longitudinal `SL(r) = ⟨δu_L³⟩`, averaged over
positions and the x/y directions. These feed [`spectral_flux`](@ref).
"""
function advective_structure_function(u::Field{T,G}, v::Field{T,G},
                                      seps::AbstractVector{<:Integer}) where {T,G}
    @assert u.grid isa Grid && v.grid isa Grid "advective_structure_function supports Cartesian Grid"
    U = u.data; Vd = v.data
    ny, nx = size(U)
    dx, dy = u.grid.dx, u.grid.dy
    px, py = u.grid.periodic_x, u.grid.periodic_y
    d = 0.5*(dx + dy)
    ns = length(seps)
    V = zeros(Float64, ns); SL = zeros(Float64, ns); r = zeros(Float64, ns)
    for (k, s) in enumerate(seps)
        if s == 0
            r[k] = 0.0; continue
        end
        accV = 0.0; accL = 0.0; cnt = 0
        @inbounds for j in 1:ny, i in 1:nx
            iiw, okx = _shift_idx(i, s, nx, px)        # x-sep: long = u, trans = v
            if okx
                dL = float(U[j, iiw]) - float(U[j, i]); dT = float(Vd[j, iiw]) - float(Vd[j, i])
                accV += (dL*dL + dT*dT)*dL; accL += dL^3; cnt += 1
            end
            jjw, oky = _shift_idx(j, s, ny, py)        # y-sep: long = v, trans = u
            if oky
                dL = float(Vd[jjw, i]) - float(Vd[j, i]); dT = float(U[jjw, i]) - float(U[j, i])
                accV += (dL*dL + dT*dT)*dL; accL += dL^3; cnt += 1
            end
        end
        V[k] = cnt > 0 ? accV/cnt : NaN
        SL[k] = cnt > 0 ? accL/cnt : NaN
        r[k] = s * d
    end
    return r, V, SL
end

"""
    spectral_flux(r, sf, K; kind=:V) -> Π

Spectral kinetic-energy flux through wavenumber `K` from a third-order structure
function (Xie & Bühler 2018, eq. 5.8), via a Bessel transform (trapezoidal in `r`):

- `kind=:V`  uses `sf = V(r)`:  `Π(K) = −(K²/4) ∫ V(r) J₂(Kr) dr`
- `kind=:SL` uses `sf = SL(r)`: `Π(K) = −(K³/12) ∫ SL(r) J₃(Kr) r dr`

`K` may be a scalar or a vector (returns a scalar or vector to match). The two exact
forms agree in the continuum. Sign convention: `Π < 0` is an upscale (inverse) energy
cascade. The relation is exact for stationary isotropic 2D turbulence; with measured,
truncated `sf(r)` it is an estimate (the `V = 2εr` part is only conditionally convergent).
"""
function spectral_flux(r::AbstractVector, sf::AbstractVector, K; kind::Symbol=:V)
    kind in (:V, :SL) || error("kind must be :V or :SL")
    Δr = _trap_widths(r)
    scalar = K isa Number
    Kv = scalar ? [float(K)] : collect(float.(K))
    Π = similar(Kv)
    for (m, Km) in enumerate(Kv)
        if kind === :V
            Π[m] = -(Km^2/4) * sum(sf[j]*_besselj2(Km*r[j])*Δr[j] for j in eachindex(r))
        else
            Π[m] = -(Km^3/12) * sum(sf[j]*_besselj3(Km*r[j])*r[j]*Δr[j] for j in eachindex(r))
        end
    end
    return scalar ? Π[1] : Π
end

# Latitude cell widths Δφ_cell (non-uniform aware), shared by the spherical SFs.
function _lat_cell_widths(ϕ::AbstractVector)
    ny = length(ϕ)
    Δϕ = Vector{Float64}(undef, ny)
    @inbounds for j in 1:ny
        Δϕ[j] = j == 1 ? (ϕ[2]-ϕ[1]) : (j == ny ? (ϕ[ny]-ϕ[ny-1]) : (ϕ[j+1]-ϕ[j-1])/2)
    end
    return Δϕ
end

"""
    structure_function_sphere(field, bins; order=2) -> (r, Sp)

Great-circle, distance-binned structure function on a `SphericalGrid`:
`Sp(r) = ⟨|f(P₂) − f(P₁)|^order⟩` where pairs are binned by great-circle distance and
**area-weighted** (∝ cosφ·Δφ). `bins` are distance bin EDGES in the units of `grid.a`
(metres); `r` returns the bin centres. Neighbour search is bounded by `max(bins)`.
"""
function structure_function_sphere(field::Field{T,G}, bins::AbstractVector;
                                   order::Integer=2) where {T,G<:SphericalGrid}
    grid = field.grid
    A = field.data
    ny = length(grid.lat); nx = length(grid.lon)
    a = grid.a; ϕ = grid.lat; λ = grid.lon
    cosϕ = cos.(ϕ); Δϕ = _lat_cell_widths(ϕ)
    dλ = nx > 1 ? (λ[end]-λ[1])/(nx-1) : 1.0
    edges = collect(float.(bins)); nb = length(edges) - 1
    rmax = edges[end]; αmax = rmax / a
    accS = zeros(Float64, nb); accW = zeros(Float64, nb)
    periodic = grid.periodic_lon
    @inbounds for j in 1:ny
        ϕj = ϕ[j]
        jmin = j; while jmin > 1 && (ϕj - ϕ[jmin-1]) <= αmax; jmin -= 1; end
        jmax = j; while jmax < ny && (ϕ[jmax+1] - ϕj) <= αmax; jmax += 1; end
        iext = min(nx ÷ 2, ceil(Int, αmax/(max(cosϕ[j], 1e-6)*dλ)) + 1)
        for i in 1:nx
            λi = λ[i]; fi = float(A[j, i])
            for jj in jmin:jmax
                area = cosϕ[jj] * Δϕ[jj]
                for dii in -iext:iext
                    ii0 = i + dii
                    if 1 <= ii0 <= nx
                        ii = ii0
                    elseif periodic
                        ii = mod(ii0 - 1, nx) + 1
                    else
                        continue
                    end
                    (jj == j && ii == i) && continue
                    dist = grid_distance(grid, ϕj, λi, ϕ[jj], λ[ii])
                    dist > rmax && continue
                    b = searchsortedlast(edges, dist)
                    (b < 1 || b > nb) && continue
                    δ = float(A[jj, ii]) - fi
                    accS[b] += area * abs(δ)^order
                    accW[b] += area
                end
            end
        end
    end
    r = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    Sp = [accW[b] > 0 ? accS[b]/accW[b] : NaN for b in 1:nb]
    return r, Sp
end

"""
    velocity_structure_function_sphere(uE, vN, bins; order=2) -> (r, SF_long, SF_trans)

Great-circle velocity structure functions on a `SphericalGrid`. The increment
`(δuE, δvN)` is projected onto the initial great-circle bearing `θ` from P₁ to P₂:
longitudinal `δu_L = δuE·sinθ + δvN·cosθ`, transverse `δu_T = −δuE·cosθ + δvN·sinθ`,
binned by great-circle distance and area-weighted. Both are averaged as signed
`⟨δ^order⟩`. (Local-frame increment — the east/north basis rotates between P₁ and P₂.)
"""
function velocity_structure_function_sphere(uE::Field{T,G}, vN::Field{T,G}, bins::AbstractVector;
                                            order::Integer=2) where {T,G<:SphericalGrid}
    grid = uE.grid
    @assert grid === vN.grid
    U = uE.data; V = vN.data
    ny = length(grid.lat); nx = length(grid.lon)
    a = grid.a; ϕ = grid.lat; λ = grid.lon
    cosϕ = cos.(ϕ); Δϕ = _lat_cell_widths(ϕ)
    dλ = nx > 1 ? (λ[end]-λ[1])/(nx-1) : 1.0
    edges = collect(float.(bins)); nb = length(edges) - 1
    rmax = edges[end]; αmax = rmax / a
    accL = zeros(Float64, nb); accT = zeros(Float64, nb); accW = zeros(Float64, nb)
    periodic = grid.periodic_lon
    @inbounds for j in 1:ny
        ϕj = ϕ[j]
        jmin = j; while jmin > 1 && (ϕj - ϕ[jmin-1]) <= αmax; jmin -= 1; end
        jmax = j; while jmax < ny && (ϕ[jmax+1] - ϕj) <= αmax; jmax += 1; end
        iext = min(nx ÷ 2, ceil(Int, αmax/(max(cosϕ[j], 1e-6)*dλ)) + 1)
        for i in 1:nx
            λi = λ[i]; uEi = float(U[j, i]); vNi = float(V[j, i])
            for jj in jmin:jmax
                area = cosϕ[jj] * Δϕ[jj]; ϕ2 = ϕ[jj]
                for dii in -iext:iext
                    ii0 = i + dii
                    if 1 <= ii0 <= nx
                        ii = ii0
                    elseif periodic
                        ii = mod(ii0 - 1, nx) + 1
                    else
                        continue
                    end
                    (jj == j && ii == i) && continue
                    Δλ = λ[ii] - λi
                    dist = grid_distance(grid, ϕj, λi, ϕ2, λ[ii])
                    dist > rmax && continue
                    b = searchsortedlast(edges, dist)
                    (b < 1 || b > nb) && continue
                    # initial bearing P1 → P2 (clockwise from north)
                    θ = atan(sin(Δλ)*cos(ϕ2), cos(ϕj)*sin(ϕ2) - sin(ϕj)*cos(ϕ2)*cos(Δλ))
                    sθ, cθ = sin(θ), cos(θ)
                    δE = float(U[jj, ii]) - uEi; δN = float(V[jj, ii]) - vNi
                    dL =  δE*sθ + δN*cθ
                    dT = -δE*cθ + δN*sθ
                    accL[b] += area * dL^order
                    accT[b] += area * dT^order
                    accW[b] += area
                end
            end
        end
    end
    r = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    SL = [accW[b] > 0 ? accL[b]/accW[b] : NaN for b in 1:nb]
    ST = [accW[b] > 0 ? accT[b]/accW[b] : NaN for b in 1:nb]
    return r, SL, ST
end
