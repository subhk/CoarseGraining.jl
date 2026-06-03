using ..CoarseGraining: Field, Grid, SphericalGrid, gradient, gradient_sphere, coarse_grain

export okuboweiss, compute_pi

"""
    okuboweiss(u, v) -> Field

Compute the Okubo–Weiss parameter W = s_n^2 + s_s^2 - ω^2,
where s_n = ∂u/∂x - ∂v/∂y, s_s = ∂v/∂x + ∂u/∂y, ω = ∂v/∂x - ∂u/∂y.
Supports Cartesian `Grid` and `SphericalGrid` (east/north gradients).
"""
function okuboweiss(u::Field{T}, v::Field{T}) where {T<:Real}
    @assert (u.grid isa Grid || u.grid isa SphericalGrid) "okuboweiss supports Cartesian Grid or SphericalGrid"
    if u.grid isa SphericalGrid
        (ux, uy) = gradient_sphere(u)
        (vx, vy) = gradient_sphere(v)
    else
        (ux, uy) = gradient(u)
        (vx, vy) = gradient(v)
    end
    sn = ux.data .- vy.data
    ss = vx.data .+ uy.data
    ω  = vx.data .- uy.data
    W = sn.^2 .+ ss.^2 .- ω.^2
    return Field(W, u.grid)
end

"""
    compute_pi(u, v, kernel) -> Field

Leonard energy transfer Π = -τ_ij S_ij for 2D resolved fields.
τ_ij = overline(u_i u_j) - ū_i ū_j, S_ij = 0.5(∂_i ū_j + ∂_j ū_i).
Supports Cartesian `Grid` and `SphericalGrid`.
"""
function compute_pi(u::Field{T,G}, v::Field{T,G}, kernel) where {T<:Real,G}
    @assert (u.grid isa Grid || u.grid isa SphericalGrid) "compute_pi supports Cartesian Grid or SphericalGrid"
    # Filter velocities and quadratic products
    ū = coarse_grain(u, kernel)
    v̄ = coarse_grain(v, kernel)
    uu = Field(u.data .* u.data, u.grid)
    vv = Field(v.data .* v.data, u.grid)
    uv = Field(u.data .* v.data, u.grid)
    ūū = coarse_grain(uu, kernel)
    v̄v̄ = coarse_grain(vv, kernel)
    ūv̄ = coarse_grain(uv, kernel)
    τxx = ūū.data .- (ū.data .* ū.data)
    τyy = v̄v̄.data .- (v̄.data .* v̄.data)
    τxy = ūv̄.data .- (ū.data .* v̄.data)
    if u.grid isa SphericalGrid
        (ux, uy) = gradient_sphere(ū)
        (vx, vy) = gradient_sphere(v̄)
    else
        (ux, uy) = gradient(ū)
        (vx, vy) = gradient(v̄)
    end
    Sxx = ux.data
    Syy = vy.data
    Sxy = 0.5 .* (uy.data .+ vx.data)
    Π = .-(τxx .* Sxx .+ 2 .* τxy .* Sxy .+ τyy .* Syy)
    return Field(Π, u.grid)
end
