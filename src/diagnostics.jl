using ..CoarseGraining: Field, Grid, gradient

export okuboweiss

"""
    okuboweiss(u, v) -> Field

Compute the Okubo–Weiss parameter W = s_n^2 + s_s^2 - ω^2 on a Cartesian grid,
where s_n = ∂u/∂x - ∂v/∂y, s_s = ∂v/∂x + ∂u/∂y, ω = ∂v/∂x - ∂u/∂y.
"""
function okuboweiss(u::Field{T}, v::Field{T}) where {T<:Real}
    @assert u.grid isa Grid "okuboweiss currently supports Cartesian Grid"
    (ux, uy) = gradient(u)
    (vx, vy) = gradient(v)
    sn = ux.data .- vy.data
    ss = vx.data .+ uy.data
    ω  = vx.data .- uy.data
    W = sn.^2 .+ ss.^2 .- ω.^2
    return Field(W, u.grid)
end

