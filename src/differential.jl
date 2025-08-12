using ..CoarseGraining: Grid, Field

export gradient, divergence, vorticity

function gradient(field::Field{T}) where {T<:Real}
    A = field.data
    ny, nx = size(A)
    dx, dy = field.grid.dx, field.grid.dy
    ∂x = similar(A)
    ∂y = similar(A)
    # x-derivative
    for j in 1:ny
        for i in 1:nx
            il = i == 1 ? (field.grid.periodic_x ? nx : 1) : i-1
            ir = i == nx ? (field.grid.periodic_x ? 1 : nx) : i+1
            ∂x[j,i] = (A[j,ir] - A[j,il])/(2dx)
        end
    end
    # y-derivative
    for j in 1:ny
        jl = j == 1 ? (field.grid.periodic_y ? ny : 1) : j-1
        jr = j == ny ? (field.grid.periodic_y ? 1 : ny) : j+1
        for i in 1:nx
            ∂y[j,i] = (A[jr,i] - A[jl,i])/(2dy)
        end
    end
    return Field(∂x, field.grid), Field(∂y, field.grid)
end

function divergence(u::Field{T}, v::Field{T}) where {T<:Real}
    (ux, uy) = gradient(u)
    (vx, vy) = gradient(v)
    div = ux.data .+ vy.data
    return Field(div, u.grid)
end

function vorticity(u::Field{T}, v::Field{T}) where {T<:Real}
    (ux, uy) = gradient(u)
    (vx, vy) = gradient(v)
    ζ = vx.data .- uy.data
    return Field(ζ, u.grid)
end

