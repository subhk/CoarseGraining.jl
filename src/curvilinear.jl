using ..CoarseGraining: Field, CurvilinearGrid

export gradient_curvilinear

"""
    gradient_curvilinear(f::Field{T,CurvilinearGrid})

Compute approximate gradient on an orthogonal curvilinear grid using local dx, dy.
Periodic handling follows `grid.periodic_x`/`periodic_y`.
"""
function gradient_curvilinear(f::Field{T,CurvilinearGrid}) where {T<:Real}
    A = f.data
    ny, nx = size(A)
    dx = f.grid.dx
    dy = f.grid.dy
    ∂x = similar(A)
    ∂y = similar(A)
    # x-derivative
    for j in 1:ny
        for i in 1:nx
            il = i == 1 ? (f.grid.periodic_x ? nx : 1) : i-1
            ir = i == nx ? (f.grid.periodic_x ? 1 : nx) : i+1
            # central difference divided by local 2*dx (choose center cell)
            ∂x[j,i] = (A[j,ir] - A[j,il]) / (dx[j,i] + dx[j,il])
        end
    end
    # y-derivative
    for j in 1:ny
        jl = j == 1 ? (f.grid.periodic_y ? ny : 1) : j-1
        jr = j == ny ? (f.grid.periodic_y ? 1 : ny) : j+1
        for i in 1:nx
            ∂y[j,i] = (A[jr,i] - A[jl,i]) / (dy[j,i] + dy[jl,i])
        end
    end
    return Field(∂x, f.grid), Field(∂y, f.grid)
end

