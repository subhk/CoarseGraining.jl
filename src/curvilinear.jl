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
    # x-derivative (one-sided at boundaries, central inside)
    for j in 1:ny
        for i in 1:nx
            if i == 1
                ∂x[j,i] = (A[j,2] - A[j,1]) / dx[j,1]
            elseif i == nx
                ∂x[j,i] = (A[j,nx] - A[j,nx-1]) / dx[j,nx-1]
            else
                ∂x[j,i] = (A[j,i+1] - A[j,i-1]) / (dx[j,i] + dx[j,i-1])
            end
        end
    end
    # y-derivative (one-sided at boundaries, central inside)
    for j in 1:ny
        for i in 1:nx
            if j == 1
                ∂y[j,i] = (A[2,i] - A[1,i]) / dy[1,i]
            elseif j == ny
                ∂y[j,i] = (A[ny,i] - A[ny-1,i]) / dy[ny-1,i]
            else
                ∂y[j,i] = (A[j+1,i] - A[j-1,i]) / (dy[j,i] + dy[j-1,i])
            end
        end
    end
    return Field(∂x, f.grid), Field(∂y, f.grid)
end
