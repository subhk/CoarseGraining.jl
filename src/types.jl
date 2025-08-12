struct Grid
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    periodic_x::Bool
    periodic_y::Bool
end

struct Field{T,G}
    data::Array{T,2}
    grid::G
end

struct Kernel
    weights::Array{Float64,2}
    radius_x::Int
    radius_y::Int
end

struct SphericalGrid
    lon::Vector{Float64}  # radians, length nx
    lat::Vector{Float64}  # radians, length ny
    a::Float64            # sphere radius (e.g., Earth 6.371e6)
    periodic_lon::Bool
end
