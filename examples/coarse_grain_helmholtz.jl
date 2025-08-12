#!/usr/bin/env julia
using CoarseGraining
using Random

"""
Example port of FlowSieve's coarse-grain + Helmholtz projection workflow.

If `ARGS[1]` is provided, it attempts to load a NetCDF variable as velocity components
from that file; otherwise, it generates synthetic data.
"""
function main()
    Random.seed!(42)
    nx, ny = 256, 256
    g = Grid(nx, ny, 1.0, 1.0, true, true)
    if length(ARGS) == 1
        path = ARGS[1]
        # Minimal example: load variables u,v from NetCDF
        u = load_netcdf_var(path, "u"; dx=g.dx, dy=g.dy, periodic_x=true, periodic_y=true)
        v = load_netcdf_var(path, "v"; dx=g.dx, dy=g.dy, periodic_x=true, periodic_y=true)
    else
        x = range(0, stop=2π, length=g.nx)
        y = range(0, stop=2π, length=g.ny)
        u = Field([sin(xx) + 0.2cos(yy) for yy in y, xx in x], g)
        v = Field([cos(xx) - 0.2sin(yy) for yy in y, xx in x], g)
    end

    # Coarse-grain velocities
    K = gaussian_kernel(2.0, 2.0)
    ū = coarse_grain(u, K)
    v̄ = coarse_grain(v, K)

    # Helmholtz decomposition
    udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(ū, v̄)

    @info "coarse_grain_helmholtz done" size(ū.data) size(udf.data)
end

main()

