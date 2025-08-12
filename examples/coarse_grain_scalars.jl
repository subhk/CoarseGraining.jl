#!/usr/bin/env julia
using CoarseGraining

"""
Coarse-grain a scalar field on a periodic Cartesian grid.
If a NetCDF path and var are provided as ARGS, it loads and filters that variable.
Otherwise, it filters a synthetic scalar.
"""
function main()
    if length(ARGS) >= 2
        path, var = ARGS[1], ARGS[2]
        f = load_netcdf_var(path, var; dx=1.0, dy=1.0, periodic_x=true, periodic_y=true)
        K = gaussian_kernel(2.0, 2.0)
        fout = coarse_grain(f, K)
        write_netcdf_field("filtered_$var.nc", "$(var)_filtered", fout)
        @info "Wrote filtered_$var.nc"
    else
        g = Grid(256, 256, 1.0, 1.0, true, true)
        x = range(0, stop=2π, length=g.nx)
        y = range(0, stop=2π, length=g.ny)
        A = [sin(xx) + cos(yy) for yy in y, xx in x]
        f = Field(A, g)
        K = gaussian_kernel(2.0, 2.0)
        fout = coarse_grain(f, K)
        @info size(fout.data)
    end
end

main()

