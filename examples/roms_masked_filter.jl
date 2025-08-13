#!/usr/bin/env julia
using CoarseGraining

"""
Usage:
  julia --project examples/roms_masked_filter.jl /path/to/his.nc varname out.nc [sigma]

Loads a ROMS/CROCO variable, applies a masked separable Gaussian (using mask_rho if present), and writes to NetCDF.
"""
function main()
    if length(ARGS) < 3
        println("Usage: julia --project examples/roms_masked_filter.jl his.nc var out.nc [sigma]")
        return
    end
    path = ARGS[1]
    var  = ARGS[2]
    outp = ARGS[3]
    σ = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 3.0
    f, meta = load_model_var(path; varname=var, model=:auto, at=:rho, t=1, z=1)
    mask = get(meta, :mask, trues(size(f.data)))
    fg = coarse_grain_gaussian_separable_masked(f, σ, σ, mask; threaded=true)
    write_netcdf_field(outp, "$(var)_filtered", fg)
    println("Wrote ", outp)
end

main()

