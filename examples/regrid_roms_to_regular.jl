#!/usr/bin/env julia
using CoarseGraining

"""
Usage:
  julia --project examples/regrid_roms_to_regular.jl /path/to/his.nc var out.nc [nx ny]

Loads a ROMS/CROCO variable at rho points, builds a regular lon/lat target grid
covering the source extent, regrids via nearest-neighbor, and writes to NetCDF.
"""
function main()
    if length(ARGS) < 3
        println("Usage: julia --project examples/regrid_roms_to_regular.jl his.nc var out.nc [nx ny]")
        return
    end
    path = ARGS[1]
    var  = ARGS[2]
    outp = ARGS[3]
    nx = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 360
    ny = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 180
    f, meta = load_model_var(path; varname=var, at=:rho, t=1, z=1)
    lon = meta[:lon]; lat = meta[:lat]
    lonmin, lonmax = minimum(lon), maximum(lon)
    latmin, latmax = minimum(lat), maximum(lat)
    lon_t = range(lonmin, lonmax, length=nx)
    lat_t = range(latmin, latmax, length=ny)
    lon_tgt = [lon_t[i] for j in 1:ny, i in 1:nx]
    lat_tgt = [lat_t[j] for j in 1:ny, i in 1:nx]
    Areg = regrid_lonlat_nearest(f, lon, lat, lon_tgt, lat_tgt)
    fret = Field(Areg, Grid(nx, ny, (lonmax-lonmin)/nx, (latmax-latmin)/ny, false, false))
    write_netcdf_field(outp, "$(var)_regular", fret)
    println("Wrote ", outp)
end

main()

