# IO & Models

## NetCDF helpers
```julia
f  = load_netcdf_var("file.nc", "temp"; dx=1.0, dy=1.0, periodic_x=true, periodic_y=true)
path = write_netcdf_field("out.nc", "temp_filtered", f)
```

## Model adapters (CROCO/ROMS/MITgcm)
```julia
# Auto-detect model, load variable at time t=1, depth level z=1
f, meta = load_model_var("his.nc"; varname="temp", model=:auto, at=:rho, t=1, z=1)
# Use mask if present
mask = get(meta, :mask, trues(size(f.data)))
```
- ROMS/CROCO: `average_to_rho(u, v)` to center C-grid velocities.
- MITgcm: `average_to_tracer_mitgcm(U, V)` to center staggered velocities.

## Mask–aware filtering
```julia
K = gaussian_kernel(3.0, 3.0)
f_sm = coarse_grain_gaussian_separable_masked(f, 3.0, 3.0, mask; threaded=true)
```

