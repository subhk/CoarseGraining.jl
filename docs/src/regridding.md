# Regridding

## Index-space bilinear (fast, approximate)
```julia
f2 = regrid_index_bilinear(f, nx*2, ny*2)
```
- Good for quick resampling when coordinates are close to uniform.

## Lon/lat nearest neighbor (no deps)
```julia
Areg = regrid_lonlat_nearest(f, lon_src, lat_src, lon_tgt, lat_tgt)
```
- Suitable for moderate sizes; O(Nsrc × Ntgt).
- For large problems, consider a k-d tree approach (not included).
