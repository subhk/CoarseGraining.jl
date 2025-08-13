# Diagnostics

## Okubo–Weiss
```julia
W = okuboweiss(u_field, v_field)
```
- Distinguish strain–dominated vs vorticity–dominated regions.

## Leonard transfer Π
```julia
K = gaussian_kernel(3.0, 3.0)
Π = compute_pi(u_field, v_field, K)
```
- Quantifies resolved-scale energy transfer driven by subfilter stress.

## KE spectrum
```julia
k, Ek = ke_spectrum_isotropic(u_field, v_field; nbins=32, normalize=:shellarea)
```
- `:counts`, `:density`, `:shellarea`, `:energy` available.

## Zonal and region stats
```julia
m, s = zonal_mean_std(f)
regions = Dict("box" => (f.data .> 0.0))
stats = region_mean_std(f, regions)
```

