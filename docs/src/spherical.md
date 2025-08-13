# Spherical & Curvilinear

## Spherical
```julia
using CoarseGraining
nx, ny = 360, 180
lon = range(0, 2π, length=nx)
lat = range(-π/2 + π/ny, π/2 - π/ny, length=ny)
sg = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
f = Field([cos(λ)*cos(ϕ) for ϕ in lat, λ in lon], sg)
fx, fy = gradient_sphere(f)
```
- Use spherical gradients and vorticity when working on lat/lon grids.

### Helmholtz–Hodge (periodic)
```julia
uE = Field(randn(ny, nx), sg); vN = Field(randn(ny, nx), sg)
udf, vdf, up, vp, φ, ψ = helmholtz_hodge_sphere(uE, vN)
```

## Curvilinear
```julia
using CoarseGraining
ny, nx = 200, 300
lon = zeros(ny, nx); lat = zeros(ny, nx)
dx = fill(1000.0, ny, nx); dy = fill(1000.0, ny, nx)
cg = CurvilinearGrid(lon, lat, dx, dy, true, true, 6.371e6)
A = randn(ny, nx)
fx, fy = gradient_curvilinear(Field(A, cg))
```
- For real–space filtering on curvilinear grids, normal filters work; prefer masked variants if land is present.

