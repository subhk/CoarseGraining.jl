# Spherical & Curvilinear

## Spherical Coordinates

### Basic Operations
```julia
using CoarseGraining
nx, ny = 360, 180
lon = range(0, 2π, length=nx)
lat = range(-π/2 + π/ny, π/2 - π/ny, length=ny)
sg = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
f = Field([cos(λ)*cos(ϕ) for ϕ in lat, λ in lon], sg)

# Spherical derivatives (include metric factors)
fx, fy = gradient_sphere(f)
vort = vorticity_sphere(uE, vN)  # Relative vorticity
div = divergence_sphere(uE, vN)   # Horizontal divergence
```

### Helmholtz Decomposition

#### Basic (FFT-based, periodic)
```julia
uE = Field(randn(ny, nx), sg); vN = Field(randn(ny, nx), sg)
udf, vdf, up, vp, φ, ψ = helmholtz_hodge_sphere(uE, vN)
```

#### Advanced (Iterative, large grids)
```julia
# For large spherical domains - memory efficient
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_iterative(
    uE, vN; max_iter=1000, tol=1e-6, method=:cg)

# Multi-grid acceleration for very large problems
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_multigrid(
    uE, vN; levels=4, max_iter=100)
```

**Key differences:**
- `helmholtz_hodge_sphere`: FFT-based, periodic, fast for moderate sizes
- `helmholtz_hodge_sphere_iterative`: Sparse iterative, handles boundaries, scalable
- `helmholtz_hodge_sphere_multigrid`: Multi-grid acceleration, best for large grids

### Ocean Modeling Example
```julia
# Realistic Gulf Stream domain
lon_range = -80.0:0.25:-60.0  # 0.25° resolution
lat_range = 30.0:0.25:50.0
lon_grid = collect(deg2rad.(lon_range))
lat_grid = collect(deg2rad.(lat_range))
ocean_grid = SphericalGrid(lon_grid, lat_grid, 6.371e6, true)

# Load ocean data with land mask
uE = load_ocean_velocity("gulf_stream_u.nc", ocean_grid)
vN = load_ocean_velocity("gulf_stream_v.nc", ocean_grid)
coastline = load_coastline_mask("coastlines.nc", ocean_grid)

# Adaptive filtering respecting coastlines
K = gaussian_kernel(3.0, 3.0)  # ~165 km filter scale
uE_filtered = coarse_grain_adaptive(uE, K; mask=coastline, boundary_mode=:adaptive)

# Spherical Helmholtz decomposition
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_iterative(
    uE_filtered, vN_filtered; max_iter=500, tol=1e-6)
```

### Coordinate Conversions
```julia
# Convert between spherical and Cartesian velocity components
u_cart, v_cart, w_cart = vel_spher_to_cart(uE, vN, zero_field, lat, lon)
uE_new, vN_new, w_rad = vel_cart_to_spher(u_cart, v_cart, w_cart, lat, lon)
```

## Curvilinear Grids

For irregularly-spaced or rotated grids common in ocean models:

```julia
using CoarseGraining
ny, nx = 200, 300

# Load grid information from model
lon = load_curvilinear_coords("model_grid.nc", "lon")  # (ny, nx)
lat = load_curvilinear_coords("model_grid.nc", "lat")  # (ny, nx)
dx = load_curvilinear_metrics("model_grid.nc", "dx")   # meters per i-step
dy = load_curvilinear_metrics("model_grid.nc", "dy")   # meters per j-step

cg = CurvilinearGrid(lon, lat, dx, dy, true, true, 6.371e6)

# Load velocity field
A = load_model_field("velocity.nc", "u")
u_field = Field(A, cg)

# Curvilinear gradients using local metrics
fx, fy = gradient_curvilinear(u_field)
```

**Filtering on curvilinear grids:**
```julia
# Real-space filters work directly
K = gaussian_kernel(2.0, 2.0)
filtered = coarse_grain(u_field, K)

# Prefer masked variants for ocean models
land_mask = load_land_mask("model_grid.nc")
filtered_masked = coarse_grain_masked(u_field, K, land_mask)

# Advanced adaptive filtering
filtered_adaptive = coarse_grain_adaptive(u_field, K; 
                                         mask=land_mask,
                                         boundary_mode=:adaptive)
```

## Performance Tips

### Grid Choice
- **Spherical**: Use for global/basin scale analysis
- **Curvilinear**: Use for high-resolution regional models
- **Cartesian**: Use for idealized studies or after regridding

### Method Selection
- **Small grids** (<100×100): Use FFT methods
- **Large grids** (>500×500): Use iterative/multi-grid methods
- **Coastal regions**: Always use adaptive boundary handling

### Memory Management
```julia
# For very large spherical grids, use hierarchical approach
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = hierarchical_helmholtz_workflow(
    uE, vN; levels=4, max_iter_per_level=50)

# This reduces memory usage and computation time significantly
```

## Theory Notes

### Spherical Metrics
The spherical Laplacian includes metric factors:
```
∇²f = 1/(a² cosφ) [∂/∂φ(cosφ ∂f/∂φ) + 1/cosφ ∂²f/∂λ²]
```

### Boundary Conditions
- **Poles**: Special handling required (singular points)
- **Coastlines**: Use adaptive filtering to avoid land contamination
- **Open boundaries**: Extend or clamp values appropriately

### Physical Scales
On Earth:
- 1° longitude ≈ 111 km × cos(latitude)
- 1° latitude ≈ 111 km (constant)
- Deformation radius ≈ 50 km (mid-latitudes)

Choose filter scales appropriately for ocean dynamics.

