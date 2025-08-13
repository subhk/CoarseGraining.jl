# Oceanic Filtering Best Practices Guide

## The Periodicity Problem

⚠️ **Critical Issue**: Many Julia implementations assume periodic boundaries, but oceanic flows are typically **non-periodic** due to:

- **Coastlines and land boundaries**
- **Regional domain boundaries** 
- **Finite basin extents**
- **Polar boundaries** (even in global models)

## Recommended Approaches by Domain Type

### 1. Regional Ocean Models (e.g., North Atlantic, Gulf Stream)
```julia
# Non-periodic in both directions
g = Grid(nx, ny, dx, dy, false, false)
field = Field(ocean_data, g)
K = gaussian_kernel(σx, σy)

# Preferred: Real-space convolution with proper boundary handling
filtered = coarse_grain(field, K)

# Alternative: Masked filtering if land mask available
if has_land_mask
    filtered = coarse_grain_masked(field, K, water_mask)
end
```

### 2. Zonal Channel Models (e.g., Southern Ocean, ACC)
```julia
# Periodic in longitude, non-periodic in latitude
g = Grid(nx, ny, dx, dy, true, false)
field = Field(ocean_data, g)
filtered = coarse_grain(field, K)
```

### 3. Global Ocean Models
```julia
# Periodic in longitude, careful handling at poles
g = Grid(nx, ny, dx, dy, true, false)
field = Field(ocean_data, g)

# May need special pole handling for spherical grids
sg = SphericalGrid(lon, lat, R_earth, true)
filtered_spherical = coarse_grain_masked(field_sph, K, water_mask)
```

## Filter Method Comparison for Ocean Applications

| Method | Boundary Handling | Edge Effects | Ocean Suitability |
|--------|------------------|---------------|-------------------|
| `coarse_grain()` | ✅ Configurable | ❌ Minimal | ✅ **Excellent** |
| `coarse_grain_masked()` | ✅ Land/water | ❌ Minimal | ✅ **Best** |
| `coarse_grain_butterworth()` | ❌ Periodic only | ❌ None | ❌ **Poor** |
| `coarse_grain_butterworth_dsp()` | ⚠️ IIR effects | ⚠️ Moderate | ⚠️ **Acceptable** |

## Common Pitfalls to Avoid

### ❌ DON'T: Use FFT filters on realistic ocean domains
```julia
# This creates artificial periodic connections!
filtered = coarse_grain_butterworth(field, kc)  # BAD for oceans
```

### ❌ DON'T: Ignore land boundaries
```julia
# This will contaminate ocean filtering with land values
g = Grid(nx, ny, dx, dy, true, true)  # BAD: assumes all water
```

### ❌ DON'T: Filter across domain boundaries
```julia
# Edge regions may need special treatment or trimming
# Consider reducing analysis domain to avoid boundary artifacts
```

## Best Practices

### ✅ DO: Use appropriate boundary conditions
```julia
# Match your grid setup to your physical domain
periodic_x = is_zonal_channel || is_global_domain
periodic_y = false  # Almost never true for ocean applications
g = Grid(nx, ny, dx, dy, periodic_x, periodic_y)
```

### ✅ DO: Use land masks when available
```julia
# Most realistic approach
filtered = coarse_grain_masked(field, K, water_mask)
```

### ✅ DO: Consider edge effects in analysis
```julia
# Trim boundary regions if artifacts are present
margin = 2 * max(kernel.radius_x, kernel.radius_y)
interior_region = filtered.data[margin+1:end-margin, margin+1:end-margin]
```

### ✅ DO: Validate filter behavior near boundaries
```julia
# Check for unrealistic values near coastlines or domain edges
boundary_check = filtered.data[1:5, :] or filtered.data[:, 1:5]
```

## Implementation Priority Fixes

1. **Update documentation** to emphasize non-periodic nature of ocean flows
2. **Add ocean-specific examples** showing proper boundary handling
3. **Improve masked filtering** to match FlowSieve capabilities
4. **Add boundary validation** utilities
5. **Consider deprecating** FFT filters for ocean applications