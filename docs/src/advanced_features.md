# Advanced Features

CoarseGraining.jl provides advanced FlowSieve-equivalent capabilities for sophisticated ocean modeling and turbulence analysis.

## Spherical Helmholtz Decomposition

For large spherical domains, use iterative solvers that are more memory-efficient than FFT methods:

```julia
# Set up spherical domain  
lon = range(0, 2π, length=360)
lat = range(-π/2, π/2, length=180)
grid = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)

uE = Field(velocity_east, grid)
vN = Field(velocity_north, grid)

# Iterative spherical Helmholtz decomposition
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_iterative(
    uE, vN; max_iter=1000, tol=1e-6, method=:cg)

# For very large grids, use multi-grid acceleration
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_multigrid(
    uE, vN; levels=4, max_iter=100)
```

**Key features:**
- Memory-efficient sparse matrix approach
- Choice of iterative solvers (CG, GMRES, BiCGStab)  
- Proper spherical metric factors
- Multi-grid acceleration for very large problems

## Complete Energy Budget Analysis

Compute all terms in the kinetic energy equation following Aluie et al. (2018):

```julia
K = gaussian_kernel(3.0, 3.0)  # Filter scale

# Complete energy budget: ∂KE/∂t + ∇·J = -Π - ε + Bc + F
budget = compute_full_energy_budget(u, v, K;
                                   pressure=P,        # Pressure field
                                   ρ=density,         # Density field  
                                   w=w_velocity,      # Vertical velocity
                                   forcing_u=wind_u,  # External forcing
                                   forcing_v=wind_v,
                                   ν=1e-3,            # Viscosity
                                   g=9.81, ρ₀=1025.0)

# Access individual budget terms
KE = budget.kinetic_energy           # Kinetic energy density
Π = budget.leonard_transfer          # Scale transfer (Leonard term)
transport = budget.transport_total   # Total transport divergence  
dissipation = budget.dissipation_total  # Viscous dissipation
baroclinic = budget.baroclinic_conversion  # Baroclinic conversion
forcing = budget.forcing_work        # External forcing work
```

**Individual diagnostic functions:**
```julia
# Energy transport terms
J_coarse, J_pressure, J_subgrid, div_J = compute_energy_transport(u, v, K; pressure=P)

# Baroclinic energy conversion  
Pi_bc, APE_flux = compute_baroclinic_transfer(u, v, ρ, w, K; g=9.81)

# Viscous dissipation (resolved + subgrid)
ε_resolved, ε_subgrid, ε_total = compute_viscous_dissipation(u, v, K; ν=1e-6)

# Spectral analysis
k, E_k = compute_kinetic_energy_spectra(u, v; method=:isotropic)
```

## Sophisticated Boundary Handling

Handle complex coastal boundaries and land masks with adaptive filtering:

```julia
# Load land/water mask (true = water, false = land)
land_mask = load_coastline_mask("coastlines.nc")

# Adaptive filtering with land-avoiding stencils
K = gaussian_kernel(2.0, 2.0)
filtered = coarse_grain_adaptive(field, K;
                                mask=land_mask,
                                boundary_mode=:adaptive,    # Use physical distances
                                deform_around_land=true,    # Avoid land contamination
                                kernel_padding=2.0)         # Integration scale factor

# For spherical grids, uses great circle distances
# For curvilinear grids, uses local metric information
```

**Boundary extension methods:**
```julia
# Extend field to reduce edge effects
extended = extend_field_to_boundaries(field, 5; method=:mirror)
# Methods: :mirror, :extrapolate, :zero, :periodic
```

**Adaptive integration bounds:**
```julia
# Compute physical distance-based bounds (used internally)
j_bounds, i_bounds = compute_adaptive_bounds(grid, j, i, kernel, padding, distance_func)
```

## Multi-Resolution Workflows

Dramatically reduce computational cost for large problems using hierarchical coarsen→solve→refine strategy:

```julia
# Automatic hierarchical Helmholtz decomposition
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = hierarchical_helmholtz_workflow(
    uE, vN; levels=4, max_iter_per_level=50)

# Manual workflow control
grids = create_multiresolution_hierarchy(base_grid, 4; coarsening_factor=2)

# Coarsen fields
uE_coarse = coarsen_field(uE, 4; method=:area_average)  # Conservative averaging
vN_coarse = coarsen_field(vN, 4; method=:subsample)     # Simple subsampling

# Solve on coarse grid
uE_div_c, vN_div_c, _, _, φ_c, ψ_c = helmholtz_hodge_sphere_iterative(uE_coarse, vN_coarse)

# Refine solution back to fine grid  
φ_fine = refine_field(φ_c, base_grid; method=:bilinear)
ψ_fine = refine_field(ψ_c, base_grid; method=:bicubic)
```

**Data persistence for workflows:**
```julia
# Save multi-resolution data for later use
save_multiresolution_data("workflow.nc", [φ_coarse, φ_fine], [coarse_grid, fine_grid])

# Load workflow data
fields, grids = load_multiresolution_data("workflow.nc")
```

**Performance benefits:**
- 10-100x speedup for large grids
- Reduced memory requirements  
- Maintained solution accuracy

## Ocean Modeling Applications

### Regional Ocean Analysis
```julia
# Gulf Stream region with coastlines
lon = range(deg2rad(-80), deg2rad(-60), length=200)
lat = range(deg2rad(30), deg2rad(50), length=200)
ocean_grid = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)

# Load velocity and create land mask
uE = load_ocean_velocity("gulf_stream_u.nc", ocean_grid)
vN = load_ocean_velocity("gulf_stream_v.nc", ocean_grid)
coastline = load_coastline_mask("coastlines.nc", ocean_grid)

# Adaptive filtering respecting coastlines
K = gaussian_kernel(3.0, 3.0)  # ~165 km scale
uE_filtered = coarse_grain_adaptive(uE, K; mask=coastline, boundary_mode=:adaptive)
vN_filtered = coarse_grain_adaptive(vN, K; mask=coastline, boundary_mode=:adaptive)

# Spherical Helmholtz decomposition
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_iterative(
    uE_filtered, vN_filtered; max_iter=500, tol=1e-6)
```

### High-Resolution Global Analysis
```julia
# 0.1° global ocean model (3600×1800 grid)
# Use multi-resolution workflow for efficiency
uE_div, vN_div, uE_pot, vN_pot, φ, ψ = hierarchical_helmholtz_workflow(
    uE_global, vN_global; levels=5, max_iter_per_level=100)

# Complete energy budget analysis
budget = compute_full_energy_budget(uE_global, vN_global, K;
                                   pressure=pressure_global,
                                   ρ=density_global,
                                   ν=1e-3)

# Identify energy transfer hotspots
energy_transfer_map = budget.leonard_transfer
upscale_regions = energy_transfer_map.data .< 0  # Energy to larger scales
downscale_regions = energy_transfer_map.data .> 0  # Energy to smaller scales
```

### Mesoscale Eddy Analysis
```julia
# Track energy pathways in eddies
eddies_mask = identify_eddies(vorticity_field)  # Your eddy detection

# Energy budget within eddies
eddy_budget = compute_full_energy_budget(u, v, K; mask=eddies_mask)

# Compute eddy transport
J_coarse, J_pressure, J_subgrid, _ = compute_energy_transport(u, v, K)
eddy_transport = J_subgrid.data[eddies_mask]

# Spectral analysis of eddy field
k, E_k = compute_kinetic_energy_spectra(u_eddies, v_eddies)
```

## Performance Tips

1. **Choose appropriate methods:**
   - Real-space filters for non-periodic domains
   - Multi-resolution for large problems  
   - Adaptive boundaries for coastal regions

2. **Memory management:**
   - Use hierarchical workflows for >1000² grids
   - Tile processing for cache efficiency
   - Sparse iterative solvers for Helmholtz

3. **Accuracy vs. speed:**
   - Increase `max_iter` for higher accuracy
   - Use multi-grid for fastest convergence
   - Conservative coarsening preserves physics

4. **Oceanic applications:**
   - Always use land masks for realistic domains
   - Prefer spherical operators for global/basin scales
   - Account for boundary effects in analysis

## Theory Background

The advanced features implement:

- **Aluie et al. (2018)**: Complete kinetic energy budget methodology
- **FlowSieve methods**: Sophisticated boundary handling and multi-resolution
- **Iterative solvers**: Memory-efficient alternatives to direct methods
- **Physical conservation**: Proper treatment of coastlines and boundaries

See the Theory page for mathematical details and the original FlowSieve papers for methodology validation.