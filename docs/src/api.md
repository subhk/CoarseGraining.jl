# API Reference

This page summarizes the main exported functions. See docstrings in the source for details.

## Types
- `Grid`, `Field`, `Kernel`, `SphericalGrid`, `CurvilinearGrid`

## Kernels
- `gaussian_kernel`, `boxcar_kernel`

## Filters (real–space and spectral)
- `coarse_grain`, `coarse_grain_gaussian_separable`, `coarse_grain_fft`
- Butterworth: `coarse_grain_butterworth*` (FFT) and `coarse_grain_butterworth*_dsp` (DSP)
- Masked: `coarse_grain_masked`, `coarse_grain_gaussian_separable_masked`
- **Advanced**: `coarse_grain_adaptive` - sophisticated boundary handling with land-avoiding stencils

## Differential & Decomposition
- `gradient`, `divergence`, `vorticity`, `helmholtz_hodge`
- Spherical: `gradient_sphere`, `vorticity_sphere`, `divergence_sphere`, `helmholtz_hodge_sphere`
- **Advanced Spherical**: `helmholtz_hodge_sphere_iterative`, `helmholtz_hodge_sphere_multigrid`
- Curvilinear: `gradient_curvilinear`

## Advanced Energy Diagnostics
- `okuboweiss`, `compute_pi` (basic)
- **Advanced**: `compute_full_energy_budget` - complete energy equation terms
- **Transport**: `compute_energy_transport` - advection, pressure, subgrid transport
- **Baroclinic**: `compute_baroclinic_transfer` - available potential energy conversion
- **Dissipation**: `compute_viscous_dissipation` - resolved and subgrid dissipation
- **Spectral**: `compute_kinetic_energy_spectra`, `compute_enstrophy_transfer`

## Multi-Resolution Workflows
- **Coarsen/Refine**: `coarsen_field`, `refine_field`
- **Hierarchical**: `hierarchical_helmholtz_workflow` - efficient multi-scale Helmholtz solve
- **Grid Management**: `create_multiresolution_hierarchy`
- **Data I/O**: `save_multiresolution_data`, `load_multiresolution_data`

## Boundary Handling
- **Adaptive Bounds**: `compute_adaptive_bounds` - physical distance-based integration
- **Land Avoidance**: `land_avoiding_stencil` - coastal boundary-aware filtering
- **Extension**: `extend_field_to_boundaries` - boundary value extension

## Stats & Analysis
- `ke_spectrum_isotropic`, `zonal_mean_std`, `region_mean_std`, `write_region_stats_and_masks`

## IO & Models
- NetCDF: `load_netcdf_var`, `write_netcdf_field`, `read_attr`, `write_attr`
- Model adapters: `detect_model`, `load_model_var`, `average_to_rho`, `average_to_tracer_mitgcm`
- Regions: `load_region_masks`, `write_region_stats_with_attrs`, `write_region_stats_and_masks_with_attrs`

## Parallel / MPI
- `mpi_init`, `mpi_finalize`, `mpi_rank`, `mpi_size`
- Real–space: `parallel_coarse_grain`, `parallel_coarse_grain_masked`
- FFT: `parallel_coarse_grain_fft`, `parallel_coarse_grain_fft_distributed`
- **Advanced**: `parallel_helmholtz_hodge` - distributed Helmholtz decomposition

## Regridding
- `regrid_index_bilinear`, `regrid_lonlat_nearest`

