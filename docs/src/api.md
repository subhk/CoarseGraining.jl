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

## Differential & Decomposition
- `gradient`, `divergence`, `vorticity`, `helmholtz_hodge`
- Spherical: `gradient_sphere`, `vorticity_sphere`, `divergence_sphere`, `helmholtz_hodge_sphere`
- Curvilinear: `gradient_curvilinear`

## Diagnostics & Stats
- `okuboweiss`, `compute_pi`, `ke_spectrum_isotropic`, `zonal_mean_std`, `region_mean_std`, `write_region_stats_and_masks`

## IO & Models
- NetCDF: `load_netcdf_var`, `write_netcdf_field`, `read_attr`, `write_attr`
- Model adapters: `detect_model`, `load_model_var`, `average_to_rho`, `average_to_tracer_mitgcm`

## Parallel / MPI
- `mpi_init`, `mpi_finalize`, `mpi_rank`, `mpi_size`
- Real–space: `parallel_coarse_grain`, `parallel_coarse_grain_masked`
- FFT: `parallel_coarse_grain_fft`, `parallel_coarse_grain_fft_distributed`

## Regridding
- `regrid_index_bilinear`, `regrid_lonlat_nearest`

