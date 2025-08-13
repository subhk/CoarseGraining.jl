module CoarseGraining

using LinearAlgebra
using FFTW
using Statistics
export coarse_grain_masked

include("types.jl")
include("kernels.jl")
include("filters.jl")
include("filters_masked.jl")
include("differential.jl")
include("helmholtz.jl")
include("io.jl")
include("mpi_utils.jl")
include("spherical.jl")
include("diagnostics.jl")
include("diagnostics_extra.jl")
include("model_io.jl")
include("curvilinear.jl")
include("regrid.jl")

# Central exports grouped by file for clarity

# types.jl
export Grid, Field, Kernel, SphericalGrid, CurvilinearGrid

# kernels.jl
export gaussian_kernel, boxcar_kernel

# filters.jl
export coarse_grain, coarse_grain_fft, coarse_grain_gaussian_separable, select_tile
export coarse_grain_butterworth, coarse_grain_butterworth_length, coarse_grain_butterworth_cycles,
       coarse_grain_butterworth_cells, coarse_grain_butterworth_nyquist
export coarse_grain_butterworth_dsp, coarse_grain_butterworth_length_dsp, coarse_grain_butterworth_cycles_dsp,
       coarse_grain_butterworth_cells_dsp, coarse_grain_butterworth_nyquist_dsp

# filters_masked.jl
export coarse_grain_masked, coarse_grain_gaussian_separable_masked

# differential.jl
export gradient, divergence, vorticity

# helmholtz.jl
export helmholtz_hodge

# spherical.jl
export gradient_sphere, vorticity_sphere, divergence_sphere, vel_spher_to_cart, vel_cart_to_spher
export helmholtz_hodge_sphere, poisson_sphere_solve

# diagnostics.jl
export okuboweiss, compute_pi

# diagnostics_extra.jl
export ke_spectrum_isotropic, zonal_mean_std, region_mean_std, write_region_stats_and_masks

# io.jl (IO submodule reexports)
export load_netcdf_var, write_netcdf_field, read_attr, write_attr,
       load_region_masks, load_vector_vars, write_vector_vars,
       write_region_stats, write_regions_file, write_region_stats_with_attrs, write_region_stats_and_masks_with_attrs

# mpi_utils.jl (MPIUtils submodule reexports)
export mpi_init, mpi_finalize, mpi_enabled, mpi_rank, mpi_size,
       parallel_coarse_grain, parallel_coarse_grain_fft, parallel_helmholtz_hodge,
       parallel_coarse_grain_fft_distributed, parallel_coarse_grain_masked

# model_io.jl (ModelIO submodule reexports)
export detect_model, load_model_var, average_to_rho, average_to_tracer_mitgcm

# curvilinear.jl
export gradient_curvilinear

# regrid.jl
export regrid_index_bilinear, regrid_lonlat_nearest

end # module
