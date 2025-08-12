module CoarseGraining

using LinearAlgebra
using FFTW
using Statistics

export Grid, Field, Kernel,
       coarse_grain, coarse_grain_fft, coarse_grain_gaussian_separable, select_tile,
       gaussian_kernel, boxcar_kernel,
       gradient, divergence, vorticity,
       gradient_curvilinear,
       helmholtz_hodge,
       load_netcdf_var, write_netcdf_field, read_attr, write_attr,
       load_region_masks, load_vector_vars, write_vector_vars, write_region_stats, write_regions_file, write_region_stats_with_attrs, write_region_stats_and_masks_with_attrs,
       mpi_init, mpi_finalize, mpi_enabled, mpi_rank, mpi_size,
       parallel_coarse_grain, parallel_coarse_grain_fft, parallel_helmholtz_hodge,
       SphericalGrid, gradient_sphere, vorticity_sphere, divergence_sphere, vel_spher_to_cart, vel_cart_to_spher,
       helmholtz_hodge_sphere, poisson_sphere_solve, CurvilinearGrid,
       okuboweiss, compute_pi, ke_spectrum_isotropic, zonal_mean_std, region_mean_std, write_region_stats_and_masks
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

end # module
