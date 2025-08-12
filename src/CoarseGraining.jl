module CoarseGraining

using LinearAlgebra
using FFTW
using Statistics

export Grid, Field, Kernel,
       coarse_grain, coarse_grain_fft,
       gaussian_kernel, boxcar_kernel,
       gradient, divergence, vorticity,
       helmholtz_hodge,
       load_netcdf_var, write_netcdf_field,
       mpi_init, mpi_finalize, mpi_enabled, mpi_rank, mpi_size,
       parallel_coarse_grain, parallel_coarse_grain_fft, parallel_helmholtz_hodge,
       SphericalGrid, gradient_sphere, vorticity_sphere, divergence_sphere, vel_spher_to_cart, vel_cart_to_spher,
       okuboweiss

include("types.jl")
include("kernels.jl")
include("filters.jl")
include("differential.jl")
include("helmholtz.jl")
include("io.jl")
include("mpi_utils.jl")
include("spherical.jl")
include("diagnostics.jl")

end # module
