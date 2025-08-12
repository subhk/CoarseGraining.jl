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

end # module
