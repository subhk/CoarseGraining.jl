#!/usr/bin/env julia
using CoarseGraining

function main()
    K = gaussian_kernel(2.0, 2.0)
    if CoarseGraining.mpi_rank() == 0
        g = Grid(63, 50, 1.0, 1.0, true, true)
        f = Field(randn(g.ny, g.nx), g)
        out = parallel_coarse_grain(f, K)
        @assert size(out.data) == (g.ny, g.nx)
        println("OK: real-space MPI filtered size = ", size(out.data))
    else
        parallel_coarse_grain(nothing, K)
    end
    CoarseGraining.mpi_finalize()
end

main()

