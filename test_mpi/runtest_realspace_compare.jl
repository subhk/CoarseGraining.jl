#!/usr/bin/env julia
using CoarseGraining

function main()
    K = gaussian_kernel(2.0, 2.0)
    nx, ny = 63, 50
    if CoarseGraining.mpi_rank() == 0
        g = Grid(nx, ny, 1.0, 1.0, true, true)
        f = Field(randn(g.ny, g.nx), g)
        out_full = coarse_grain(f, K; threaded=true)
        out_mpi  = parallel_coarse_grain(f, K; threaded=true)
        err = mean(abs.(out_full.data .- out_mpi.data))
        @assert err < 1e-10
        println("OK: MPI real-space matches serial (mean abs err = ", err, ")")
    else
        parallel_coarse_grain(nothing, K; threaded=true)
    end
    CoarseGraining.mpi_finalize()
end

main()

