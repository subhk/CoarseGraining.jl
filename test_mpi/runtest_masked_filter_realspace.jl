#!/usr/bin/env julia
using CoarseGraining

function main()
    K = gaussian_kernel(2.0, 2.0)
    if CoarseGraining.mpi_rank() == 0
        g = Grid(63, 50, 1.0, 1.0, true, true)
        f = Field(randn(g.ny, g.nx), g)
        mask = trues(g.ny, g.nx)
        mask[:, 1:5] .= false  # mask a stripe
        out_serial = coarse_grain_masked(f, K, mask)
        out_mpi = parallel_coarse_grain_masked(f, K, mask)
        err = mean(abs.(out_serial.data .- out_mpi.data))
        @assert err < 1e-10
        println("OK: MPI masked filter matches serial (mean abs err = ", err, ")")
    else
        parallel_coarse_grain_masked(nothing, K, nothing)
    end
    CoarseGraining.mpi_finalize()
end

main()

