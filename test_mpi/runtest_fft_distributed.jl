#!/usr/bin/env julia
using CoarseGraining

function main()
    σx, σy = 1.5, 1.5
    if CoarseGraining.mpi_rank() == 0
        g = Grid(64, 48, 1.0, 1.0, true, true)
        f = Field(randn(g.ny, g.nx), g)
        out = parallel_coarse_grain_fft_distributed(f, σx, σy)
        @assert size(out.data) == (g.ny, g.nx)
        println("OK: distributed FFT filtered size = ", size(out.data))
    else
        parallel_coarse_grain_fft_distributed(nothing, σx, σy)
    end
    CoarseGraining.mpi_finalize()
end

main()

