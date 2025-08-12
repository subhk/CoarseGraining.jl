#!/usr/bin/env julia
using CoarseGraining

function main()
    σx, σy = 1.2, 0.8
    nx, ny = 60, 48
    if CoarseGraining.mpi_rank() == 0
        g = Grid(nx, ny, 1.0, 1.0, true, true)
        f = Field(randn(g.ny, g.nx), g)
        out_full = coarse_grain_fft(f, σx, σy)
        out_dist = parallel_coarse_grain_fft_distributed(f, σx, σy)
        err = mean(abs.(out_full.data .- out_dist.data))
        @assert err < 1e-8
        println("OK: distributed FFT matches serial (mean abs err = ", err, ")")
    else
        parallel_coarse_grain_fft_distributed(nothing, σx, σy)
    end
    CoarseGraining.mpi_finalize()
end

main()

