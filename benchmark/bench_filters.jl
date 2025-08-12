#!/usr/bin/env julia
using CoarseGraining
using Printf

function bench()
    nx, ny = 1024, 1024
    g = Grid(nx, ny, 1.0, 1.0, true, true)
    f = Field(randn(ny, nx), g)
    K = gaussian_kernel(3.0, 3.0)
    t1 = @elapsed coarse_grain(f, K)
    @printf("Real-space coarse_grain: %.3f s\n", t1)
    t2 = @elapsed coarse_grain_fft(f, 2.0, 2.0)
    @printf("FFT Gaussian coarse_grain_fft: %.3f s\n", t2)
end

bench()

