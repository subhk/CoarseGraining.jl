#!/usr/bin/env julia
using CoarseGraining
using Printf
using Random

function time_filter(f::Function)
    # Warmup
    f()
    t = @elapsed f()
    return t
end

function main()
    Random.seed!(123)
    nx = parse(Int, get(ENV, "NX", "1024"))
    ny = parse(Int, get(ENV, "NY", "1024"))
    dx = parse(Float64, get(ENV, "DX", "1.0"))
    dy = parse(Float64, get(ENV, "DY", "1.0"))
    σx = parse(Float64, get(ENV, "SIGX", "3.0"))
    σy = parse(Float64, get(ENV, "SIGY", "3.0"))
    g = Grid(nx, ny, dx, dy, true, true)
    fld = Field(randn(ny, nx), g)
    K = gaussian_kernel(σx, σy)

    println("== Autotune tiles for coarse_grain ==")
    suggested = select_tile(fld, K)
    @printf("Suggested tile: (%d,%d)\n", suggested...)

    candidates = [(0,0), (32,32), (64,64), (128,64), (64,128), (128,128), suggested]
    seen = Set{Tuple{Int,Int}}()
    best_t = Inf
    best_tile = (0,0)
    for tile in candidates
        tile in seen && continue
        push!(seen, tile)
        t = time_filter(() -> coarse_grain(fld, K; threaded=true, tile=tile))
        @printf("tile=(%d,%d) time=%.4f s\n", tile..., t)
        if t < best_t
            best_t = t
            best_tile = tile
        end
    end
    @printf("Best tile: (%d,%d) time=%.4f s\n", best_tile..., best_t)

    println("\n== Compare methods ==")
    t_sep = time_filter(() -> coarse_grain_gaussian_separable(fld, σx, σy; threaded=true))
    t_fft = time_filter(() -> coarse_grain_fft(fld, σx, σy))
    t_2d  = time_filter(() -> coarse_grain(fld, K; threaded=true, tile=best_tile))
    @printf("separable: %.4f s, fft: %.4f s, 2d-kernel: %.4f s\n", t_sep, t_fft, t_2d)
end

main()

