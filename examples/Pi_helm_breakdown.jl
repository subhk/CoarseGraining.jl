#!/usr/bin/env julia
using CoarseGraining
using Statistics

"""
Compute Leonard energy transfer Π on a periodic Cartesian grid and print stats.
"""
function main()
    nx, ny = 256, 256
    g = Grid(nx, ny, 1.0, 1.0, true, true)
    x = range(0, stop=2π, length=g.nx)
    y = range(0, stop=2π, length=g.ny)
    u = Field([sin(xx) + 0.3cos(yy) for yy in y, xx in x], g)
    v = Field([cos(xx) - 0.3sin(yy) for yy in y, xx in x], g)
    K = gaussian_kernel(3.0, 3.0)
    Π = compute_pi(u, v, K)
    @info "Pi stats" mean=mean(Π.data) std=std(Π.data)

    # Optional: compare with Helmholtz filtered fields
    ū = coarse_grain(u, K)
    v̄ = coarse_grain(v, K)
    udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(ū, v̄)
    # Could compute Pi using only divergence-free part, if desired
end

main()

