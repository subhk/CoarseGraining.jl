#!/usr/bin/env julia
using CoarseGraining

# Example: coarse-grain a synthetic velocity field and compute Helmholtz-Hodge

nx, ny = 256, 256
g = Grid(nx, ny, 1.0, 1.0, true, true)
u = Field(randn(ny, nx), g)
v = Field(randn(ny, nx), g)

# Coarse-grain velocities
K = gaussian_kernel(2.0, 2.0)
ū = coarse_grain(u, K)
v̄ = coarse_grain(v, K)

# Helmholtz decomposition
udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(ū, v̄)

@info "Done" size(ū.data) size(udf.data)

