# Theory Overview

This section summarizes the underlying ideas behind coarse–graining and related diagnostics.

## Filters
A filter of width ℓ smooths a field f to produce a coarse–grained field f̄.

- Real–space convolution:
  f̄(x) = ∫ G_ℓ(x − y) f(y) dy, where G_ℓ is a normalized kernel (e.g., Gaussian).
- Spectral multiplication:
  f̂̄(k) = H_ℓ(k) f̂(k), with transfer function H_ℓ (e.g., Butterworth).

Properties to consider:
- Support and smoothness of G_ℓ
- Commutation with derivatives (e.g., under periodicity and constant spacing)
- Boundary conditions (periodic vs clamped) and masking

## Helmholtz–Hodge Decomposition

A 2D vector field u = (u, v) can be decomposed into a divergence–free part and a potential (irrotational) part.

- In periodic Cartesian domains, compute via FFT:
  Solve Poisson problems for scalar potentials φ and ψ from divergence and vorticity, then reconstruct components.

This helps interpret fluxes and energy pathways at scale.

## Energy Transfer Π (Leonard term)

For resolved (filtered) fields (ū, v̄), define subfilter stress τ_ij = overline(u_i u_j) − ū_i ū_j and resolved strain S_ij.

- Leonard transfer Π = − τ_ij S_ij measures scale–to–scale energy transfer.
- Sign of Π indicates local forward/backward transfer.

## Spherical and Curvilinear Effects

On the sphere or curvilinear grids, derivatives must include metric terms. This package provides:
- Spherical gradients and vorticity (lat/lon grid)
- Curvilinear gradients using per–cell dx, dy approximations

For uniform spacing assumptions (FFT), prefer regridding to uniform grids when appropriate.

