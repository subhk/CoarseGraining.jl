# Theory Overview

This section summarizes the underlying ideas behind coarse–graining and related diagnostics, including advanced energy budget analysis.

## Filters
A filter of width ℓ smooths a field f to produce a coarse–grained field f̄.

- **Real–space convolution**:
  f̄(x) = ∫ G_ℓ(x − y) f(y) dy, where G_ℓ is a normalized kernel (e.g., Gaussian).
- **Spectral multiplication**:
  f̂̄(k) = H_ℓ(k) f̂(k), with transfer function H_ℓ (e.g., Butterworth).

Properties to consider:
- Support and smoothness of G_ℓ
- Commutation with derivatives (e.g., under periodicity and constant spacing)
- Boundary conditions (periodic vs clamped) and masking
- **Physical distance** for spherical/curvilinear grids

## Helmholtz–Hodge Decomposition

A 2D vector field **u** = (u, v) can be decomposed into:
- **Divergence-free part**: **u**_div = ∇ × ψ ẑ (rotational, ∇ · **u**_div = 0)
- **Potential part**: **u**_pot = ∇φ (irrotational, ∇ × **u**_pot = 0)

### Cartesian (Periodic)
Use FFT to solve Poisson equations:
- ∇²φ = ∇ · **u** → **u**_pot = ∇φ
- ∇²ψ = (∇ × **u**) · ẑ → **u**_div = ∇ × ψ ẑ

### Spherical (Non-Periodic)
Use iterative solvers for spherical Laplacian:
- ∇²φ = ∇ · **u** where ∇² includes metric factors (cosφ terms)
- Sparse matrix representation for memory efficiency
- Multi-grid acceleration for large problems

## Complete Energy Budget

Following Aluie et al. (2018), the kinetic energy equation is:

**∂/∂t(½ρ₀|ū|²) + ∇ · J = -Π - ε + Bc + F**

### Terms:

1. **Leonard Transfer (Π)**:
   Π = -τᵢⱼSᵢⱼ = -½(∂ūᵢ/∂xⱼ + ∂ūⱼ/∂xᵢ)(overline(uᵢuⱼ) - ūᵢūⱼ)
   - Measures scale-to-scale energy transfer
   - Π < 0: upscale transfer (inverse cascade)
   - Π > 0: downscale transfer (forward cascade)

2. **Transport Divergence (∇ · J)**:
   - **Advection by coarse scales**: ∇ · (½ρ₀|ū|²ū)
   - **Pressure transport**: ∇ · (P̄ū) 
   - **Subgrid advection**: ∇ · (ρ₀ū · τ)

3. **Viscous Dissipation (ε)**:
   - **Resolved**: ε_res = 2ρ₀νSᵢⱼSᵢⱼ
   - **Subgrid**: ε_sgs = 2ρ₀ν_sgs|S̄|²

4. **Baroclinic Conversion (Bc)**:
   Bc = -g⟨w'ρ'⟩ (available potential energy → kinetic energy)

5. **External Forcing (F)**:
   F = ρ₀F̄ · ū (work by external forces)

## Multi-Resolution Methods

For computational efficiency on large grids:

### Strategy
1. **Coarsen** to manageable resolution (conservative averaging)
2. **Solve** on coarse grid (fast)
3. **Refine** solution to target resolution (interpolation)  
4. **Correct** using coarse solution as initial guess
5. **Repeat** until convergence

### Benefits
- Reduces computational cost by ~10-100x
- Maintains solution accuracy
- Enables analysis of high-resolution models

### Theory
Multi-grid methods exploit the fact that:
- Large-scale features converge quickly on coarse grids
- Small-scale features need fine grids but benefit from good initial guesses
- Hierarchical approach captures all scales efficiently

## Sophisticated Boundary Handling

### Land-Avoiding Stencils
For coastal regions, avoid contamination from land values:
- **Adaptive integration bounds** based on physical distance
- **Kernel deformation** around land boundaries  
- **Normalization** by effective water area only

### Distance Functions
- **Spherical**: Great circle distance using Haversine formula
- **Curvilinear**: Local metric approximation
- **Cartesian**: Euclidean distance

### Boundary Conditions
- **Periodic**: Wrap-around for global domains
- **Clamped**: Extend edge values for regional domains
- **Masked**: Zero land, normalize by water area
- **Adaptive**: Physical distance-based inclusion

## Spherical and Curvilinear Effects

### Spherical Coordinates
Gradients include metric factors:
- ∇f = (1/(a cosφ) ∂f/∂λ, 1/a ∂f/∂φ)
- Laplacian: ∇²f = 1/(a² cosφ)[∂/∂φ(cosφ ∂f/∂φ) + 1/cosφ ∂²f/∂λ²]

### Curvilinear Grids  
Use local metrics dx(i,j), dy(i,j) for finite differences.

### Practical Considerations
- **Real-space filters** work on any grid type
- **FFT methods** require uniform spacing → prefer regridding
- **Spherical operators** essential for realistic ocean domains

## Oceanic Applications

### Scale Separation
Typical ocean scales:
- **Mesoscale eddies**: 50-200 km (most energetic)
- **Submesoscale**: 1-10 km (rapid evolution)
- **Large-scale circulation**: 1000+ km (slow, persistent)

### Energy Pathways
- **Baroclinic instability**: Available PE → mesoscale KE
- **Eddy saturation**: Mesoscale → large-scale transfer
- **Submesoscale**: Forward cascade to dissipation

### Boundary Effects
- **Coastlines**: Complex geometry requires adaptive filtering
- **Topography**: Influences energy pathways
- **Open boundaries**: Need careful treatment in regional models

## References

- **Aluie et al. (2018)**: "Mapping the energy cascade in the North Atlantic Ocean" - *J. Phys. Oceanogr.*
- **Storer et al. (2022)**: "Global energy spectrum of the general oceanic circulation" - *Nature Communications*
- **Germano (1992)**: "Turbulence: the filtering approach" - *J. Fluid Mech.*
- **Leonard (1974)**: "Energy Cascade in Large-Eddy Simulations" - *Adv. Geophys.*
- **Original FlowSieve**: Aluie & Eyink methodology implementation

