# CoarseGraining.jl

Welcome! CoarseGraining.jl provides practical, efficient tools for coarse–graining turbulent and geophysical flows in Julia — from serial and threaded filtering to MPI domain decomposition, spherical/curvilinear operators, diagnostics, and model IO adapters (CROCO/ROMS/MITgcm).

This guide starts with a brief theory tour, then shows how to use the package with clear code examples. You do not need prior experience with coarse–graining to follow along.

- If you are new, read Theory → Quick Start → Filters.
- If you have model data, jump to IO & Models.
- For scaling out, see MPI & Parallel.

## What is coarse–graining?

Coarse–graining constructs a scale–dependent representation of a flow by convolving fields with a smoothing kernel of width ℓ (or cutoff wavenumber k_c):

- Real–space: f̄(x) = ∫ G_ℓ(x − y) f(y) dy
- Spectral: f̂̄(k) = H_ℓ(k) f̂(k)

Common choices:
- Gaussian (real–space or spectral)
- Boxcar
- Butterworth (spectral/IIR)

Applications:
- Separate “large–scale” and “small–scale” contributions
- Study scale–to–scale energy transfer Π
- Build scale–aware diagnostics and maps (zonal stats, regions)

See the Theory page for details, including Helmholtz–Hodge decomposition and the Leonard transfer Π.

## Key features

- Serial + threaded filters (Gaussian, Butterworth, boxcar)
- Mask–aware filtering for land/sea and missing data
- FFT and DSP engines for spectral filtering
- MPI paths (halo exchange) and distributed FFT
- Spherical and curvilinear operators
- Diagnostics (Okubo–Weiss, Π) and spectra
- NetCDF IO helpers + model adapters (CROCO/ROMS/MITgcm)
- Regridding utilities (index bilinear, lon/lat nearest)

## Next steps

- Installation: how to add and build docs
- Quick Start: a first end–to–end example
- Filters: pick the right filter and engine
- IO & Models: read model outputs the simple way
- MPI & Parallel: run on many cores and nodes

