# Quick Start

This example filters a random field on a periodic grid and computes basic diagnostics.

```julia
using CoarseGraining

# Grid and field
nx, ny = 256, 256
g = Grid(nx, ny, 1.0, 1.0, true, true)
f = Field(randn(ny, nx), g)

# Gaussian kernel and filtering
K = gaussian_kernel(3.0, 3.0)
f_real = coarse_grain(f, K; threaded=true)
f_fft  = coarse_grain_fft(f, 3.0, 3.0)

# Butterworth (spectral) with cutoff length ℓc = 16
f_bw = coarse_grain_butterworth_length(f, 16; order=2)

# Gradient and vorticity
fx, fy = gradient(f_real)
ζ = vorticity(fx, fy)

# Okubo–Weiss and KE spectrum
W = okuboweiss(fx, fy)
k, Ek = ke_spectrum_isotropic(fx, fy; nbins=32)
```

For mask–aware and MPI examples, see the corresponding pages.

