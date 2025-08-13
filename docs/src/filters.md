# Filters

CoarseGraining.jl provides several filtering paths. Choose based on domain, boundaries, and performance needs.

## Gaussian (real–space)
```julia
K = gaussian_kernel(σx, σy; truncate=3.0)
f_sm = coarse_grain(f, K; threaded=true, tile=(64,64))
```
- Works with periodic or clamped edges.
- `threaded` and `tile` improve performance.

## Gaussian (FFT)
```julia
f_fft = coarse_grain_fft(f, σx, σy)  # periodic
```
- Periodic boundaries, uniform spacing.

## Butterworth (spectral)
```julia
# Cutoff specified by length ℓc or wavenumber kc
f_bw = coarse_grain_butterworth_length(f, 16; order=2)
# DSP-based (IIR) zero-phase filtering
f_bw_dsp = coarse_grain_butterworth_dsp(f, 2π/16; order=2, zero_phase=true)
```

## Mask–aware filtering
```julia
mask = trues(size(f.data))
mask[:, 1:5] .= false
f_masked = coarse_grain_masked(f, K, mask)
```

### Masked separable Gaussian (fast)
```julia
f_msep = coarse_grain_gaussian_separable_masked(f, 3.0, 3.0, mask; threaded=true)
```

## Tips
- Prefer separable Gaussian for speed; use tiled real–space for general kernels.
- Use Butterworth for sharper spectral cutoffs; DSP version avoids phase shift.
- For curvilinear/spherical grids, prefer real–space filters.

