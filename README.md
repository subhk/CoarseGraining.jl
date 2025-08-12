# CoarseGraining.jl

Julia implementation scaffolding for FlowSieve-style coarse graining of turbulent flows with optional MPI parallelism.

Features implemented:
- Core types: `Grid`, `Field`, `Kernel`.
- Real-space filters: Gaussian and boxcar kernels via `coarse_grain`.
- FFT-based Gaussian filtering via `coarse_grain_fft` (periodic) and `coarse_grain_gaussian_separable` (fast separable real-space).
  - `coarse_grain(...; threaded=true, tile=(bj,bi))`: threaded 2D kernel with optional cache tiling (defaults to `(64,64)` on small grids, auto-tunes on large).
  - `coarse_grain_gaussian_separable(...; threaded=true)`: multi-threaded 1D passes (x then y) with SIMD.
  - Tile helper: `select_tile(ny, nx, ry, rx)` returns a suggested `(bj,bi)`.
  - Butterworth low-pass: `coarse_grain_butterworth(field, kc; order=2)`, where `kc` is cutoff wavenumber (or `(kcx,kcy)`), assuming periodic boundaries.
  - Butterworth by length: `coarse_grain_butterworth_length(field, ℓc; order=2)`, where `ℓc` is cutoff length (or `(ℓx,ℓy)`).
  - Butterworth by cycles-per-domain: `coarse_grain_butterworth_cycles(field, cycles; order=2)`, where `cycles` is scalar or `(cx,cy)`.
  - Butterworth by Nyquist fraction: `coarse_grain_butterworth_nyquist(field, frac; order=2)`, where `frac` is scalar or `(fx,fy)` in (0,1].
  - DSP-based Butterworth (IIR, zero-phase): `coarse_grain_butterworth_dsp(field, kc; order=2, zero_phase=true)` and wrappers `*_length_dsp`, `*_cycles_dsp`, `*_cells_dsp`, `*_nyquist_dsp`.
- Basic differential operators: `gradient`, `divergence`, `vorticity` (Cartesian).
- NetCDF IO helpers: `load_netcdf_var`, `write_netcdf_field`.
  - Vector IO: `load_vector_vars`, `write_vector_vars`.
  - Region IO: `load_region_masks`, `write_region_stats`, `write_regions_file`.
  - One-shot: `write_region_stats_and_masks(field, regions, stats_path; masks_path, lon, lat)`.
  - With attributes: `write_region_stats_with_attrs`, `write_region_stats_and_masks_with_attrs`.
- MPI utilities: `mpi_init`, `parallel_coarse_grain` for domain-decomposed filtering along x; gather/compute/scatter paths for FFT filtering (`parallel_coarse_grain_fft`) and Helmholtz-Hodge (`parallel_helmholtz_hodge`).
  - Also: `parallel_coarse_grain_fft_distributed` for equal-slab, periodic, truly distributed FFT filtering (requires `nx % nprocs == 0` and `ny % nprocs == 0`).
  - Real-space MPI supports a threaded local filter via `parallel_coarse_grain(...; threaded=true)`.
- Spherical grid support: `SphericalGrid`, `gradient_sphere`, `vorticity_sphere`, and velocity conversions.
- Spherical Helmholtz–Hodge: `helmholtz_hodge_sphere` (FFT-in-lon + FD-in-lat baseline).
- Diagnostics: `okuboweiss`, `compute_pi` (Leonard transfer).
  - Extra diagnostics: `ke_spectrum_isotropic`, `zonal_mean_std`, `region_mean_std`.
  - `ke_spectrum_isotropic(...; normalize=:counts|:density|:shellarea|:energy, bins=:linear|:log)`: normalization options, linear or log bins; set `return_edges=true` for bin edges.

Quick start:
```
using CoarseGraining
g = Grid(256, 256, 1.0, 1.0, true, true)
A = randn(g.ny, g.nx)
f = Field(A, g)
K = gaussian_kernel(2.0, 2.0)
fg = coarse_grain(f, K)
```

Benchmark and tile auto-tuning:
```
# Tune tile and compare methods (env overrides: NX, NY, SIGX, SIGY)
julia --project benchmark/tile_autotune.jl
```

Recommend a tile and optionally write a config:
```
# Default nx,ny=1024,1024; σx,σy=3.0
julia --project examples/recommend_tile.jl --nx 2048 --ny 1024 --sigx 2.5 --sigy 2.5 --write CoarseGraining.toml
```

MPI usage (run with `mpiexec -n 4 julia --project -e 'using CoarseGraining; ...'`):
```
using CoarseGraining
K = gaussian_kernel(3.0, 3.0)
if CoarseGraining.mpi_rank() == 0
   g = Grid(1024, 1024, 1.0, 1.0, true, true)
   f = Field(randn(g.ny, g.nx), g)
   out = parallel_coarse_grain(f, K)
   @info size(out.data)
else
   parallel_coarse_grain(nothing, K)
end
CoarseGraining.mpi_finalize()
```

Distributed FFT Gaussian (equal slab):
```
using CoarseGraining
Kσx, Kσy = 3.0, 3.0
if CoarseGraining.mpi_rank() == 0
   g = Grid(1024, 1024, 1.0, 1.0, true, true)
   f = Field(randn(g.ny, g.nx), g)
   out = parallel_coarse_grain_fft_distributed(f, Kσx, Kσy)
   @info "done" size(out.data)
else
   parallel_coarse_grain_fft_distributed(nothing, Kσx, Kσy)
end
CoarseGraining.mpi_finalize()
```

MPI smoke tests (set `RUN_MPI=true` to run within `Pkg.test`):
- Real-space filter: `mpiexec -n 3 julia --project test_mpi/runtest_filter_realspace.jl`
- Distributed FFT: `mpiexec -n 4 julia --project test_mpi/runtest_fft_distributed.jl`
- Compare real-space MPI vs serial: `mpiexec -n 5 julia --project test_mpi/runtest_realspace_compare.jl`
- Compare distributed FFT vs serial: `mpiexec -n 6 julia --project test_mpi/runtest_fft_compare.jl`

Spherical gradients:
```
using CoarseGraining
nx, ny = 360, 180
lon = range(0, stop=2π, length=nx)
lat = range(-π/2 + π/ny, stop=π/2 - π/ny, length=ny)
sg = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
f = Field([cos(λ)*cos(ϕ) for ϕ in lat, λ in lon], sg)
gx, gy = gradient_sphere(f)
```

Helmholtz–Hodge (periodic Cartesian):
```
See examples:
- `examples/coarse_grain_helmholtz.jl`
- `examples/helmholtz_workflow.jl`
- `examples/coarse_grain_scalars.jl`
- `examples/Pi_helm_breakdown.jl`
g = Grid(128, 128, 1.0, 1.0, true, true)
u = Field(randn(g.ny, g.nx), g)
v = Field(randn(g.ny, g.nx), g)
udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(u, v)
```

Region stats + masks with attributes:
```
using CoarseGraining
# Assume f is Field and regions is Dict{String,BitArray{2}}
stats_attrs = Dict("filter" => "gaussian", "sigma" => 3.0)
masks_attrs = Dict("source" => "coastlines", "version" => "v1")
write_region_stats_and_masks_with_attrs(
    f, regions,
    "region_stats.nc";
    masks_path="region_masks.nc", lon=1:f.grid.nx, lat=1:f.grid.ny,
    stats_attrs=stats_attrs, masks_attrs=masks_attrs,
)
```

Note: This is an initial scaffold. Additional FlowSieve features (spherical grids, Helmholtz projections, advanced IO and postprocessing) can be layered on with the current modular structure.
This build already includes spherical gradients, a periodic FFT-based Helmholtz–Hodge, and MPI gather/compute/scatter wrappers; true distributed FFTs can be added later.
