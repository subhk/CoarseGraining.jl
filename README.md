# CoarseGraining.jl

Julia implementation scaffolding for FlowSieve-style coarse graining of turbulent flows with optional MPI parallelism.

Features implemented:
- Core types: `Grid`, `Field`, `Kernel`.
- Real-space filters: Gaussian and boxcar kernels via `coarse_grain`.
- FFT-based Gaussian filtering via `coarse_grain_fft` (periodic).
- Basic differential operators: `gradient`, `divergence`, `vorticity` (Cartesian).
- NetCDF IO helpers: `load_netcdf_var`, `write_netcdf_field`.
- MPI utilities: `mpi_init`, `parallel_coarse_grain` for domain-decomposed filtering along x; gather/compute/scatter paths for FFT filtering (`parallel_coarse_grain_fft`) and Helmholtz-Hodge (`parallel_helmholtz_hodge`).
  - Also: `parallel_coarse_grain_fft_distributed` for equal-slab, periodic, truly distributed FFT filtering (requires `nx % nprocs == 0` and `ny % nprocs == 0`).
- Spherical grid support: `SphericalGrid`, `gradient_sphere`, `vorticity_sphere`, and velocity conversions.
- Diagnostics: `okuboweiss`.

Quick start:
```
using CoarseGraining
g = Grid(256, 256, 1.0, 1.0, true, true)
A = randn(g.ny, g.nx)
f = Field(A, g)
K = gaussian_kernel(2.0, 2.0)
fg = coarse_grain(f, K)
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
g = Grid(128, 128, 1.0, 1.0, true, true)
u = Field(randn(g.ny, g.nx), g)
v = Field(randn(g.ny, g.nx), g)
udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(u, v)
```

Note: This is an initial scaffold. Additional FlowSieve features (spherical grids, Helmholtz projections, advanced IO and postprocessing) can be layered on with the current modular structure.
This build already includes spherical gradients, a periodic FFT-based Helmholtz–Hodge, and MPI gather/compute/scatter wrappers; true distributed FFTs can be added later.
