# CoarseGraining.jl

Julia implementation scaffolding for FlowSieve-style coarse graining of turbulent flows with optional MPI parallelism.

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

## References
- Germano, M. (1992). Turbulence: the filtering approach. Journal of Fluid Mechanics, 238, 325–336. doi:10.1017/S0022112092001733
- Leonard, A. (1974). Energy Cascade in Large-Eddy Simulations of Turbulent Fluid Flows. Advances in Geophysics, 18A, 237–248. doi:10.1016/S0065-2687(08)60464-1
- Eyink, G. L. (2005). Locality of turbulent cascades. Physical Review E, 72(6), 066302. doi:10.1103/PhysRevE.72.066302
- Aluie, H. (2019). Coarse-grained incompressible MHD: inviscid invariants and k−4 spectrum. Physical Review Fluids, 4, 114603. doi:10.1103/PhysRevFluids.4.114603
- Helmholtz–Hodge decomposition: Chorin, A. J., & Marsden, J. E. (1993). A Mathematical Introduction to Fluid Mechanics (3rd ed.). Springer.
- Butterworth filtering: Butterworth, S. (1930). On the theory of filter amplifiers. Wireless Engineer, 7, 536–541.
- FlowSieve (coarse-graining toolkit): https://github.com/janinejanoski/FlowSieve (original C++ reference implementation and documentation)
