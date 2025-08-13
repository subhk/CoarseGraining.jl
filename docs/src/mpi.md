# MPI & Parallel

CoarseGraining.jl supports domain-decomposed real–space filtering and distributed FFTs.

## Real–space filtering with halos
```bash
mpiexec -n 4 julia --project -e '
using CoarseGraining
nx, ny = 1024, 1024; g = Grid(nx, ny, 1.0, 1.0, true, true)
f = Field(randn(ny, nx), g)
K = gaussian_kernel(3.0, 3.0)
out = parallel_coarse_grain(f, K; halosize=K.radius_x, threaded=true)
if mpi_rank() == 0; @show size(out.data); end
mpi_finalize()'
```

## Mask–aware MPI filtering
```bash
mpiexec -n 4 julia --project -e '
using CoarseGraining
nx, ny = 512, 512; g = Grid(nx, ny, 1.0, 1.0, true, true)
f = Field(randn(ny, nx), g)
mask = trues(ny, nx); mask[:, 1:8] .= false
K = gaussian_kernel(2.0, 2.0)
out = parallel_coarse_grain_masked(f, K, mask)
if mpi_rank() == 0; @show size(out.data); end
mpi_finalize()'
```

## Distributed FFT Gaussian (gather/scatter and fully distributed)
- Gather/compute/scatter:
```bash
mpiexec -n 4 julia --project -e '
using CoarseGraining
nx, ny = 512, 512; g = Grid(nx, ny, 1.0, 1.0, true, true)
f = Field(randn(ny, nx), g)
out = parallel_coarse_grain_fft(f, 2.0, 2.0)
mpi_finalize()'
```
- Fully distributed (Alltoallv):
```bash
mpiexec -n 4 julia --project -e '
using CoarseGraining
nx, ny = 512, 512; g = Grid(nx, ny, 1.0, 1.0, true, true)
f = Field(randn(ny, nx), g)
out = parallel_coarse_grain_fft_distributed(f, 2.0, 2.0)
mpi_finalize()'
```

