# Installation

CoarseGraining.jl is a standard Julia package. If you are working from this repository, activate its environment and instantiate dependencies.

## Prerequisites
- Julia 1.9+
- For MPI features: an MPI implementation (e.g., OpenMPI) and MPI.jl configured
- For NetCDF IO: NCDatasets.jl (installed by the package)

## Using this repo
```julia
julia --project
using Pkg
Pkg.instantiate()
```

## Basic usage
```julia
using CoarseGraining
```

## Building the docs locally
```bash
julia --project=docs docs/make.jl
```

To deploy to GitHub Pages, set the `DOCS_REPO` environment variable to your repo (e.g., `github.com:USER/CoarseGraining.jl.git`) and run `docs/make.jl` on CI.

