using Documenter
using CoarseGraining

DocMeta.setdocmeta!(CoarseGraining, :DocTestSetup, :(using CoarseGraining); recursive=true)

makedocs(
    sitename = "CoarseGraining.jl",
    modules = [CoarseGraining],
    format = Documenter.HTML(
        assets=["assets/custom.css"], 
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://subhk.github.io/CoarseGraining.jl/stable/",
        edit_link="main"
    ),
    authors = "CoarseGraining.jl Contributors",
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md",
        "Theory" => "theory.md",
        "Filters" => "filters.md",
        "Advanced Features" => "advanced_features.md",
        "MPI & Parallel" => "mpi.md",
        "IO & Models" => "io.md",
        "Regridding" => "regridding.md",
        "Spherical & Curvilinear" => "spherical.md",
        "Diagnostics" => "diagnostics.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/subhk/CoarseGraining.jl.git",
    devbranch = "main",
    push_preview = true,
    target = "build",
    deps = nothing,
    make = nothing,
)

