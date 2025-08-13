using Documenter
using CoarseGraining

DocMeta.setdocmeta!(CoarseGraining, :DocTestSetup, :(using CoarseGraining); recursive=true)

makedocs(
    sitename = "CoarseGraining.jl",
    modules = [CoarseGraining],
    format = Documenter.HTML(assets=["assets/custom.css"], prettyurls=get(ENV, "CI", "false") == "true"),
    authors = "CoarseGraining.jl Contributors",
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick Start" => "quickstart.md",
        "Theory" => "theory.md",
        "Filters" => "filters.md",
        "MPI & Parallel" => "mpi.md",
        "IO & Models" => "io.md",
        "Regridding" => "regridding.md",
        "Spherical & Curvilinear" => "spherical.md",
        "Diagnostics" => "diagnostics.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = get(ENV, "DOCS_REPO", ""), # set e.g. "github.com:USER/CoarseGraining.jl.git"
    devbranch = "main",
    push_preview = true,
)

