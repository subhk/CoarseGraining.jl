# Documentation Website Setup Guide

This guide helps you deploy the CoarseGraining.jl documentation to your GitHub Pages site at `subhk.github.io`.

## 🎯 Quick Setup

### 1. Repository Setup
Ensure your repository is at `https://github.com/subhk/CoarseGraining.jl`

### 2. GitHub Pages Configuration
1. Go to your repository → **Settings** → **Pages**
2. Under "Source", select **GitHub Actions**
3. This allows the automated workflow to deploy documentation

### 3. Required Secrets (Optional but Recommended)
For automatic deployment, set up a `DOCUMENTER_KEY`:

```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -C "documenter" -f documenter_key -N ""

# Add public key as deploy key
# Repository → Settings → Deploy keys → Add deploy key
# Title: "Documenter"
# Key: paste content of documenter_key.pub
# Check "Allow write access"

# Add private key as repository secret
# Repository → Settings → Secrets and variables → Actions → New repository secret
# Name: DOCUMENTER_KEY
# Value: paste content of documenter_key (private key)
```

### 4. Trigger Documentation Build
1. **Push to main branch**: Documentation builds automatically
2. **Manual trigger**: Go to Actions tab → Documentation workflow → Run workflow

## 📍 Website URLs

After setup, your documentation will be available at:
- **Stable**: https://subhk.github.io/CoarseGraining.jl/stable/
- **Development**: https://subhk.github.io/CoarseGraining.jl/dev/

## 🔧 Local Development

### Build Documentation Locally
```bash
# Install documentation dependencies
julia --project=docs -e 'using Pkg; Pkg.instantiate()'

# Build documentation
julia --project=docs docs/make.jl

# View locally (opens in browser)
cd docs/build && python -m http.server 8000
# Visit: http://localhost:8000
```

### Preview Changes
```bash
# After making changes to docs/src/*.md files
julia --project=docs docs/make.jl
```

## 📂 Documentation Structure

```
docs/
├── make.jl                    # Build configuration
├── Project.toml              # Documentation dependencies
└── src/
    ├── index.md              # Homepage
    ├── installation.md       # Installation guide
    ├── quickstart.md         # Quick start tutorial
    ├── theory.md             # Mathematical theory
    ├── filters.md            # Filtering methods
    ├── advanced_features.md  # NEW: Advanced FlowSieve features
    ├── spherical.md          # Spherical operations
    ├── diagnostics.md        # Diagnostic functions
    ├── mpi.md               # Parallel computing
    ├── io.md                # Input/output
    ├── regridding.md        # Grid operations
    ├── api.md               # API reference
    └── assets/
        └── custom.css       # Custom styling
```

## 🚀 GitHub Actions Workflow

The `.github/workflows/Documentation.yml` file automatically:
1. **Triggers** on pushes to `main` branch and pull requests
2. **Installs** Julia and package dependencies
3. **Builds** documentation using Documenter.jl
4. **Deploys** to GitHub Pages

### Manual Workflow Trigger
If needed, you can manually trigger documentation builds:
1. Go to your repository's **Actions** tab
2. Select **Documentation** workflow
3. Click **Run workflow**

## 🎨 Customization

### Site Branding
Edit `docs/make.jl`:
```julia
makedocs(
    sitename = "CoarseGraining.jl",  # Change site title
    authors = "Your Name",           # Change author
    # ... other settings
)
```

### Custom Styling
Edit `docs/src/assets/custom.css` for custom appearance.

### Add New Pages
1. Create new `.md` file in `docs/src/`
2. Add to `pages` array in `docs/make.jl`
3. Rebuild documentation

## 🔍 Troubleshooting

### Common Issues

**Documentation not updating:**
- Check GitHub Actions logs in the Actions tab
- Ensure GitHub Pages source is set to "GitHub Actions"
- Verify the workflow completed successfully

**Build errors:**
- Check that all examples in documentation run correctly
- Ensure all new functions are exported in the main module
- Verify markdown syntax is correct

**Missing dependencies:**
- Run `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
- Check that all required packages are in `docs/Project.toml`

### Debug Local Builds
```bash
# Run with verbose output
julia --project=docs -e '
    ENV["JULIA_DEBUG"] = "Documenter"
    include("docs/make.jl")
'
```

## 📚 Advanced Features Documented

The documentation now covers all advanced FlowSieve-equivalent features:

- ✅ **Spherical Helmholtz decomposition** with iterative solvers
- ✅ **Complete energy budget analysis** (transport, baroclinic, dissipation)
- ✅ **Sophisticated boundary handling** with land-avoiding stencils  
- ✅ **Multi-resolution workflows** for computational efficiency
- ✅ **Realistic ocean modeling** examples and workflows

## 🌊 Ready to Deploy!

Your documentation website will showcase the comprehensive FlowSieve-equivalent capabilities of CoarseGraining.jl for advanced ocean modeling and turbulence analysis.

**Next steps:**
1. Push changes to your `main` branch
2. Check GitHub Actions for successful build
3. Visit https://subhk.github.io/CoarseGraining.jl/stable/

Happy documenting! 🚀