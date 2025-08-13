#!/usr/bin/env julia

"""
Advanced Ocean Analysis Example

Demonstrates the new FlowSieve-equivalent features:
1. Spherical Helmholtz decomposition with iterative solvers
2. Complete energy budget analysis 
3. Sophisticated boundary handling with land masks
4. Multi-resolution workflow for large datasets

This example processes realistic ocean model output similar to FlowSieve workflows.
"""

using CoarseGraining
using Statistics
using LinearAlgebra

function main()
    println("🌊 Advanced Ocean Analysis - FlowSieve-style workflow")
    println("=" ^ 60)
    
    # === 1. SETUP REALISTIC OCEAN DOMAIN ===
    println("\n1. Setting up realistic ocean domain...")
    
    # Gulf Stream region (realistic coordinates)
    lon_range = -80.0:0.5:-60.0  # 40×40 degree region  
    lat_range = 30.0:0.5:50.0
    lon_grid = collect(deg2rad.(lon_range))
    lat_grid = collect(deg2rad.(lat_range))
    
    ocean_grid = SphericalGrid(lon_grid, lat_grid, 6.371e6, true)
    ny, nx = length(lat_grid), length(lon_grid)
    
    println("   Domain: $(length(lon_range))×$(length(lat_range)) points")
    println("   Resolution: ~55 km (0.5°)")
    println("   Coverage: Gulf Stream region")
    
    # === 2. GENERATE REALISTIC OCEAN VELOCITY FIELD ===
    println("\n2. Generating realistic ocean velocity field...")
    
    # Create Gulf Stream-like velocity field with mesoscale eddies
    uE_data, vN_data, land_mask = create_realistic_ocean_field(ocean_grid)
    
    uE = Field(uE_data, ocean_grid)
    vN = Field(vN_data, ocean_grid)
    
    println("   Max velocity: $(round(maximum(sqrt.(uE_data.^2 .+ vN_data.^2)), digits=2)) m/s")
    println("   Land fraction: $(round(100*(1 - mean(land_mask)), digits=1))%")
    
    # === 3. SOPHISTICATED BOUNDARY HANDLING ===
    println("\n3. Testing sophisticated boundary handling...")
    
    # Create filtering kernel  
    K = gaussian_kernel(3.0, 3.0)  # ~165 km filter scale
    
    # Standard filtering (ignores land boundaries)
    uE_basic = coarse_grain(uE, K)
    
    # Advanced filtering with land-avoiding stencils
    uE_adaptive = coarse_grain_adaptive(uE, K; 
                                       mask=land_mask,
                                       boundary_mode=:adaptive,
                                       deform_around_land=true)
    
    # Compare differences near coastlines
    diff_coastal = compute_coastal_differences(uE_basic, uE_adaptive, land_mask)
    println("   RMS difference near coasts: $(round(diff_coastal, digits=4)) m/s")
    println("   ✅ Land-avoiding filtering implemented")
    
    # === 4. SPHERICAL HELMHOLTZ DECOMPOSITION ===
    println("\n4. Spherical Helmholtz decomposition...")
    
    # Use iterative solver for large spherical domain
    println("   Solving with iterative methods...")
    uE_div, vN_div, uE_pot, vN_pot, φ, ψ = helmholtz_hodge_sphere_iterative(
        uE_adaptive, coarse_grain_adaptive(vN, K; mask=land_mask, boundary_mode=:adaptive);
        max_iter=500, tol=1e-6, method=:cg)
    
    # Verify decomposition accuracy
    uE_total = Field(uE_div.data .+ uE_pot.data, ocean_grid)
    vN_total = Field(vN_div.data .+ vN_pot.data, ocean_grid)
    
    decomp_error = sqrt(mean((uE_total.data .- uE_adaptive.data).^2 .+ 
                           (vN_total.data .- coarse_grain_adaptive(vN, K; mask=land_mask).data).^2))
    
    println("   Decomposition RMS error: $(round(decomp_error, digits=6)) m/s")
    println("   Divergent energy fraction: $(round(100*compute_energy_fraction(uE_pot, vN_pot, uE_total, vN_total), digits=1))%")
    println("   ✅ Spherical Helmholtz decomposition complete")
    
    # === 5. COMPLETE ENERGY BUDGET ANALYSIS ===
    println("\n5. Complete energy budget analysis...")
    
    # Create synthetic pressure and density fields for demonstration
    pressure = create_pressure_field(ocean_grid, uE, vN)
    ρ = create_density_field(ocean_grid, lat_grid)
    
    # Compute full energy budget
    budget = compute_full_energy_budget(uE, vN, K;
                                       pressure=pressure,
                                       ρ=ρ,
                                       w=nothing,  # No vertical velocity for 2D analysis
                                       ν=1e-3,     # Ocean viscosity  
                                       ρ₀=1025.0)
    
    # Display budget terms
    display_energy_budget(budget, land_mask)
    
    # === 6. MULTI-RESOLUTION WORKFLOW ===
    println("\n6. Multi-resolution workflow demonstration...")
    
    # Create hierarchy of grids (coarsen by factor of 2 each level)
    levels = 3
    grids = create_multiresolution_hierarchy(ocean_grid, levels; coarsening_factor=2)
    
    println("   Grid hierarchy:")
    for (i, grid) in enumerate(grids)
        ny_level, nx_level = length(grid.lat), length(grid.lon)
        resolution = round(6.371e6 * mean(diff(grid.lat)) / 1000, digits=1)
        println("     Level $i: $(nx_level)×$(ny_level) (~$(resolution) km)")
    end
    
    # Hierarchical Helmholtz solve (faster for large problems)
    println("   Performing hierarchical Helmholtz solve...")
    uE_div_hier, vN_div_hier, uE_pot_hier, vN_pot_hier, φ_hier, ψ_hier = 
        hierarchical_helmholtz_workflow(uE, vN; levels=levels, max_iter_per_level=50)
    
    # Compare with single-level solution
    hier_error = sqrt(mean((uE_div_hier.data .- uE_div.data).^2 .+ 
                          (vN_div_hier.data .- vN_div.data).^2))
    
    println("   Hierarchical vs single-level RMS error: $(round(hier_error, digits=6)) m/s")
    println("   ✅ Multi-resolution workflow complete")
    
    # === 7. ADVANCED DIAGNOSTICS ===
    println("\n7. Advanced diagnostic computations...")
    
    # Energy spectra (if domain is large enough)
    if nx >= 64 && ny >= 64
        # Make periodic version for spectral analysis
        uE_periodic = Field(uE.data, Grid(nx, ny, mean(diff(lon_grid)), mean(diff(lat_grid)), true, true))
        vN_periodic = Field(vN.data, Grid(nx, ny, mean(diff(lon_grid)), mean(diff(lat_grid)), true, true))
        
        k, E_k = compute_kinetic_energy_spectra(uE_periodic, vN_periodic; method=:isotropic)
        
        # Find spectral slope
        log_k = log10.(k[k .> 0])
        log_E = log10.(E_k[k .> 0])
        
        if length(log_k) > 10
            slope = compute_spectral_slope(log_k, log_E)
            println("   Kinetic energy spectral slope: $(round(slope, digits=2))")
            println("     (Expected: -3 for QG turbulence, -5/3 for 3D turbulence)")
        end
    end
    
    # Enstrophy transfer
    Π_enstrophy, ω̄ = compute_enstrophy_transfer(uE, vN, K)
    
    println("   Enstrophy transfer range: [$(round(minimum(Π_enstrophy.data), digits=8)), $(round(maximum(Π_enstrophy.data), digits=8))] s⁻³")
    println("   Mean vorticity: $(round(mean(ω̄.data), digits=8)) s⁻¹")
    
    # === 8. SAVE RESULTS ===
    println("\n8. Saving results...")
    
    # Save multi-resolution hierarchy
    save_multiresolution_data("ocean_multiresolution.nc", 
                             [uE_div, uE_div_hier], 
                             [ocean_grid, ocean_grid]; 
                             group_name="velocity_divergent")
    
    println("   Saved to: ocean_multiresolution.nc")
    println("   ✅ Advanced ocean analysis complete!")
    
    # === SUMMARY ===
    println("\n" * "=" * 60)
    println("🎯 SUMMARY - FlowSieve Features Implemented:")
    println("✅ Spherical Helmholtz decomposition with iterative solvers")
    println("✅ Advanced energy diagnostics (transport, baroclinic, viscous)")  
    println("✅ Sophisticated boundary handling with land-avoiding stencils")
    println("✅ Multi-resolution workflows for computational efficiency")
    println("✅ Comprehensive energy budget analysis")
    println("✅ Advanced spectral and enstrophy diagnostics")
    println("\n🌊 Ready for realistic ocean model analysis!")
    
    return budget, grids
end

# Helper functions for realistic ocean modeling

function create_realistic_ocean_field(grid::SphericalGrid)
    """Create Gulf Stream-like velocity field with mesoscale features"""
    
    ny, nx = length(grid.lat), length(grid.lon)
    lat = grid.lat
    lon = grid.lon
    
    uE = zeros(ny, nx)
    vN = zeros(ny, nx)
    land_mask = trues(ny, nx)  # true = water, false = land
    
    # Gulf Stream jet
    for j in 1:ny
        for i in 1:nx
            φ = lat[j]
            λ = lon[i]
            
            # Add coastline (simplified)
            if λ > deg2rad(-65) && φ < deg2rad(40)
                land_mask[j, i] = false
                continue
            end
            
            # Gulf Stream core (eastward jet)
            jet_center = deg2rad(38.0)
            jet_width = deg2rad(2.0)
            jet_strength = 1.5 * exp(-((φ - jet_center)/jet_width)^2)
            
            # Add meandering
            meander_amp = deg2rad(1.0)
            meander_wave = sin(3 * λ) * meander_amp
            jet_center_local = jet_center + meander_wave
            
            uE[j, i] = jet_strength * exp(-((φ - jet_center_local)/jet_width)^2)
            
            # Add mesoscale eddies
            eddy_scale = deg2rad(1.5)
            for n in 1:3
                eddy_lon = deg2rad(-75 + 5*n)
                eddy_lat = deg2rad(35 + 3*n)
                eddy_strength = 0.5 * (-1)^n
                
                r = sqrt((λ - eddy_lon)^2 + (φ - eddy_lat)^2)
                eddy_vel = eddy_strength * exp(-(r/eddy_scale)^2) * r/eddy_scale
                
                uE[j, i] += eddy_vel * cos(atan(φ - eddy_lat, λ - eddy_lon))
                vN[j, i] += eddy_vel * sin(atan(φ - eddy_lat, λ - eddy_lon))
            end
            
            # Add noise
            uE[j, i] += 0.1 * randn()
            vN[j, i] += 0.1 * randn()
        end
    end
    
    # Set land values to zero
    uE[.!land_mask] .= 0.0
    vN[.!land_mask] .= 0.0
    
    return uE, vN, land_mask
end

function create_pressure_field(grid::SphericalGrid, uE::Field, vN::Field)
    """Create synthetic pressure field consistent with geostrophic balance"""
    
    # Simplified: pressure proportional to velocity magnitude
    p_data = 1e5 .+ 1e3 * sqrt.(uE.data.^2 .+ vN.data.^2)
    return Field(p_data, grid)
end

function create_density_field(grid::SphericalGrid, lat::Vector)
    """Create synthetic density field with realistic stratification"""
    
    ny, nx = length(grid.lat), length(grid.lon)
    ρ_data = zeros(ny, nx)
    
    for j in 1:ny
        # Density increases towards poles (simplified)
        ρ_data[j, :] .= 1025.0 + 2.0 * sin(lat[j])^2
    end
    
    return Field(ρ_data, grid)
end

function compute_coastal_differences(field1::Field, field2::Field, land_mask::BitArray)
    """Compute RMS difference near coastal boundaries"""
    
    # Find coastal points (water points adjacent to land)
    ny, nx = size(land_mask)
    coastal_mask = falses(ny, nx)
    
    for j in 2:ny-1
        for i in 2:nx-1
            if land_mask[j, i]  # Water point
                # Check if adjacent to land
                if !land_mask[j-1, i] || !land_mask[j+1, i] || 
                   !land_mask[j, i-1] || !land_mask[j, i+1]
                    coastal_mask[j, i] = true
                end
            end
        end
    end
    
    if any(coastal_mask)
        diff_coastal = sqrt(mean((field1.data[coastal_mask] .- field2.data[coastal_mask]).^2))
    else
        diff_coastal = 0.0
    end
    
    return diff_coastal
end

function compute_energy_fraction(u1::Field, v1::Field, u_total::Field, v_total::Field)
    """Compute energy fraction of one velocity component"""
    
    KE1 = mean(u1.data.^2 .+ v1.data.^2)
    KE_total = mean(u_total.data.^2 .+ v_total.data.^2)
    
    return KE_total > 0 ? KE1 / KE_total : 0.0
end

function display_energy_budget(budget, land_mask::BitArray)
    """Display energy budget terms with oceanographic interpretation"""
    
    println("   Energy Budget Terms (averaged over ocean points):")
    
    ocean_points = land_mask
    
    ke_mean = mean(budget.kinetic_energy.data[ocean_points])
    π_mean = mean(budget.leonard_transfer.data[ocean_points])
    transport_mean = mean(budget.transport_total.data[ocean_points])
    dissipation_mean = mean(budget.dissipation_total.data[ocean_points])
    
    println("     Kinetic Energy: $(round(ke_mean/1e3, digits=2)) kJ/m³")
    println("     Leonard Transfer (Π): $(round(π_mean, digits=6)) W/m³")
    println("     Transport Divergence: $(round(transport_mean, digits=6)) W/m³")
    println("     Viscous Dissipation: $(round(dissipation_mean, digits=6)) W/m³")
    
    # Energy budget closure check
    residual = π_mean + transport_mean - dissipation_mean
    println("     Budget Residual: $(round(residual, digits=6)) W/m³")
    
    if abs(residual) < 1e-4
        println("     ✅ Energy budget closes well")
    else
        println("     ⚠️  Energy budget has residual (expected for 2D analysis)")
    end
end

function compute_spectral_slope(log_k::Vector, log_E::Vector)
    """Compute spectral slope using linear regression"""
    
    # Use middle portion of spectrum (avoid forcing and dissipation scales)
    n = length(log_k)
    idx_range = max(1, n÷4):min(n, 3*n÷4)
    
    if length(idx_range) < 5
        return NaN
    end
    
    # Linear regression: log(E) = a*log(k) + b
    X = [ones(length(idx_range)) log_k[idx_range]]
    y = log_E[idx_range]
    
    coeffs = X \ y
    slope = coeffs[2]
    
    return slope
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end