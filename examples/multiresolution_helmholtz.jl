#!/usr/bin/env julia

"""
Multi-Resolution Helmholtz Decomposition Example

Demonstrates FlowSieve's coarsen → solve → refine strategy for efficient
Helmholtz decomposition on large grids. This approach dramatically reduces
computational cost while maintaining accuracy.

Workflow:
1. Coarsen velocity field to manageable resolution
2. Solve Helmholtz decomposition on coarse grid
3. Refine solution to higher resolution
4. Use coarse solution as initial guess for fine solve
5. Repeat until reaching target resolution
"""

using CoarseGraining
using LinearAlgebra
using Printf

function main()
    println("🔄 Multi-Resolution Helmholtz Decomposition")
    println("=" ^ 50)
    
    # Test different grid sizes to show scaling benefits
    test_sizes = [
        (64, 64, "Small"),
        (128, 128, "Medium"), 
        (256, 256, "Large"),
        (512, 512, "Very Large")
    ]
    
    for (nx, ny, size_name) in test_sizes
        println("\n📊 Testing $size_name Grid: $(nx)×$(ny)")
        println("-" ^ 40)
        
        # Create test velocity field
        uE, vN, grid = create_test_velocity_field(nx, ny)
        
        # Method 1: Direct spherical Helmholtz (baseline)
        println("Method 1: Direct iterative solver")
        time_direct = @elapsed begin
            try
                uE_div_direct, vN_div_direct, uE_pot_direct, vN_pot_direct, φ_direct, ψ_direct = 
                    helmholtz_hodge_sphere_iterative(uE, vN; max_iter=200, tol=1e-6)
                direct_success = true
            catch e
                println("   ❌ Direct method failed: $e")
                direct_success = false
                uE_div_direct = vN_div_direct = uE_pot_direct = vN_pot_direct = nothing
                φ_direct = ψ_direct = nothing
            end
        end
        
        if direct_success
            println(@sprintf("   ⏱️  Time: %.2f seconds", time_direct))
        end
        
        # Method 2: Multi-resolution approach
        println("Method 2: Multi-resolution hierarchy")
        levels = determine_optimal_levels(nx, ny)
        
        time_multiresolution = @elapsed begin
            try
                uE_div_multi, vN_div_multi, uE_pot_multi, vN_pot_multi, φ_multi, ψ_multi = 
                    hierarchical_helmholtz_workflow(uE, vN; levels=levels, max_iter_per_level=50)
                multi_success = true
            catch e
                println("   ❌ Multi-resolution method failed: $e")
                multi_success = false
                uE_div_multi = vN_div_multi = uE_pot_multi = vN_pot_multi = nothing
                φ_multi = ψ_multi = nothing
            end
        end
        
        if multi_success
            println(@sprintf("   ⏱️  Time: %.2f seconds", time_multiresolution))
            println(@sprintf("   📈 Grid levels: %d", levels))
            
            if direct_success
                speedup = time_direct / time_multiresolution
                println(@sprintf("   🚀 Speedup: %.1fx", speedup))
                
                # Compare accuracy
                error_div = sqrt(mean((uE_div_direct.data .- uE_div_multi.data).^2 .+ 
                                    (vN_div_direct.data .- vN_div_multi.data).^2))
                error_pot = sqrt(mean((uE_pot_direct.data .- uE_pot_multi.data).^2 .+ 
                                    (vN_pot_direct.data .- vN_pot_multi.data).^2))
                
                println(@sprintf("   🎯 RMS Error (divergent): %.2e m/s", error_div))
                println(@sprintf("   🎯 RMS Error (potential): %.2e m/s", error_pot))
                
                if error_div < 1e-3 && error_pot < 1e-3
                    println("   ✅ Excellent accuracy maintained")
                elseif error_div < 1e-2 && error_pot < 1e-2
                    println("   ✅ Good accuracy maintained")
                else
                    println("   ⚠️  Some accuracy loss detected")
                end
            end
        end
        
        # Method 3: Show the grid hierarchy
        if multi_success
            println("\nGrid Hierarchy Details:")
            grids = create_multiresolution_hierarchy(grid, levels; coarsening_factor=2)
            
            for (i, g) in enumerate(grids)
                ny_level, nx_level = length(g.lat), length(g.lon)
                resolution_km = round(6.371e6 * mean(diff(g.lat)) / 1000, digits=1)
                memory_mb = round(8 * ny_level * nx_level / 1e6, digits=2)
                
                println(@sprintf("   Level %d: %dx%d grid (~%.1f km, %.2f MB)", 
                                i, nx_level, ny_level, resolution_km, memory_mb))
            end
        end
        
        # Save results for largest successful case
        if size_name == "Large" && multi_success
            save_decomposition_results(uE, vN, uE_div_multi, vN_div_multi, 
                                     uE_pot_multi, vN_pot_multi, φ_multi, ψ_multi, grid)
        end
    end
    
    # Demonstrate workflow with realistic ocean case
    println("\n\n🌊 Realistic Ocean Application")
    println("=" ^ 50)
    demonstrate_ocean_workflow()
    
    return nothing
end

function create_test_velocity_field(nx::Int, ny::Int)
    """Create test velocity field with multiple scales"""
    
    # Create spherical grid (global-like)
    lon = range(0, 2π, length=nx)
    lat = range(-π/2 + π/ny, π/2 - π/ny, length=ny)
    grid = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
    
    uE_data = zeros(ny, nx)
    vN_data = zeros(ny, nx)
    
    # Add multiple scales of motion
    for j in 1:ny
        for i in 1:nx
            φ = lat[j]
            λ = lon[i]
            
            # Large-scale flow (zonal jet)
            uE_data[j, i] += 20.0 * cos(2*φ) * (1 + 0.1*cos(4*λ))
            
            # Mesoscale eddies
            uE_data[j, i] += 5.0 * sin(8*φ) * cos(6*λ)
            vN_data[j, i] += 5.0 * cos(8*φ) * sin(6*λ)
            
            # Small-scale turbulence
            uE_data[j, i] += 1.0 * sin(16*φ) * sin(12*λ)
            vN_data[j, i] += 1.0 * cos(16*φ) * cos(12*λ)
            
            # Add some noise
            uE_data[j, i] += 0.1 * randn()
            vN_data[j, i] += 0.1 * randn()
        end
    end
    
    uE = Field(uE_data, grid)
    vN = Field(vN_data, grid)
    
    return uE, vN, grid
end

function determine_optimal_levels(nx::Int, ny::Int)
    """Determine optimal number of levels for multi-resolution"""
    
    min_size = min(nx, ny)
    
    if min_size <= 64
        return 2
    elseif min_size <= 128
        return 3
    elseif min_size <= 256
        return 4
    elseif min_size <= 512
        return 5
    else
        return 6
    end
end

function save_decomposition_results(uE, vN, uE_div, vN_div, uE_pot, vN_pot, φ, ψ, grid)
    """Save Helmholtz decomposition results"""
    
    println("\n💾 Saving decomposition results...")
    
    # Save original and decomposed fields
    fields = [uE, vN, uE_div, vN_div, uE_pot, vN_pot, φ, ψ]
    field_names = ["uE_original", "vN_original", "uE_divergent", "vN_divergent", 
                   "uE_potential", "vN_potential", "velocity_potential", "stream_function"]
    
    try
        save_multiresolution_data("helmholtz_decomposition.nc", fields, 
                                 fill(grid, length(fields)); group_name="decomposition")
        
        println("   ✅ Saved to: helmholtz_decomposition.nc")
        
        # Display decomposition statistics
        total_ke = mean(uE.data.^2 .+ vN.data.^2)
        div_ke = mean(uE_div.data.^2 .+ vN_div.data.^2)  
        pot_ke = mean(uE_pot.data.^2 .+ vN_pot.data.^2)
        
        println(@sprintf("   📊 Total KE: %.2e m²/s²", total_ke))
        println(@sprintf("   📊 Divergent KE: %.2e m²/s² (%.1f%%)", div_ke, 100*div_ke/total_ke))
        println(@sprintf("   📊 Potential KE: %.2e m²/s² (%.1f%%)", pot_ke, 100*pot_ke/total_ke))
        
    catch e
        println("   ❌ Failed to save: $e")
    end
end

function demonstrate_ocean_workflow()
    """Demonstrate workflow with realistic ocean parameters"""
    
    # High-resolution ocean model setup
    println("Setting up high-resolution ocean model scenario...")
    
    # North Atlantic domain
    lon_range = -80.0:0.25:-20.0  # 0.25° resolution (~25 km)
    lat_range = 20.0:0.25:65.0
    
    nx = length(lon_range)
    ny = length(lat_range)
    
    println(@sprintf("Domain: %.1f°E to %.1f°E, %.1f°N to %.1f°N", 
                    lon_range[1], lon_range[end], lat_range[1], lat_range[end]))
    println(@sprintf("Resolution: %.2f° (~25 km)", step(lon_range)))
    println(@sprintf("Grid size: %dx%d = %d points", nx, ny, nx*ny))
    
    # Estimate computational requirements
    memory_gb = 8 * nx * ny * 10 / 1e9  # 10 fields, 8 bytes each
    println(@sprintf("Memory requirement: ~%.1f GB", memory_gb))
    
    # Create simplified velocity field for demonstration
    println("\nCreating simplified velocity field...")
    lon_grid = collect(deg2rad.(lon_range))
    lat_grid = collect(deg2rad.(lat_range))
    grid = SphericalGrid(lon_grid, lat_grid, 6.371e6, true)
    
    # Gulf Stream-like jet
    uE_data = zeros(ny, nx)
    vN_data = zeros(ny, nx)
    
    for j in 1:ny
        for i in 1:nx
            φ = lat_grid[j]
            λ = lon_grid[i]
            
            # Simplified Gulf Stream
            if deg2rad(30) < φ < deg2rad(45) && deg2rad(-75) < λ < deg2rad(-50)
                jet_strength = 1.0 * exp(-((φ - deg2rad(38))/deg2rad(3))^2)
                uE_data[j, i] = jet_strength
                
                # Add meandering
                vN_data[j, i] = 0.2 * sin(3*(λ - deg2rad(-60))) * jet_strength
            end
        end
    end
    
    uE = Field(uE_data, grid)
    vN = Field(vN_data, grid)
    
    # Multi-resolution strategy
    println("\nMulti-resolution strategy:")
    
    levels = 4
    grids = create_multiresolution_hierarchy(grid, levels)
    
    total_cost_direct = nx * ny  # Simplified cost metric
    total_cost_multi = 0
    
    for (i, g) in enumerate(grids)
        ny_level, nx_level = length(g.lat), length(g.lon)
        level_cost = nx_level * ny_level
        total_cost_multi += level_cost
        
        resolution_km = round(6.371e6 * mean(diff(g.lat)) / 1000, digits=1)
        
        println(@sprintf("   Level %d: %dx%d (~%.0f km) - Cost: %d", 
                        i, nx_level, ny_level, resolution_km, level_cost))
    end
    
    cost_ratio = total_cost_direct / total_cost_multi
    println(@sprintf("\nComputational cost reduction: %.1fx", cost_ratio))
    println("💡 Multi-resolution approach enables analysis of high-resolution ocean models!")
    
    # Show memory usage at each level
    println("\nMemory usage per level:")
    for (i, g) in enumerate(grids)
        ny_level, nx_level = length(g.lat), length(g.lon)
        memory_mb = 8 * ny_level * nx_level * 10 / 1e6
        println(@sprintf("   Level %d: %.1f MB", i, memory_mb))
    end
    
    println("\n✅ Realistic ocean workflow demonstration complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end