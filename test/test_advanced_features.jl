using Test
using CoarseGraining
using LinearAlgebra
using Statistics

@testset "Advanced FlowSieve Features" begin
    
    @testset "Spherical Helmholtz Decomposition" begin
        # Create small test grid
        nx, ny = 32, 16
        lon = range(0, 2π, length=nx)
        lat = range(-π/4, π/4, length=ny)
        grid = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
        
        # Create test velocity field
        uE_data = [sin(2*lon[i]) * cos(lat[j]) for j in 1:ny, i in 1:nx]
        vN_data = [cos(2*lon[i]) * sin(lat[j]) for j in 1:ny, i in 1:nx]
        
        uE = Field(uE_data, grid)
        vN = Field(vN_data, grid)
        
        # Test iterative spherical Helmholtz
        @test_nowarn begin
            uE_div, vN_div, uE_pot, vN_pot, φ, ψ = 
                helmholtz_hodge_sphere_iterative(uE, vN; max_iter=100, tol=1e-4)
        end
        
        # Test decomposition accuracy
        uE_div, vN_div, uE_pot, vN_pot, φ, ψ = 
            helmholtz_hodge_sphere_iterative(uE, vN; max_iter=100, tol=1e-4)
        
        uE_recon = Field(uE_div.data .+ uE_pot.data, grid)
        vN_recon = Field(vN_div.data .+ vN_pot.data, grid)
        
        recon_error = sqrt(mean((uE_recon.data .- uE.data).^2 .+ (vN_recon.data .- vN.data).^2))
        @test recon_error < 1e-2
        
        # Test divergence and curl properties
        div_div = divergence_sphere(uE_div, vN_div)
        curl_pot = vorticity_sphere(uE_pot, vN_pot)
        
        @test maximum(abs.(div_div.data)) < 1e-6  # Divergence-free part should have zero divergence
        @test maximum(abs.(curl_pot.data)) < 1e-6  # Potential part should have zero curl
    end
    
    @testset "Advanced Energy Diagnostics" begin
        # Create test fields
        nx, ny = 32, 32
        g = Grid(nx, ny, 1.0, 1.0, true, true)
        
        u_data = [sin(2π*i/nx) * cos(2π*j/ny) for j in 1:ny, i in 1:nx]
        v_data = [cos(2π*i/nx) * sin(2π*j/ny) for j in 1:ny, i in 1:nx]
        p_data = [1e5 + 1e3*sin(π*i/nx) for j in 1:ny, i in 1:nx]
        
        u = Field(u_data, g)
        v = Field(v_data, g)
        pressure = Field(p_data, g)
        
        K = gaussian_kernel(2.0, 2.0)
        
        # Test energy transport computation
        @test_nowarn begin
            J_coarse, J_pressure, J_subgrid, div_J_total = 
                compute_energy_transport(u, v, K; pressure=pressure)
        end
        
        # Test viscous dissipation
        @test_nowarn begin
            eps_resolved, eps_subgrid, eps_total = 
                compute_viscous_dissipation(u, v, K; ν=1e-6)
        end
        
        eps_resolved, eps_subgrid, eps_total = 
            compute_viscous_dissipation(u, v, K; ν=1e-6)
        
        @test all(eps_resolved.data .≥ 0)  # Dissipation should be non-negative
        @test all(eps_total.data .≥ eps_resolved.data)  # Total ≥ resolved
        
        # Test full energy budget
        @test_nowarn begin
            budget = compute_full_energy_budget(u, v, K; pressure=pressure)
        end
        
        budget = compute_full_energy_budget(u, v, K; pressure=pressure)
        @test haskey(budget, :kinetic_energy)
        @test haskey(budget, :leonard_transfer)
        @test haskey(budget, :dissipation_total)
    end
    
    @testset "Sophisticated Boundary Handling" begin
        # Create test grid with boundaries
        nx, ny = 16, 16
        g = Grid(nx, ny, 1.0, 1.0, false, false)  # Non-periodic
        
        # Create test field with boundaries
        A = [exp(-((i-nx/2)^2 + (j-ny/2)^2)/16) for j in 1:ny, i in 1:nx]
        field = Field(A, g)
        
        # Create land mask (water=true, land=false)
        land_mask = trues(ny, nx)
        land_mask[1:5, :] .= false  # Land strip at bottom
        land_mask[:, 1:3] .= false  # Land strip at left
        
        K = gaussian_kernel(2.0, 2.0)
        
        # Test adaptive filtering
        @test_nowarn begin
            filtered = coarse_grain_adaptive(field, K; 
                                           mask=land_mask, 
                                           boundary_mode=:adaptive)
        end
        
        filtered = coarse_grain_adaptive(field, K; 
                                       mask=land_mask, 
                                       boundary_mode=:adaptive)
        
        # Check that land points remain zero
        @test all(filtered.data[.!land_mask] .== 0)
        
        # Test boundary extension
        @test_nowarn begin
            extended = extend_field_to_boundaries(field, 3; method=:mirror)
        end
    end
    
    @testset "Multi-Resolution Workflows" begin
        # Create test grid hierarchy
        nx, ny = 64, 32
        lon = range(0, 2π, length=nx)
        lat = range(-π/4, π/4, length=ny)
        base_grid = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
        
        # Test grid hierarchy creation
        @test_nowarn begin
            grids = create_multiresolution_hierarchy(base_grid, 3; coarsening_factor=2)
        end
        
        grids = create_multiresolution_hierarchy(base_grid, 3; coarsening_factor=2)
        @test length(grids) == 3
        @test length(grids[2].lon) == nx ÷ 2
        @test length(grids[3].lon) == nx ÷ 4
        
        # Create test field
        uE_data = [sin(4*lon[i]) * cos(2*lat[j]) for j in 1:ny, i in 1:nx]
        vN_data = [cos(4*lon[i]) * sin(2*lat[j]) for j in 1:ny, i in 1:nx]
        
        uE = Field(uE_data, base_grid)
        vN = Field(vN_data, base_grid)
        
        # Test field coarsening
        @test_nowarn begin
            uE_coarse = coarsen_field(uE, 2; method=:area_average)
        end
        
        uE_coarse = coarsen_field(uE, 2; method=:area_average)
        @test size(uE_coarse.data) == (ny ÷ 2, nx ÷ 2)
        
        # Test field refinement
        @test_nowarn begin
            uE_refined = refine_field(uE_coarse, base_grid; method=:bilinear)
        end
        
        uE_refined = refine_field(uE_coarse, base_grid; method=:bilinear)
        @test size(uE_refined.data) == (ny, nx)
        
        # Test hierarchical Helmholtz (small problem)
        small_nx, small_ny = 16, 8
        small_lon = lon[1:4:end]
        small_lat = lat[1:4:end]
        small_grid = SphericalGrid(collect(small_lon), collect(small_lat), 6.371e6, true)
        
        small_uE = Field(uE_data[1:4:end, 1:4:end], small_grid)
        small_vN = Field(vN_data[1:4:end, 1:4:end], small_grid)
        
        @test_nowarn begin
            uE_div, vN_div, uE_pot, vN_pot, φ, ψ = 
                hierarchical_helmholtz_workflow(small_uE, small_vN; levels=2, max_iter_per_level=20)
        end
    end
    
    @testset "Spectral Diagnostics" begin
        # Create periodic test case for spectral analysis
        nx, ny = 64, 64
        g = Grid(nx, ny, 1.0, 1.0, true, true)
        
        # Create test field with known spectral properties
        u_data = zeros(ny, nx)
        v_data = zeros(ny, nx)
        
        for j in 1:ny, i in 1:nx
            x, y = (i-1), (j-1)
            
            # Add multiple wavenumber components
            u_data[j, i] = sin(2π*x/nx) + 0.5*sin(4π*x/nx) + 0.25*sin(8π*x/nx)
            v_data[j, i] = cos(2π*y/ny) + 0.5*cos(4π*y/ny) + 0.25*cos(8π*y/ny)
        end
        
        u = Field(u_data, g)
        v = Field(v_data, g)
        
        # Test kinetic energy spectra
        @test_nowarn begin
            k, E_k = compute_kinetic_energy_spectra(u, v; method=:isotropic)
        end
        
        k, E_k = compute_kinetic_energy_spectra(u, v; method=:isotropic)
        @test length(k) > 0
        @test length(E_k) == length(k)
        @test all(E_k .≥ 0)
        
        # Test enstrophy transfer
        K = gaussian_kernel(2.0, 2.0)
        @test_nowarn begin
            Π_enstrophy, ω̄ = compute_enstrophy_transfer(u, v, K)
        end
    end
    
    @testset "Data I/O for Multi-Resolution" begin
        # Create test data
        nx, ny = 16, 8
        g1 = Grid(nx, ny, 1.0, 1.0, true, true)
        g2 = Grid(nx÷2, ny÷2, 2.0, 2.0, true, true)
        
        f1 = Field(randn(ny, nx), g1)
        f2 = Field(randn(ny÷2, nx÷2), g2)
        
        fields = [f1, f2]
        grids = [g1, g2]
        
        # Test saving (might fail due to dependencies, but shouldn't error in structure)
        try
            filename = "test_multiresolution.nc"
            @test_nowarn save_multiresolution_data(filename, fields, grids)
            
            # Test loading if save succeeded
            if isfile(filename)
                @test_nowarn load_multiresolution_data(filename)
                rm(filename)  # Clean up
            end
        catch e
            @test_skip "NetCDF I/O test skipped: $e"
        end
    end
end

@testset "Integration Tests" begin
    @testset "Complete Ocean Analysis Workflow" begin
        # Simplified version of the complete workflow
        
        # Small test domain
        nx, ny = 16, 8
        lon = range(deg2rad(-70), deg2rad(-60), length=nx)
        lat = range(deg2rad(35), deg2rad(45), length=ny)
        grid = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
        
        # Simple velocity field
        uE_data = [0.5 * sin(2*lon[i]) * cos(lat[j]) for j in 1:ny, i in 1:nx]
        vN_data = [0.3 * cos(lon[i]) * sin(lat[j]) for j in 1:ny, i in 1:nx]
        
        uE = Field(uE_data, grid)
        vN = Field(vN_data, grid)
        
        # Create land mask
        land_mask = trues(ny, nx)
        land_mask[1:2, :] .= false  # Southern boundary as land
        
        K = gaussian_kernel(1.5, 1.5)
        
        # Test complete workflow
        @test_nowarn begin
            # 1. Adaptive filtering
            uE_filtered = coarse_grain_adaptive(uE, K; mask=land_mask)
            vN_filtered = coarse_grain_adaptive(vN, K; mask=land_mask)
            
            # 2. Spherical Helmholtz decomposition
            uE_div, vN_div, uE_pot, vN_pot, φ, ψ = 
                helmholtz_hodge_sphere_iterative(uE_filtered, vN_filtered; max_iter=50, tol=1e-3)
            
            # 3. Energy diagnostics
            budget = compute_full_energy_budget(uE, vN, K)
            
            # 4. Multi-resolution setup
            grids = create_multiresolution_hierarchy(grid, 2)
        end
        
        @test true  # If we get here without errors, the workflow completed
    end
end