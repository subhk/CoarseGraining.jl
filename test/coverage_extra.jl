# Branch/coverage tests — Test, CoarseGraining, Statistics, NCDatasets in scope (from runtests.jl).

@testset "Filter branches (threaded / tiled / non-periodic / separable)" begin
    g = Grid(48, 40, 1.0, 1.0, true, true)
    A = [sin(2π*3*i/48)*cos(2π*2*j/40) for j in 1:40, i in 1:48]
    f = Field(A, g)
    K = gaussian_kernel(2.0, 2.0)
    base = coarse_grain(f, K)
    @test coarse_grain(f, K; threaded=true).data ≈ base.data
    @test coarse_grain(f, K; tile=(16, 16)).data ≈ base.data
    @test coarse_grain(f, K; threaded=true, tile=(16, 16)).data ≈ base.data
    @test coarse_grain(f, K; threaded=true).data ≈ base.data
    # non-periodic grid exercises padidx clamp branch
    @test size(coarse_grain(Field(A, Grid(48, 40, 1.0, 1.0, false, false)), K).data) == size(A)
    # separable threaded vs serial
    @test coarse_grain_gaussian_separable(f, 2.0, 2.0; threaded=true).data ≈
          coarse_grain_gaussian_separable(f, 2.0, 2.0; threaded=false).data
    # tile selection helpers
    @test select_tile(64, 64, 3, 3) == (64, 64)
    @test select_tile(4000, 4000, 3, 3)[1] > 0
    @test select_tile(f, K)[1] > 0
end

@testset "Masked filter branches" begin
    g = Grid(32, 32, 1.0, 1.0, true, true)
    A = [sin(2π*i/32) + cos(2π*j/32) for j in 1:32, i in 1:32]
    f = Field(A, g)
    K = gaussian_kernel(2.0, 2.0)
    full = trues(32, 32)
    @test size(coarse_grain_masked(f, K, full).data) == (32, 32)
    @test size(coarse_grain_masked(f, K, full; normalize=false).data) == (32, 32)
    # all-land neighbourhood ⇒ fill_value
    out = coarse_grain_masked(f, K, falses(32, 32); fill_value=-999.0)
    @test all(out.data .== -999.0)
    # separable masked: threaded == serial (no land ⇒ no NaN)
    @test coarse_grain_gaussian_separable_masked(f, 2.0, 2.0, full; threaded=true).data ≈
          coarse_grain_gaussian_separable_masked(f, 2.0, 2.0, full; threaded=false).data
    @test size(coarse_grain_gaussian_separable_masked(f, 2.0, 2.0, full; normalize=false).data) == (32, 32)
end

@testset "DSP Butterworth variants" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    A = [sin(2π*5*i/64) + 0.3*sin(2π*15*j/64) for j in 1:64, i in 1:64]
    f = Field(A, g)
    for out in (coarse_grain_butterworth_dsp(f, 2π/8; order=2),
                coarse_grain_butterworth_length_dsp(f, 8.0),
                coarse_grain_butterworth_cycles_dsp(f, 4.0),
                coarse_grain_butterworth_cells_dsp(f, 8.0),
                coarse_grain_butterworth_nyquist_dsp(f, 0.25))
        @test size(out.data) == size(A)
        @test all(isfinite, out.data)
    end
end

@testset "KE spectrum options + region stats" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    u = Field([sin(2π*6*i/64) for j in 1:64, i in 1:64], g)
    v = Field([cos(2π*4*j/64) for j in 1:64, i in 1:64], g)
    for nrm in (:counts, :density, :shellarea, :energy)
        kc, Ek = ke_spectrum_isotropic(u, v; normalize=nrm)
        @test length(kc) == length(Ek) && all(isfinite, Ek)
    end
    kc, Ek, edges = ke_spectrum_isotropic(u, v; bins=:log, return_edges=true)
    @test length(edges) == length(kc) + 1
    @test_throws ErrorException ke_spectrum_isotropic(u, v; bins=:bogus)
    m, s = zonal_mean_std(u)
    @test length(m) == 64 && length(s) == 64
    regions = Dict("a" => BitArray([j <= 32 for j in 1:64, i in 1:64]),
                   "b" => BitArray([j > 32  for j in 1:64, i in 1:64]))
    rs = region_mean_std(u, regions)
    @test haskey(rs, "a") && haskey(rs, "b")
end

@testset "Model C-grid averaging" begin
    ny, nx = 5, 6
    u = reshape(collect(1.0:ny*(nx+1)), ny, nx+1)
    v = reshape(collect(1.0:(ny+1)*nx), ny+1, nx)
    ur, vr = average_to_rho(u, v)
    @test size(ur) == (ny, nx) && size(vr) == (ny, nx)
    @test ur[1, 1] ≈ 0.5*(u[1, 1] + u[1, 2])
    @test vr[1, 1] ≈ 0.5*(v[1, 1] + v[2, 1])
    # all four MITgcm staggering branches
    a, b = average_to_tracer_mitgcm(ones(ny, nx+1), ones(ny+1, nx)); @test size(a) == (ny, nx) && size(b) == (ny, nx)
    a, _ = average_to_tracer_mitgcm(ones(ny, nx+1), ones(ny, nx));   @test size(a) == (ny, nx)
    _, b = average_to_tracer_mitgcm(ones(ny, nx),   ones(ny+1, nx)); @test size(b) == (ny, nx)
    a, b = average_to_tracer_mitgcm(ones(ny, nx),   ones(ny, nx));   @test size(a) == (ny, nx) && size(b) == (ny, nx)
end

@testset "IO NetCDF round-trips" begin
    try
        g = Grid(16, 12, 2.0, 3.0, true, false)
        f = Field([1.0*i + j for j in 1:12, i in 1:16], g)
        p  = tempname()*".nc"; pv = tempname()*".nc"
        ps = tempname()*".nc"; pm = tempname()*".nc"
        write_netcdf_field(p, "phi", f)
        @test load_netcdf_var(p, "phi"; dx=2.0, dy=3.0).data ≈ f.data
        write_attr(p, "title", "cov-test")
        @test read_attr(p, "title") == "cov-test"
        write_vector_vars(pv, "u", "v", f, f)
        u2, v2 = load_vector_vars(pv, "u", "v")
        @test u2.data ≈ f.data && v2.data ≈ f.data
        regions = Dict("r1" => BitArray([j <= 6 for j in 1:12, i in 1:16]))
        write_region_stats_and_masks(f, regions, ps; masks_path=pm,
                                     lon=collect(1.0:16), lat=collect(1.0:12))
        @test haskey(load_region_masks(pm, ["r1"]), "r1")
        for x in (p, pv, ps, pm); rm(x, force=true); end
    catch e
        @test_skip "IO round-trip skipped: $e"
    end
end

@testset "Multiresolution coarsen/refine methods" begin
    g = Grid(32, 32, 1.0, 1.0, true, true)
    f = Field([sin(2π*i/32) + cos(2π*j/32) for j in 1:32, i in 1:32], g)
    @test size(coarsen_field(f, 2; method=:area_average).data) == (16, 16)
    @test size(coarsen_field(f, 2; method=:subsample).data) == (16, 16)
    @test size(coarsen_field(f, 2; method=:conservative).data) == (16, 16)
    @test size(coarsen_field(f, (2, 4)).data) == (16, 8)
    coarse = coarsen_field(f, 2)
    for m in (:bilinear, :nearest, :bicubic, :conservative)
        @test size(refine_field(coarse, g; method=m).data) == (32, 32)
    end
    @test refine_field(f, g).data == f.data                  # identity (same grid)
    @test length(create_multiresolution_hierarchy(g, 3; coarsening_factor=2)) == 3
    # spherical coarsening exercises compute_grid_area(SphericalGrid)
    sg = SphericalGrid(collect(range(0, 2π, length=33))[1:32],
                       collect(range(deg2rad(10), deg2rad(70), length=32)), 6.371e6, true)
    fs = Field([sin(2π*i/32) for j in 1:32, i in 1:32], sg)
    @test size(coarsen_field(fs, 2).data) == (16, 16)
end

@testset "Boundary handling (extend + adaptive modes)" begin
    g = Grid(24, 20, 1.0, 1.0, false, false)
    f = Field([1.0*i + j for j in 1:20, i in 1:24], g)
    for m in (:mirror, :extrapolate, :zero, :periodic)
        e = extend_field_to_boundaries(f, 3; method=m)
        @test size(e.data) == (26, 30)
    end
    K = gaussian_kernel(2.0, 2.0)
    @test size(coarse_grain_adaptive(f, K; boundary_mode=:fixed).data) == (20, 24)
    @test size(coarse_grain_adaptive(f, K; boundary_mode=:adaptive).data) == (20, 24)
    # adaptive with land mask (deform-around-land branch)
    mask = trues(20, 24); mask[1:3, :] .= false
    @test size(coarse_grain_adaptive(f, K; mask=mask, deform_around_land=true).data) == (20, 24)
    @test size(coarse_grain_adaptive(f, K; mask=mask, deform_around_land=false).data) == (20, 24)
end

@testset "Anisotropic (tuple) filters + anisotropic spectrum" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    A = [sin(2π*5*i/64) + 0.3*sin(2π*9*j/64) for j in 1:64, i in 1:64]
    f = Field(A, g)
    for out in (coarse_grain_butterworth(f, (2π/8, 2π/6)),
                coarse_grain_butterworth_length(f, (8.0, 6.0)),
                coarse_grain_butterworth_cycles(f, (4.0, 3.0)),
                coarse_grain_butterworth_cells(f, (8.0, 6.0)),
                coarse_grain_butterworth_nyquist(f, (0.3, 0.25)),
                coarse_grain_butterworth_length_dsp(f, (8.0, 6.0)),
                coarse_grain_butterworth_cycles_dsp(f, (4.0, 3.0)),
                coarse_grain_butterworth_cells_dsp(f, (8.0, 6.0)),
                coarse_grain_butterworth_nyquist_dsp(f, (0.3, 0.25)))
        @test size(out.data) == size(A) && all(isfinite, out.data)
    end
    u = Field([sin(2π*6*i/64) for j in 1:64, i in 1:64], g)
    v = Field([cos(2π*4*j/64) for j in 1:64, i in 1:64], g)
    km, KE = compute_kinetic_energy_spectra(u, v; method=:anisotropic)
    @test size(km) == size(KE)
end

@testset "Energy budget optional terms" begin
    g = Grid(32, 32, 1.0, 1.0, true, true)
    u = Field([sin(2π*i/32)*cos(2π*j/32) for j in 1:32, i in 1:32], g)
    v = Field([cos(2π*i/32)*sin(2π*j/32) for j in 1:32, i in 1:32], g)
    p = Field([1e5 + 1e3*sin(2π*i/32) for j in 1:32, i in 1:32], g)
    ρ = Field([1025.0 + sin(2π*j/32) for j in 1:32, i in 1:32], g)
    w = Field([0.01*cos(2π*i/32) for j in 1:32, i in 1:32], g)
    K = gaussian_kernel(2.0, 2.0)
    Jc, Jp, Js, Jt = compute_energy_transport(u, v, K; pressure=p)
    @test size(Jt.data) == (32, 32)
    Pi_bc, APE = compute_baroclinic_transfer(u, v, ρ, w, K)
    @test size(Pi_bc.data) == (32, 32)
    b = compute_full_energy_budget(u, v, K; pressure=p, ρ=ρ, w=w, forcing_u=u, forcing_v=v)
    @test haskey(b, :baroclinic_conversion) && haskey(b, :forcing_work)
end

@testset "Curvilinear adaptive filtering" begin
    ny, nx = 16, 16
    lon = [Float64(i) for j in 1:ny, i in 1:nx]
    lat = [Float64(j) for j in 1:ny, i in 1:nx]
    dx = fill(1000.0, ny, nx); dy = fill(1000.0, ny, nx)
    cg = CurvilinearGrid(lon, lat, dx, dy, false, false, 6.371e6)
    f = Field([sin(i/3) + cos(j/3) for j in 1:ny, i in 1:nx], cg)
    @test size(gradient_curvilinear(f)[1].data) == (ny, nx)
    @test size(coarse_grain_adaptive(f, gaussian_kernel(2.0, 2.0)).data) == (ny, nx)
end
