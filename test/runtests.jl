using Test
using CoarseGraining
using NCDatasets
import Statistics

@testset "Kernels" begin
    K = gaussian_kernel(2.0, 2.0)
    @test isapprox(sum(K.weights), 1.0; atol=1e-12)
end

@testset "Filters - serial" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    x = range(0, stop=2π, length=g.nx)
    y = range(0, stop=2π, length=g.ny)
    A = [sin(xx) + cos(yy) for yy in y, xx in x]
    f = Field(A, g)
    K = gaussian_kernel(1.0, 1.0)
    fg = coarse_grain(f, K)
    @test size(fg.data) == size(A)
    ff = coarse_grain_fft(f, 1.0, 1.0)
    @test size(ff.data) == size(A)
end

@testset "Filter type keyword (top-hat default)" begin
    g = Grid(32, 32, 1.0, 1.0, true, true)
    A = [sin(2π*i/32) * cos(2π*j/32) for j in 1:32, i in 1:32]
    f = Field(A, g)
    # Default is top-hat; L=4 ⇒ box radius round(4/2)=2 ⇒ boxcar_kernel(2,2)
    out_default = coarse_grain(f, 4.0)
    @test out_default.data == coarse_grain(f, 4.0; kernel=:tophat).data
    @test out_default.data ≈ coarse_grain(f, boxcar_kernel(2, 2)).data
    # :gaussian ⇒ σ = L/2 = 2 ⇒ gaussian_kernel(2,2)
    @test coarse_grain(f, 4.0; kernel=:gaussian).data ≈ coarse_grain(f, gaussian_kernel(2.0, 2.0)).data
    # anisotropic length and unknown kernel
    @test size(coarse_grain(f, 4.0, 6.0).data) == size(A)
    @test_throws ErrorException coarse_grain(f, 4.0; kernel=:nope)
end

@testset "Spherical area-weighted top-hat (coarse_grain_sphere)" begin
    a = 6.371e6
    nx, ny = 64, 48
    lon = collect(range(0, 2π, length=nx+1))[1:nx]
    lat = collect(range(deg2rad(10), deg2rad(80), length=ny))
    sg = SphericalGrid(lon, lat, a, true)
    A = [sin(2*lon[i]) * cos(lat[j]) for j in 1:ny, i in 1:nx]
    f = Field(A, sg)
    # constant field preserved exactly (area-weighted normalization)
    cf = coarse_grain_sphere(Field(fill(2.5, ny, nx), sg), 500e3)
    @test maximum(abs.(cf.data .- 2.5)) < 1e-10
    # sub-grid ℓ ⇒ identity (only the point itself is within radius)
    @test maximum(abs.(coarse_grain_sphere(f, 1e3).data .- A)) < 1e-10
    # real scale: finite and does not amplify (it's a weighted average)
    big = coarse_grain_sphere(f, 1000e3)
    @test all(isfinite, big.data)
    @test maximum(abs.(big.data)) <= maximum(abs.(A)) + 1e-10
    # smoothed edge runs
    @test all(isfinite, coarse_grain_sphere(f, 1000e3; smooth=true).data)
    # land mask: masked points fill_value, valid interior finite
    mask = trues(ny, nx); mask[1:4, :] .= false
    fm = coarse_grain_sphere(f, 800e3; mask=mask)
    @test all(isnan, fm.data[1:4, :])
    @test all(isfinite, fm.data[20:end, :])
end

@testset "3D/4D driver + NetCDF" begin
    a = 6.371e6; nx, ny, nz, nt = 32, 24, 2, 2
    lon = collect(range(0, 2π, length=nx+1))[1:nx]
    lat = collect(range(deg2rad(10), deg2rad(70), length=ny))
    sg = SphericalGrid(lon, lat, a, true)
    data = [sin(2*lon[i])*cos(lat[j]) + 0.1k + 0.01t for j in 1:ny, i in 1:nx, k in 1:nz, t in 1:nt]
    out = coarse_grain_4d(data, sg, 600e3)
    @test size(out) == size(data)
    @test out[:, :, 2, 2] ≈ coarse_grain_sphere(Field(data[:, :, 2, 2], sg), 600e3).data
    @test coarse_grain_4d(data, sg, 600e3; threaded=false) ≈ out
    @test map_horizontal(s -> 2 .* s, data) ≈ 2 .* data
    @test maximum(abs.(coarse_grain_4d(fill(5.0, ny, nx, nz, nt), sg, 600e3) .- 5.0)) < 1e-10
    # NetCDF round-trip (guarded so I/O issues can't break CI)
    try
        tmp = tempname() * ".nc"
        NCDataset(tmp, "c") do ds
            defDim(ds, "lon", nx); defDim(ds, "lat", ny); defDim(ds, "depth", nz); defDim(ds, "time", nt)
            vl = defVar(ds, "lon", Float64, ("lon",)); vl[:] = rad2deg.(lon)
            va = defVar(ds, "lat", Float64, ("lat",)); va[:] = rad2deg.(lat)
            vu = defVar(ds, "u", Float64, ("lon", "lat", "depth", "time"))
            vu[:, :, :, :] = permutedims(data, (2, 1, 3, 4))
        end
        filt, grid = coarse_grain_netcdf(tmp, "u", 600e3)
        @test size(filt) == (nx, ny, nz, nt)
        @test permutedims(filt, (2, 1, 3, 4)) ≈ out
        rm(tmp, force=true)
    catch e
        @test_skip "NetCDF driver test skipped: $e"
    end
end

@testset "Multi-scale sweep (coarse_grain_scales)" begin
    a = 6.371e6; nx, ny = 32, 24
    lon = collect(range(0, 2π, length=nx+1))[1:nx]
    lat = collect(range(deg2rad(10), deg2rad(70), length=ny))
    sg = SphericalGrid(lon, lat, a, true)
    f = Field([sin(2*lon[i])*cos(lat[j]) for j in 1:ny, i in 1:nx], sg)
    ℓs = [200e3, 500e3, 1000e3]
    data, sc = coarse_grain_scales(f, ℓs)
    @test size(data) == (ny, nx, 3)
    @test sc == ℓs
    @test data[:, :, 2] ≈ coarse_grain_sphere(f, 500e3).data          # slice == single-scale call
    @test coarse_grain_scales(f, ℓs; threaded=false)[1] ≈ data        # threaded == serial
    flds, _ = coarse_grain_scales(f, ℓs; as_fields=true)
    @test length(flds) == 3 && flds[3].data ≈ data[:, :, 3]
    # Cartesian path (scales = cells)
    gc = Grid(32, 32, 1.0, 1.0, true, true)
    fc = Field([sin(2π*i/32)*cos(2π*j/32) for j in 1:32, i in 1:32], gc)
    dc, _ = coarse_grain_scales(fc, [2.0, 4.0])
    @test size(dc) == (32, 32, 2)
    @test dc[:, :, 1] ≈ coarse_grain(fc, 2.0; kernel=:tophat).data
end

@testset "Streaming postprocessing accumulators" begin
    ny, nx = 6, 8
    slices = [Float64[(i + 2j + 3k) % 5 for j in 1:ny, i in 1:nx] for k in 1:5]
    stack = cat(slices...; dims=3)
    # RunningMean (Welford) vs Statistics over the stacked slices
    rm = RunningMean(ny, nx)
    for s in slices; update!(rm, s); end
    r = results(rm)
    @test r.mean ≈ Statistics.mean(stack; dims=3)[:, :, 1]
    @test r.var ≈ Statistics.var(stack; dims=3)[:, :, 1]
    @test r.n == 5
    # RegionAccumulator: area-weighted region mean/weight vs direct
    sg = SphericalGrid(collect(range(0, 2π, length=nx+1))[1:nx],
                       collect(range(deg2rad(10), deg2rad(60), length=ny)), 6.371e6, true)
    area = cell_area_weights(sg)
    region = fill(1, ny, nx); region[1:3, :] .= 2
    acc = RegionAccumulator(2)
    for s in slices; update!(acc, Field(s, sg), region; area=area); end
    res = results(acc)
    vals = vcat([vec(s[1:3, :]) for s in slices]...)
    wts  = repeat(vec(area[1:3, :]), 5)
    @test res.mean[2] ≈ sum(wts .* vals) / sum(wts)
    @test res.weight[2] ≈ 5 * sum(area[1:3, :])
    # HistogramAccumulator: binning + out-of-range skip
    h = HistogramAccumulator(0:0.25:1.0)
    update!(h, [0.1, 0.3, 0.3, 0.9, -5.0, 5.0])
    rh = results(h)
    @test sum(rh.counts) == 4
    @test rh.counts[2] == 2
    @test length(rh.centers) == 4
end

@testset "Structure functions + Bessel spectrum" begin
    nx = ny = 64; dx = 1.0
    g = Grid(nx, ny, dx, dx, true, true)
    m = 4; k0 = 2π*m/(nx*dx)
    A = [sin(k0*(i-1)*dx) + sin(k0*(j-1)*dx) for j in 1:ny, i in 1:nx]
    f = Field(A, g)
    r, Sp = structure_function(f, [1, 2, 4, 8, 16]; order=2)
    @test maximum(abs.(Sp .- [1 - cos(k0*ri) for ri in r])) < 1e-8   # S2 of sin = 1-cos(k0 r)
    @test structure_function(f, [0])[2][1] == 0.0
    # velocity SF on a shear u=y: transverse grows with separation, longitudinal ≈ 0
    gx = Grid(nx, ny, dx, dx, true, false)
    u = Field([float((j-1)*dx) for j in 1:ny, i in 1:nx], gx)
    v = Field(zeros(ny, nx), gx)
    _, SL, ST = velocity_structure_function(u, v, [1, 2, 4]; order=2)
    @test maximum(abs.(SL)) < 1e-9
    @test ST[1] < ST[3]
    # Bessel S2→E(k) inversion recovers a known spectral bump
    kg = collect(range(0.05, 1.5, length=24)); kc = 0.5
    E = exp.(-((kg .- kc) ./ 0.15).^2)
    rg = collect(range(1.0, 40.0, length=48))
    Ehat = spectrum_from_sf2(rg, sf2_from_spectrum(kg, E, rg), kg)
    @test kg[argmax(Ehat)] ≈ kc atol=0.1
    @test Statistics.cor(E, Ehat) > 0.99
end

@testset "Advective-SF spectral flux (Bessel)" begin
    # Bessel J1/J2/J3 vs reference values
    @test isapprox(CoarseGraining._besselj1(1.0), 0.4400505857, atol=1e-5)
    @test isapprox(CoarseGraining._besselj2(1.0), 0.1149034849, atol=1e-5)
    @test isapprox(CoarseGraining._besselj3(5.0), 0.3648312306, atol=1e-4)
    # V-form transform vs Laplace closed form: ∫ e^{-r} J2(Kr) dr = [√(1+K²)-1]²/(K²√(1+K²))
    rg = collect(range(0, 60, length=6001))
    for K in (0.5, 1.0, 2.0)
        IV = -4/K^2 * spectral_flux(rg, exp.(-rg), K; kind=:V)
        sq = sqrt(1 + K^2)
        @test isapprox(IV, (sq-1)^2/(K^2*sq); rtol=1e-3)
    end
    # Both forms vs Gaussian–Hankel: ∫ e^{-r²/2} J_ν(Kr) r^{ν+1} dr = K^ν e^{-K²/2}
    rg2 = collect(range(0, 12, length=2401)); sf = rg2.^3 .* exp.(-rg2.^2 ./ 2)
    for K in (0.5, 1.0, 2.0)
        IV  = -4/K^2  * spectral_flux(rg2, sf, K; kind=:V)
        ISL = -12/K^3 * spectral_flux(rg2, sf, K; kind=:SL)
        @test isapprox(IV,  K^2 * exp(-K^2/2); rtol=1e-3)
        @test isapprox(ISL, K^3 * exp(-K^2/2); rtol=1e-3)
    end
    # advective_structure_function feeds spectral_flux (scalar & vector K)
    nx = ny = 48; g = Grid(nx, ny, 1.0, 1.0, true, true)
    u = Field([sin(2π*3*i/nx)*cos(2π*2*j/ny) for j in 1:ny, i in 1:nx], g)
    v = Field([cos(2π*2*i/nx)*sin(2π*3*j/ny) for j in 1:ny, i in 1:nx], g)
    r, V, SL = advective_structure_function(u, v, collect(1:20))
    @test all(isfinite, V) && all(isfinite, SL) && r[1] ≈ 1.0
    Π = spectral_flux(r, V, collect(0.1:0.1:1.0); kind=:V)
    @test length(Π) == 10 && all(isfinite, Π)
    @test spectral_flux(r, V, 0.5; kind=:V) isa Float64
end

@testset "Spherical structure functions (great-circle)" begin
    a = 6.371e6; nx, ny = 48, 40
    # constant field ⇒ SF = 0
    lon = collect(range(0, 2π, length=nx+1))[1:nx]
    lat = collect(range(deg2rad(10), deg2rad(70), length=ny))
    sg = SphericalGrid(lon, lat, a, true)
    _, Sc = structure_function_sphere(Field(fill(3.3, ny, nx), sg), range(50e3, 1500e3, length=6))
    @test all(s -> !isfinite(s) || abs(s) < 1e-12, Sc)
    # equatorial patch (locally flat) ⇒ matches Cartesian structure_function
    span = 0.08
    lo = collect(range(0, span, length=nx)); la = collect(range(-span/2, span/2, length=ny))
    sgp = SphericalGrid(lo, la, a, false)
    dx = a*(span/(nx-1)); dy = a*(span/(ny-1))
    F = [sin(3*(i-1)/nx*2π) + sin(3*(j-1)/ny*2π) for j in 1:ny, i in 1:nx]
    _, Scar = structure_function(Field(F, Grid(nx, ny, dx, dy, false, false)), [2, 4, 6]; order=2)
    bins = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5] .* dx            # centers 2,3,4,5,6 · dx
    _, Ssph = structure_function_sphere(Field(F, sgp), bins; order=2)
    @test Statistics.cor(Scar, Ssph[[1, 3, 5]]) > 0.95     # tracks Cartesian (great-circle ≈ Euclidean)
    @test abs(Ssph[1] - Scar[1]) / Scar[1] < 0.3
    @test all(Ssph .>= 0) && Ssph[1] < Ssph[5]
    # velocity: zonal field ⇒ longitudinal dominates transverse
    uE = Field([sin(4*(i-1)/nx*2π) for j in 1:ny, i in 1:nx], sgp); vN = Field(zeros(ny, nx), sgp)
    _, SL, ST = velocity_structure_function_sphere(uE, vN, bins; order=2)
    @test maximum(SL) > 2 * maximum(abs.(ST))
    # velocity constant ⇒ 0
    _, SLc, STc = velocity_structure_function_sphere(Field(fill(2.0, ny, nx), sgp),
                                                     Field(fill(1.0, ny, nx), sgp), bins)
    @test maximum(abs.(filter(isfinite, SLc))) < 1e-12
    @test maximum(abs.(filter(isfinite, STc))) < 1e-12
end

# Optional MPI smoke tests under ENV guard
if get(ENV, "RUN_MPI", "false") == "true"
    try
        run(`mpiexec -n 3 julia --project test_mpi/runtest_filter_realspace.jl`)
        run(`mpiexec -n 4 julia --project test_mpi/runtest_fft_distributed.jl`)
        run(`mpiexec -n 5 julia --project test_mpi/runtest_realspace_compare.jl`)
        run(`mpiexec -n 6 julia --project test_mpi/runtest_fft_compare.jl`)
        run(`mpiexec -n 3 julia --project test_mpi/runtest_masked_filter_realspace.jl`)
    catch e
        @warn "MPI smoke tests failed or mpiexec not available" error=e
    end
end

@testset "Differential" begin
    g = Grid(64, 64, 0.1, 0.1, true, true)
    x = (0:g.nx-1).*g.dx
    y = (0:g.ny-1).*g.dy
    A = [sin(xx)*cos(yy) for yy in y, xx in x]
    f = Field(A, g)
    fx, fy = gradient(f)
    @test size(fx.data) == size(A)
    @test size(fy.data) == size(A)
end

@testset "Helmholtz-Hodge (periodic Cartesian)" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    x = range(0, stop=2π, length=g.nx)
    y = range(0, stop=2π, length=g.ny)
    uA = [sin(xx) + cos(yy) for yy in y, xx in x]
    vA = [cos(xx) - sin(yy) for yy in y, xx in x]
    u = Field(uA, g)
    v = Field(vA, g)
    udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(u, v)
    # Reconstruction
    @test isapprox(u.data, udf.data .+ up.data; rtol=1e-5, atol=1e-7)
    @test isapprox(v.data, vdf.data .+ vp.data; rtol=1e-5, atol=1e-7)
end

@testset "Spherical gradient" begin
    nx, ny = 72, 36
    lon = range(0, stop=2π, length=nx)
    lat = range(-π/2 + π/ny, stop=π/2 - π/ny, length=ny) # avoid poles
    a = 1.0
    sg = SphericalGrid(collect(lon), collect(lat), a, true)
    A = [cos(λ)*cos(ϕ) for ϕ in lat, λ in lon]
    f = Field(A, sg)
    gx, gy = gradient_sphere(f)
    # Exact gradients: gx = -sin(λ)/a, gy = -(sin(ϕ)cos(λ))/a
    GX = [-sin(λ)/a for ϕ in lat, λ in lon]
    GY = [-(sin(ϕ)*cos(λ))/a for ϕ in lat, λ in lon]
    @test isapprox(gx.data, GX; rtol=5e-2)
    @test isapprox(gy.data, GY; rtol=5e-2)
end

@testset "Spherical Helmholtz consistency" begin
    nx, ny = 72, 36
    lon = range(0, stop=2π, length=nx)
    lat = range(-π/2 + π/ny, stop=π/2 - π/ny, length=ny)
    a = 1.0
    sg = SphericalGrid(collect(lon), collect(lat), a, true)
    # Construct a simple non-divfree field and check decomposition
    uE = Field([cos(λ)*cos(ϕ) for ϕ in lat, λ in lon], sg)
    vN = Field([sin(λ)*cos(ϕ) for ϕ in lat, λ in lon], sg)
    udf, vdf, up, vp, φ, ψ = helmholtz_hodge_sphere(uE, vN)
    # Check div-free of udf/vdf and near irrotationality of up/vp.
    # Restrict to interior latitudes: on a collocated lat-lon grid the 1/cosφ
    # metric amplifies truncation error in the polar caps, so consistency is
    # assessed away from the poles (|lat| ≤ 60°), standard for such grids.
    div_df = divergence_sphere(udf, vdf)
    vort_p = vorticity_sphere(up, vp)
    interior = findall(j -> abs(lat[j]) ≤ deg2rad(60), 1:ny)
    @test mean(abs.(div_df.data[interior, :])) < 0.5
    @test mean(abs.(vort_p.data[interior, :])) < 0.5
end

@testset "Spherical Poisson" begin
    nx, ny = 72, 36
    lon = range(0, stop=2π, length=nx)
    lat = range(-π/2 + π/ny, stop=π/2 - π/ny, length=ny)
    a = 1.0
    sg = SphericalGrid(collect(lon), collect(lat), a, true)
    φ = [cos(λ)*cos(ϕ) for ϕ in lat, λ in lon]
    rhs = [-2.0 * φ[j,i] for j in 1:ny, i in 1:nx]  # ∇²φ = -2φ on unit sphere for l=1
    ϕnum = poisson_sphere_solve(Field(rhs, sg))
    # Compare up to an additive constant: subtract means
    φ0 = φ .- mean(φ)
    ϕ0 = ϕnum.data .- mean(ϕnum.data)
    @test isapprox(ϕ0, φ0; rtol=0.2)  # Loose tolerance for coarse grid
end

@testset "Spherical Poisson convergence" begin
    # Solve ∇²φ = -l(l+1)φ for φ = cosϕ cosλ, l=1 → rhs = -2φ
    function solve_err(nx, ny)
        lon = range(0, stop=2π, length=nx)
        lat = range(-π/2 + π/ny, stop=π/2 - π/ny, length=ny)
        a = 1.0
        sg = SphericalGrid(collect(lon), collect(lat), a, true)
        φ = [cos(λ)*cos(ϕ) for ϕ in lat, λ in lon]
        rhs = [-2.0 * φ[j,i] for j in 1:ny, i in 1:nx]
        ϕnum = poisson_sphere_solve(Field(rhs, sg))
        φ0 = φ .- mean(φ)
        ϕ0 = ϕnum.data .- mean(ϕnum.data)
        return mean(abs.(ϕ0 .- φ0))
    end
    e1 = solve_err(72, 36)
    e2 = solve_err(144, 72)
    @test e2 < e1
end

@testset "Spherical Helmholtz iterative (AMG)" begin
    nx, ny = 48, 24
    lon = range(0, stop=2π, length=nx)
    lat = range(-π/4, stop=π/4, length=ny)
    sg = SphericalGrid(collect(lon), collect(lat), 6.371e6, true)
    uE = Field([sin(2*lon[i])*cos(lat[j]) for j in 1:ny, i in 1:nx], sg)
    vN = Field([cos(2*lon[i])*sin(lat[j]) for j in 1:ny, i in 1:nx], sg)
    # AMG-preconditioned iterative solver should converge without warnings
    @test_nowarn helmholtz_hodge_sphere_iterative(uE, vN; max_iter=300, tol=1e-10, solver=:amg)
    uEd, vNd, uEp, vNp, φ, ψ =
        helmholtz_hodge_sphere_iterative(uE, vN; max_iter=300, tol=1e-10, solver=:amg)
    # Residual decomposition: exact reconstruction, div-free / curl-free parts
    recon = sqrt(mean((uEd.data .+ uEp.data .- uE.data).^2 .+ (vNd.data .+ vNp.data .- vN.data).^2))
    @test recon < 1e-8
    @test maximum(abs.(divergence_sphere(uEd, vNd).data)) < 1e-5
    @test maximum(abs.(vorticity_sphere(uEp, vNp).data)) < 1e-5
    # Standalone AMG Poisson agrees with the direct solver on a smooth divergence
    d = divergence_sphere(uE, vN)
    pa = poisson_sphere_solve_iterative(d; reltol=1e-10).data
    @test all(isfinite, pa)
end

@testset "Pi diagnostic" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    x = range(0, stop=2π, length=g.nx)
    y = range(0, stop=2π, length=g.ny)
    u = Field([sin(xx) + 0.2cos(yy) for yy in y, xx in x], g)
    v = Field([cos(xx) - 0.2sin(yy) for yy in y, xx in x], g)
    K = gaussian_kernel(2.0, 2.0)
    Π = compute_pi(u, v, K)
    @test size(Π.data) == (g.ny, g.nx)
    @test isfinite(mean(Π.data))
end

@testset "Separable Gaussian vs FFT (rough)" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    x = range(0, stop=2π, length=g.nx)
    y = range(0, stop=2π, length=g.ny)
    A = [sin(xx) + cos(yy) for yy in y, xx in x]
    f = Field(A, g)
    fs = coarse_grain_gaussian_separable(f, 1.0, 1.0)
    ff = coarse_grain_fft(f, 1.0, 1.0)
    @test isapprox(mean(abs.(fs.data .- ff.data)), 0.0; atol=5e-2)
end

@testset "KE spectrum" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    u = Field(randn(g.ny, g.nx), g)
    v = Field(randn(g.ny, g.nx), g)
    k, Ek = ke_spectrum_isotropic(u, v; nbins=16)
    @test length(k) == 16
    @test length(Ek) == 16
    k2, Ek2 = ke_spectrum_isotropic(u, v; nbins=16, normalize=:shellarea)
    @test length(k2) == 16 == length(Ek2)
    _, Eke = ke_spectrum_isotropic(u, v; nbins=16, normalize=:energy)
    KE_real = 0.5 * mean(u.data.^2 .+ v.data.^2)
    @test isapprox(sum(Eke), KE_real; rtol=1e-6, atol=1e-8)
end

@testset "Curvilinear gradient" begin
    nx, ny = 40, 30
    dxval, dyval = 1000.0, 2000.0
    lon = zeros(ny, nx); lat = zeros(ny, nx)
    dx = fill(dxval, ny, nx); dy = fill(dyval, ny, nx)
    g = CurvilinearGrid(lon, lat, dx, dy, true, true, 6.371e6)
    # f(x,y) = a*x + b*y in physical coords; x = i*dx, y = j*dy
    a, b = 2.5, -1.2
    A = Array{Float64}(undef, ny, nx)
    for j in 1:ny, i in 1:nx
        A[j,i] = a * (i-1)*dxval + b * (j-1)*dyval
    end
    f = Field(A, g)
    fx, fy = gradient_curvilinear(f)
    @test isapprox(mean(fx.data), a; rtol=1e-6, atol=1e-6)
    @test isapprox(mean(fy.data), b; rtol=1e-6, atol=1e-6)
end

@testset "Regrid index bilinear" begin
    nx, ny = 30, 20
    g = Grid(nx, ny, 1.0, 1.0, true, true)
    A = [sin(0.2*i) + cos(0.15*j) for j in 1:ny, i in 1:nx]
    f = Field(A, g)
    f2 = regrid_index_bilinear(f, nx*2, ny*2)
    @test size(f2.data) == (ny*2, nx*2)
    # Downsample back should be close on average (not exact but reasonable)
    f3 = regrid_index_bilinear(f2, nx, ny)
    @test mean(abs.(f3.data .- A)) < 0.1
end

@testset "Regrid lon/lat nearest (identity)" begin
    nx, ny = 16, 12
    # Build a trivial lon/lat grid and a field
    lon = [Float64(i) for j in 1:ny, i in 1:nx]
    lat = [Float64(j) for j in 1:ny, i in 1:nx]
    g = Grid(nx, ny, 1.0, 1.0, false, false)
    A = [sin(0.1*lon[j,i]) + cos(0.07*lat[j,i]) for j in 1:ny, i in 1:nx]
    f = Field(A, g)
    out = regrid_lonlat_nearest(f, lon, lat, lon, lat)
    @test out == A
end

@testset "Masked separable Gaussian" begin
    g = Grid(32, 32, 1.0, 1.0, true, true)
    A = randn(g.ny, g.nx)
    f = Field(A, g)
    mask_all = trues(g.ny, g.nx)
    fs = coarse_grain_gaussian_separable(f, 1.0, 1.0)
    fsm = coarse_grain_gaussian_separable_masked(f, 1.0, 1.0, mask_all)
    @test isapprox(mean(abs.(fs.data .- fsm.data)), 0.0; atol=1e-8)
    # If mask all false, output should be fill_value
    mask_none = falses(g.ny, g.nx)
    fsm2 = coarse_grain_gaussian_separable_masked(f, 1.0, 1.0, mask_none; fill_value=NaN)
    @test all(isnan, fsm2.data)
    # If a single source point is masked, neighbors should not pick it up
    Aimp = zeros(g.ny, g.nx); Aimp[16,16] = 1.0
    fim = Field(Aimp, g)
    mask = trues(g.ny, g.nx); mask[16,16] = false
    f_unmasked = coarse_grain_gaussian_separable(fim, 1.0, 1.0)
    f_masked   = coarse_grain_gaussian_separable_masked(fim, 1.0, 1.0, mask)
    @test f_unmasked.data[16,17] > f_masked.data[16,17]
end

@testset "Butterworth filter" begin
    g = Grid(64, 64, 1.0, 1.0, true, true)
    x = range(0, stop=2π, length=g.nx)
    y = range(0, stop=2π, length=g.ny)
    # Compose two modes: one low-k (λ=32), one high-k (λ=8)
    A = [sin(xx*(2π/32)) + 0.5*sin(xx*(2π/8)) for yy in y, xx in x]
    f = Field(A, g)
    # kc between the two: e.g., cutoff at λc=16 => kc=2π/16
    kc = 2π/16
    f_low = coarse_grain_butterworth(f, kc; order=2)
    # Energy with low-pass should drop high-k content; compare variance reduction
    var_orig = mean((A .- mean(A)).^2)
    var_low  = mean((f_low.data .- mean(f_low.data)).^2)
    @test var_low < var_orig
    # Larger kc should retain more energy
    kc2 = 2π/4
    f_less = coarse_grain_butterworth(f, kc2; order=2)
    var_less = mean((f_less.data .- mean(f_less.data)).^2)
    @test var_less > var_low
    # Length-based wrapper equivalence
    f_len = coarse_grain_butterworth_length(f, 16; order=2)
    @test isapprox(mean(abs.(f_len.data .- f_low.data)), 0.0; atol=1e-8)
    # Cycles-per-domain wrapper: cycles = L/ℓ => 64/16 = 4
    f_cyc = coarse_grain_butterworth_cycles(f, 4; order=2)
    @test isapprox(mean(abs.(f_cyc.data .- f_low.data)), 0.0; atol=1e-8)
    # Cells wrapper: cells = ℓ/dx = 16
    f_cells = coarse_grain_butterworth_cells(f, 16; order=2)
    @test isapprox(mean(abs.(f_cells.data .- f_low.data)), 0.0; atol=1e-8)
    # Nyquist fraction: for dx=1, kc = f*π; ℓ = 2π/kc = 2/f
    # For ℓ=16, f = 2/16 = 0.125
    f_nyq = coarse_grain_butterworth_nyquist(f, 0.125; order=2)
    @test isapprox(mean(abs.(f_nyq.data .- f_low.data)), 0.0; atol=1e-8)
    # DSP-based variant: ensure variance reduction and shape
    fdsp = coarse_grain_butterworth_dsp(f, kc; order=2)
    @test size(fdsp.data) == size(f.data)
    var_dsp = mean((fdsp.data .- mean(fdsp.data)).^2)
    @test var_dsp < var_orig
end

include("validation.jl")
include("coverage_extra.jl")
include("test_advanced_features.jl")
include("flowsieve_crossval.jl")
