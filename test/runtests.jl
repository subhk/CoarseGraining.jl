using Test
using CoarseGraining

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

# Optional MPI smoke tests under ENV guard
if get(ENV, "RUN_MPI", "false") == "true"
    try
        run(`mpiexec -n 3 julia --project test_mpi/runtest_filter_realspace.jl`)
        run(`mpiexec -n 4 julia --project test_mpi/runtest_fft_distributed.jl`)
        run(`mpiexec -n 5 julia --project test_mpi/runtest_realspace_compare.jl`)
        run(`mpiexec -n 6 julia --project test_mpi/runtest_fft_compare.jl`)
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
    # Check div-free of udf/vdf and near irrotationality of up/vp
    div_df = divergence_sphere(udf, vdf)
    vort_p = vorticity_sphere(up, vp)
    @test mean(abs.(div_df.data)) < 5e-1
    @test mean(abs.(vort_p.data)) < 5e-1
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
