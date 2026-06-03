# Analytic validation suite — locks in the operator/Helmholtz/filter correctness.
# Included from runtests.jl (Test, CoarseGraining, Statistics already in scope).

@testset "Validation: spherical operators vs analytic" begin
    a = 6.371e6; nx, ny = 96, 72
    lon = collect(range(0, 2π, length=nx+1))[1:nx]
    lat = collect(range(-π/3, π/3, length=ny))
    sg = SphericalGrid(lon, lat, a, true)
    relerr(num, ana) = sqrt(sum((num[3:end-2, :] .- ana[3:end-2, :]).^2)) /
                       sqrt(sum(ana[3:end-2, :].^2))
    # gradient: f = sinφ ⇒ north component = cosφ/a
    _, gN = gradient_sphere(Field([sin(lat[j]) for j in 1:ny, i in 1:nx], sg))
    @test relerr(gN.data, [cos(lat[j])/a for j in 1:ny, i in 1:nx]) < 0.01
    # divergence: uniform northward flow ⇒ -tanφ/a
    dn = divergence_sphere(Field(zeros(ny, nx), sg), Field(ones(ny, nx), sg)).data
    @test relerr(dn, [-tan(lat[j])/a for j in 1:ny, i in 1:nx]) < 0.01
    # vorticity: v_N = sinλ ⇒ cosλ/(a cosφ)
    vt = vorticity_sphere(Field(zeros(ny, nx), sg),
                          Field([sin(lon[i]) for j in 1:ny, i in 1:nx], sg)).data
    @test relerr(vt, [cos(lon[i])/(a*cos(lat[j])) for j in 1:ny, i in 1:nx]) < 0.01
    # Laplacian on Y₂² ∝ cos²φ cos2λ ⇒ -6/a² · Y
    f2 = [cos(lat[j])^2 * cos(2*lon[i]) for j in 1:ny, i in 1:nx]
    L = CoarseGraining.build_spherical_laplacian(sg)
    Lf = reshape(L * vec(f2), ny, nx)              # idx is column-major ⇒ vec/reshape match
    @test relerr(Lf, -6/a^2 .* f2) < 0.02
end

@testset "Validation: FFT Gaussian transfer function" begin
    nx = ny = 64; dx = 1.0
    g = Grid(nx, ny, dx, dx, true, true)
    m = 4; k0 = 2π*m/(nx*dx); σ = 2.0
    A = [sin(k0*(i-1)*dx) for j in 1:ny, i in 1:nx]
    ff = coarse_grain_fft(Field(A, g), σ, σ)
    damp = exp(-0.5 * σ^2 * k0^2)                  # analytic Gaussian transfer
    @test maximum(abs.(ff.data .- damp .* A)) < 1e-6
end

@testset "Validation: Helmholtz reconstruction is exact" begin
    nx = ny = 32
    # Cartesian periodic (spectral projection)
    g = Grid(nx, ny, 1.0, 1.0, true, true)
    u = Field([cos(2π*i/nx) + 0.3sin(2π*j/ny) for j in 1:ny, i in 1:nx], g)
    v = Field([sin(2π*j/ny) - 0.2cos(2π*i/nx) for j in 1:ny, i in 1:nx], g)
    udf, vdf, up, vp, _, _ = helmholtz_hodge(u, v)
    @test maximum(abs.(udf.data .+ up.data .- u.data)) < 1e-8
    @test maximum(abs.(vdf.data .+ vp.data .- v.data)) < 1e-8
    # Spherical residual reconstruction
    a = 6.371e6
    lon = collect(range(0, 2π, length=nx+1))[1:nx]
    lat = collect(range(-π/4, π/4, length=ny))
    sg = SphericalGrid(lon, lat, a, true)
    uE = Field([sin(2*lon[i])*cos(lat[j]) for j in 1:ny, i in 1:nx], sg)
    vN = Field([cos(2*lon[i])*sin(lat[j]) for j in 1:ny, i in 1:nx], sg)
    ud, vd, upp, vpp, _, _ = helmholtz_hodge_sphere(uE, vN)
    @test maximum(abs.(ud.data .+ upp.data .- uE.data)) < 1e-8
    @test maximum(abs.(vd.data .+ vpp.data .- vN.data)) < 1e-8
end

@testset "Validation: KE spectrum peaks at the input mode" begin
    nx = ny = 64
    g = Grid(nx, ny, 1.0, 1.0, true, true)
    m = 8; k0 = 2π*m/nx
    u = Field([sin(k0*(i-1)) for j in 1:ny, i in 1:nx], g)
    v = Field(zeros(ny, nx), g)
    k, Ek = compute_kinetic_energy_spectra(u, v; method=:isotropic)
    @test k[argmax(Ek)] ≈ k0 atol=0.15
end

# FlowSieve reference-data cross-validation is a follow-up: it needs the published
# FlowSieve sample datasets, which can't be bundled here.
