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
