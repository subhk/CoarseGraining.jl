# Cross-validation against FlowSieve (husseinaluie/FlowSieve), transcribing its exact
# formulas: Tests/distance_formulas.cpp, Functions/kernel.cpp, Tests/filtering_tests_spher.cpp.
# Test, CoarseGraining in scope from runtests.jl.

const _FS_R_EARTH = 6371e3

# FlowSieve Tests/distance_formulas.cpp :: distance_high_precision
function _fs_distance(lon1, lat1, lon2, lat2)
    Δλ = lon2 - lon1
    cl2, sl2, cl1, sl1, cΔ = cos(lat2), sin(lat2), cos(lat1), sin(lat1), cos(Δλ)
    numer = sqrt((cl2*sin(Δλ))^2 + (cl1*sl2 - sl1*cl2*cΔ)^2)
    denom = sl1*sl2 + cl1*cl2*cΔ
    return _FS_R_EARTH * atan(numer, denom)
end

# FlowSieve Functions/kernel.cpp (D = dist/(scale/2))
function _fs_kernel(dist, scale, kind)
    D = scale > 0 ? dist/(scale/2) : (dist == 0 ? 1.0 : 0.0)
    kind == :tophat         ? (D < 1 ? 1.0 : 0.0) :
    kind == :gaussian       ? exp(-D^2) :
    kind == :hyper_gaussian ? exp(-D^4) :
    kind == :smooth_hat     ? 0.5*(1 - tanh((D-1)/0.1)) : error("kernel")
end

@testset "FlowSieve cross-validation" begin
    sg = SphericalGrid([0.0], [0.0], _FS_R_EARTH, true)
    # 1. Great-circle distance matches FlowSieve's exact formula to machine precision
    for (λ1, ϕ1, λ2, ϕ2) in [(0.1, 0.2, 0.3, -0.1), (1.0, 0.5, -2.0, 0.9),
                             (-3.0, -1.4, 3.0, 1.4), (0.0, 0.0, π/2, 0.0)]
        @test isapprox(CoarseGraining.grid_distance(sg, ϕ1, λ1, ϕ2, λ2),
                       _fs_distance(λ1, ϕ1, λ2, ϕ2); rtol=1e-12, atol=1e-3)
    end
    # 2. Kernel weights match FlowSieve kernel.cpp exactly (all four shapes)
    scale = 2000e3; rad = scale/2
    for kind in (:tophat, :gaussian, :hyper_gaussian, :smooth_hat)
        for dist in (0.0, 0.3rad, 0.9rad, rad, 1.5rad, 3rad)
            @test isapprox(CoarseGraining._sphere_kernel(dist/rad, kind),
                           _fs_kernel(dist, scale, kind); atol=1e-13)
        end
    end
    # 3. Filtering test (replicates Tests/filtering_tests_spher.cpp): global grid,
    #    cell-centered lat/lon; constant conserved, smooth linear-lat preserved.
    Nlat, Nlon = 64, 128
    lat = [(-π/2) + (i-0.5)*(π/Nlat)  for i in 1:Nlat]
    lon = [(-π)   + (i-0.5)*(2π/Nlon) for i in 1:Nlon]
    sgf = SphericalGrid(collect(lon), collect(lat), _FS_R_EARTH, true)
    for kind in (:tophat, :gaussian)
        cf = coarse_grain_sphere(Field(fill(100.0, Nlat, Nlon), sgf), scale; kernel=kind)
        @test maximum(abs.(cf.data .- 100.0)) < 1e-6          # mass conservation
    end
    latfield = Field([2*lat[j] for j in 1:Nlat, i in 1:Nlon], sgf)
    filt = coarse_grain_sphere(latfield, scale; kernel=:tophat).data
    intr = 12:Nlat-11
    @test maximum(abs.(filt[intr, :] .- [2*lat[j] for j in intr, i in 1:Nlon])) < 0.05
end
