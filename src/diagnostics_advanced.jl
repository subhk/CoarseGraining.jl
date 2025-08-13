using ..CoarseGraining: Field, Grid, SphericalGrid, gradient, gradient_sphere, coarse_grain
using ..CoarseGraining: divergence, vorticity, divergence_sphere, vorticity_sphere

export compute_energy_transport, compute_baroclinic_transfer, compute_viscous_dissipation,
       compute_pressure_transport, compute_advection_transport, compute_full_energy_budget,
       compute_kinetic_energy_spectra, compute_enstrophy_transfer

"""
    compute_energy_transport(u, v, kernel; ρ₀=1025.0)

Compute full kinetic energy transport terms following FlowSieve methodology.
Returns the divergence of kinetic energy flux: ∇·J_transport

From Aluie et al. (2018), the transport includes:
- Advection by coarse-scale flow: ½ρ₀|ū|²ū  
- Pressure transport: P̄ū
- Advection by fine-scale flow: ρ₀ū·τ(u,u)
- Diffusion: -ν∇(½ρ₀|ū|²) [if viscosity provided]

Parameters:
- `u`, `v`: Velocity components  
- `kernel`: Coarse-graining kernel
- `ρ₀`: Reference density (kg/m³)

Returns:
- `J_coarse`: Advection by coarse-scale flow
- `J_pressure`: Pressure-induced transport  
- `J_subgrid`: Advection by fine-scale (subgrid) flow
- `div_J_total`: Total transport divergence
"""
function compute_energy_transport(u::Field{T,G}, v::Field{T,G}, kernel; 
                                 ρ₀::Real=1025.0, pressure::Union{Field,Nothing}=nothing) where {T<:Real,G}
    
    # Filter velocities
    ū = coarse_grain(u, kernel)
    v̄ = coarse_grain(v, kernel)
    
    # Compute kinetic energy
    KE_coarse = Field(0.5 * ρ₀ * (ū.data.^2 .+ v̄.data.^2), ū.grid)
    
    # 1. Advection by coarse-scale flow: ∇·(½ρ₀|ū|²ū)
    KEu_field = Field(KE_coarse.data .* ū.data, ū.grid)
    KEv_field = Field(KE_coarse.data .* v̄.data, ū.grid)
    
    if ū.grid isa SphericalGrid
        J_coarse = divergence_sphere(KEu_field, KEv_field)
    else
        J_coarse = divergence(KEu_field, KEv_field)
    end
    
    # 2. Pressure transport: ∇·(P̄ū)
    if pressure !== nothing
        P̄ = coarse_grain(pressure, kernel)
        Pu_field = Field(P̄.data .* ū.data, ū.grid)
        Pv_field = Field(P̄.data .* v̄.data, ū.grid)
        
        if ū.grid isa SphericalGrid
            J_pressure = divergence_sphere(Pu_field, Pv_field)
        else
            J_pressure = divergence(Pu_field, Pv_field)
        end
    else
        J_pressure = Field(zeros(T, size(ū.data)), ū.grid)
    end
    
    # 3. Subgrid transport: ∇·(ρ₀ū·τ)
    # Compute Reynolds stress τᵢⱼ = ūᵢuⱼ - ūᵢūⱼ
    uu = Field(u.data .* u.data, u.grid)
    uv = Field(u.data .* v.data, u.grid)
    vv = Field(v.data .* v.data, u.grid)
    
    ūū = coarse_grain(uu, kernel)
    ūv̄ = coarse_grain(uv, kernel)
    v̄v̄ = coarse_grain(vv, kernel)
    
    τ₁₁ = Field(ūū.data .- ū.data .* ū.data, ū.grid)  # τₓₓ
    τ₁₂ = Field(ūv̄.data .- ū.data .* v̄.data, ū.grid)  # τₓᵧ  
    τ₂₂ = Field(v̄v̄.data .- v̄.data .* v̄.data, ū.grid)  # τᵧᵧ
    
    # ∇·(ρ₀ū·τ) = ρ₀[∂(ūτₓₓ + v̄τₓᵧ)/∂x + ∂(ūτₓᵧ + v̄τᵧᵧ)/∂y]
    subgrid_x = Field(ρ₀ * (ū.data .* τ₁₁.data .+ v̄.data .* τ₁₂.data), ū.grid)
    subgrid_y = Field(ρ₀ * (ū.data .* τ₁₂.data .+ v̄.data .* τ₂₂.data), ū.grid)
    
    if ū.grid isa SphericalGrid
        J_subgrid = divergence_sphere(subgrid_x, subgrid_y)
    else
        J_subgrid = divergence(subgrid_x, subgrid_y)
    end
    
    # Total transport divergence
    div_J_total = Field(J_coarse.data .+ J_pressure.data .+ J_subgrid.data, ū.grid)
    
    return J_coarse, J_pressure, J_subgrid, div_J_total
end

"""
    compute_baroclinic_transfer(u, v, ρ, kernel; g=9.81, ρ₀=1025.0)

Compute baroclinic energy transfer terms for stratified flows.
Implements the pressure-density correlation terms from FlowSieve.

For the barotropic-baroclinic energy exchange:
Π_bc = -⟨w'ρ'⟩g = potential energy conversion to kinetic energy

Parameters:
- `u`, `v`: Horizontal velocity components
- `ρ`: Density field  
- `kernel`: Coarse-graining kernel
- `g`: Gravitational acceleration
- `ρ₀`: Reference density

Returns:
- `Pi_bc`: Baroclinic conversion term
- `APE_flux`: Available potential energy flux
"""
function compute_baroclinic_transfer(u::Field{T,G}, v::Field{T,G}, ρ::Field{T,G}, 
                                   w::Union{Field{T,G},Nothing}, kernel; 
                                   g::Real=9.81, ρ₀::Real=1025.0) where {T<:Real,G}
    
    # Filter fields
    ū = coarse_grain(u, kernel)
    v̄ = coarse_grain(v, kernel)
    ρ̄ = coarse_grain(ρ, kernel)
    
    if w !== nothing
        w̄ = coarse_grain(w, kernel)
        
        # Compute density fluctuations
        ρ_prime = Field(ρ.data .- ρ̄.data, ρ.grid)
        w_prime = Field(w.data .- w̄.data, w.grid)
        
        # Baroclinic conversion: -g⟨w'ρ'⟩
        wρ = Field(w.data .* ρ.data, w.grid)
        w̄ρ̄_mean = coarse_grain(wρ, kernel)
        
        Pi_bc = Field(-g * (w̄ρ̄_mean.data .- w̄.data .* ρ̄.data), ū.grid)
        
        # APE flux (simplified)
        APE_flux = Field(g * ρ_prime.data .* w_prime.data, ū.grid)
    else
        # No vertical velocity provided - set to zero
        Pi_bc = Field(zeros(T, size(ū.data)), ū.grid)
        APE_flux = Field(zeros(T, size(ū.data)), ū.grid)
    end
    
    return Pi_bc, APE_flux
end

"""
    compute_viscous_dissipation(u, v, kernel; ν=1e-6, ρ₀=1025.0)

Compute viscous dissipation terms for the kinetic energy budget.
Includes both resolved and subgrid-scale dissipation.

Viscous dissipation rate: ε = ρ₀ν(∂uᵢ/∂xⱼ)(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)

Parameters:
- `u`, `v`: Velocity components
- `kernel`: Coarse-graining kernel  
- `ν`: Kinematic viscosity (m²/s)
- `ρ₀`: Reference density

Returns:
- `eps_resolved`: Resolved-scale dissipation
- `eps_subgrid`: Subgrid-scale dissipation  
- `eps_total`: Total dissipation
"""
function compute_viscous_dissipation(u::Field{T,G}, v::Field{T,G}, kernel; 
                                   ν::Real=1e-6, ρ₀::Real=1025.0) where {T<:Real,G}
    
    # Filter velocities
    ū = coarse_grain(u, kernel)
    v̄ = coarse_grain(v, kernel)
    
    # Compute velocity gradients  
    if ū.grid isa SphericalGrid
        ∂ū∂E, ∂ū∂N = gradient_sphere(ū)
        ∂v̄∂E, ∂v̄∂N = gradient_sphere(v̄)
    else
        ∂ū∂x, ∂ū∂y = gradient(ū)
        ∂v̄∂x, ∂v̄∂y = gradient(v̄)
        ∂ū∂E, ∂ū∂N = ∂ū∂x, ∂ū∂y
        ∂v̄∂E, ∂v̄∂N = ∂v̄∂x, ∂v̄∂y
    end
    
    # Strain rate tensor components for filtered velocity
    S₁₁ = ∂ū∂E.data
    S₂₂ = ∂v̄∂N.data  
    S₁₂ = 0.5 * (∂ū∂N.data .+ ∂v̄∂E.data)
    
    # Resolved-scale dissipation: ε_res = 2ρ₀νSᵢⱼSᵢⱼ
    strain_mag_sq = S₁₁.^2 .+ S₂₂.^2 .+ 2*S₁₂.^2
    eps_resolved = Field(2 * ρ₀ * ν * strain_mag_sq, ū.grid)
    
    # Subgrid-scale dissipation (using Smagorinsky-like model)
    # This would require a subgrid model - simplified version here
    if ū.grid isa Grid
        Δx = ū.grid.dx
        Δy = ū.grid.dy
        Δ = sqrt(Δx * Δy)  # Grid scale
    else
        # For spherical grids, estimate grid scale
        a = ū.grid.a
        dλ = mean(diff(ū.grid.lon))
        dφ = mean(diff(ū.grid.lat))
        Δ = sqrt((a*dλ)^2 + (a*dφ)^2)
    end
    
    # Smagorinsky subgrid viscosity: ν_sgs = (Cs*Δ)²|S̄|
    Cs = 0.1  # Smagorinsky constant
    strain_mag = sqrt.(strain_mag_sq)
    ν_sgs = (Cs * Δ)^2 * strain_mag
    eps_subgrid = Field(2 * ρ₀ * ν_sgs .* strain_mag_sq, ū.grid)
    
    # Total dissipation
    eps_total = Field(eps_resolved.data .+ eps_subgrid.data, ū.grid)
    
    return eps_resolved, eps_subgrid, eps_total
end

"""
    compute_full_energy_budget(u, v, kernel; pressure=nothing, ρ=nothing, w=nothing, 
                              ν=1e-6, g=9.81, ρ₀=1025.0)

Compute complete kinetic energy budget following Aluie et al. (2018):
∂/∂t(½ρ₀|ū|²) + ∇·J = -Π - ε + Bc + F

Where:
- Π: Scale transfer (Leonard term)
- ε: Viscous dissipation  
- Bc: Baroclinic conversion
- F: External forcing
- J: Transport terms

Returns a named tuple with all budget terms.
"""
function compute_full_energy_budget(u::Field{T,G}, v::Field{T,G}, kernel;
                                   pressure::Union{Field,Nothing}=nothing,
                                   ρ::Union{Field,Nothing}=nothing, 
                                   w::Union{Field,Nothing}=nothing,
                                   forcing_u::Union{Field,Nothing}=nothing,
                                   forcing_v::Union{Field,Nothing}=nothing,
                                   ν::Real=1e-6, g::Real=9.81, ρ₀::Real=1025.0) where {T<:Real,G}
    
    # Leonard energy transfer (scale interactions)
    Π = compute_pi(u, v, kernel)
    
    # Transport terms
    J_coarse, J_pressure, J_subgrid, div_J_total = 
        compute_energy_transport(u, v, kernel; ρ₀=ρ₀, pressure=pressure)
    
    # Viscous dissipation
    eps_resolved, eps_subgrid, eps_total = 
        compute_viscous_dissipation(u, v, kernel; ν=ν, ρ₀=ρ₀)
    
    # Baroclinic conversion
    if ρ !== nothing && w !== nothing
        Pi_bc, APE_flux = compute_baroclinic_transfer(u, v, ρ, w, kernel; g=g, ρ₀=ρ₀)
    else
        Pi_bc = Field(zeros(T, size(Π.data)), Π.grid)
        APE_flux = Field(zeros(T, size(Π.data)), Π.grid)
    end
    
    # External forcing work
    if forcing_u !== nothing && forcing_v !== nothing
        ū = coarse_grain(u, kernel)
        v̄ = coarse_grain(v, kernel)
        F̄u = coarse_grain(forcing_u, kernel)
        F̄v = coarse_grain(forcing_v, kernel)
        
        forcing_work = Field(ρ₀ * (ū.data .* F̄u.data .+ v̄.data .* F̄v.data), ū.grid)
    else
        forcing_work = Field(zeros(T, size(Π.data)), Π.grid)
    end
    
    # Kinetic energy
    ū = coarse_grain(u, kernel)
    v̄ = coarse_grain(v, kernel)
    KE = Field(0.5 * ρ₀ * (ū.data.^2 .+ v̄.data.^2), ū.grid)
    
    return (
        kinetic_energy = KE,
        leonard_transfer = Π,
        transport_coarse = J_coarse,
        transport_pressure = J_pressure, 
        transport_subgrid = J_subgrid,
        transport_total = div_J_total,
        dissipation_resolved = eps_resolved,
        dissipation_subgrid = eps_subgrid,
        dissipation_total = eps_total,
        baroclinic_conversion = Pi_bc,
        ape_flux = APE_flux,
        forcing_work = forcing_work
    )
end

"""
    compute_kinetic_energy_spectra(u, v; method=:isotropic)

Compute kinetic energy spectral density following FlowSieve methodology.
Returns wavenumber and corresponding spectral density.

Parameters:
- `u`, `v`: Velocity components on Grid (must be periodic)
- `method`: :isotropic for radially-averaged spectrum, :anisotropic for 2D spectrum

Returns:
- `k`: Wavenumber array
- `E_k`: Kinetic energy spectral density
"""
function compute_kinetic_energy_spectra(u::Field{T,G}, v::Field{T,G}; 
                                       method::Symbol=:isotropic) where {T<:Real,G<:Grid}
    @assert u.grid.periodic_x && u.grid.periodic_y "Spectral analysis requires periodic boundaries"
    
    ny, nx = size(u.data)
    dx, dy = u.grid.dx, u.grid.dy
    
    # Fourier transform of velocity components
    û = fft(u.data)
    v̂ = fft(v.data)
    
    # Kinetic energy in spectral space: ½|û|² + ½|v̂|²
    KE_spec = 0.5 * (abs2.(û) .+ abs2.(v̂))
    
    # Wavenumber grids
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*dy))
    
    if method == :isotropic
        # Radially average the spectrum
        k_max = min(π/dx, π/dy)
        k_bins = range(0, k_max, length=min(nx, ny)÷2)
        E_k = zeros(length(k_bins)-1)
        k_centers = 0.5 * (k_bins[1:end-1] .+ k_bins[2:end])
        
        for j in 1:ny, i in 1:nx
            k_mag = sqrt(kx[i]^2 + ky[j]^2)
            if k_mag > 0
                bin_idx = searchsortedfirst(k_bins, k_mag) - 1
                if 1 ≤ bin_idx ≤ length(E_k)
                    E_k[bin_idx] += KE_spec[j, i]
                end
            end
        end
        
        return k_centers, E_k
    else
        # Return full 2D spectrum
        KX = reshape(kx, 1, :)
        KY = reshape(ky, :, 1)
        k_mag = sqrt.(KX.^2 .+ KY.^2)
        return k_mag, KE_spec
    end
end

"""
    compute_enstrophy_transfer(u, v, kernel)

Compute enstrophy (vorticity squared) transfer across scales.
Similar to kinetic energy transfer but for enstrophy conservation.

Returns enstrophy transfer rate and transport terms.
"""
function compute_enstrophy_transfer(u::Field{T,G}, v::Field{T,G}, kernel) where {T<:Real,G}
    
    # Compute vorticity
    if u.grid isa SphericalGrid
        ω = vorticity_sphere(u, v)
    else
        ω = vorticity(u, v)
    end
    
    # Filter vorticity and velocity
    ω̄ = coarse_grain(ω, kernel)
    ū = coarse_grain(u, kernel)
    v̄ = coarse_grain(v, kernel)
    
    # Enstrophy transfer: Π_ω = -⟨ω'∇'⟩·∇ω̄
    # Simplified computation similar to Leonard term
    ωω = Field(ω.data .* ω.data, ω.grid)
    ω̄ω̄_mean = coarse_grain(ωω, kernel)
    
    Π_enstrophy = Field(ω̄ω̄_mean.data .- ω̄.data.^2, ω̄.grid)
    
    return Π_enstrophy, ω̄
end