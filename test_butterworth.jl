#!/usr/bin/env julia

using CoarseGraining
using FFTW
using Plots

"""
Test script to compare DSP.jl and FFT-based Butterworth implementations
"""

function test_butterworth_implementations()
    # Create test data
    nx, ny = 256, 256
    dx, dy = 1.0, 1.0
    g = Grid(nx, ny, dx, dy, true, true)
    
    # Create test signal with multiple frequency components
    x = range(0, stop=2π, length=nx)
    y = range(0, stop=2π, length=ny)
    
    # Test signal: sum of sinusoids at different scales
    A = [sin(2*xx) + 0.5*sin(8*xx) + 0.25*sin(16*yy) + 0.1*randn() for yy in y, xx in x]
    field = Field(A, g)
    
    # Test parameters
    cutoff_length = 8.0  # cutoff length scale
    kc = 2π/cutoff_length  # corresponding wavenumber
    order = 4
    
    println("Testing Butterworth filter implementations...")
    println("Domain size: $(nx)×$(ny), dx=$(dx), dy=$(dy)")
    println("Cutoff length: $(cutoff_length), cutoff wavenumber: $(kc)")
    println("Filter order: $(order)")
    
    # FFT-based Butterworth
    println("\n1. FFT-based Butterworth (periodic boundaries):")
    fft_result = coarse_grain_butterworth_length(field, cutoff_length; order=order)
    println("   Result range: [$(minimum(fft_result.data)), $(maximum(fft_result.data))]")
    
    # DSP.jl Butterworth (zero-phase)
    println("\n2. DSP.jl Butterworth (zero-phase, separable):")
    dsp_result = coarse_grain_butterworth_length_dsp(field, cutoff_length; order=order, zero_phase=true)
    println("   Result range: [$(minimum(dsp_result.data)), $(maximum(dsp_result.data))]")
    
    # DSP.jl Butterworth (with phase)
    println("\n3. DSP.jl Butterworth (with phase, separable):")
    dsp_phase_result = coarse_grain_butterworth_length_dsp(field, cutoff_length; order=order, zero_phase=false)
    println("   Result range: [$(minimum(dsp_phase_result.data)), $(maximum(dsp_phase_result.data))]")
    
    # Compare results
    diff_fft_dsp = fft_result.data .- dsp_result.data
    diff_fft_dsp_phase = fft_result.data .- dsp_phase_result.data
    
    println("\n4. Comparison:")
    println("   RMS difference (FFT vs DSP zero-phase): $(sqrt(mean(diff_fft_dsp.^2)))")
    println("   Max difference (FFT vs DSP zero-phase): $(maximum(abs.(diff_fft_dsp)))")
    println("   RMS difference (FFT vs DSP with-phase): $(sqrt(mean(diff_fft_dsp_phase.^2)))")
    println("   Max difference (FFT vs DSP with-phase): $(maximum(abs.(diff_fft_dsp_phase)))")
    
    # Test frequency response
    println("\n5. Frequency Response Analysis:")
    test_frequency_response(g, cutoff_length, order)
    
    return fft_result, dsp_result, dsp_phase_result
end

function test_frequency_response(grid, cutoff_length, order)
    """Test the frequency response of both implementations"""
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    kc = 2π/cutoff_length
    
    # Create impulse response by filtering a delta function
    delta = zeros(ny, nx)
    delta[ny÷2, nx÷2] = 1.0
    delta_field = Field(delta, grid)
    
    # Filter the delta function
    fft_impulse = coarse_grain_butterworth_length(delta_field, cutoff_length; order=order)
    dsp_impulse = coarse_grain_butterworth_length_dsp(delta_field, cutoff_length; order=order, zero_phase=true)
    
    # Compute power spectral density
    fft_spectrum = abs2.(fft(fft_impulse.data))
    dsp_spectrum = abs2.(fft(dsp_impulse.data))
    
    # Theoretical Butterworth response
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*dy))
    KX = reshape(kx, 1, :)
    KY = reshape(ky, :, 1)
    K_mag = @. sqrt(KX^2 + KY^2)
    theoretical_response = @. 1.0 / (1.0 + (K_mag/kc)^(2*order))
    
    # Compare at cutoff frequency
    cutoff_idx_x = argmin(abs.(kx .- kc))
    cutoff_idx_y = argmin(abs.(ky))
    
    println("   At cutoff frequency (kc = $(kc)):")
    println("   Theoretical response: $(theoretical_response[cutoff_idx_y, cutoff_idx_x])")
    println("   FFT implementation: $(fft_spectrum[cutoff_idx_y, cutoff_idx_x] / fft_spectrum[1,1])")
    println("   DSP implementation: $(dsp_spectrum[cutoff_idx_y, cutoff_idx_x] / dsp_spectrum[1,1])")
end

function test_edge_effects()
    """Test edge effects and boundary handling"""
    println("\n6. Edge Effects Test:")
    
    # Small domain to emphasize edge effects
    nx, ny = 64, 64
    g = Grid(nx, ny, 1.0, 1.0, false, false)  # Non-periodic
    
    # Step function to test edge handling
    A = zeros(ny, nx)
    A[:, 1:nx÷2] .= 1.0
    field = Field(A, g)
    
    cutoff_length = 8.0
    
    # DSP.jl will have edge effects due to IIR nature
    dsp_result = coarse_grain_butterworth_length_dsp(field, cutoff_length; order=2, zero_phase=true)
    
    println("   Original step function range: [$(minimum(field.data)), $(maximum(field.data))]")
    println("   DSP filtered range: [$(minimum(dsp_result.data)), $(maximum(dsp_result.data))]")
    println("   Edge effects visible: $(any(dsp_result.data .< -0.01) || any(dsp_result.data .> 1.01))")
    
    return dsp_result
end

function test_parameter_conversion()
    """Test parameter conversion consistency"""
    println("\n7. Parameter Conversion Test:")
    
    nx, ny = 128, 128
    g = Grid(nx, ny, 2.0, 1.5, true, true)  # Non-unit spacing
    
    # Test data
    A = randn(ny, nx)
    field = Field(A, g)
    
    # Different parameter specifications should give equivalent results
    ℓc = 10.0
    kc = 2π/ℓc
    cycles = ℓc/(nx*g.dx)  # cycles per domain in x
    cells = ℓc/g.dx        # cells in x
    frac = kc/(π/g.dx)     # fraction of Nyquist
    
    # Test FFT implementations
    r1 = coarse_grain_butterworth(field, kc; order=2)
    r2 = coarse_grain_butterworth_length(field, ℓc; order=2)
    r3 = coarse_grain_butterworth_cycles(field, (cycles, cycles*g.dx/g.dy); order=2)
    r4 = coarse_grain_butterworth_cells(field, (cells, cells*g.dx/g.dy); order=2)
    r5 = coarse_grain_butterworth_nyquist(field, (frac, frac*g.dx/g.dy); order=2)
    
    println("   FFT parameter consistency:")
    println("   RMS diff (kc vs length): $(sqrt(mean((r1.data .- r2.data).^2)))")
    println("   RMS diff (kc vs cycles): $(sqrt(mean((r1.data .- r3.data).^2)))")
    println("   RMS diff (kc vs cells): $(sqrt(mean((r1.data .- r4.data).^2)))")
    println("   RMS diff (kc vs nyquist): $(sqrt(mean((r1.data .- r5.data).^2)))")
    
    # Test DSP implementations  
    d2 = coarse_grain_butterworth_length_dsp(field, ℓc; order=2)
    d3 = coarse_grain_butterworth_cycles_dsp(field, (cycles, cycles*g.dx/g.dy); order=2)
    d4 = coarse_grain_butterworth_cells_dsp(field, (cells, cells*g.dx/g.dy); order=2)
    d5 = coarse_grain_butterworth_nyquist_dsp(field, (frac, frac*g.dx/g.dy); order=2)
    
    println("   DSP parameter consistency:")
    println("   RMS diff (length vs cycles): $(sqrt(mean((d2.data .- d3.data).^2)))")
    println("   RMS diff (length vs cells): $(sqrt(mean((d2.data .- d4.data).^2)))")
    println("   RMS diff (length vs nyquist): $(sqrt(mean((d2.data .- d5.data).^2)))")
end

# Run all tests
println("=" ^ 60)
println("BUTTERWORTH FILTER IMPLEMENTATION TEST")
println("=" ^ 60)

fft_result, dsp_result, dsp_phase_result = test_butterworth_implementations()
edge_result = test_edge_effects()
test_parameter_conversion()

println("\n" * "=" * 60)
println("TEST COMPLETE")
println("=" * 60)