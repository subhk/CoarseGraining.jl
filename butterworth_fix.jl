"""
Corrected DSP.jl Butterworth filter implementation
"""

function coarse_grain_butterworth_dsp_fixed(field::Field{T,G}, kc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    @assert order >= 1
    A = Float64.(field.data)
    ny, nx = size(A)
    dx = field.grid.dx; dy = field.grid.dy
    
    if kc isa Tuple
        kcx, kcy = kc
    else
        kcx = kcy = Float64(kc)  # FIX: Correct assignment
    end
    
    # FIX: Correct frequency normalization for DSP.jl
    # DSP.jl expects normalized frequency ω ∈ [0, 1] where 1 = Nyquist
    # Nyquist frequency in our case is π/dx and π/dy
    # So for cutoff kcx, normalized freq is: ωx = kcx/(π/dx) = kcx*dx/π
    # But we need to be careful about the 2π factor difference
    
    # The relationship should be: ω_normalized = k_physical / k_nyquist
    # where k_nyquist = π/dx (since we're dealing with real signals)
    wx = clamp(kcx * dx / π, 0.0, 1.0)
    wy = clamp(kcy * dy / π, 0.0, 1.0)
    
    # Note: This creates separable filters, not the same as 2D isotropic
    fx = digitalfilter(Lowpass(wx), Butterworth(order))
    fy = digitalfilter(Lowpass(wy), Butterworth(order))
    
    if zero_phase
        B = DSP.filtfilt(fx, A; dims=2)  # Filter along x (columns)
        C = DSP.filtfilt(fy, B; dims=1)  # Filter along y (rows)
    else
        B = DSP.filt(fx, A; dims=2)
        C = DSP.filt(fy, B; dims=1)
    end
    
    return Field(T.(C), field.grid)
end

"""
Alternative: True 2D Butterworth using FFT to match the FFT implementation exactly
"""
function coarse_grain_butterworth_dsp_2d(field::Field{T,G}, kc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2) where {T<:Real,G}
    # This is essentially the same as the existing FFT implementation
    # but using DSP.jl conventions if needed
    return coarse_grain_butterworth(field, kc; order=order)
end

"""
Hybrid approach: Use separable approximation but with correct frequency mapping
"""
function coarse_grain_butterworth_dsp_separable_corrected(field::Field{T,G}, kc::Union{Real,Tuple{<:Real,<:Real}}; order::Integer=2, zero_phase::Bool=true) where {T<:Real,G}
    @assert order >= 1
    A = Float64.(field.data)
    ny, nx = size(A)
    dx = field.grid.dx; dy = field.grid.dy
    
    if kc isa Tuple
        kcx, kcy = kc
    else
        # For separable approximation of isotropic filter:
        # Use kcx = kcy = kc/sqrt(2) to better approximate 2D response
        kc_sep = Float64(kc) / sqrt(2.0)
        kcx = kcy = kc_sep
    end
    
    # Correct frequency normalization
    wx = clamp(kcx * dx / π, 0.0, 1.0)
    wy = clamp(kcy * dy / π, 0.0, 1.0)
    
    fx = digitalfilter(Lowpass(wx), Butterworth(order))
    fy = digitalfilter(Lowpass(wy), Butterworth(order))
    
    if zero_phase
        B = DSP.filtfilt(fx, A; dims=2)
        C = DSP.filtfilt(fy, B; dims=1)
    else
        B = DSP.filt(fx, A; dims=2)
        C = DSP.filt(fy, B; dims=1)
    end
    
    return Field(T.(C), field.grid)
end