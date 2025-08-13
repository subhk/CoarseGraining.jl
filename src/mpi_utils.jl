module MPIUtils

using MPI
using ..CoarseGraining: Field, Grid, Kernel, coarse_grain, coarse_grain_fft, coarse_grain_masked
using ..CoarseGraining: helmholtz_hodge

export mpi_init, mpi_finalize, mpi_enabled, mpi_rank, mpi_size,
       parallel_coarse_grain, parallel_coarse_grain_fft, parallel_helmholtz_hodge,
       parallel_coarse_grain_fft_distributed, parallel_coarse_grain_masked

const _MPI_ENABLED = Ref(false)

function mpi_init()
    if !_MPI_ENABLED[]
        MPI.Init()
        _MPI_ENABLED[] = true
    end
    return nothing
end

function mpi_finalize()
    if _MPI_ENABLED[]
        MPI.Barrier(MPI.COMM_WORLD)
        MPI.Finalize()
        _MPI_ENABLED[] = false
    end
    return nothing
end

mpi_enabled() = _MPI_ENABLED[]
mpi_rank() = mpi_enabled() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
mpi_size() = mpi_enabled() ? MPI.Comm_size(MPI.COMM_WORLD) : 1

function decompose_x(nx::Int, size::Int)
    counts = fill(nx ÷ size, size)
    for i in 1:(nx % size)
        counts[i] += 1
    end
    offsets = cumsum([0; counts[1:end-1]])
    return counts, offsets
end

function scatter_field(field::Field)
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    ny, nx = size(field.data)
    counts, offsets = decompose_x(nx, p)
    nx_loc = counts[r+1]
    A_loc = Array{eltype(field.data),2}(undef, ny, nx_loc)
    send = r == 0 ? field.data : Array{eltype(field.data),2}(undef, 0, 0)
    sendcounts = [counts[i]*ny for i in 1:p]
    displs = [offsets[i]*ny for i in 1:p]
    MPI.Scatterv!(send, A_loc, 0, comm; counts=sendcounts, displs=displs)
    grid_loc = Grid(nx_loc, ny, field.grid.dx, field.grid.dy, field.grid.periodic_x, field.grid.periodic_y)
    return Field(A_loc, grid_loc), (counts, offsets)
end

function exchange_halos!(A::Array{T,2}, halosize::Int, periodic::Bool=true) where {T}
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    left = r == 0 ? (periodic ? p-1 : MPI.PROC_NULL) : r-1
    right = r == p-1 ? (periodic ? 0 : MPI.PROC_NULL) : r+1
    ny, nxloc = size(A)
    # send/recv right halo to/from right neighbor
    if halosize > 0 && p > 1
        sendbuf_r = @view A[:, (nxloc-halosize+1):nxloc]
        recvbuf_l = similar(sendbuf_r)
        req1 = MPI.Isend(sendbuf_r, right, 0, comm)
        req2 = MPI.Irecv!(recvbuf_l, left, 0, comm)
        MPI.Wait(req1); MPI.Wait(req2)
        # prepend left halo
        A = hcat(recvbuf_l, A)

        # send/recv left halo to/from left neighbor
        sendbuf_l = @view A[:, (1+halosize):(2*halosize)]
        recvbuf_r = similar(sendbuf_l)
        req3 = MPI.Isend(sendbuf_l, left, 1, comm)
        req4 = MPI.Irecv!(recvbuf_r, right, 1, comm)
        MPI.Wait(req3); MPI.Wait(req4)
        # append right halo
        A = hcat(A, recvbuf_r)
    end
    return A
end

function strip_halos(A::Array{T,2}, halosize::Int) where {T}
    ny, nx = size(A)
    return @view A[:, (1+halosize):(nx-halosize)]
end

function parallel_coarse_grain(field::Union{Field,Nothing}, kernel::Kernel; halosize::Int=max(kernel.radius_x, kernel.radius_y), threaded::Bool=true)
    mpi_init()
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    # Broadcast grid metadata from root (rank 0)
    if r == 0 && field === nothing
        error("On rank 0, 'field' must be provided.")
    end
    ints = zeros(Int64, 4) # nx, ny, px, py
    floats = zeros(Float64, 2) # dx, dy
    if r == 0
        ints .= (field.grid.nx, field.grid.ny, field.grid.periodic_x ? 1 : 0, field.grid.periodic_y ? 1 : 0)
        floats .= (field.grid.dx, field.grid.dy)
    end
    MPI.Bcast!(ints, 0, comm)
    MPI.Bcast!(floats, 0, comm)
    nx, ny = Int(ints[1]), Int(ints[2])
    px, py = (ints[3] == 1), (ints[4] == 1)
    dx, dy = floats[1], floats[2]
    fullgrid = Grid(nx, ny, dx, dy, px, py)

    counts, offsets = decompose_x(nx, p)
    nx_loc = counts[r+1]
    A_loc = Array{Float64,2}(undef, ny, nx_loc)
    send = r == 0 ? Float64.(field.data) : Array{Float64,2}(undef, 0, 0)
    sendcounts = [counts[i]*ny for i in 1:p]
    displs = [offsets[i]*ny for i in 1:p]
    MPI.Scatterv!(send, A_loc, 0, comm; counts=sendcounts, displs=displs)
    fld_loc = Field(A_loc, Grid(nx_loc, ny, dx, dy, px, py))

    # add halos in x and filter locally
    Ahalo = exchange_halos!(fld_loc.data, halosize, px)
    fld_halo = Field(Ahalo, Grid(size(Ahalo,2), size(Ahalo,1), dx, dy, px, py))
    out_halo = coarse_grain(fld_halo, kernel; threaded=threaded)
    out_loc = Field(copy(strip_halos(out_halo.data, halosize)), fld_loc.grid)

    # gather results to root
    recv = r == 0 ? Array{Float64,2}(undef, ny, nx) : Array{Float64,2}(undef, 0, 0)
    MPI.Gatherv!(out_loc.data, recv, 0, comm; counts=sendcounts, displs=displs)
    if r == 0
        return Field(recv, fullgrid)
    else
        return nothing
    end
end

function parallel_coarse_grain_masked(field::Union{Field,Nothing}, kernel::Kernel, mask::Union{BitArray{2},Nothing}; halosize::Int=max(kernel.radius_x, kernel.radius_y), threaded::Bool=true, normalize::Bool=true, fill_value=NaN)
    mpi_init()
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    if r == 0 && (field === nothing || mask === nothing)
        error("On rank 0, 'field' and 'mask' must be provided.")
    end
    ints = zeros(Int64, 4)
    floats = zeros(Float64, 2)
    if r == 0
        ints .= (field.grid.nx, field.grid.ny, field.grid.periodic_x ? 1 : 0, field.grid.periodic_y ? 1 : 0)
        floats .= (field.grid.dx, field.grid.dy)
    end
    MPI.Bcast!(ints, 0, comm)
    MPI.Bcast!(floats, 0, comm)
    nx, ny = Int(ints[1]), Int(ints[2])
    px, py = (ints[3] == 1), (ints[4] == 1)
    dx, dy = floats[1], floats[2]
    counts, offsets = decompose_x(nx, p)
    nx_loc = counts[r+1]
    sendcounts = [counts[i]*ny for i in 1:p]
    displs = [offsets[i]*ny for i in 1:p]
    # Scatter field
    A_loc = Array{Float64,2}(undef, ny, nx_loc)
    MPI.Scatterv!(r==0 ? Float64.(field.data) : Array{Float64,2}(undef,0,0), A_loc, 0, comm; counts=sendcounts, displs=displs)
    # Scatter mask
    M_loc = BitArray(undef, ny, nx_loc)
    if r == 0
        Mall = reshape(mask, ny, nx)
    else
        Mall = BitArray(undef, 0, 0)
    end
    MPI.Scatterv!(Mall, M_loc, 0, comm; counts=sendcounts, displs=displs)
    # Halos
    Ahalo = exchange_halos!(A_loc, halosize, px)
    Mhalo = exchange_halos!(M_loc, halosize, px)
    fld_halo = Field(Ahalo, Grid(size(Ahalo,2), size(Ahalo,1), dx, dy, px, py))
    out_halo = coarse_grain_masked(fld_halo, kernel, Mhalo; normalize=normalize, fill_value=fill_value)
    out_loc = Field(copy(strip_halos(out_halo.data, halosize)), Grid(nx_loc, ny, dx, dy, px, py))
    # gather
    recv = r == 0 ? Array{Float64,2}(undef, ny, nx) : Array{Float64,2}(undef, 0, 0)
    MPI.Gatherv!(out_loc.data, recv, 0, comm; counts=sendcounts, displs=displs)
    if r == 0
        return Field(recv, Grid(nx, ny, dx, dy, px, py))
    else
        return nothing
    end
end

"""
    parallel_coarse_grain_fft(field_or_nothing, σx, σy)

MPI driver that gathers the field to rank 0, applies `coarse_grain_fft`, and scatters back.
Suitable for periodic domains. This is a gather/compute/scatter implementation.
"""
function parallel_coarse_grain_fft(field::Union{Field,Nothing}, σx::Real, σy::Real)
    mpi_init()
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    if r == 0 && field === nothing
        error("On rank 0, 'field' must be provided.")
    end
    ints = zeros(Int64, 4)
    floats = zeros(Float64, 2)
    if r == 0
        ints .= (field.grid.nx, field.grid.ny, field.grid.periodic_x ? 1 : 0, field.grid.periodic_y ? 1 : 0)
        floats .= (field.grid.dx, field.grid.dy)
    end
    MPI.Bcast!(ints, 0, comm)
    MPI.Bcast!(floats, 0, comm)
    nx, ny = Int(ints[1]), Int(ints[2])
    px, py = (ints[3] == 1), (ints[4] == 1)
    dx, dy = floats[1], floats[2]
    counts, offsets = decompose_x(nx, p)
    nx_loc = counts[r+1]
    A_loc = Array{Float64,2}(undef, ny, nx_loc)
    sendcounts = [counts[i]*ny for i in 1:p]
    displs = [offsets[i]*ny for i in 1:p]
    MPI.Scatterv!(r==0 ? Float64.(field.data) : Array{Float64,2}(undef,0,0), A_loc, 0, comm; counts=sendcounts, displs=displs)
    A_full = r == 0 ? Array{Float64,2}(undef, ny, nx) : Array{Float64,2}(undef, 0, 0)
    MPI.Gatherv!(A_loc, A_full, 0, comm; counts=sendcounts, displs=displs)
    if r == 0
        f = Field(A_full, Grid(nx, ny, dx, dy, px, py))
        out = coarse_grain_fft(f, σx, σy)
        A_out = out.data
    else
        A_out = Array{Float64,2}(undef, 0, 0)
    end
    B_loc = Array{Float64,2}(undef, ny, nx_loc)
    MPI.Scatterv!(A_out, B_loc, 0, comm; counts=sendcounts, displs=displs)
    return Field(B_loc, Grid(nx_loc, ny, dx, dy, px, py))
end

"""
    parallel_helmholtz_hodge(u_or_nothing, v_or_nothing)

Gather u,v to root, compute serial FFT-based Helmholtz-Hodge, scatter back both components.
Returns divergence-free and potential parts for the local slab on each rank.
"""
function parallel_helmholtz_hodge(u::Union{Field,Nothing}, v::Union{Field,Nothing})
    mpi_init()
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    if r == 0 && (u === nothing || v === nothing)
        error("On rank 0, fields must be provided.")
    end
    ints = zeros(Int64, 4)
    floats = zeros(Float64, 2)
    if r == 0
        ints .= (u.grid.nx, u.grid.ny, u.grid.periodic_x ? 1 : 0, u.grid.periodic_y ? 1 : 0)
        floats .= (u.grid.dx, u.grid.dy)
    end
    MPI.Bcast!(ints, 0, comm)
    MPI.Bcast!(floats, 0, comm)
    nx, ny = Int(ints[1]), Int(ints[2])
    px, py = (ints[3] == 1), (ints[4] == 1)
    dx, dy = floats[1], floats[2]
    counts, offsets = decompose_x(nx, p)
    nx_loc = counts[r+1]
    sendcounts = [counts[i]*ny for i in 1:p]
    displs = [offsets[i]*ny for i in 1:p]
    u_loc = Array{Float64,2}(undef, ny, nx_loc)
    v_loc = Array{Float64,2}(undef, ny, nx_loc)
    MPI.Scatterv!(r==0 ? Float64.(u.data) : Array{Float64,2}(undef,0,0), u_loc, 0, comm; counts=sendcounts, displs=displs)
    MPI.Scatterv!(r==0 ? Float64.(v.data) : Array{Float64,2}(undef,0,0), v_loc, 0, comm; counts=sendcounts, displs=displs)
    U_full = r == 0 ? Array{Float64,2}(undef, ny, nx) : Array{Float64,2}(undef, 0, 0)
    V_full = r == 0 ? Array{Float64,2}(undef, ny, nx) : Array{Float64,2}(undef, 0, 0)
    MPI.Gatherv!(u_loc, U_full, 0, comm; counts=sendcounts, displs=displs)
    MPI.Gatherv!(v_loc, V_full, 0, comm; counts=sendcounts, displs=displs)
    if r == 0
        grid = Grid(nx, ny, dx, dy, px, py)
        udf, vdf, up, vp, ϕ, ψ = helmholtz_hodge(Field(U_full, grid), Field(V_full, grid))
        Udf = udf.data; Vdf = vdf.data; Up = up.data; Vp = vp.data
    else
        Udf = Array{Float64,2}(undef, 0, 0)
        Vdf = Array{Float64,2}(undef, 0, 0)
        Up  = Array{Float64,2}(undef, 0, 0)
        Vp  = Array{Float64,2}(undef, 0, 0)
    end
    udf_loc = Array{Float64,2}(undef, ny, nx_loc)
    vdf_loc = Array{Float64,2}(undef, ny, nx_loc)
    up_loc  = Array{Float64,2}(undef, ny, nx_loc)
    vp_loc  = Array{Float64,2}(undef, ny, nx_loc)
    MPI.Scatterv!(Udf, udf_loc, 0, comm; counts=sendcounts, displs=displs)
    MPI.Scatterv!(Vdf, vdf_loc, 0, comm; counts=sendcounts, displs=displs)
    MPI.Scatterv!(Up,  up_loc,  0, comm; counts=sendcounts, displs=displs)
    MPI.Scatterv!(Vp,  vp_loc,  0, comm; counts=sendcounts, displs=displs)
    grid_loc = Grid(nx_loc, ny, dx, dy, px, py)
    return Field(udf_loc, grid_loc), Field(vdf_loc, grid_loc), Field(up_loc, grid_loc), Field(vp_loc, grid_loc)
end

"""
    parallel_coarse_grain_fft_distributed(field_or_nothing, σx, σy)

Distributed 2D FFT Gaussian filtering with equal-slab decomposition.
Requirements: periodic in x and y; nx % nprocs == 0; ny % nprocs == 0.
Returns full filtered field on rank 0 and nothing on others.
"""
function parallel_coarse_grain_fft_distributed(field::Union{Field,Nothing}, σx::Real, σy::Real)
    mpi_init()
    comm = MPI.COMM_WORLD
    r = MPI.Comm_rank(comm)
    p = MPI.Comm_size(comm)
    if r == 0 && field === nothing
        error("On rank 0, 'field' must be provided.")
    end
    ints = zeros(Int64, 4)
    floats = zeros(Float64, 2)
    if r == 0
        ints .= (field.grid.nx, field.grid.ny, field.grid.periodic_x ? 1 : 0, field.grid.periodic_y ? 1 : 0)
        floats .= (field.grid.dx, field.grid.dy)
    end
    MPI.Bcast!(ints, 0, comm)
    MPI.Bcast!(floats, 0, comm)
    nx, ny = Int(ints[1]), Int(ints[2])
    px, py = (ints[3] == 1), (ints[4] == 1)
    dx, dy = floats[1], floats[2]
    if !px || !py
        error("parallel_coarse_grain_fft_distributed requires periodic boundaries in both x and y")
    end
    # Decompositions (possibly non-uniform)
    xcounts, xoffsets = decompose_x(nx, p)
    ycounts, yoffsets = decompose_x(ny, p)
    nx_loc = xcounts[r+1]
    ny_loc = ycounts[r+1]

    # Scatter x-slabs
    A_loc = Array{Float64,2}(undef, ny, nx_loc)
    if r == 0
        A = Float64.(field.data)
    else
        A = Array{Float64,2}(undef, 0, 0)
    end
    sendcounts = [xcounts[i]*ny for i in 1:p]
    displs = [xoffsets[i]*ny for i in 1:p]
    MPI.Scatterv!(A, A_loc, 0, comm; counts=sendcounts, displs=displs)

    # 1) FFT along y locally
    F_y = fft(A_loc, 1)

    # 2) Alltoallv to y-slabs
    # Pack send buffer as concatenation over dest q of rows yoffsets[q]..yoffsets[q]+ycounts[q]-1, all local columns
    sendcounts_v = [ycounts[q]*nx_loc for q in 1:p]
    sdispls_v = cumsum([0; sendcounts_v[1:end-1]])
    sendbuf = Vector{ComplexF64}(undef, sum(sendcounts_v))
    for q in 1:p
        r0 = yoffsets[q] + 1
        r1 = yoffsets[q] + ycounts[q]
        block = @view F_y[r0:r1, 1:nx_loc]
        copyto!(sendbuf, sdispls_v[q] + 1, vec(block), 1, sendcounts_v[q])
    end
    # Prepare recv buffer sized ny_loc * nx (sum over sources of ny_loc*xcounts[src])
    recvcounts_v = [ny_loc * xcounts[s] for s in 1:p]
    rdispls_v = cumsum([0; recvcounts_v[1:end-1]])
    recvbuf = Vector{ComplexF64}(undef, sum(recvcounts_v))
    MPI.Alltoallv!(sendbuf, sendcounts_v, sdispls_v, recvbuf, recvcounts_v, rdispls_v, comm)
    # Reassemble y-slab (ny_loc x nx)
    Fy_yslab = Array{ComplexF64,2}(undef, ny_loc, nx)
    for s in 1:p
        nx_s = xcounts[s]
        offx = xoffsets[s] + 1
        start_idx = rdispls_v[s] + 1
        end_idx = rdispls_v[s] + recvcounts_v[s]
        block = reshape(@view(recvbuf[start_idx:end_idx]), ny_loc, nx_s)
        Fy_yslab[:, offx:offx+nx_s-1] .= block
    end

    # 3) FFT along x on y-slabs
    Fyx = fft(Fy_yslab, 2)

    # 4) Apply Gaussian transfer function G(kx, ky)
    kx = [0:floor(Int,nx/2); ceil(Int,-nx/2)+1:-1] .* (2π/(nx*dx))
    ky = [0:floor(Int,ny/2); ceil(Int,-ny/2)+1:-1] .* (2π/(ny*dy))
    ky_loc_vec = ky[yoffsets[r+1]+1 : yoffsets[r+1]+ny_loc]
    KX = reshape(kx, 1, :)
    KY = reshape(ky_loc_vec, :, 1)
    G = @. exp(-0.5*((σx^2)*(KX^2) + (σy^2)*(KY^2)))
    Ffilt = Fyx .* G

    # 5) Inverse FFT along x
    fy_yslab = ifft(Ffilt, 2)

    # 6) Alltoallv back to x-slabs
    sendcounts2 = [ny_loc * xcounts[q] for q in 1:p]
    sdispls2 = cumsum([0; sendcounts2[1:end-1]])
    sendbuf2 = Vector{ComplexF64}(undef, sum(sendcounts2))
    for q in 1:p
        cx0 = xoffsets[q] + 1
        cx1 = xoffsets[q] + xcounts[q]
        block = @view fy_yslab[:, cx0:cx1]
        copyto!(sendbuf2, sdispls2[q] + 1, vec(block), 1, sendcounts2[q])
    end
    recvcounts2 = [ycounts[s] * nx_loc for s in 1:p]
    rdispls2 = cumsum([0; recvcounts2[1:end-1]])
    recvbuf2 = Vector{ComplexF64}(undef, sum(recvcounts2))
    MPI.Alltoallv!(sendbuf2, sendcounts2, sdispls2, recvbuf2, recvcounts2, rdispls2, comm)
    # Reassemble x-slab (ny x nx_loc)
    fy_xslab = Array{ComplexF64,2}(undef, ny, nx_loc)
    for s in 1:p
        ny_sloc = ycounts[s]
        offy = yoffsets[s] + 1
        start_idx2 = rdispls2[s] + 1
        end_idx2 = rdispls2[s] + recvcounts2[s]
        block = reshape(@view(recvbuf2[start_idx2:end_idx2]), ny_sloc, nx_loc)
        fy_xslab[offy:offy+ny_sloc-1, :] .= block
    end

    # 7) Inverse FFT along y locally
    out_loc = real(ifft(fy_xslab, 1))

    # Gather to root
    recv = r == 0 ? Array{Float64,2}(undef, ny, nx) : Array{Float64,2}(undef, 0, 0)
    MPI.Gatherv!(Float64.(out_loc), recv, 0, comm; counts=sendcounts, displs=displs)
    if r == 0
        return Field(recv, Grid(nx, ny, dx, dy, px, py))
    else
        return nothing
    end
end

end # module MPIUtils

using .MPIUtils: mpi_init, mpi_finalize, mpi_enabled, mpi_rank, mpi_size, parallel_coarse_grain, parallel_coarse_grain_fft, parallel_helmholtz_hodge, parallel_coarse_grain_fft_distributed, parallel_coarse_grain_masked
export mpi_init, mpi_finalize, mpi_enabled, mpi_rank, mpi_size,
       parallel_coarse_grain, parallel_coarse_grain_fft, parallel_helmholtz_hodge,
       parallel_coarse_grain_fft_distributed, parallel_coarse_grain_masked
