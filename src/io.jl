module IO

using NCDatasets
using ..CoarseGraining: Field, Grid

export load_netcdf_var, write_netcdf_field

function load_netcdf_var(path::AbstractString, varname::AbstractString;
                         xdim::AbstractString="x", ydim::AbstractString="y",
                         dx::Real=1.0, dy::Real=1.0,
                         periodic_x::Bool=true, periodic_y::Bool=true)
    ds = NCDataset(path)
    A = permutedims(Array(ds[varname]), (2,1)) # assume vars ordered (y,x)
    close(ds)
    grid = Grid(size(A,2), size(A,1), float(dx), float(dy), periodic_x, periodic_y)
    return Field(A, grid)
end

function write_netcdf_field(path::AbstractString, varname::AbstractString, field::Field)
    ny, nx = size(field.data)
    ds = NCDataset(path, "c")
    defDim(ds, "y", ny)
    defDim(ds, "x", nx)
    v = defVar(ds, varname, eltype(field.data), ("y", "x"))
    v[:] = permutedims(field.data, (1,2))
    close(ds)
    return path
end

end # module IO

using .IO: load_netcdf_var, write_netcdf_field

