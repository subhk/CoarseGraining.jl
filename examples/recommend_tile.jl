#!/usr/bin/env julia
using CoarseGraining

function parse_args(args)
    opts = Dict{String,Any}(
        "nx" => 1024,
        "ny" => 1024,
        "dx" => 1.0,
        "dy" => 1.0,
        "sigx" => 3.0,
        "sigy" => 3.0,
        "truncate" => 3.0,
        "write" => nothing,
    )
    i = 1
    while i <= length(args)
        k = lowercase(args[i])
        if k in ["--nx","--ny","--sigx","--sigy","--dx","--dy","--truncate","--write"]
            i += 1
            if i > length(args)
                error("Missing value for $k")
            end
            v = args[i]
            if k in ["--nx","--ny"]
                opts[replace(k,"--"=>"")] = parse(Int, v)
            elseif k in ["--write"]
                opts[replace(k,"--"=>"")] = v
            else
                opts[replace(k,"--"=>"")] = parse(Float64, v)
            end
        else
            error("Unknown option: $k. Use --nx, --ny, --dx, --dy, --sigx, --sigy, --truncate, --write")
        end
        i += 1
    end
    return opts
end

function main()
    opts = parse_args(ARGS)
    nx = opts["nx"]; ny = opts["ny"]
    dx = opts["dx"]; dy = opts["dy"]
    sigx = opts["sigx"]; sigy = opts["sigy"]
    trunc = opts["truncate"]
    # Build a kernel to infer radii consistent with the main APIs
    K = gaussian_kernel(sigx, sigy; truncate=trunc)
    bj, bi = select_tile(ny, nx, K.radius_y, K.radius_x)
    println("Recommended tile: (", bj, ", ", bi, ") for nx=", nx, ", ny=", ny,
            ", σx=", sigx, ", σy=", sigy, ", truncate=", trunc, ")")
    if opts["write"] !== nothing
        path = String(opts["write"])
        open(path, "w") do io
            println(io, "[coarse_grain]")
            println(io, "tile_bj = ", bj)
            println(io, "tile_bi = ", bi)
            println(io, "# nx = ", nx)
            println(io, "# ny = ", ny)
            println(io, "# dx = ", dx)
            println(io, "# dy = ", dy)
            println(io, "# sigx = ", sigx)
            println(io, "# sigy = ", sigy)
            println(io, "# truncate = ", trunc)
        end
        println("Wrote config to ", path)
    end
end

main()

