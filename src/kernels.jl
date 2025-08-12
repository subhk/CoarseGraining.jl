export gaussian_kernel, boxcar_kernel

function gaussian_kernel(σx::Real, σy::Real; truncate::Real=3.0)
    rx = max(1, Int(ceil(truncate*σx)))
    ry = max(1, Int(ceil(truncate*σy)))
    xs = collect(-rx:rx)
    ys = collect(-ry:ry)
    K = [exp(-0.5*((x/σx)^2 + (y/σy)^2)) for y in ys, x in xs]
    K ./= sum(K)
    return Kernel(K, rx, ry)
end

function boxcar_kernel(rx::Integer, ry::Integer)
    K = fill(1.0, 2*ry+1, 2*rx+1)
    K ./= sum(K)
    return Kernel(K, rx, ry)
end

