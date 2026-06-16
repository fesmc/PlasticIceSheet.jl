module PlasticIceSheetNetCDFExt

using PlasticIceSheet
using NCDatasets

function PlasticIceSheet.load_plastic_inputs(path::AbstractString;
        z_b = "z_b", tau = "tau", mask = "mask", x = "x", y = "y")
    NCDataset(path, "r") do ds
        z_bv  = Array{Float64}(ds[z_b][:, :])
        tauv  = Array{Float64}(ds[tau][:, :])
        maskv = BitMatrix(ds[mask][:, :] .!= 0)
        xv    = Array{Float64}(ds[x][:])
        yv    = Array{Float64}(ds[y][:])
        dx = length(xv) > 1 ? abs(xv[2] - xv[1]) : one(eltype(xv))
        dy = length(yv) > 1 ? abs(yv[2] - yv[1]) : one(eltype(yv))
        return (; z_b = z_bv, τ_b = tauv, mask = maskv, dx, dy, x = xv, y = yv)
    end
end

function PlasticIceSheet.save_reconstruction(path::AbstractString, z_s, H;
        x = nothing, y = nothing, attrib = Dict{String,Any}())
    nx, ny = size(z_s)
    NCDataset(path, "c") do ds
        defDim(ds, "x", nx)
        defDim(ds, "y", ny)
        if x !== nothing
            defVar(ds, "x", collect(x), ("x",))
        end
        if y !== nothing
            defVar(ds, "y", collect(y), ("y",))
        end
        zv = defVar(ds, "z_s", Float64, ("x", "y"))
        hv = defVar(ds, "H", Float64, ("x", "y"))
        zv[:, :] = z_s
        hv[:, :] = H
        zv.attrib["long_name"] = "ice surface elevation"
        zv.attrib["units"] = "m"
        hv.attrib["long_name"] = "ice thickness"
        hv.attrib["units"] = "m"
        for (k, v) in attrib
            ds.attrib[k] = v
        end
    end
    return path
end

end # module
