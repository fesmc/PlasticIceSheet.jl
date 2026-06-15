module PlasticIceSheetNetCDFExt

using PlasticIceSheet
using NCDatasets

function PlasticIceSheet.load_plastic_inputs(path::AbstractString;
        bed = "bed", tau = "tau", mask = "mask", x = "x", y = "y")
    NCDataset(path, "r") do ds
        bedv  = Array{Float64}(ds[bed][:, :])
        tauv  = Array{Float64}(ds[tau][:, :])
        maskv = BitMatrix(ds[mask][:, :] .!= 0)
        xv    = Array{Float64}(ds[x][:])
        yv    = Array{Float64}(ds[y][:])
        dx = length(xv) > 1 ? abs(xv[2] - xv[1]) : one(eltype(xv))
        dy = length(yv) > 1 ? abs(yv[2] - yv[1]) : one(eltype(yv))
        return (; bed = bedv, τ = tauv, mask = maskv, dx, dy, x = xv, y = yv)
    end
end

function PlasticIceSheet.save_reconstruction(path::AbstractString, E, H;
        x = nothing, y = nothing, attrib = Dict{String,Any}())
    nx, ny = size(E)
    NCDataset(path, "c") do ds
        defDim(ds, "x", nx)
        defDim(ds, "y", ny)
        if x !== nothing
            defVar(ds, "x", collect(x), ("x",))
        end
        if y !== nothing
            defVar(ds, "y", collect(y), ("y",))
        end
        ev = defVar(ds, "E", Float64, ("x", "y"))
        hv = defVar(ds, "H", Float64, ("x", "y"))
        ev[:, :] = E
        hv[:, :] = H
        ev.attrib["long_name"] = "ice surface elevation"
        ev.attrib["units"] = "m"
        hv.attrib["long_name"] = "ice thickness"
        hv.attrib["units"] = "m"
        for (k, v) in attrib
            ds.attrib[k] = v
        end
    end
    return path
end

end # module
