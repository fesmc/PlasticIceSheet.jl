# Route c: long-lived Julia worker driven over two named pipes with raw binary files
# (put the files on node-local /dev/shm). Request line: "solve nx ny dx dy mode in out" or "quit".
# in : zb, tb (Float64, nx*ny each) then mask (Int8); out: zs, H (Float64).
using KryosEmbed, PlasticIceSheet
cmdf, respf = ARGS[1], ARGS[2]
# warm the compiled code so the first request is not penalised
KryosEmbed.warmup()
open(respf, "w") do io; println(io, "ready"); end
while true
    line = open(readline, cmdf)
    f = split(line)
    f[1] == "quit" && break
    nx, ny = parse(Int, f[2]), parse(Int, f[3]); dx, dy = parse(Float64, f[4]), parse(Float64, f[5])
    mode = parse(Int, f[6]) == 0 ? :flat : :full
    resp = try
        zb = Matrix{Float64}(undef, nx, ny); tb = similar(zb); m8 = Matrix{Int8}(undef, nx, ny)
        open(f[7], "r") do io; read!(io, zb); read!(io, tb); read!(io, m8); end
        zs, H = solve(zb, tb, m8 .!= 0; dx, dy, mode)
        open(f[8], "w") do io; write(io, zs); write(io, H); end
        "ok"
    catch e
        "error " * sprint(showerror, e)
    end
    open(respf, "w") do io; println(io, resp); end
end
