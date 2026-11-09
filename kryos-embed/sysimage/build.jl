# Build a sysimage holding KryosEmbed + PlasticIceSheet + AD stack, precompile-executed
# on the warmup case so the first solve/gradient call has no JIT latency.
#   julia --project=kryos-embed/env kryos-embed/sysimage/build.jl
using PackageCompiler
out = joinpath(@__DIR__, "kryos_embed.so")
create_sysimage(["KryosEmbed"]; sysimage_path = out,
                precompile_execution_file = joinpath(@__DIR__, "warmup.jl"))
println("wrote ", out, "  (", round(filesize(out) / 2^20; digits = 1), " MiB)")
