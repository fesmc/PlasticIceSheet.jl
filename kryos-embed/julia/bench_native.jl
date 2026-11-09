# Julia-native reference timings: julia --project=kryos-embed/KryosEmbed bench_native.jl nx ny ncalls
using KryosEmbed, PlasticIceSheet, Printf, Statistics
nx, ny, nc = parse.(Int, ARGS[1:3])
t0 = time_ns()
zb, tb, m = synthetic_case(nx, ny)
for mode in (:flat, :full)
    t = @elapsed (zs, H) = solve(zb, tb, m; dx = 32e3, dy = 32e3, mode)
    ts = [@elapsed solve(zb, tb, m; dx = 32e3, dy = 32e3, mode) for _ in 1:nc]
    @printf("native %s %dx%d  first=%.3f s  steady median=%.4f s  min=%.4f  Hmax=%.3f Hmean=%.6f\n",
            mode, nx, ny, t, median(ts), minimum(ts), maximum(H), sum(H) / count(m))
end
