# Resolution test for the tiled τ_b inversion: how many bed-derived tiles does the data
# actually warrant? Sweeps the number of roughness classes (× with/without the sediment
# split), optimizes one τ_b scalar per tile against the observed surface, and plots the
# surface-misfit rmse vs the number of free parameters. The knee of that curve is the
# principled, automatic replacement for ICESHEET's hand-chosen domain count (Gowan et al.).
#
# Run with:  julia --project=examples examples/invert_taub_sweep_greenland.jl
#   env knobs:  GRID=GRL-32KM  N_ITER=100  MODE=flat

using PlasticIceSheet
using NCDatasets
using CairoMakie
using Printf, Statistics, LinearAlgebra
using ImplicitDifferentiation, ADTypes, ForwardDiff
using Zygote

# --- configuration ----------------------------------------------------------------
const GRID = get(ENV, "GRID", "GRL-32KM")
const TOPO_NAME = Dict("GRL-16KM" => "GRL-16KM_TOPO-M17-v5.nc",
                       "GRL-32KM" => "GRL-32KM_TOPO-M17.nc",
                       "GRL-8KM"  => "GRL-8KM_TOPO-M17.nc")[GRID]
const DATADIR   = joinpath(@__DIR__, "ice_data", "Greenland", GRID)
const TOPO_FILE = joinpath(DATADIR, TOPO_NAME)
const SED_FILE  = joinpath(DATADIR, "$(GRID)_SED-L97.nc")
const OUTDIR    = joinpath(@__DIR__, "output")

const ROUGH_LIST = [1, 2, 3, 4, 6, 8]   # roughness-class counts to sweep
const ROUGH_KM   = 64.0
const SED_THRESH = 50.0
const N_ITER     = parse(Int, get(ENV, "N_ITER", "100"))
const LR         = 5.0e3
const τ_MIN, τ_MAX = 5.0e3, 4.0e5
const MODE   = Symbol(get(ENV, "MODE", "flat"))
const SOLVER = (; mode = MODE, params = PlasticParams(),
                max_sweeps = 600, tol = 1.0e-9, n_outer = 500, outer_tol = 1.0e-5, relax = 0.4)

read2d(ds, name, nx, ny) =
    (a = Float64.(coalesce.(ds[name][:, :], NaN)); size(a) == (nx, ny) ? a : permutedims(a))
function windowed_std(A, r)
    nx, ny = size(A); R = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        s = 0.0; s2 = 0.0; n = 0
        for dj in -r:r, di in -r:r
            ii, jj = i + di, j + dj
            if 1 <= ii <= nx && 1 <= jj <= ny && isfinite(A[ii, jj])
                s += A[ii, jj]; s2 += A[ii, jj]^2; n += 1
            end
        end
        n >= 2 && (R[i, j] = sqrt(max(s2 / n - (s / n)^2, 0.0)))
    end
    return R
end

# --- load inputs ------------------------------------------------------------------
println("Loading $GRID from $DATADIR")
xc, yc, z_bed, H_obs, mask_raw = NCDataset(TOPO_FILE, "r") do ds
    x = Array{Float64}(ds["xc"][:]); y = Array{Float64}(ds["yc"][:])
    nx, ny = length(x), length(y)
    (x, y, read2d(ds, "z_bed", nx, ny), read2d(ds, "H_ice", nx, ny), read2d(ds, "mask", nx, ny))
end
const NX, NY = length(xc), length(yc)
const DX = abs(xc[2] - xc[1]) * 1000.0
const DY = abs(yc[2] - yc[1]) * 1000.0
z_bed[z_bed .<= -9990] .= NaN
H_obs[H_obs .<= -9990] .= NaN
const MASK    = round.(mask_raw) .== 2
const Z_B     = map(z -> isnan(z) ? 0.0 : z, z_bed)
const Z_S_OBS = Z_B .+ map(h -> isnan(h) ? 0.0 : h, H_obs)
const USE_SED = isfile(SED_FILE)
z_sed = USE_SED ? (NCDataset(SED_FILE, "r") do ds; read2d(ds, "z_sed", NX, NY); end) : fill(NaN, NX, NY)
const ROUGH = windowed_std(z_bed, max(1, round(Int, ROUGH_KM * 1000.0 / DX / 2)))
@printf("grid %d×%d, dx=%.0f m, grounded cells %d; sediment: %s\n",
        NX, NY, DX, count(MASK), USE_SED ? "yes" : "no")

# --- one inversion for a given tiling ---------------------------------------------
# Tiles from bed roughness (n_rough quantile classes) × optional sediment split.
function build_tiles(n_rough, use_sed)
    redges = n_rough > 1 ? quantile(filter(isfinite, ROUGH[MASK]), (1:n_rough-1) ./ n_rough) : Float64[]
    roughbin(r) = isnan(r) ? 1 : clamp(1 + count(<(r), redges), 1, n_rough)
    nsed = use_sed ? 2 : 1
    sedclass(z) = (use_sed && isfinite(z) && z >= SED_THRESH) ? 1 : 0
    ndom = nsed * n_rough
    label = [sedclass(z_sed[i, j]) * n_rough + roughbin(ROUGH[i, j]) for i in 1:NX, j in 1:NY]
    onehot = zeros(NX * NY, ndom)
    for (k, l) in enumerate(vec(label)); onehot[k, l] = 1.0; end
    populated = length(unique(label[MASK]))
    return label, onehot, ndom, populated
end

function invert(onehot, ndom; n_iter = N_ITER)
    taub(θ) = clamp.(reshape(onehot * θ, NX, NY), τ_MIN, τ_MAX)
    loss(θ) = surface_misfit(differentiable_thickness(taub(θ), Z_B, MASK; dx = DX, dy = DY, SOLVER...) .+ Z_B,
                             Z_S_OBS, MASK)
    θ = fill(1.0e5, ndom); m = zero(θ); v = zero(θ)
    β1, β2, ϵ = 0.9, 0.999, 1.0e-8
    local L = 0.0
    for it in 1:n_iter
        L, back = Zygote.pullback(loss, θ)
        g = back(1.0)[1]
        m = β1 .* m .+ (1 - β1) .* g
        v = β2 .* v .+ (1 - β2) .* g .^ 2
        θ = θ .- LR .* (m ./ (1 - β1^it)) ./ (sqrt.(v ./ (1 - β2^it)) .+ ϵ)
    end
    return sqrt(L), clamp.(θ, τ_MIN, τ_MAX)
end

# --- sweep ------------------------------------------------------------------------
results = NamedTuple[]   # (n_rough, use_sed, ntiles, rmse)
println("\nn_rough  sed   tiles   rmse(m)   elapsed(s)")
t0 = time()
for use_sed in (USE_SED ? (false, true) : (false,)), nr in ROUGH_LIST
    _, onehot, ndom, pop = build_tiles(nr, use_sed)
    rmse, _ = invert(onehot, ndom)
    push!(results, (; n_rough = nr, use_sed, ntiles = pop, rmse))
    @printf("%6d  %4s  %5d   %7.1f   %8.1f\n", nr, use_sed ? "yes" : "no", pop, rmse, time() - t0)
end

# --- figure: rmse vs number of free parameters ------------------------------------
mkpath(OUTDIR)
fig = Figure(size = (760, 560))
ax = Axis(fig[1, 1]; title = "τ_b tiling resolution test — Greenland $GRID",
          xlabel = "number of τ_b tiles (free parameters)", ylabel = "surface-misfit rmse (m)")
for (us, lab, col) in ((false, "roughness only", :steelblue), (true, "roughness × sediment", :darkorange))
    r = filter(x -> x.use_sed == us, results)
    isempty(r) && continue
    sort!(r, by = x -> x.ntiles)
    scatterlines!(ax, [x.ntiles for x in r], [x.rmse for x in r]; color = col, label = lab, markersize = 10)
end
axislegend(ax)
figout = joinpath(OUTDIR, "invert_taub_sweep_$GRID.png")
save(figout, fig)
println("\nwrote $figout")
