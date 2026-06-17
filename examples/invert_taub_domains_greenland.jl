# Gowan-style basal-shear-stress inversion with a low-dimensional, *paleo-transferable*
# parameterisation — the elegant/automatic version of ICESHEET's hand-tuned shear-stress
# tiles (Gowan et al. 2016/2021).
#
# Design constraint: the τ_b domains are built ONLY from data we would have for an unknown
# (e.g. paleo) ice sheet — bedrock topography and a sediment mask. We never use the ice
# thickness or ice-derived drainage basins to define the tiles. The observed surface enters
# *only* as the optimisation target (the modern-surface calibration Gowan used for GrIS/AIS);
# in step 3 that target is replaced by relative sea level through a GIA model.
#
# Tiles: roughness(z_bed) binned into N_ROUGH classes × {soft, hard} sediment classes.
# Each tile carries ONE τ_b scalar (≈ a dozen free parameters total), optimised by
# reverse-mode AD (differentiable_thickness) + Adam to fit the observed surface.
#
# Run with:  julia --project=examples examples/invert_taub_domains_greenland.jl
#   env knobs:  GRID=GRL-16KM  N_ITER=300  MODE=flat

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

const N_ROUGH    = 4             # bed-roughness classes (quantile bins)
const ROUGH_KM   = 64.0          # window (full width ~) for the roughness metric
const SED_THRESH = 50.0          # sediment thickness (m) splitting soft (≥) / hard (<) beds
const N_ITER     = parse(Int, get(ENV, "N_ITER", "200"))
const LR         = 5.0e3
const τ_MIN, τ_MAX = 5.0e3, 4.0e5
const MODE   = Symbol(get(ENV, "MODE", "flat"))
const SOLVER = (; mode = MODE, params = PlasticParams(),
                max_sweeps = 600, tol = 1.0e-9, n_outer = 500, outer_tol = 1.0e-5, relax = 0.4)

read2d(ds, name, nx, ny) =
    (a = Float64.(coalesce.(ds[name][:, :], NaN)); size(a) == (nx, ny) ? a : permutedims(a))

# Windowed standard deviation of `A` (NaN-aware) — the bed-roughness metric.
function windowed_std(A, r)
    nx, ny = size(A)
    R = fill(NaN, nx, ny)
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
nx, ny = length(xc), length(yc)
dx = abs(xc[2] - xc[1]) * 1000.0
dy = abs(yc[2] - yc[1]) * 1000.0
z_bed[z_bed .<= -9990] .= NaN
H_obs[H_obs .<= -9990] .= NaN

mask    = round.(mask_raw) .== 2                       # grounded ice = prescribed margin
z_b     = map(z -> isnan(z) ? 0.0 : z, z_bed)
z_s_obs = z_b .+ map(h -> isnan(h) ? 0.0 : h, H_obs)   # calibration target (observed surface)

z_sed = isfile(SED_FILE) ? (NCDataset(SED_FILE, "r") do ds; read2d(ds, "z_sed", nx, ny); end) :
        fill(NaN, nx, ny)
const USE_SED = isfile(SED_FILE)
@printf("grid %d×%d, dx=%.0f m, grounded cells %d; sediment file: %s\n",
        nx, ny, dx, count(mask), USE_SED ? basename(SED_FILE) : "none")

# --- build τ_b tiles from bed roughness (+ sediment) ------------------------------
rr = max(1, round(Int, ROUGH_KM * 1000.0 / dx / 2))
rough = windowed_std(z_bed, rr)
redges = quantile(filter(isfinite, rough[mask]), (1:N_ROUGH-1) ./ N_ROUGH)
roughbin(r) = isnan(r) ? 1 : clamp(1 + count(<(r), redges), 1, N_ROUGH)
sedclass(z) = (USE_SED && isfinite(z) && z >= SED_THRESH) ? 1 : 0   # 1 = soft, 0 = hard
const N_SED = USE_SED ? 2 : 1
const NDOM  = N_SED * N_ROUGH

label = [sedclass(z_sed[i, j]) * N_ROUGH + roughbin(rough[i, j]) for i in 1:nx, j in 1:ny]  # 1..NDOM
onehot = zeros(nx * ny, NDOM)                          # linear, Zygote-friendly tile selector
for (k, l) in enumerate(vec(label)); onehot[k, l] = 1.0; end

@printf("tiles: %d roughness × %d sediment = %d domains (roughness edges %s m)\n",
        N_ROUGH, N_SED, NDOM, string(round.(Int, redges)))

# --- objective: one τ_b scalar per tile -------------------------------------------
taub(θ) = clamp.(reshape(onehot * θ, nx, ny), τ_MIN, τ_MAX)
function loss(θ)
    H = differentiable_thickness(taub(θ), z_b, mask; dx, dy, SOLVER...)
    return surface_misfit(H .+ z_b, z_s_obs, mask)
end

# --- Adam inversion ---------------------------------------------------------------
θ = fill(1.0e5, NDOM)
m = zero(θ); v = zero(θ)
β1, β2, ϵ = 0.9, 0.999, 1.0e-8
rmse = Float64[]
println("iter   rmse(m)   elapsed(s)")
t0 = time()
for it in 1:N_ITER
    L, back = Zygote.pullback(loss, θ)
    push!(rmse, sqrt(L))
    g = back(1.0)[1]
    global m = β1 .* m .+ (1 - β1) .* g
    global v = β2 .* v .+ (1 - β2) .* g .^ 2
    m̂ = m ./ (1 - β1^it); v̂ = v ./ (1 - β2^it)
    global θ = θ .- LR .* m̂ ./ (sqrt.(v̂) .+ ϵ)
    (it == 1 || it % 20 == 0 || it == N_ITER) &&
        @printf("%4d   %7.1f   %8.1f\n", it, sqrt(L), time() - t0)
end

# --- final reconstruction + per-tile report ---------------------------------------
τ_b = taub(θ)
z_s, H = solve(z_b, τ_b, mask; dx, dy, SOLVER...)
dH = H .- H_obs
@printf("\nfinal rmse %.1f m;  H obs/mod mean %.0f / %.0f m;  ΔH mean %.0f rms %.0f m\n",
        last(rmse), mean(H_obs[mask]), mean(H[mask]), mean(dH[mask]), sqrt(mean(dH[mask] .^ 2)))
println("\ntile   sed   rough-bin   cells   τ_b(kPa)")
for d in 1:NDOM
    cells = count(k -> label[k] == d && mask[k], eachindex(label))
    cells == 0 && continue
    sed = d > N_ROUGH ? "soft" : "hard"
    rb  = ((d - 1) % N_ROUGH) + 1
    @printf("%4d   %4s   %9d   %5d   %7.1f\n", d, sed, rb, cells, clamp(θ[d], τ_MIN, τ_MAX) / 1e3)
end

# --- write NetCDF -----------------------------------------------------------------
mkpath(OUTDIR)
ncout = joinpath(OUTDIR, "invert_taub_domains_$GRID.nc")
isfile(ncout) && rm(ncout)
NCDataset(ncout, "c") do ds
    defDim(ds, "xc", nx); defDim(ds, "yc", ny)
    defVar(ds, "xc", xc, ("xc",)).attrib["units"] = "km"
    defVar(ds, "yc", yc, ("yc",)).attrib["units"] = "km"
    function put(name, A, long, units)
        v = defVar(ds, name, Float64, ("xc", "yc")); v[:, :] = A
        v.attrib["long_name"] = long; v.attrib["units"] = units
    end
    put("z_s",    z_s,            "reconstructed ice surface elevation", "m")
    put("z_bed",  z_b,            "bedrock elevation (input)",           "m")
    put("H",      H,              "reconstructed ice thickness",         "m")
    put("H_obs",  H_obs,          "observed ice thickness (target)",     "m")
    put("tau_b",  τ_b,            "inverted basal shear stress",         "Pa")
    put("domain", Float64.(label), "τ_b tile id (from bed roughness + sediment)", "1")
    ds.attrib["title"] = "Tiled τ_b inversion (Gowan-style), Greenland $GRID"
    ds.attrib["n_domains"] = NDOM
end
println("wrote $ncout")

# --- figure -----------------------------------------------------------------------
nan_outside(A) = ifelse.(mask, Float64.(A), NaN)
Hobs_p = nan_outside(H_obs); Hmod_p = nan_outside(H); dH_p = nan_outside(dH)
hmax  = max(maximum(Hobs_p[mask]), maximum(Hmod_p[mask]))
dHlim = max(maximum(abs.(dH_p[mask])), 1.0)

fig = Figure(size = (1500, 1000))
function hpanel(pos, A, title, cmap, crange, cbar)
    ax = Axis(fig[pos...]; title, aspect = DataAspect(), xlabel = "x (km)", ylabel = "y (km)")
    hm = heatmap!(ax, xc, yc, A; colormap = cmap, colorrange = crange, nan_color = :transparent)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label = cbar, width = 12)
end
hpanel((1, 1), Hobs_p, "H_obs", :viridis, (0, hmax), "m")
hpanel((1, 3), Hmod_p, "H_mod", :viridis, (0, hmax), "m")
hpanel((1, 5), dH_p, "H_mod − H_obs", :balance, (-dHlim, dHlim), "m")
hpanel((2, 1), nan_outside(label), "τ_b tiles (bed+sediment)", :tab20, (1, NDOM), "tile id")
hpanel((2, 3), nan_outside(τ_b ./ 1e3), "inverted τ_b", :turbo, (0, maximum(τ_b[mask]) / 1e3), "kPa")

axc = Axis(fig[2, 5]; title = "surface misfit", xlabel = "iteration", ylabel = "rmse (m)")
lines!(axc, 1:length(rmse), rmse)

Label(fig[0, :], "Tiled τ_b inversion — Greenland $GRID  ($NDOM domains from bed+sediment)",
      fontsize = 20)
figout = joinpath(OUTDIR, "invert_taub_domains_$GRID.png")
save(figout, fig)
println("wrote $figout")
