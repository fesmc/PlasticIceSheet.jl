# Basal-shear-stress inversion on real data — the original ICESHEET-style purpose.
#
# Given the observed bed and the grounded mask, recover the basal shear stress τ_b whose
# plastic reconstruction best fits the observed ice surface (the plastic solve always fills
# the whole mask, so this fits the observed *geometry* on a fixed extent — not the extent).
#
# Two parameterizations of τ_b, selected with METHOD — switch between them with one env var:
#
#   METHOD=tiled    (default)  Gowan-style tiles: ~8 domains defined ONLY from bedrock
#                              roughness × a sediment mask (data available for an unknown /
#                              paleo ice sheet — never the ice thickness or ice-derived
#                              basins). One τ_b scalar per tile. Few, interpretable, and
#                              transferable to a paleo (RSL) target.
#   METHOD=percell             Full per-cell τ_b field, regularized to REG_KM by a Gaussian-
#                              blur reparameterization. Many parameters; fits the dense
#                              present-day surface much more closely.
#
# In both cases the observed surface enters only as the optimisation target, and τ_b is
# recovered by reverse-mode AD (differentiable_thickness) + Adam. See docs/inversion.qmd.
#
# Run with:  julia --project=examples examples/invert_taub_greenland.jl
#   env knobs:  METHOD=percell  GRID=GRL-16KM  N_ITER=250  MODE=flat

using PlasticIceSheet
using NCDatasets
using CairoMakie
using Printf, Statistics, LinearAlgebra
using ImplicitDifferentiation, ADTypes, ForwardDiff    # activate the reverse-mode ext
using Zygote

# --- configuration ----------------------------------------------------------------
const METHOD = Symbol(get(ENV, "METHOD", "tiled"))      # :tiled | :percell
const GRID   = get(ENV, "GRID", "GRL-32KM")
const TOPO_NAME = Dict("GRL-16KM" => "GRL-16KM_TOPO-M17-v5.nc",
                       "GRL-32KM" => "GRL-32KM_TOPO-M17.nc",
                       "GRL-8KM"  => "GRL-8KM_TOPO-M17.nc")[GRID]
const DATADIR   = joinpath(@__DIR__, "ice_data", "Greenland", GRID)
const TOPO_FILE = joinpath(DATADIR, TOPO_NAME)
const SED_FILE  = joinpath(DATADIR, "$(GRID)_SED-L97.nc")
const OUTDIR    = joinpath(@__DIR__, "output")

# tiled-method tiling controls (default N_ROUGH × 2 sediment = 8 tiles)
const N_ROUGH    = 4
const ROUGH_KM   = 64.0          # window (~full width) for the bed-roughness metric
const SED_THRESH = 50.0          # sediment thickness (m): soft (≥) vs hard (<) bed
# per-cell-method regularization length
const REG_KM     = 32.0          # τ_b smoothing length scale (km)

const N_ITER = parse(Int, get(ENV, "N_ITER", "250"))
const LR     = 5.0e3
const τ_MIN, τ_MAX = 5.0e3, 4.0e5
# :flat gives a robust, well-conditioned eikonal adjoint; :full's dense direct adjoint is
# ill-posed on big grids (singular) — see docs/inversion.qmd.
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
println("Loading $GRID from $DATADIR  (METHOD=$METHOD)")
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
mask    = round.(mask_raw) .== 2                        # grounded ice = prescribed margin
z_b     = map(z -> isnan(z) ? 0.0 : z, z_bed)
z_s_obs = z_b .+ map(h -> isnan(h) ? 0.0 : h, H_obs)    # calibration target (observed surface)
@printf("grid %d×%d, dx=%.0f m, grounded cells %d\n", nx, ny, dx, count(mask))

# --- build the τ_b parameterization -----------------------------------------------
label = nothing                                         # populated only for METHOD=:tiled
if METHOD === :tiled
    z_sed = isfile(SED_FILE) ? (NCDataset(SED_FILE, "r") do ds; read2d(ds, "z_sed", nx, ny); end) :
            fill(NaN, nx, ny)
    use_sed = isfile(SED_FILE)
    rr = max(1, round(Int, ROUGH_KM * 1000.0 / dx / 2))
    rough = windowed_std(z_bed, rr)
    redges = N_ROUGH > 1 ? quantile(filter(isfinite, rough[mask]), (1:N_ROUGH-1) ./ N_ROUGH) : Float64[]
    roughbin(r) = isnan(r) ? 1 : clamp(1 + count(<(r), redges), 1, N_ROUGH)
    sedclass(z) = (use_sed && isfinite(z) && z >= SED_THRESH) ? 1 : 0
    nsed = use_sed ? 2 : 1
    NDOM = nsed * N_ROUGH
    global label = [sedclass(z_sed[i, j]) * N_ROUGH + roughbin(rough[i, j]) for i in 1:nx, j in 1:ny]
    onehot = zeros(nx * ny, NDOM)
    for (k, l) in enumerate(vec(label)); onehot[k, l] = 1.0; end
    taub(p) = clamp.(reshape(onehot * p, nx, ny), τ_MIN, τ_MAX)
    p0 = fill(1.0e5, NDOM)
    @printf("tiles: %d roughness × %d sediment = %d domains\n", N_ROUGH, nsed, NDOM)
elseif METHOD === :percell
    σ = REG_KM * 1000.0 / dx
    gaussmat(n) = (K = [exp(-((i - j)^2) / (2σ^2)) for i in 1:n, j in 1:n]; K ./ sum(K, dims = 2))
    Kx = gaussmat(nx); Ky = gaussmat(ny)
    taub(p) = clamp.(Kx * p * Ky, τ_MIN, τ_MAX)
    p0 = fill(1.0e5, nx, ny)
    @printf("per-cell field, %.0f km Gaussian regularization (σ=%.2f cells)\n", REG_KM, σ)
else
    error("METHOD must be :tiled or :percell, got :$METHOD")
end

# --- objective + Adam inversion (generic over the parameter shape) ----------------
loss(p) = surface_misfit(differentiable_thickness(taub(p), z_b, mask; dx, dy, SOLVER...) .+ z_b,
                         z_s_obs, mask)
p = copy(p0); m = zero(p); v = zero(p)
β1, β2, ϵ = 0.9, 0.999, 1.0e-8
rmse = Float64[]
println("iter   rmse(m)   elapsed(s)")
t0 = time()
for it in 1:N_ITER
    L, back = Zygote.pullback(loss, p)
    push!(rmse, sqrt(L))
    g = back(1.0)[1]
    global m = β1 .* m .+ (1 - β1) .* g
    global v = β2 .* v .+ (1 - β2) .* g .^ 2
    global p = p .- LR .* (m ./ (1 - β1^it)) ./ (sqrt.(v ./ (1 - β2^it)) .+ ϵ)
    (it == 1 || it % 20 == 0 || it == N_ITER) &&
        @printf("%4d   %7.1f   %8.1f\n", it, sqrt(L), time() - t0)
end

# --- final reconstruction ---------------------------------------------------------
τ_b = taub(p)
z_s, H = solve(z_b, τ_b, mask; dx, dy, SOLVER...)
dH = H .- H_obs
@printf("\nfinal rmse %.1f m;  H obs/mod mean %.0f / %.0f m;  ΔH mean %.0f rms %.0f m\n",
        last(rmse), mean(H_obs[mask]), mean(H[mask]), mean(dH[mask]), sqrt(mean(dH[mask] .^ 2)))
@printf("τ_b over mask [kPa]: median %.1f  [%.1f, %.1f]\n",
        median(τ_b[mask]) / 1e3, minimum(τ_b[mask]) / 1e3, maximum(τ_b[mask]) / 1e3)
if METHOD === :tiled
    println("\ntile   sed    rough-bin   cells   τ_b(kPa)")
    for d in 1:maximum(label)
        cells = count(k -> label[k] == d && mask[k], eachindex(label))
        cells == 0 && continue
        @printf("%4d   %4s   %9d   %5d   %7.1f\n", d, d > N_ROUGH ? "soft" : "hard",
                ((d - 1) % N_ROUGH) + 1, cells, clamp(p[d], τ_MIN, τ_MAX) / 1e3)
    end
end

# --- write NetCDF -----------------------------------------------------------------
mkpath(OUTDIR)
tag = "$(GRID)_$(METHOD)"
ncout = joinpath(OUTDIR, "invert_taub_$tag.nc")
isfile(ncout) && rm(ncout)
NCDataset(ncout, "c") do ds
    defDim(ds, "xc", nx); defDim(ds, "yc", ny)
    defVar(ds, "xc", xc, ("xc",)).attrib["units"] = "km"
    defVar(ds, "yc", yc, ("yc",)).attrib["units"] = "km"
    function put(name, A, long, units)
        v = defVar(ds, name, Float64, ("xc", "yc")); v[:, :] = A
        v.attrib["long_name"] = long; v.attrib["units"] = units
    end
    put("z_s",   z_s,   "reconstructed ice surface elevation", "m")
    put("z_bed", z_b,   "bedrock elevation (input)",           "m")
    put("H",     H,     "reconstructed ice thickness",         "m")
    put("H_obs", H_obs, "observed ice thickness (target)",     "m")
    put("tau_b", τ_b,   "inverted basal shear stress",         "Pa")
    METHOD === :tiled && put("domain", Float64.(label), "τ_b tile id (bed roughness + sediment)", "1")
    ds.attrib["title"]  = "τ_b inversion ($METHOD) fitting observed surface, Greenland $GRID"
    ds.attrib["method"] = String(METHOD)
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
hpanel((2, 1), nan_outside(τ_b ./ 1e3), "inverted τ_b", :turbo, (0, maximum(τ_b[mask]) / 1e3), "kPa")

if METHOD === :tiled
    hpanel((2, 3), nan_outside(label), "τ_b tiles (bed+sediment)", :tab20, (1, maximum(label)), "tile id")
else
    axs = Axis(fig[2, 3]; title = "thickness 1:1", xlabel = "H_obs (m)", ylabel = "H_mod (m)")
    scatter!(axs, H_obs[mask], H[mask]; markersize = 2, color = (:steelblue, 0.3))
    lines!(axs, [0, hmax], [0, hmax]; color = :black)
end
axc = Axis(fig[2, 5]; title = "surface misfit", xlabel = "iteration", ylabel = "rmse (m)")
lines!(axc, 1:length(rmse), rmse)

Label(fig[0, :], "τ_b inversion ($METHOD) — Greenland $GRID", fontsize = 20)
figout = joinpath(OUTDIR, "invert_taub_$tag.png")
save(figout, fig)
println("wrote $figout")
