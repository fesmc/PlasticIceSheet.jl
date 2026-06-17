# Basal-shear-stress inversion on real data: the original ICESHEET-style purpose.
#
# Given the observed bed `z_bed`, the grounded mask, and the observed ice surface
# `z_s_obs = z_bed + H_obs`, recover the per-cell basal shear stress `τ_b` whose plastic
# reconstruction best fits that surface. The plastic solve always fills the whole mask, so
# this fits the observed *geometry* on the fixed extent — not the extent itself.
#
# τ_b is regularised to a length scale REG_KM by a Gaussian-blur reparameterisation:
# we optimise a raw field φ and set τ_b = clamp(blur_σ(φ)), with blur a separable linear
# smoother (σ = REG_KM/dx cells). Being linear it differentiates cleanly under Zygote, and
# it *guarantees* τ_b varies only on scales ≳ REG_KM regardless of the data fit.
#
# Reverse-mode gradients of the misfit w.r.t. τ_b come from the implicit-differentiation
# extension (the converged Godunov fixed point differentiated via the IFT).
#
# Run with:  julia --project=examples examples/invert_taub_greenland.jl
#   env knobs:  GRID=GRL-32KM  N_ITER=300

using PlasticIceSheet
using NCDatasets
using CairoMakie
using Printf, Statistics, LinearAlgebra
using ImplicitDifferentiation, ADTypes, ForwardDiff    # activate the reverse-mode ext
using Zygote

# --- configuration ----------------------------------------------------------------
const GRID = get(ENV, "GRID", "GRL-16KM")
const TOPO_NAME = Dict("GRL-16KM" => "GRL-16KM_TOPO-M17-v5.nc",
                       "GRL-32KM" => "GRL-32KM_TOPO-M17.nc",
                       "GRL-8KM"  => "GRL-8KM_TOPO-M17.nc")[GRID]
const DATADIR   = joinpath(@__DIR__, "ice_data", "Greenland", GRID)
const TOPO_FILE = joinpath(DATADIR, TOPO_NAME)
const OUTDIR    = joinpath(@__DIR__, "output")

const H_MASK_MIN = 10.0          # grounded-ice threshold on observed H_ice (m)
const REG_KM     = 32.0          # τ_b smoothing length scale (km)
const N_ITER     = parse(Int, get(ENV, "N_ITER", "250"))
const LR         = 4.0e3         # Adam step in τ_b units (Pa); Adam rescales by grad rms
const τ_MIN, τ_MAX = 5.0e3, 4.0e5

# forward / adjoint solve controls. :flat (|∇H|≈|∇z_s|) gives a robust, well-conditioned
# eikonal adjoint; :full is bed-aware but its dense direct adjoint is ill-posed on big grids.
const MODE   = Symbol(get(ENV, "MODE", "flat"))
const SOLVER = (; mode = MODE, params = PlasticParams(),
                max_sweeps = 600, tol = 1.0e-9, n_outer = 500, outer_tol = 1.0e-5, relax = 0.4)

read2d(ds, name, nx, ny) =
    (a = Float64.(coalesce.(ds[name][:, :], NaN)); size(a) == (nx, ny) ? a : permutedims(a))

# --- load inputs ------------------------------------------------------------------
println("Loading $GRID from $DATADIR")
xc, yc, z_bed, H_obs = NCDataset(TOPO_FILE, "r") do ds
    x = Array{Float64}(ds["xc"][:]); y = Array{Float64}(ds["yc"][:])
    (x, y, read2d(ds, "z_bed", length(x), length(y)),
            read2d(ds, "H_ice", length(x), length(y)))
end
nx, ny = length(xc), length(yc)
dx = abs(xc[2] - xc[1]) * 1000.0
dy = abs(yc[2] - yc[1]) * 1000.0
z_bed[z_bed .<= -9990] .= NaN          # guard the -9999 fill value
H_obs[H_obs .<= -9990] .= NaN
mask    = (H_obs .> H_MASK_MIN) .& .!isnan.(H_obs)
z_b     = map(z -> isnan(z) ? 0.0 : z, z_bed)
z_s_obs = z_b .+ map(h -> isnan(h) ? 0.0 : h, H_obs)
@printf("grid %d×%d, dx=%.0f m, grounded cells %d\n", nx, ny, dx, count(mask))

# --- τ_b regularisation: separable Gaussian blur ----------------------------------
σ = REG_KM * 1000.0 / dx                                   # smoothing σ in cells
gaussmat(n) = (K = [exp(-((i - j)^2) / (2σ^2)) for i in 1:n, j in 1:n]; K ./ sum(K, dims = 2))
const Kx = gaussmat(nx)
const Ky = gaussmat(ny)
blur(φ) = Kx * φ * Ky                                       # smooth both axes (Ky symmetric)
taub(φ) = clamp.(blur(φ), τ_MIN, τ_MAX)

# --- objective --------------------------------------------------------------------
function loss(φ)
    H = differentiable_thickness(taub(φ), z_b, mask; dx, dy, SOLVER...)
    return surface_misfit(H .+ z_b, z_s_obs, mask)
end

# --- Adam inversion ---------------------------------------------------------------
φ = fill(1.0e5, nx, ny)          # uniform 100 kPa first guess (blur preserves constants)
m = zero(φ); v = zero(φ)
β1, β2, ϵ = 0.9, 0.999, 1.0e-8
rmse = Float64[]
println("iter   rmse(m)   elapsed(s)")
t0 = time()
for it in 1:N_ITER
    L, back = Zygote.pullback(loss, φ)
    push!(rmse, sqrt(L))
    g = back(1.0)[1]
    global m = β1 .* m .+ (1 - β1) .* g
    global v = β2 .* v .+ (1 - β2) .* g .^ 2
    m̂ = m ./ (1 - β1^it); v̂ = v ./ (1 - β2^it)
    global φ = φ .- LR .* m̂ ./ (sqrt.(v̂) .+ ϵ)
    (it == 1 || it % 10 == 0 || it == N_ITER) &&
        @printf("%4d   %7.1f   %8.1f\n", it, sqrt(L), time() - t0)
end

# --- final reconstruction ---------------------------------------------------------
τ_b = taub(φ)
z_s, H = solve(z_b, τ_b, mask; dx, dy, SOLVER...)
dH = H .- H_obs
@printf("\nfinal rmse %.1f m;  H obs/mod mean %.0f / %.0f m;  ΔH mean %.0f rms %.0f m\n",
        last(rmse), mean(H_obs[mask]), mean(H[mask]), mean(dH[mask]), sqrt(mean(dH[mask] .^ 2)))
@printf("τ_b over mask [kPa]: median %.1f  [%.1f, %.1f]\n",
        median(τ_b[mask]) / 1e3, minimum(τ_b[mask]) / 1e3, maximum(τ_b[mask]) / 1e3)

# --- write NetCDF -----------------------------------------------------------------
mkpath(OUTDIR)
ncout = joinpath(OUTDIR, "invert_taub_$GRID.nc")
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
    ds.attrib["title"] = "τ_b inversion to fit observed surface, Greenland $GRID"
    ds.attrib["reg_length_km"] = REG_KM
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

axc = Axis(fig[2, 3]; title = "surface misfit", xlabel = "iteration", ylabel = "rmse (m)")
lines!(axc, 1:length(rmse), rmse)

axs = Axis(fig[2, 5]; title = "thickness 1:1", xlabel = "H_obs (m)", ylabel = "H_mod (m)")
scatter!(axs, H_obs[mask], H[mask]; markersize = 2, color = (:steelblue, 0.3))
lines!(axs, [0, hmax], [0, hmax]; color = :black)

Label(fig[0, :], "τ_b inversion → surface fit — Greenland $GRID  (reg $(Int(REG_KM)) km)", fontsize = 20)
figout = joinpath(OUTDIR, "invert_taub_$GRID.png")
save(figout, fig)
println("wrote $figout")
