# SMB-driven reconstruction on a real ice sheet: Greenland, GRL-16KM grid.
#
# Reads bed + observed thickness from a topo file, surface mass balance from a climate
# file, and observed surface velocity for comparison. Builds the grounded mask from the
# observed thickness, reconstructs the steady geometry that the SMB demands with
# `diva_reconstruct`, then reads its balance velocity off the reconstructed geometry.
#
# Outputs (under examples/output/):
#   - reconstruct_GRL-16KM.nc  : z_s, z_bed, H, smb, uxy_bal (+ ux/uy_bal)
#   - reconstruct_GRL-16KM.png : H_obs / H_mod / ΔH  (top),  u_obs / u_bal / Δu  (bottom)
#
# Run with:  julia --project=examples examples/reconstruct_greenland.jl

using PlasticIceSheet
using NCDatasets
using CairoMakie
using Printf
using Statistics

# --- configuration ----------------------------------------------------------------
const DATADIR = joinpath(@__DIR__, "ice_data", "Greenland", "GRL-16KM")
# Use the underlying data files (the GRL-16KM_TOPO.nc etc. convenience names are symlinks).
const TOPO_FILE = joinpath(DATADIR, "GRL-16KM_TOPO-M17-v5.nc")
const CLIM_FILE = joinpath(DATADIR, "GRL-16KM_MARv3.11-ERA_annmean_1961-1990.nc")
const VEL_FILE  = joinpath(DATADIR, "GRL-16KM_VEL-J18.nc")
const OUTDIR    = joinpath(@__DIR__, "output")

const H_MASK_MIN = 10.0          # grounded-ice threshold on observed H_ice (m)
const ρ_i        = 917.0         # ice density (kg m⁻³), matches PlasticParams default
const g          = 9.80665

# Rheology + sliding for the DIVA flux law inverted inside the reconstruction.
const RHEOLOGY = GlenRheology(A = 1.0e-16, n = 3.0)   # ≈ -10 °C ice, per-year ⇒ m/yr
# Regularized-Coulomb bed. `C` is a single scalar drag cap; we set it from a representative
# overburden ρ_i g H_ref (see note in the run summary — full overburden nearly switches
# sliding off, so this is the primary knob to tune).
const U0_SLIDE = 100.0           # reference (threshold) sliding velocity (m/yr)
const M_SLIDE  = 3.0

# --- helpers ----------------------------------------------------------------------

"Read a 2-D variable as (nx, ny) = (xc, yc), filling missing with NaN and permuting if
NCDatasets hands back (yc, xc)."
function read2d(ds, name, nx, ny)
    a = Float64.(coalesce.(ds[name][:, :], NaN))
    size(a) == (nx, ny) && return a
    size(a) == (ny, nx) && return permutedims(a)
    error("$name has unexpected size $(size(a)); expected ($nx,$ny) or ($ny,$nx)")
end

# --- load inputs ------------------------------------------------------------------
println("Loading inputs from $DATADIR")
xc, yc, z_bed, H_obs = NCDataset(TOPO_FILE, "r") do ds
    x = Array{Float64}(ds["xc"][:]); y = Array{Float64}(ds["yc"][:])   # km
    (x, y, read2d(ds, "z_bed", length(x), length(y)),
            read2d(ds, "H_ice", length(x), length(y)))
end
nx, ny = length(xc), length(yc)
dx = abs(xc[2] - xc[1]) * 1000.0   # km → m
dy = abs(yc[2] - yc[1]) * 1000.0
@printf("grid %d×%d, dx=%.0f m, dy=%.0f m\n", nx, ny, dx, dy)

# SMB (mm w.e. / yr) → m ice-equiv / yr.  1 mm w.e. = 1 kg m⁻², so m i.e. = (mm w.e.)/ρ_i.
smb_mma = NCDataset(CLIM_FILE, "r") do ds
    read2d(ds, "smb", nx, ny)
end
smb = smb_mma ./ ρ_i
nbad = count(isnan, smb)
nbad > 0 && (@printf("zeroing %d NaN smb cells\n", nbad); smb[isnan.(smb)] .= 0.0)

# Observed surface speed for comparison (m/yr); gaps come back as NaN.
uxy_obs = NCDataset(VEL_FILE, "r") do ds
    read2d(ds, "uxy_srf", nx, ny)
end

# --- mask from observed thickness -------------------------------------------------
mask = (H_obs .> H_MASK_MIN) .& .!isnan.(H_obs)
@printf("grounded cells (H_obs > %.0f m): %d of %d\n", H_MASK_MIN, count(mask), nx * ny)
@printf("smb over mask [m i.e./yr]: min %.3f  max %.3f  mean %.3f\n",
        minimum(smb[mask]), maximum(smb[mask]), mean(smb[mask]))

# --- reconstruction ---------------------------------------------------------------
# Representative overburden as the scalar Coulomb cap.
H_ref = mean(H_obs[mask])
C_cap = ρ_i * g * H_ref
sliding = RegularizedCoulomb(C = C_cap, u_0 = U0_SLIDE, m = M_SLIDE)
@printf("Coulomb cap C = ρ_i g H_ref = %.3e Pa  (H_ref = %.0f m, mean grounded H_obs)\n",
        C_cap, H_ref)

println("Running diva_reconstruct …")
t0 = time()
z_s, H = diva_reconstruct(z_bed, mask, smb, dx, dy;
                          rheology = RHEOLOGY, sliding = sliding,
                          n_outer = 1000, relax = 0.05, tol = 1.0)
@printf("done in %.1f s\n", time() - t0)

# --- balance velocity on the reconstructed geometry -------------------------------
bal = balance_velocity(z_s, H, mask, smb, dx, dy; H_min = H_MASK_MIN)
# Centred-gradient flow direction ⇒ velocity already on cell centres; the unstaggered
# magnitude is hypot(ux, uy) (== bal.speed up to the slope_eps regularisation).
uxy_bal = hypot.(bal.ux, bal.uy)

# Balance velocity is only meaningful where the reconstruction actually carries ice. With
# raw (net-negative at the margin) SMB and a fixed mask, the ablation fringe reconstructs to
# H≈0; those cells are still flux outlets, so Q/(wH) blows up there. Compare velocities only
# where H_mod > threshold (thickness is still compared over the full observed mask below).
mask_mod = mask .& (H .> H_MASK_MIN)

# --- diagnostics ------------------------------------------------------------------
dH = H .- H_obs
@printf("\nH    : obs mean %.0f m, mod mean %.0f m  (deglaciated fringe: %d of %d cells)\n",
        mean(H_obs[mask]), mean(H[mask]), count(mask) - count(mask_mod), count(mask))
@printf("ΔH   : mean %.0f m, rms %.0f m, max|Δ| %.0f m\n",
        mean(dH[mask]), sqrt(mean(dH[mask] .^ 2)), maximum(abs.(dH[mask])))
qtl(a, p) = (s = sort(filter(isfinite, a)); s[clamp(ceil(Int, p * length(s)), 1, length(s))])
uo = uxy_obs[mask_mod]; ub = uxy_bal[mask_mod]
@printf("|u|  : obs      median %.1f  p90 %.1f  p99 %.1f  max %.0f m/yr  (over H_mod>%.0f)\n",
        qtl(uo, 0.5), qtl(uo, 0.9), qtl(uo, 0.99), maximum(filter(isfinite, uo)), H_MASK_MIN)
@printf("|u|  : balance  median %.1f  p90 %.1f  p99 %.1f  max %.0f m/yr\n",
        qtl(ub, 0.5), qtl(ub, 0.9), qtl(ub, 0.99), maximum(ub))

# --- write NetCDF -----------------------------------------------------------------
mkpath(OUTDIR)
ncout = joinpath(OUTDIR, "reconstruct_GRL-16KM.nc")
isfile(ncout) && rm(ncout)
NCDataset(ncout, "c") do ds
    defDim(ds, "xc", nx); defDim(ds, "yc", ny)
    defVar(ds, "xc", xc, ("xc",)).attrib["units"] = "km"
    defVar(ds, "yc", yc, ("yc",)).attrib["units"] = "km"
    function put(name, A, long, units)
        v = defVar(ds, name, Float64, ("xc", "yc"))
        v[:, :] = A
        v.attrib["long_name"] = long
        v.attrib["units"] = units
    end
    put("z_s",     z_s,     "reconstructed ice surface elevation", "m")
    put("z_bed",   z_bed,   "bedrock elevation (input)",           "m")
    put("H",       H,       "reconstructed ice thickness",         "m")
    put("smb",     smb,     "surface mass balance (input)",        "m ice-equiv yr-1")
    put("uxy_bal", uxy_bal, "balance velocity magnitude (cell centre)", "m yr-1")
    put("ux_bal",  bal.ux,  "balance velocity, x-component",       "m yr-1")
    put("uy_bal",  bal.uy,  "balance velocity, y-component",       "m yr-1")
    ds.attrib["title"] = "SMB-driven plastic reconstruction, Greenland GRL-16KM"
    ds.attrib["source_topo"] = basename(TOPO_FILE)
    ds.attrib["source_clim"] = basename(CLIM_FILE)
end
println("wrote $ncout")

# --- figure -----------------------------------------------------------------------
nan_outside(A, msk) = ifelse.(msk, Float64.(A), NaN)
logv(A) = map(a -> isnan(a) ? NaN : log10(max(a, 1.0)), A)
# thickness over the observed mask; velocity over the reconstructed-ice mask
Hobs_p = nan_outside(H_obs, mask); Hmod_p = nan_outside(H, mask); dH_p = nan_outside(dH, mask)
uobs_p = nan_outside(uxy_obs, mask_mod); ubal_p = nan_outside(uxy_bal, mask_mod)
du_p   = nan_outside(uxy_bal .- uxy_obs, mask_mod)

fig = Figure(size = (1500, 1000))
hmax  = max(maximum(Hobs_p[mask]), maximum(Hmod_p[mask]))
dHlim = maximum(abs.(dH_p[mask]))
vlim  = (0.0, 3.5)   # log10 m/yr ⇒ up to ~3000 m/yr
dulim = 1000.0

function panel(pos, A, title, cmap, crange; cbarlabel = "")
    ax = Axis(fig[pos...]; title, aspect = DataAspect(),
              xlabel = "x (km)", ylabel = "y (km)")
    hm = heatmap!(ax, xc, yc, A; colormap = cmap, colorrange = crange,
                  nan_color = :transparent)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label = cbarlabel, width = 12)
    return ax
end

panel((1, 1), Hobs_p, "H_obs",  :viridis, (0, hmax); cbarlabel = "m")
panel((1, 3), Hmod_p, "H_mod",  :viridis, (0, hmax); cbarlabel = "m")
panel((1, 5), dH_p,   "H_mod − H_obs", :balance, (-dHlim, dHlim); cbarlabel = "m")
panel((2, 1), logv(uobs_p), "u_obs (surface)", :turbo, vlim; cbarlabel = "log₁₀ m/yr")
panel((2, 3), logv(ubal_p), "u_balance",       :turbo, vlim; cbarlabel = "log₁₀ m/yr")
panel((2, 5), du_p, "u_balance − u_obs", :balance, (-dulim, dulim); cbarlabel = "m/yr")

Label(fig[0, :], "SMB-driven reconstruction — Greenland GRL-16KM", fontsize = 20)
figout = joinpath(OUTDIR, "reconstruct_GRL-16KM.png")
save(figout, fig)
println("wrote $figout")
