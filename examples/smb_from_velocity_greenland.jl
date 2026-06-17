# Closure B on real data: recover surface mass balance from observed velocity + thickness.
#
# Assuming uxy_srf ≈ ū (surface speed = depth-averaged speed), the steady-state implied SMB
# is the flux divergence ḃ = ∇·(ū H), routed down the observed surface. This is a pure
# diagnostic on observations — no reconstruction, no rheology — so it tests whether the
# observed (velocity, thickness) pair is consistent with the observed climate SMB.
#
# Compares smb_obs (CLIM) / smb_mod (implied) / difference.
#
# Run with:  julia --project=examples examples/smb_from_velocity_greenland.jl

using PlasticIceSheet
using NCDatasets
using CairoMakie
using Printf
using Statistics

# --- configuration ----------------------------------------------------------------
const GRID = "GRL-16KM"          # "GRL-8KM" or "GRL-16KM"
# underlying topo file (the GRID_TOPO.nc convenience name is a symlink to these)
const TOPO_NAME = Dict("GRL-16KM" => "GRL-16KM_TOPO-M17-v5.nc",
                       "GRL-8KM"  => "GRL-8KM_TOPO-M17.nc")[GRID]
const DATADIR   = joinpath(@__DIR__, "ice_data", "Greenland", GRID)
const TOPO_FILE = joinpath(DATADIR, TOPO_NAME)
const CLIM_FILE = joinpath(DATADIR, "$(GRID)_MARv3.11-ERA_annmean_1961-1990.nc")
const VEL_FILE  = joinpath(DATADIR, "$(GRID)_VEL-J18.nc")
const OUTDIR    = joinpath(@__DIR__, "output")

const H_MASK_MIN   = 10.0        # grounded-ice threshold on observed H_ice (m)
const ρ_i          = 917.0       # ice density (kg m⁻³)
const SMOOTH_PASSES = 5          # mask-aware smoothing passes on the velocity (≈ √n cells)

read2d(ds, name, nx, ny) =
    (a = Float64.(coalesce.(ds[name][:, :], NaN)); size(a) == (nx, ny) ? a : permutedims(a))

# In-mask 5-point smoother that also fills NaN gaps: each masked cell becomes the mean of
# its finite masked neighbours (+ itself). Repeated `npass` times ⇒ diffusive low-pass.
function smooth_masked(A, mask, npass)
    nx, ny = size(A)
    B = copy(A)
    for _ in 1:npass
        C = copy(B)
        @inbounds for j in 1:ny, i in 1:nx
            mask[i, j] || continue
            s = 0.0; n = 0
            for (di, dj) in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
                ii, jj = i + di, j + dj
                if 1 <= ii <= nx && 1 <= jj <= ny && mask[ii, jj] && isfinite(B[ii, jj])
                    s += B[ii, jj]; n += 1
                end
            end
            n > 0 && (C[i, j] = s / n)
        end
        B = C
    end
    return B
end

# --- load inputs ------------------------------------------------------------------
println("Loading inputs from $DATADIR")
xc, yc, z_bed, H_obs = NCDataset(TOPO_FILE, "r") do ds
    x = Array{Float64}(ds["xc"][:]); y = Array{Float64}(ds["yc"][:])
    (x, y, read2d(ds, "z_bed", length(x), length(y)),
            read2d(ds, "H_ice", length(x), length(y)))
end
nx, ny = length(xc), length(yc)
dx = abs(xc[2] - xc[1]) * 1000.0   # km → m
dy = abs(yc[2] - yc[1]) * 1000.0

smb_mma = NCDataset(CLIM_FILE, "r") do ds; read2d(ds, "smb", nx, ny); end
smb_mma[smb_mma .<= -9990] .= NaN                       # guard the -9999 fill value
smb_obs = smb_mma ./ ρ_i                                # m i.e./yr
uxy_obs = NCDataset(VEL_FILE, "r") do ds; read2d(ds, "uxy_srf", nx, ny); end          # m/yr

# --- assemble fields --------------------------------------------------------------
mask = (H_obs .> H_MASK_MIN) .& .!isnan.(H_obs) .& isfinite.(smb_obs)
z_s  = z_bed .+ H_obs                                   # observed surface for flux routing

ngap = count(isnan, uxy_obs[mask])
@printf("smoothing velocity: %d passes; %d of %d cells were data-gaps (filled by smoother)\n",
        SMOOTH_PASSES, ngap, count(mask))
speed = smooth_masked(uxy_obs, mask, SMOOTH_PASSES)
speed = ifelse.(mask .& isfinite.(speed), speed, 0.0)   # any cell with no finite neighbours

# --- closure B: implied SMB = ∇·(ū H) ---------------------------------------------
smb_mod = smb_from_velocity((; speed = speed), H_obs, z_s, mask, dx, dy)

# --- diagnostics ------------------------------------------------------------------
o = smb_obs[mask]; m = smb_mod[mask]
@printf("grid %d×%d, dx=%.0f m, grounded cells %d\n", nx, ny, dx, count(mask))
@printf("smb_obs  mean %.3f  std %.3f  [%.2f, %.2f] m i.e./yr\n",
        mean(o), std(o), minimum(o), maximum(o))
@printf("smb_mod  mean %.3f  std %.3f  [%.2f, %.2f] m i.e./yr\n",
        mean(m), std(m), minimum(m), maximum(m))
@printf("diff     mean %.3f  rms %.3f  corr %.3f\n",
        mean(m .- o), sqrt(mean((m .- o) .^ 2)), cor(o, m))

# --- figure -----------------------------------------------------------------------
mkpath(OUTDIR)
nan_outside(A) = ifelse.(mask, Float64.(A), NaN)
clim  = round(quantile(abs.(o), 0.98), digits = 1)        # symmetric SMB range
dclim = round(quantile(abs.(m .- o), 0.98), digits = 1)

fig = Figure(size = (1500, 560))
function panel(col, A, title, crange)
    ax = Axis(fig[1, col]; title, aspect = DataAspect(), xlabel = "x (km)", ylabel = "y (km)")
    hm = heatmap!(ax, xc, yc, A; colormap = :balance, colorrange = crange,
                  nan_color = :transparent)
    Colorbar(fig[1, col + 1], hm; label = "m i.e./yr", width = 12)
end
panel(1, nan_outside(smb_obs),           "smb_obs (CLIM)",      (-clim, clim))
panel(3, nan_outside(smb_mod),           "smb_mod = ∇·(ū H)",   (-clim, clim))
panel(5, nan_outside(smb_mod .- smb_obs), "smb_mod − smb_obs",  (-dclim, dclim))
Label(fig[0, :], "SMB from observed velocity — Greenland $GRID", fontsize = 20)

figout = joinpath(OUTDIR, "smb_from_velocity_$GRID.png")
save(figout, fig)
println("wrote $figout")
