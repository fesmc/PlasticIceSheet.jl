# Physical parameters for the perfectly-plastic ice-sheet reconstruction.
#
# These are held as plain (non-differentiated) scalars: the AD target is the basal
# shear stress field `τ`, not these constants. Defaults match ICESHEET 1.0
# (Gowan et al. 2016, global_parameters.f90).

"""
    PlasticParams(; ρ_i, ρ_w, g, sea_level, H_min)

Physical constants and reconstruction settings.

- `ρ_i`      : ice density (kg m⁻³), default 917.0
- `ρ_w`      : sea-water density (kg m⁻³), default 1025.0
- `g`        : gravitational acceleration (m s⁻²), default 9.80665
- `sea_level`: sea-surface elevation (m), default 0.0
- `H_min`    : minimum thickness floor (m) used to keep the driving-stress slope
               `τ / (ρ_i g H)` finite in `:hj` mode, default 1.0
"""
Base.@kwdef struct PlasticParams{T<:Real}
    ρ_i::T       = 917.0
    ρ_w::T       = 1025.0
    g::T         = 9.80665
    sea_level::T = 0.0
    H_min::T     = 1.0
end

# Flotation (minimum grounding) thickness at a marine margin: the thickness of ice
# whose weight balances the local water column. Zero where the bed is at or above
# sea level. Matches ICESHEET's marine grounding-line treatment (H = -B ρ_w/ρ_i).
@inline flotation_thickness(bed, p::PlasticParams) =
    (p.ρ_w / p.ρ_i) * max(p.sea_level - bed, zero(bed))
