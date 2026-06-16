# Physical parameters for the perfectly-plastic ice-sheet reconstruction.
#
# These are held as plain (non-differentiated) scalars: the AD target is the basal
# shear stress field `τ_b`, not these constants. Defaults match ICESHEET
# (Gowan et al. 2016, global_parameters.f90).

"""
    PlasticParams(; ρ_i, ρ_w, g, z_ss, H_min)

Physical constants and reconstruction settings.

- `ρ_i`   : ice density (kg m⁻³), default 917.0
- `ρ_w`   : sea-water density (kg m⁻³), default 1025.0
- `g`     : gravitational acceleration (m s⁻²), default 9.80665
- `z_ss`  : sea-surface elevation (sea level) (m), default 0.0
- `H_min` : minimum thickness floor (m) used to keep the driving-stress slope
            `τ_b / (ρ_i g H)` finite in `:full` mode, default 1.0
"""
Base.@kwdef struct PlasticParams{T<:Real}
    ρ_i::T   = 917.0
    ρ_w::T   = 1025.0
    g::T     = 9.80665
    z_ss::T  = 0.0
    H_min::T = 1.0
end

# Flotation (minimum grounding) thickness `H_flt` at a marine margin: the thickness of ice
# whose weight balances the local water column. Zero where the bed is at or above sea
# level. Matches ICESHEET's marine grounding-line treatment (H_flt = (ρ_w/ρ_i)(z_ss − z_b)).
@inline flotation_thickness(z_b, p::PlasticParams) =
    (p.ρ_w / p.ρ_i) * max(p.z_ss - z_b, zero(z_b))
