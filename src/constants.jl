# Physical parameters for the perfectly-plastic ice-sheet reconstruction.
#
# These are held as plain (non-differentiated) scalars: the AD target is the basal
# shear stress field `τ_b`, not these constants. Defaults match ICESHEET
# (Gowan et al. 2016, global_parameters.f90).

"""
    PlasticParams(; ρ_i, ρ_w, g, H_min)

Physical constants and reconstruction settings.

- `ρ_i`   : ice density (kg m⁻³), default 917.0
- `ρ_w`   : sea-water density (kg m⁻³), default 1025.0
- `g`     : gravitational acceleration (m s⁻²), default 9.80665
- `H_min` : minimum thickness floor (m) used to keep the driving-stress slope
            `τ_b / (ρ_i g H)` finite in `:full` mode, default 1.0

Sea level is an input like the bed, not a constant: pass `z_ss` (scalar or field) to
[`solve`](@ref) and the other solvers.
"""
Base.@kwdef struct PlasticParams{T<:Real}
    ρ_i::T   = 917.0
    ρ_w::T   = 1025.0
    g::T     = 9.80665
    H_min::T = 1.0
end

"""
    flotation_thickness(z_b, z_ss, params)

Flotation (minimum grounding) thickness `H_flt = (ρ_w/ρ_i) max(z_ss − z_b, 0)` at a marine
margin: the ice thickness whose weight balances the local water column, zero where the bed
is at or above sea level `z_ss`. Elementwise; broadcast over fields.
"""
@inline flotation_thickness(z_b, z_ss, p::PlasticParams) =
    (p.ρ_w / p.ρ_i) * max(z_ss - z_b, zero(z_b))

# A scalar input broadcast to a field of `shape`, or a field checked against `shape`.
function _as_field(x, shape, name)
    x isa Number && return fill(float(x), shape)
    size(x) == shape || throw(DimensionMismatch("$name must be a scalar or match the grid $shape"))
    return x
end
