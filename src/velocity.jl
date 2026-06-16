# Closure B — read a velocity, then a surface-mass-balance estimate, off a fixed plastic
# geometry by assuming a rheology and a sliding law.
#
# The plastic state is the membrane-free limit of SSA/DIVA (τ_d = τ_b, with the
# longitudinal-stress divergence vanishing), so the DIVA velocity here is purely local and
# analytical: flow follows surface descent, with magnitude split into internal deformation
# (Glen) plus basal sliding (the sliding law), both evaluated at the local driving stress
# τ = τ_b. In steady state the implied SMB is the flux divergence ∇·(ū H).

"""
    flow_direction(z_s, dx, dy; slope_eps = 1.0e-6)

Unit ice-flow direction `(eˣ, eʸ) = −∇z_s / |∇z_s|` on every cell, from the central
surface gradient. The magnitude is regularized as `√(|∇z_s|² + slope_eps²)` so the field
is smooth through ice divides (where `|∇z_s| → 0`); there the direction simply tapers to
zero, matching the vanishing horizontal velocity at a dome.
"""
function flow_direction(z_s, dx, dy; slope_eps = 1.0e-6)
    z_sx, z_sy = _grad(z_s, dx, dy)
    ex = similar(z_sx)
    ey = similar(z_sy)
    @inbounds for k in eachindex(z_sx)
        g = sqrt(z_sx[k]^2 + z_sy[k]^2 + slope_eps^2)
        ex[k] = -z_sx[k] / g
        ey[k] = -z_sy[k] / g
    end
    return ex, ey
end

"""
    diva_velocity(z_s, H, τ_b, mask, dx, dy; rheology, sliding, slope_eps = 1.0e-6)

Local DIVA depth-averaged velocity on the fixed plastic geometry. Flow follows
[`flow_direction`](@ref); the speed is the sum of an internal-deformation part
([`deformational_velocity`](@ref) with the driving stress `τ = τ_b`) and a basal-sliding
part ([`basal_velocity`](@ref) for `sliding`). `τ_b` may be a scalar or a field. Ice-free
cells return zero velocity.

Returns a named tuple with everything exposed:
- `ux`, `uy` : velocity vector components (`speed · eˣ`, `speed · eʸ`)
- `speed`    : total depth-averaged speed `u_b + u_def`
- `u_b`      : basal-sliding speed (`NaN` where a `RegularizedCoulomb` bed is at its cap)
- `u_def`    : internal-deformation speed
- `ex`, `ey` : unit flow direction

Pass the result to [`smb_from_velocity`](@ref) for the implied steady-state SMB. All
quantities are differentiable w.r.t. `τ_b`, so the tuple doubles as an observation
operator for `τ_b` inversion.
"""
function diva_velocity(z_s, H, τ_b, mask, dx, dy;
                       rheology::GlenRheology, sliding::SlidingLaw, slope_eps = 1.0e-6)
    size(z_s) == size(H) == size(mask) ||
        throw(DimensionMismatch("z_s, H, mask must share shape"))
    τ_bf = τ_b isa Number ? fill(float(τ_b), size(z_s)) : τ_b
    size(τ_bf) == size(z_s) || throw(DimensionMismatch("τ_b and z_s must match"))

    ex, ey = flow_direction(z_s, dx, dy; slope_eps)
    T = promote_type(eltype(H), eltype(τ_bf), eltype(ex))
    nx, ny = size(z_s)
    ux = zeros(T, nx, ny)
    uy = zeros(T, nx, ny)
    speed = zeros(T, nx, ny)
    u_b = zeros(T, nx, ny)
    u_def = zeros(T, nx, ny)
    @inbounds for k in eachindex(z_s)
        mask[k] || continue
        udk = deformational_velocity(rheology, τ_bf[k], H[k])
        ubk = basal_velocity(sliding, τ_bf[k])
        s = ubk + udk
        u_def[k] = udk
        u_b[k] = ubk
        speed[k] = s
        ux[k] = s * ex[k]
        uy[k] = s * ey[k]
    end
    return (; ux, uy, speed, u_b, u_def, ex, ey)
end

# `smb_from_velocity` (the implied SMB, ∇·(ūH)) lives in `balance.jl`, where it shares the
# finite-volume MFD flux operator with `balance_flux` so the two are exact inverses.
