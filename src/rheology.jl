# Constitutive ingredients for the DIVA velocity closure layered on a plastic
# reconstruction: internal deformation (Glen) and basal sliding.
#
# These are *not* needed by the plastic solve itself — that turns `(mask, τ_b)` into
# geometry via force balance alone. They enter only when reading a velocity (and hence a
# surface-mass-balance estimate) off the fixed geometry; see `velocity.jl` / `balance.jl`.
#
# Unit convention: velocities come out in the time unit carried by `A` and by the sliding
# coefficients. Use a per-year rate factor and per-year sliding coefficients for m yr⁻¹.

# --- internal deformation (Glen) --------------------------------------------------

"""
    GlenRheology(; A, n)

Isothermal Glen flow law for the depth-averaged deformational velocity.

- `A` : rate factor (Pa⁻ⁿ time⁻¹), default `1.0e-16` (≈ ice near −10 °C, per year)
- `n` : flow-law exponent, default `3`

The depth-averaged deformational speed under a driving stress `τ` over thickness `H`,
using the plastic constraint `τ_d = τ_b` at yield, is the SIA closed form

    ū_def = 2A/(n+2) · τⁿ · H .
"""
struct GlenRheology{T<:Real}
    A::T
    n::T
end
GlenRheology(; A = 1.0e-16, n = 3.0) = GlenRheology(promote(float(A), float(n))...)

"""
    deformational_velocity(rheology, τ, H)

Depth-averaged internal-deformation speed `2A/(n+2)·τⁿ·H` (always ≥ 0). `τ` is the local
driving stress, which equals `τ_b` at yield in a plastic reconstruction.
"""
deformational_velocity(r::GlenRheology, τ, H) = (2 * r.A / (r.n + 2)) * τ^r.n * H

# --- basal sliding ----------------------------------------------------------------

"""
    SlidingLaw

Abstract supertype for basal sliding laws. Concrete laws implement
[`basal_velocity`](@ref) `(law, τ_b) -> u_b`, the sliding speed that produces basal drag
`τ_b`. In a plastic reconstruction the drag equals the local driving stress, so
`basal_velocity` is a pointwise read-out of the solved `τ_b` field.
"""
abstract type SlidingLaw end

"""
    LinearSliding(β)

Linear (viscous) sliding `τ_b = β u_b`, inverted as `u_b = τ_b / β`. `β` has units
Pa / (velocity). Total: defined for every `τ_b ≥ 0`.
"""
struct LinearSliding{T<:Real} <: SlidingLaw
    β::T
end
LinearSliding(; β) = LinearSliding(float(β))

"""
    WeertmanSliding(β, m)

Weertman power-law sliding `τ_b = β u_b^{1/m}`, inverted as `u_b = (τ_b/β)ᵐ`, with `m` the
stress exponent (`u_b ∝ τ_bᵐ`; `m = 1` recovers [`LinearSliding`](@ref), `m ≈ 3` is
typical). `β` has units Pa / (velocity)^{1/m}. Total: defined for every `τ_b ≥ 0`.
"""
struct WeertmanSliding{T<:Real} <: SlidingLaw
    β::T
    m::T
end
WeertmanSliding(β, m) = WeertmanSliding(promote(float(β), float(m))...)
WeertmanSliding(; β, m) = WeertmanSliding(β, m)

"""
    RegularizedCoulomb(C, u_0, m)

Regularized-Coulomb sliding (Joughin et al. 2019)

    τ_b = C (u_b / (u_b + u_0))^{1/m},   inverted as   u_b = u_0 · (τ_b/C)ᵐ / (1 − (τ_b/C)ᵐ).

- `C`   : Coulomb cap (Pa) — the basal drag the bed can sustain. **Must exceed the local
          operating `τ_b`** for a finite sliding speed.
- `u_0` : reference (threshold) velocity
- `m`   : stress exponent of the low-speed power-law limit (`u_b ∝ τ_bᵐ`)

This is the only law here that can represent a bed *at* its Coulomb plateau — exactly the
perfectly-plastic bed the reconstruction assumes. There the sliding speed is genuinely
undetermined, so `basal_velocity` returns `NaN` for any cell with `τ_b ≥ C` rather than
fabricating a value; those cells should take their speed from the balance velocity
([`balance_velocity`](@ref)) instead.
"""
struct RegularizedCoulomb{T<:Real} <: SlidingLaw
    C::T
    u_0::T
    m::T
end
RegularizedCoulomb(C, u_0, m) = RegularizedCoulomb(promote(float(C), float(u_0), float(m))...)
RegularizedCoulomb(; C, u_0, m) = RegularizedCoulomb(C, u_0, m)

"""
    basal_velocity(law, τ_b)

Sliding speed `u_b ≥ 0` that produces basal drag `τ_b` under `law`. For
[`RegularizedCoulomb`](@ref) returns `NaN` where `τ_b ≥ C` (bed at/over the Coulomb cap,
speed undetermined). See each law for its inversion.
"""
basal_velocity(l::LinearSliding, τ_b) = τ_b / l.β
basal_velocity(l::WeertmanSliding, τ_b) = (τ_b / l.β)^l.m
function basal_velocity(l::RegularizedCoulomb, τ_b)
    x = (τ_b / l.C)^l.m
    return τ_b < l.C ? l.u_0 * x / (1 - x) : oftype(x, NaN)
end
