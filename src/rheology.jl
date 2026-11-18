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

The law's coefficient (`β` for [`LinearSliding`](@ref) and [`WeertmanSliding`](@ref), `C`
for [`RegularizedCoulomb`](@ref)) is a scalar or a per-cell field (`nx × ny`); the other
parameters are scalars. Functions taking a grid evaluate a field coefficient per cell
(and, on a cell face, as the mean of its two cells).
"""
abstract type SlidingLaw end

"""
    LinearSliding(β)

Linear (viscous) sliding `τ_b = β u_b`, inverted as `u_b = τ_b / β`. `β` (scalar or field)
has units Pa / (velocity). Total: defined for every `τ_b ≥ 0`.
"""
struct LinearSliding{B} <: SlidingLaw
    β::B
end
LinearSliding(; β) = LinearSliding(float(β))

"""
    WeertmanSliding(β, m)

Weertman power-law sliding `τ_b = β u_b^{1/m}`, inverted as `u_b = (τ_b/β)ᵐ`, with `m` the
stress exponent (`u_b ∝ τ_bᵐ`; `m = 1` recovers [`LinearSliding`](@ref), `m ≈ 3` is
typical). `β` (scalar or field) has units Pa / (velocity)^{1/m}. Total: defined for every
`τ_b ≥ 0`.
"""
struct WeertmanSliding{B, T<:Real} <: SlidingLaw
    β::B
    m::T
end
WeertmanSliding(; β, m) = WeertmanSliding(float(β), float(m))

"""
    RegularizedCoulomb(C, u_0, m; u_max = Inf)

Regularized-Coulomb sliding (Joughin et al. 2019)

    τ_b = C (u_b / (u_b + u_0))^{1/m},   inverted as   u_b = u_0 · (τ_b/C)ᵐ / (1 − (τ_b/C)ᵐ).

- `C`     : Coulomb cap (Pa, scalar or field) — the basal drag the bed can sustain
- `u_0`   : reference (threshold) velocity
- `m`     : stress exponent of the low-speed power-law limit (`u_b ∝ τ_bᵐ`)
- `u_max` : sliding-speed cap; `Inf` (default) for none

This is the only law here that can represent a bed *at* its Coulomb plateau — exactly the
perfectly-plastic bed the reconstruction assumes. There the sliding speed is genuinely
undetermined: without a cap `basal_velocity` returns `NaN` for any cell with `τ_b ≥ C`
rather than fabricating a value (those cells should take their speed from the balance
velocity, [`balance_velocity`](@ref)); with a finite `u_max` the speed is
`min(u_b, u_max)`, and `u_max` at or above the cap.
"""
struct RegularizedCoulomb{C, T<:Real} <: SlidingLaw
    C::C
    u_0::T
    m::T
    u_max::T
end
RegularizedCoulomb(C, u_0, m; u_max = Inf) =
    RegularizedCoulomb(float(C), promote(float(u_0), float(m), float(u_max))...)
RegularizedCoulomb(; C, u_0, m, u_max = Inf) = RegularizedCoulomb(C, u_0, m; u_max)

"""
    basal_velocity(law, τ_b)

Sliding speed `u_b ≥ 0` that produces basal drag `τ_b` under `law` (with a scalar
coefficient; see [`SlidingLaw`](@ref) for fields). For [`RegularizedCoulomb`](@ref)
without a speed cap returns `NaN` where `τ_b ≥ C` (bed at/over the Coulomb cap, speed
undetermined). See each law for its inversion.
"""
basal_velocity(l::LinearSliding, τ_b) = τ_b / l.β
basal_velocity(l::WeertmanSliding, τ_b) = (τ_b / l.β)^l.m
function basal_velocity(l::RegularizedCoulomb, τ_b)
    x = (τ_b / l.C)^l.m
    τ_b < l.C && return min(l.u_0 * x / (1 - x), oftype(x, l.u_max))
    return isfinite(l.u_max) ? oftype(x, l.u_max) : oftype(x, NaN)
end

# The law with its coefficient at cell `k` (a field coefficient indexed, a scalar kept),
# and on the face between cells `k1`, `k2` (mean of the two cells).
_coef_at(c::Number, k) = c
_coef_at(c::AbstractArray, k) = c[k]
_coef_at(c::Number, k1, k2) = c
_coef_at(c::AbstractArray, k1, k2) = (c[k1] + c[k2]) / 2
_law_at(l::LinearSliding, k...) = LinearSliding(_coef_at(l.β, k...))
_law_at(l::WeertmanSliding, k...) = WeertmanSliding(_coef_at(l.β, k...), l.m)
_law_at(l::RegularizedCoulomb, k...) = RegularizedCoulomb(_coef_at(l.C, k...), l.u_0, l.m, l.u_max)

# A field coefficient must match the grid.
_check_law(l::SlidingLaw, shape) = nothing
_check_coef(c, shape) = c isa Number || size(c) == shape ||
    throw(DimensionMismatch("sliding coefficient field must match the grid $shape"))
_check_law(l::Union{LinearSliding, WeertmanSliding}, shape) = _check_coef(l.β, shape)
_check_law(l::RegularizedCoulomb, shape) = _check_coef(l.C, shape)
