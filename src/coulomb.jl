# Coulomb yield law in mode :surface (port of the Fortran `plastic_solve_coulomb`):
#
#     τ_b = μ N_eff,   N = ρ_i g H max((1 − H_f/H)^p, δ),   1/N_eff = 1/N + 1/(ρ_i g H_N),
#
# with H_f the flotation thickness ((1 − H_f/H)^p = 0 at or below flotation, Leguy et al.
# 2014) and the floor δ of the overburden (as Bueler and van Pelt 2015). The plastic
# equation ρ_i g H |∇z_s| = τ_b becomes a slope law |∇z_s| = s(H) = μ N_eff/(ρ_i g H).

"""
    CoulombYield(μ; H_N, p = 1.0, δ = 0.05)

Coulomb yield law for the basal shear stress, `τ_b = μ N_eff(H)`, passed to
[`solve`](@ref) in place of `τ_b`:

    N = ρ_i g H max((1 − H_f/H)^p, δ),    1/N_eff = 1/N + 1/(ρ_i g H_N),

where `H_f` is the flotation thickness (0 on land). `μ ≥ 0` (scalar or field) is the
friction coefficient times any bed factor. `p ∈ [0, 1]` is the exponent of the ocean
connection, `H_N > 0` (m) caps `N_eff/(ρ_i g)`, and `δ ∈ (0, 1]` floors `N` as a fraction
of the overburden. The surface slope is `μ N_eff/(ρ_i g H)`: a constant `μ` for
`H ≪ H_N` on land (a wedge), the Nye profile of `τ_b = μ ρ_i g H_N` for `H ≫ H_N`, and
`μ δ H_N/(δ H + H_N)` at flotation. The floor `δ > 0` makes the solution unique and
continuous in the bed and sea level.
"""
struct CoulombYield{M, T <: AbstractFloat}
    μ::M
    p::T
    H_N::T
    δ::T
    function CoulombYield(μ; H_N, p = 1.0, δ = 0.05)
        p, H_N, δ = promote(float(p), float(H_N), float(δ))
        0 <= p <= 1 ||
            throw(ArgumentError("p must be in [0, 1] (the update is monotone for p ≤ 1), got $p"))
        0 < H_N < Inf || throw(ArgumentError("H_N must be positive and finite, got $H_N"))
        0 < δ <= 1 ||
            throw(ArgumentError("δ must be in (0, 1] (the floor makes the solution unique), got $δ"))
        return new{typeof(μ), typeof(p)}(μ, p, H_N, δ)
    end
end

# N_eff/(ρ_i g) (m) of the law at thickness H with flotation thickness hf; 0 for H ≤ 0.
@inline function _coulomb_neff(H, hf, law)
    H > 0 || return zero(H)
    if hf > 0
        n = H > hf ? (law.p == 1 ? H - hf : H * (1 - hf / H)^law.p) : zero(H)
        n = max(n, law.δ * H)
    else
        n = H
    end
    return n * law.H_N / (n + law.H_N)
end

# q = s/μ = N_eff/(ρ_i g h) at h and d ln q/dh (on the floor: d ln q/dh = (c − 1)/h).
@inline function _coulomb_ratio(h, hf, law)
    if hf > 0
        r = h > hf ? (law.p == 1 ? 1 - hf / h : (1 - hf / h)^law.p) : zero(h)
        if r < law.δ
            c = law.H_N / (h * law.δ + law.H_N)
            return law.δ * c, (c - 1) / h
        end
        c = law.H_N / (h * r + law.H_N)
        return r * c, ((1 - law.p) / h + law.p / (h - hf)) * c - 1 / h
    end
    return law.H_N / (h + law.H_N), -1 / (h + law.H_N)
end

# Add one axis's T² = ((u − a)₊/(d s(h)))², h = (u − zb + (a − zb)₊)/2, and its u-derivative.
@inline function _coulomb_axis(f, df, u, a, d, zb, μ, hf, law)
    u > a || return f, df
    h = (u - zb + max(a - zb, zero(a))) / 2
    q, dlq = _coulomb_ratio(h, hf, law)
    q > 0 || return oftype(f, Inf), df
    t = (u - a) / (d * μ * q)
    return f + t * t, df + 2t * (1 - (u - a) * dlq / 2) / (d * μ * q)
end

# φ(u) = T_a² + T_b² − 1 of the Coulomb update and its u-derivative.
@inline function _coulomb_phi(u, ha, ac, da, hb, bc, db, zb, μ, hf, law)
    f, df = -one(u), zero(u)
    ha && ((f, df) = _coulomb_axis(f, df, u, ac, da, zb, μ, hf, law))
    hb && ((f, df) = _coulomb_axis(f, df, u, bc, db, zb, μ, hf, law))
    return f, df
end

"""
    godunov_coulomb(a, da, b, db, zb, μ, H_f, law)

Godunov update of the plastic surface equation with the Coulomb yield law
([`CoulombYield`](@ref)): the surface `u = z_s` of a cell with bed `zb`, friction
coefficient `μ` and flotation thickness `H_f`, from the upwind neighbour surfaces `a`
(x, spacing `da`) and `b` (y, spacing `db`), either may be `Inf`. Only `law.p`,
`law.H_N` and `law.δ` are used (the cell's `μ` is an argument).

Each axis carries the slope at its edge-mean thickness `h_a = (H + (a − zb)₊)/2`,

    T_a(u) = (u − a)₊ / (da s(h_a)),    T_a² + T_b² = 1,

which is the equation of [`godunov_surface`](@ref) for `s = k/h`. On a marine cell
(`H_f > 0`) a neighbour below the cell's flotation surface is raised to it, so
`u ≥ zb + H_f`. For `p ≤ 1` and `δ > 0` the root is unique and the update nondecreasing
in `a` and `b` (a monotone scheme). The root is bracketed by the smaller neighbour and
`min(a + μ da, b + μ db)` and found by Newton safeguarded by bisection.
"""
@inline function godunov_coulomb(a, da, b, db, zb, μ, H_f, law)
    T = float(promote_type(typeof(a), typeof(da), typeof(b), typeof(db), typeof(zb),
                           typeof(μ), typeof(H_f)))
    ha, hb = a < Inf, b < Inf
    (ha || hb) || return T(Inf)
    ac = T(ha && H_f > 0 ? max(a, zb + H_f) : a)
    bc = T(hb && H_f > 0 ? max(b, zb + H_f) : b)
    lo = T(max(min(ac, bc), zb + H_f))
    μ > 0 || return lo
    hi = T(Inf)
    ha && (hi = min(hi, ac + μ * da))
    hb && (hi = min(hi, bc + μ * db))
    φ(u) = _coulomb_phi(u, ha, ac, da, hb, bc, db, zb, μ, H_f, law)
    f, _ = φ(lo)
    (hi <= lo || f >= 0) && return lo

    # Newton from the upper bracket; bisection when a step leaves (lo, hi) or does not halve
    # the step before last (rtsafe: the kink of s at the floor can make plain Newton cycle).
    u = hi
    f, df = φ(u)
    dxo = dxs = hi - lo
    for _ in 1:100
        f >= 0 ? (hi = u) : (lo = u)
        un = (lo + hi) / 2
        if df > 0
            un_newton = u - f / df
            lo < un_newton < hi && abs(2f) <= abs(dxo * df) && (un = un_newton)
        end
        dxo, dxs = dxs, abs(un - u)
        tol = 1e-12 * max(one(T), u - zb)
        if abs(un - u) <= tol || hi - lo <= tol
            u = un
            break
        end
        u = un
        f, df = φ(u)
    end
    # One Newton step at the root: no change of the value beyond the tolerance, but under
    # forward-mode AD the derivative becomes the implicit one, −(∂φ/∂θ)/(∂φ/∂u), whatever
    # path (bisection or Newton) reached the root.
    f, df = φ(u)
    return df > 0 && isfinite(f) ? u - f / df : u
end

# Kernel of the sweep: per-cell data (μ, H_f), see `_kernel` in solve.jl.
struct _Coulomb{L}
    law::L
end
@inline _kernel(k::_Coulomb, a, da, b, db, c::Tuple, zb) =
    godunov_coulomb(a, da, b, db, zb, c[1], c[2], k.law)

"""
    solve(z_b, law::CoulombYield, mask; dx, dy, z_ss=0.0, margin=:face, mode=:surface,
          params=PlasticParams(), max_sweeps=200, tol=1e-6)

Plastic reconstruction with the basal shear stress given by a Coulomb yield law,
`τ_b = μ N_eff(H)` (see [`CoulombYield`](@ref)), in one fast-sweeping solve with
[`godunov_coulomb`](@ref). Mode `:surface` only. Other inputs and the result `(z_s, H)` as
for the constant-`τ_b` [`solve`](@ref); [`coulomb_taub`](@ref) gives the `τ_b` field of the
result.
"""
function solve(z_b::AbstractMatrix, law::CoulombYield, mask::AbstractMatrix{Bool};
               dx::Real, dy::Real, z_ss = 0.0, margin = :face, mode::Symbol = :surface,
               params::PlasticParams = PlasticParams(), max_sweeps::Int = 200, tol = 1e-6)
    mode === :surface ||
        throw(ArgumentError("CoulombYield needs mode = :surface (the Coulomb law has a surface kernel only), got :$mode"))
    size(z_b) == size(mask) || throw(DimensionMismatch("z_b and mask must match"))
    μ = _as_field(law.μ, size(z_b), "μ")
    all(k -> !mask[k] || 0 <= μ[k] < Inf, eachindex(mask)) ||
        throw(ArgumentError("μ must be finite and ≥ 0 on the ice"))
    bc = _bc(z_b, _as_field(z_ss, size(z_b), "z_ss"), params, :surface)
    cells = map((m, z, zb) -> (m, z - zb), μ, bc, z_b)      # (μ, H_f) per cell
    u = _sweep(_Coulomb(law), cells, z_b, mask, bc, _margin_dfac(mask, margin), dx, dy;
               max_sweeps, tol)
    return _geometry(u, z_b, mask, :surface)
end

"""
    coulomb_taub(H, z_b, mask, law::CoulombYield; z_ss=0.0, params=PlasticParams())

Basal shear stress (Pa) of the Coulomb yield law at thickness `H`, `μ ρ_i g N_eff(H)/(ρ_i g)`
on the ice cells and 0 elsewhere, e.g. for the `H` of `solve(z_b, law, mask; ...)`.
"""
function coulomb_taub(H, z_b, mask, law::CoulombYield; z_ss = 0.0,
                      params::PlasticParams = PlasticParams())
    sz = size(z_b)
    μ = _as_field(law.μ, sz, "μ")
    zs = _as_field(z_ss, sz, "z_ss")
    ρg = params.ρ_i * params.g
    return [mask[k] ? μ[k] * ρg * _coulomb_neff(H[k], flotation_thickness(z_b[k], zs[k], params), law) :
                      zero(ρg * μ[k] * H[k]) for k in CartesianIndices(z_b)]
end
