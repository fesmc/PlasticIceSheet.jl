# The SMB-driven reconstruction — the dual of `solve`. Where `solve` takes a yield stress
# `τ_b` and returns geometry, this takes a surface mass balance and returns the geometry
# consistent with it on the *same prescribed margin*. Two methods:
#
# :steady (default) — steady mass conservation ∇·q = smb with the membrane-free DIVA flux,
#   solved directly (pseudo-transient continuation + semismooth Newton; `steady.jl`).
#
# :routing — the plastic solver with `τ_b` derived each iteration:
#   1. route the SMB down the current surface          → required throughput `Q`
#   2. invert the DIVA flux law pointwise for `τ_d`     such that q = H·(u_b(τ_d)+u_def(τ_d,H))
#   3. sweep the plastic equation with that `τ_d` field → updated thickness
#   4. under-relax and repeat.
#   A plastic solution fed its own closure-B SMB is an exact fixed point (routing is the
#   exact inverse of `smb_from_velocity`), but the map is only conditionally stable: its
#   Jacobian has eigenvalues λ ~ −H/(slope·Δx), so it converges only for
#   relax < 2/(1 − λ), small and shrinking with resolution. Kept for comparison.

# Maximum drag a sliding law can carry (the Coulomb cap; ∞ for the unbounded power laws).
_drag_cap(::SlidingLaw) = Inf
_drag_cap(l::RegularizedCoulomb) = l.C

# Solve q = H·(u_b(τ) + u_def(τ,H)) for the driving stress τ ≥ 0 (monotone increasing in
# τ), by bisection. Saturates at the bracket top where even that cannot carry `q` — a
# flux-limited margin, or a `RegularizedCoulomb` bed driven to its cap.
function _invert_driving_stress(q, H, sliding::SlidingLaw, rheology::GlenRheology, τ_hi)
    T = promote_type(typeof(q), typeof(H), typeof(τ_hi))
    (q <= 0 || H <= 0) && return zero(T)
    cap = _drag_cap(sliding)
    hi = isfinite(cap) ? min(T(τ_hi), T(0.999 * cap)) : T(τ_hi)
    fl(τ) = H * (basal_velocity(sliding, τ) + deformational_velocity(rheology, τ, H))
    fl(hi) <= q && return hi
    lo = zero(T)
    for _ in 1:60
        mid = (lo + hi) / 2
        fl(mid) < q ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    diva_reconstruct(z_b, mask, smb, dx, dy; rheology, sliding, method = :steady,
                     z_ss = 0.0, margin = :face, params = PlasticParams(),
                     H_init = nothing, τ_init = 5.0e4,
                     # method = :steady
                     res_tol = 1e-6, n_steady = 500, dt0 = 1.0, dt_max = 1e8, verbose = false,
                     # method = :routing
                     inner_mode = :surface, n_outer = 1000, relax = 0.05, tol = 1.0,
                     τ_hi = 5.0e5, max_sweeps = 400, sweep_tol = 1e-9)

Reconstruct a steady-state ice surface and thickness **driven by surface mass balance
`smb`** (m ice yr⁻¹, scalar or field) rather than by a prescribed yield stress, on the
fixed grounded `mask`, with the DIVA flux law of `rheology` + `sliding`. Inputs mirror
[`solve`](@ref) (`z_b`, `mask`, `dx`, `dy`, `z_ss`, `margin`, `params`).

`method = :steady` (default) solves steady mass conservation directly,

    ∇·q = smb,  q = −H (u_b(τ) + 2A/(n+2) τⁿ H) ∇z_s/|∇z_s|,  τ = ρ_i g H |∇z_s|,  H ≥ 0,

by pseudo-transient continuation with a semismooth Newton step per pseudo-time step
(cells where ablation exceeds the inflow end ice-free). It converges when the
complementarity residual `max|min(H, ∇·q − smb)| < res_tol` (m yr⁻¹), within `n_steady`
pseudo-time steps from `dt0` (grown up to `dt_max`, yr). The geometry `H` is a root under
its own [`smb_flux_divergence`](@ref).

`method = :routing` iterates route → invert for `τ_d` → plastic sweep (`inner_mode`) →
under-relax (`relax`) until `max|ΔH| < tol` (m) or `n_outer`; `τ_hi` bounds the
driving-stress inversion. It is only conditionally stable (needs `relax` ≲ slope·Δx/H) and
is kept for comparison.

The start is `H_init`, or a plastic solve (`:surface`, or `inner_mode` for routing) at the
uniform `τ_b = τ_init`. Returns `(z_s, H)` like [`solve`](@ref); warns if the method did
not converge.
"""
function diva_reconstruct(z_b::AbstractMatrix, mask::AbstractMatrix{Bool}, smb, dx::Real, dy::Real;
                          rheology::GlenRheology, sliding::SlidingLaw, method::Symbol = :steady,
                          z_ss = 0.0, margin = :face, params::PlasticParams = PlasticParams(),
                          H_init = nothing, τ_init = 5.0e4,
                          res_tol = 1e-6, n_steady::Int = 500, dt0 = 1.0, dt_max = 1e8,
                          verbose::Bool = false,
                          inner_mode::Symbol = :surface, n_outer::Int = 1000, relax = 0.05,
                          tol = 1.0, τ_hi = 5.0e5, max_sweeps::Int = 400, sweep_tol = 1e-9)
    size(z_b) == size(mask) || throw(DimensionMismatch("z_b and mask must match"))
    method in (:steady, :routing) || throw(ArgumentError("method must be :steady or :routing, got :$method"))
    smbf = _as_field(smb, size(z_b), "smb")
    _check_law(sliding, size(z_b))
    start_mode = method === :steady ? :surface : inner_mode
    H = H_init === nothing ?
        Matrix{Float64}(last(solve(z_b, τ_init, mask; dx, dy, z_ss, margin, params,
                                   mode = start_mode, max_sweeps, tol = sweep_tol))) :
        Matrix{Float64}(_as_field(H_init, size(z_b), "H_init"))

    if method === :steady
        P = _SteadyProblem(z_b, mask, dx, dy; rheology, sliding, z_ss, margin, params, τ_cap = nothing)
        n, res, ok = _steady_solve!(H, smbf, P; res_tol, n_steady, dt0, dt_max, verbose)
        ok || @warn "diva_reconstruct (:steady): residual $(round(res; sigdigits = 3)) m/yr above res_tol after $n steps"
    else
        ok = _reconstruct_routing!(H, z_b, mask, smbf, dx, dy, rheology, sliding, z_ss, margin, params;
                                   inner_mode, n_outer, relax, tol, τ_hi, max_sweeps, sweep_tol)
        ok || @warn "diva_reconstruct (:routing): max|ΔH| above tol after $n_outer iterations"
    end
    z_s = @. ifelse(mask, H + z_b, z_b)
    return z_s, H
end

# The routing iteration, from H (modified in place); returns whether it converged.
function _reconstruct_routing!(H, z_b, mask, smbf, dx, dy, rheology, sliding, z_ss, margin, p;
                               inner_mode, n_outer, relax, tol, τ_hi, max_sweeps, sweep_tol)
    bc = _bc(z_b, _as_field(z_ss, size(z_b), "z_ss"), p, inner_mode)
    dfac = _margin_dfac(mask, margin)
    w = _flux_width(dx, dy)
    τ_d = similar(H)
    for _ in 1:n_outer
        z_s = @. ifelse(mask, H + z_b, z_b)
        Q = balance_flux(z_s, mask, smbf, dx, dy)
        @inbounds for k in eachindex(H)
            τ_d[k] = mask[k] ?
                _invert_driving_stress(Q[k] / w, H[k], _law_at(sliding, k), rheology, τ_hi) : 0.0
        end
        u = _solve_u(τ_d, z_b, mask, bc, dfac, dx, dy, p, inner_mode;
                     max_sweeps, tol = sweep_tol, n_outer = 1, outer_tol = 1.0)
        Hnew = last(_geometry(u, z_b, mask, inner_mode))
        Δ = maximum(abs.(Hnew .- H))
        @. H = (1 - relax) * H + relax * Hnew
        Δ < tol && return true
    end
    return false
end
