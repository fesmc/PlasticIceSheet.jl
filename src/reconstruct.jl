# Feature (2): the SMB-driven reconstruction — the dual of `solve`. Where `solve` takes a
# yield stress `τ_b` and returns geometry, this takes a surface mass balance and returns
# the geometry (and velocity) consistent with it on the *same prescribed margin*.
#
# It is the plastic solver with `τ_b` no longer prescribed but *derived* each iteration:
#
#   1. route the SMB down the current surface          → required throughput `Q`
#   2. invert the DIVA flux law pointwise for `τ_d`     such that q = H·(u_b(τ_d)+u_def(τ_d,H))
#   3. sweep the plastic eikonal with that `τ_d` field  → updated thickness
#   4. under-relax and repeat.
#
# Because routing (`balance_flux`) is the exact inverse of the implied-SMB divergence
# (`smb_from_velocity`), a plastic solution fed its own closure-B SMB is a clean fixed
# point: step 1 recovers exactly `ū H`, step 2 returns exactly the original `τ_b`, and step
# 3 reproduces the geometry. The eikonal in step 3 also makes the thickness non-local, so
# the interior (where the local flux→thickness map is singular at divides) is set by
# integrating from the margin inward rather than cell-by-cell.

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
    diva_reconstruct(z_b, mask, smb, dx, dy; rheology, sliding, params = PlasticParams(),
                     H_init = nothing, n_outer = 300, relax = 0.3, tol = 0.5,
                     τ_hi = 5.0e5, max_sweeps = 400, sweep_tol = 1e-9)

Reconstruct a steady-state ice surface and thickness **driven by surface mass balance
`smb`** rather than by a prescribed yield stress, on the fixed grounded `mask`. Internally
iterates: route `smb` down the surface for the required flux, invert the DIVA flow law
(`rheology` + `sliding`) for the driving stress that carries it, and sweep the plastic
eikonal — see the file header. `smb` may be a scalar or a field.

Inputs mirror [`solve`](@ref) (`z_b`, `mask`, `dx`, `dy`, `params`). Reconstruction
controls:
- `H_init`    : initial thickness (defaults to a plastic solve at a nominal uniform `τ_b`)
- `n_outer`   : maximum outer iterations
- `relax`     : under-relaxation factor in (0,1]; lower is more robust, slower
- `tol`       : convergence threshold on `max|ΔH|` (m)
- `τ_hi`      : upper bracket (Pa) for the driving-stress inversion (flux-limited above it)
- `max_sweeps`, `sweep_tol` : inner eikonal controls

Returns `(z_s, H)` like [`solve`](@ref). The plastic solution is a fixed point when `smb`
is its own [`smb_from_velocity`](@ref); for a general `smb` the result is the plastic-style
geometry whose DIVA flux balances that climate. Bed relief is handled to first order (the
inner sweep runs in `:flat` mode).

Like the `:full` plastic mode, the outer "route, re-invert, re-sweep" map is *not*
contractive, so it must be under-relaxed: the default `relax = 0.05` converges the test
configurations, but stiffer or finer grids need a smaller `relax` and more `n_outer`
(the trade is robustness vs. iterations).
"""
function diva_reconstruct(z_b::AbstractMatrix, mask::AbstractMatrix{Bool}, smb, dx::Real, dy::Real;
                          rheology::GlenRheology, sliding::SlidingLaw,
                          params::PlasticParams = PlasticParams(), H_init = nothing,
                          n_outer::Int = 1000, relax = 0.05, tol = 1.0, τ_hi = 5.0e5,
                          max_sweeps::Int = 400, sweep_tol = 1e-9)
    size(z_b) == size(mask) || throw(DimensionMismatch("z_b and mask must match"))
    p = params
    smbf = smb isa Number ? fill(float(smb), size(z_b)) : smb
    size(smbf) == size(z_b) || throw(DimensionMismatch("smb and z_b must match"))

    H = H_init === nothing ?
        last(solve(z_b, 5.0e4, mask; dx, dy, mode = :flat, max_sweeps, tol = sweep_tol)) :
        copy(float.(H_init))
    bc = _bc_flat.(z_b, Ref(p))
    w = _flux_width(dx, dy)
    τ_d = similar(H)
    z_s = @. ifelse(mask, H + z_b, z_b)

    for _ in 1:n_outer
        z_s = @. ifelse(mask, H + z_b, z_b)
        Q = balance_flux(z_s, mask, smbf, dx, dy)
        @inbounds for k in eachindex(H)
            τ_d[k] = mask[k] ?
                _invert_driving_stress(Q[k] / w, H[k], sliding, rheology, τ_hi) : zero(eltype(H))
        end
        wnew = _solve_w(τ_d, z_b, mask, bc, dx, dy, p, :flat;
                        max_sweeps, tol = sweep_tol, n_outer = 1, outer_tol = 1.0)
        Hnew = @. ifelse(mask, sqrt(max(wnew, zero(eltype(wnew)))), zero(eltype(wnew)))
        Δ = maximum(abs.(Hnew .- H))
        @. H = (1 - relax) * H + relax * Hnew
        Δ < tol && break
    end

    z_s = @. ifelse(mask, H + z_b, z_b)
    return z_s, H
end
