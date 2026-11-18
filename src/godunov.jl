# Pure, AD-agnostic Godunov upwind update for the static eikonal equation
#
#     |∇u| = F      (F ≥ 0)
#
# discretised on a regular grid with spacings `da`, `db` in the two axes. `a` and `b`
# are the smaller of the two neighbour values along each axis (the upwind values).
# This is the building block of the fast-sweeping solver; it contains all the
# numerics and is deliberately free of any mutation, indexing, or AD-specific code so
# that ForwardDiff / Enzyme can differentiate straight through it.

"""
    godunov_eikonal(a, da, b, db, F)

Smallest `u` satisfying the Godunov discretisation of `|∇u| = F`:

    ((u - a)₊ / da)² + ((u - b)₊ / db)² = F²

where `(·)₊ = max(·, 0)`. `a`, `b` are the upwind (smaller) neighbour values along the
two axes with grid spacings `da`, `db`. Either neighbour may be `Inf` (no upwind
information yet), in which case the update degenerates to the one-sided form.
"""
@inline function godunov_eikonal(a, da, b, db, F)
    # One-sided fall-backs when a neighbour carries no information.
    if !isfinite(a) && !isfinite(b)
        return oftype(F * da, Inf)
    elseif !isfinite(a)
        return b + F * db
    elseif !isfinite(b)
        return a + F * da
    end

    # Two-sided update: solve the quadratic
    #   A u² + B u + C = 0,  A = 1/da² + 1/db², for the case where both axes are upwind.
    ida2 = inv(da * da)
    idb2 = inv(db * db)
    A = ida2 + idb2
    B = -2 * (a * ida2 + b * idb2)
    C = a * a * ida2 + b * b * idb2 - F * F
    disc = B * B - 4 * A * C
    if disc >= 0
        u = (-B + sqrt(disc)) / (2 * A)
        # Accept only if both neighbours are genuinely upwind (u ≥ a and u ≥ b),
        # i.e. both (u - ·)₊ terms were active.
        if u >= a && u >= b
            return u
        end
    end

    # Otherwise the smaller single-axis update governs.
    return min(a + F * da, b + F * db)
end

"""
    godunov_surface(a, da, b, db, zb, k)

Godunov update of the plastic surface equation `|∇z_s| = k / H`, `H = z_s − zb`,
`k = τ_b/(ρ_i g)`, for the surface `u = z_s` of a cell with bed `zb`, from the upwind
(smaller) neighbour surfaces `a` (spacing `da`) and `b` (spacing `db`); `Inf` means no
upwind information.

Each axis contributes the difference `(u − a)₊` times the edge-mean thickness
`(H + (a − zb)₊)/2`, with the neighbour thickness measured from this cell's bed:

    P_a(H) = (H − α)₊ (H + α₊) / 2,   α = a − zb,
    (P_a/da)² + (P_b/db)² = k².

`P_a` increases with `H` (the Hamiltonian is proper) and does not increase with `a`
(monotone scheme), so fast sweeping applies directly, with no outer iteration. The edge
mean integrates the margin singularity exactly on a flat bed (the Nye profile, as the
`w = H²` eikonal); a cell-centred `k/H` would put the first ice cell at `√(k dx)` instead
of `√(2k dx)`. Where both neighbours stand above `zb` the update is the `w`-eikonal with
`w_a = α²`, `F = 2k`, in closed form (for the increment over the smaller `w_a`, free of
cancellation); otherwise the two-sided root is found by Newton from the smaller
one-sided root (`P²` is convex and increasing there, so Newton decreases monotonically).
"""
@inline function godunov_surface(a, da, b, db, zb, k)
    fa, fb = isfinite(a), isfinite(b)
    T = float(promote_type(typeof(a), typeof(da), typeof(b), typeof(db), typeof(zb), typeof(k)))
    (fa || fb) || return T(Inf)
    k > 0 || return T(max(zb, min(a, b)))

    al = fa ? a - zb : zero(T)
    be = fb ? b - zb : zero(T)

    # Both neighbour surfaces at or above this bed: the w-eikonal (w = H², F = 2k).
    if al >= 0 && be >= 0
        !fa && return T(zb + sqrt(be * be + 2 * k * db))
        !fb && return T(zb + sqrt(al * al + 2 * k * da))
        wa, wb = al * al, be * be
        wm, dm, c, dother = wa <= wb ? (wa, da, wb - wa, db) : (wb, db, wa - wb, da)
        dw = 2 * k * dm                                   # one-sided in the smaller neighbour
        if dw > c                                         # both axes upwind: root in (c, dw)
            qa = 1 / (dm * dm) + 1 / (dother * dother)
            dw = (c / (dother * dother) +
                  sqrt(max(qa * (2k)^2 - (c / (dm * dother))^2, zero(c)))) / qa
        end
        return T(zb + sqrt(wm + dw))
    end

    # One-sided roots P(H) = k d; the smaller one bounds the two-sided root.
    hA = fa ? _surface_one_sided(al, k * da) : T(Inf)
    hB = fb ? _surface_one_sided(be, k * db) : T(Inf)
    h0, other = hA <= hB ? (hA, b) : (hB, a)
    (!isfinite(other) || zb + h0 <= other) && return T(zb + h0)

    # Two-sided: Newton on φ(H) = (P_a/da)² + (P_b/db)² − k² from H = h0.
    h = h0
    alp, bep = max(al, zero(al)), max(be, zero(be))
    for _ in 1:60
        pa = (h - al) * (h + alp) / 2
        pb = (h - be) * (h + bep) / 2
        dpa = (2h + alp - al) / 2
        dpb = (2h + bep - be) / 2
        φ = (pa / da)^2 + (pb / db)^2 - k * k
        dφ = 2 * (pa * dpa / (da * da) + pb * dpb / (db * db))
        (φ > 0 && dφ > 0) || break
        dh = φ / dφ
        h -= dh
        dh <= 1e-14 * h && break
    end
    return T(zb + h)
end

# Root H > 0 of (H − α)(H + α₊)/2 = k d (written without cancellation for α < 0).
@inline _surface_one_sided(al, kd) =
    al >= 0 ? sqrt(al * al + 2kd) : 4kd / (sqrt(al * al + 8kd) - al)
