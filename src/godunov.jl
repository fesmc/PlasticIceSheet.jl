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
