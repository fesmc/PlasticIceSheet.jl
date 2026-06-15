# Fast-sweeping eikonal solver and the top-level plastic reconstruction.
#
# The grounded-ice domain is given by a boolean `mask` (true = grounded ice). The
# margin is the interface between mask=true and mask=false cells; the Dirichlet
# boundary value is carried in the non-ice cells themselves, so neighbour fetches
# only ever read the working array. Cells outside the array are treated as ice-free
# land (clamp-to-edge); pad the mask with at least one ice-free cell for a clean
# margin everywhere.

# --- core fast sweeping -----------------------------------------------------------

# One Gauss–Seidel sweep in a given (xdir, ydir) ordering. Updates only ice cells,
# keeps the monotone min, and returns the largest decrease seen this sweep.
function _sweep_once!(u, F, mask, dx, dy, xdir::Int, ydir::Int)
    nx, ny = size(u)
    change = zero(eltype(u))
    irange = xdir > 0 ? (1:nx) : (nx:-1:1)
    jrange = ydir > 0 ? (1:ny) : (ny:-1:1)
    @inbounds for j in jrange, i in irange
        mask[i, j] || continue
        im = i > 1  ? i - 1 : 1
        ip = i < nx ? i + 1 : nx
        jm = j > 1  ? j - 1 : 1
        jp = j < ny ? j + 1 : ny
        a = min(u[im, j], u[ip, j])
        b = min(u[i, jm], u[i, jp])
        unew = godunov_eikonal(a, dx, b, dy, F[i, j])
        if unew < u[i, j]
            change = max(change, u[i, j] - unew)
            u[i, j] = unew
        end
    end
    return change
end

"""
    sweep_eikonal(F, mask, bc, dx, dy; max_sweeps=200, tol=1e-6)

Solve `|∇u| = F` on the cells where `mask` is true, with Dirichlet values supplied by
`bc` on the ice-free cells. `F`, `bc` are per-cell arrays; `dx`, `dy` the spacings.
Returns the field `u` (ice-free cells retain their `bc` value).

The four diagonal sweep orderings are alternated so the scheme converges in O(N) for
any characteristic direction. A fixed `max_sweeps` makes the routine reproducible
under AD; `tol` provides an early exit for the forward solve.
"""
function sweep_eikonal(F, mask, bc, dx, dy; max_sweeps::Int = 200, tol = 1e-6)
    T = promote_type(eltype(F), eltype(bc), typeof(dx), typeof(dy))
    u = Matrix{T}(undef, size(mask)...)
    @inbounds for k in eachindex(u)
        u[k] = mask[k] ? T(Inf) : T(bc[k])
    end
    dirs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    for s in 1:max_sweeps
        xdir, ydir = dirs[mod1(s, 4)]
        change = _sweep_once!(u, F, mask, dx, dy, xdir, ydir)
        # require a full cycle of four orderings before trusting the early exit
        if s >= 4 && isfinite(change) && change < tol
            break
        end
    end
    return u
end

# --- boundary values --------------------------------------------------------------

# Both modes solve for u = H² (which is non-singular at the margin, unlike the surface
# E whose driving slope τ/(ρ_i g H) blows up as H→0). The margin Dirichlet value is the
# squared flotation thickness (0 on land).
_bc_flat(bed, p) = let h = flotation_thickness(bed, p); h * h end

# Central-difference gradient of a field with clamp-to-edge (one-sided at array edges).
function _grad(f, dx, dy)
    nx, ny = size(f)
    fx = similar(f)
    fy = similar(f)
    @inbounds for j in 1:ny, i in 1:nx
        im = max(i - 1, 1); ip = min(i + 1, nx)
        jm = max(j - 1, 1); jp = min(j + 1, ny)
        fx[i, j] = (f[ip, j] - f[im, j]) / ((ip - im) * dx)
        fy[i, j] = (f[i, jp] - f[i, jm]) / ((jp - jm) * dy)
    end
    return fx, fy
end

# --- driving stress, forward solve, and residual (shared by solve and the AD ext) --

# Per-cell right-hand side of the w = H² eikonal, |∇w| = F.
#   :flat → F = G = 2τ/(ρ_i g)        (independent of w)
#   :hj   → F = √(G² − 4√w(∇B·∇w) − 4w|∇B|²)   (depends on w through the gradient term)
# Both the forward sweep and the implicit-diff residual call this, so they cannot
# drift apart.
function _driving_rhs(w, τf, bed, dx, dy, p::PlasticParams, mode::Symbol)
    G = @. 2 * τf / (p.ρ_i * p.g)
    mode === :flat && return G
    mode === :hj || throw(ArgumentError("mode must be :hj or :flat, got $mode"))
    Bx, By = _grad(bed, dx, dy)
    wx, wy = _grad(w, dx, dy)
    # Floor at H_min² so √w stays differentiable at w = 0 (ice-free cells) — sqrt(0)
    # has an infinite derivative that otherwise poisons the implicit-diff VJP.
    sqrtw = @. sqrt(max(w, p.H_min^2))
    gradB2 = @. Bx * Bx + By * By
    dotBw = @. Bx * wx + By * wy
    # Floor the squared RHS to a small positive fraction of G² so it stays real (a
    # near-flat surface where the bed is too steep for plastic ice to thicken).
    return @. sqrt(max(G * G - 4 * sqrtw * dotBw - 4 * w * gradB2, (1e-3 * G)^2))
end

# Forward solve for w = H². Warm-starts the `:hj` outer fixed point from `:flat`.
#
# The `:hj` map "freeze Geff(w), re-solve the eikonal" is NOT contractive for steep beds
# (Geff carries ∇w, so the undamped iteration oscillates). Under-relaxation by `relax`
# converges it to the same fixed point — i.e. to a vanishing `godunov_residual`, which is
# also what makes the implicit-diff gradient exact. Smaller `relax` is more robust on
# steep relief but needs more iterations.
function _solve_w(τf, bed, mask, bc, dx, dy, p::PlasticParams, mode::Symbol;
                  max_sweeps::Int, tol, n_outer::Int, outer_tol, relax = 0.5)
    G = _driving_rhs(τf, τf, bed, dx, dy, p, :flat)   # flat RHS (w arg unused for :flat)
    w = sweep_eikonal(G, mask, bc, dx, dy; max_sweeps, tol)
    if mode === :hj
        for _ in 1:n_outer
            F = _driving_rhs(w, τf, bed, dx, dy, p, :hj)
            wc = sweep_eikonal(F, mask, bc, dx, dy; max_sweeps, tol)
            wnew = @. (1 - relax) * w + relax * wc
            conv = maximum(abs.(sqrt.(max.(wnew, 0)) .- sqrt.(max.(w, 0))))
            w = wnew
            conv < outer_tol && break
        end
    end
    return w
end

# Godunov update at a single cell from the current field (clamp-to-edge, upwind min).
@inline function _godunov_at(w, F, i::Int, j::Int, dx, dy)
    nx, ny = size(w)
    im = i > 1  ? i - 1 : 1
    ip = i < nx ? i + 1 : nx
    jm = j > 1  ? j - 1 : 1
    jp = j < ny ? j + 1 : ny
    a = min(w[im, j], w[ip, j])
    b = min(w[i, jm], w[i, jp])
    return godunov_eikonal(a, dx, b, dy, F[i, j])
end

"""
    godunov_residual(w, τf, bed, mask, bc, dx, dy, params, mode)

Optimality conditions satisfied (≈ 0) by the converged field `w = H²`, returned with the
same shape as `w`: `w − (Godunov update)` on ice cells, `w − bc` on ice-free cells. This
is the residual the implicit-differentiation extension differentiates; it shares
`_driving_rhs` and `_godunov_at` with the forward solver so the two cannot diverge.
"""
function godunov_residual(w, τf, bed, mask, bc, dx, dy, p::PlasticParams, mode::Symbol)
    F = _driving_rhs(w, τf, bed, dx, dy, p, mode)
    nx, ny = size(w)
    # Promote so the residual carries duals whether τ or w is the differentiation seed.
    T = promote_type(eltype(w), eltype(F), eltype(bc))
    r = Matrix{T}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        r[i, j] = mask[i, j] ? w[i, j] - _godunov_at(w, F, i, j, dx, dy) : w[i, j] - bc[i, j]
    end
    return r
end

# --- top-level solve --------------------------------------------------------------

"""
    solve(bed, τ, mask; dx, dy, mode=:hj, params=PlasticParams(),
          max_sweeps=200, tol=1e-6, n_outer=25, outer_tol=1e-3)

Reconstruct a perfectly-plastic, steady-state ice sheet.

Inputs (all matrices share the same `(nx, ny)` shape unless noted):
- `bed`  : bed elevation `B` (m)
- `τ`    : basal shear stress (Pa); a scalar is broadcast to a uniform field
- `mask` : `true` where grounded ice exists; the margin is the mask boundary
- `dx`, `dy` : grid spacings (m)

Options:
- `mode`  : `:hj` (default, full Hamilton–Jacobi over arbitrary bed relief) or
            `:flat` (flat-bed eikonal, `|∇H| ≈ |∇E|`; exact when bed relief ≪ H)
- `params`: `PlasticParams`
- `max_sweeps`, `tol`        : inner eikonal fast-sweeping controls
- `n_outer`, `outer_tol`, `relax` : `:hj` damped fixed-point controls (ignored for
  `:flat`). `relax` ∈ (0,1] is the under-relaxation factor; lower is more robust on steep
  beds but slower. The undamped map (`relax=1`) can oscillate over strong bed relief.

Returns `(E, H)` — ice surface elevation and thickness (m). Ice-free cells have
`H = 0`, `E = bed`.

The whole computation is differentiable w.r.t. `τ`; see `docs/PLAN.md`.
"""
function solve(bed::AbstractMatrix, τ, mask::AbstractMatrix{Bool};
               dx::Real, dy::Real, mode::Symbol = :hj,
               params::PlasticParams = PlasticParams(),
               max_sweeps::Int = 200, tol = 1e-6,
               n_outer::Int = 100, outer_tol = 1e-3, relax = 0.5)

    size(bed) == size(mask) || throw(DimensionMismatch("bed and mask must match"))
    p = params
    τf = τ isa Number ? fill(float(τ), size(bed)) : τ
    size(τf) == size(bed) || throw(DimensionMismatch("τ and bed must match"))

    bc = _bc_flat.(bed, Ref(p))
    w = _solve_w(τf, bed, mask, bc, dx, dy, p, mode; max_sweeps, tol, n_outer, outer_tol, relax)

    H = @. ifelse(mask, sqrt(max(w, zero(eltype(w)))), zero(eltype(w)))
    E = @. ifelse(mask, H + bed, bed)
    return E, H
end

# Reverse-mode differentiable thickness solve w.r.t. a full `τ` field. The real method
# is provided by the package extension loaded with `ImplicitDifferentiation` (and an
# AD backend, e.g. ForwardDiff). Without it, this fallback explains what to load.
function differentiable_thickness end
