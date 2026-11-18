# Fast-sweeping solver (eikonal and plastic-surface updates) and the top-level plastic
# reconstruction.
#
# The grounded-ice domain is given by a boolean `mask` (true = grounded ice). The
# margin is the interface between mask=true and mask=false cells; the Dirichlet
# boundary value is carried in the non-ice cells themselves, so neighbour fetches
# only ever read the working array. Cells outside the array are treated as ice-free
# land (clamp-to-edge); pad the mask with at least one ice-free cell for a clean
# margin everywhere.

# --- margin position --------------------------------------------------------------

# Floor of the spacing factor to an ice-free neighbour, (1 − margin_frac) ≥ MARGIN_DMIN:
# keeps the update well conditioned as the margin approaches an ice-cell centre.
const MARGIN_DMIN = 0.01

# Spacing factor to each cell as a neighbour: 1 on ice cells; on ice-free cells
# max(1 − f, MARGIN_DMIN), where the margin lies f·Δ from the ice-free cell centre:
# f = 0.5 for `:face` (margin on the cell face), 0 for `:centre` (on the ice-free cell
# centre), or a per-cell field `margin_frac` ∈ [0, 1) on the ice-free cells.
function _margin_dfac(mask, margin)
    if margin isa Symbol
        f0 = margin === :face   ? 0.5 :
             margin === :centre ? 0.0 :
             throw(ArgumentError("margin must be :face, :centre or a margin_frac field, got :$margin"))
        return [m ? 1.0 : max(1 - f0, MARGIN_DMIN) for m in mask]
    end
    size(margin) == size(mask) || throw(DimensionMismatch("margin_frac and mask must match"))
    all(k -> mask[k] || 0 <= margin[k] < 1, eachindex(mask)) ||
        throw(ArgumentError("margin_frac must be in [0, 1) on ice-free cells"))
    return [mask[k] ? 1.0 : max(1 - float(margin[k]), MARGIN_DMIN) for k in eachindex(IndexCartesian(), mask)]
end

"""
    margin_frac_from_sdist(sdist, mask, dx, dy)

Margin position from a signed distance to the margin, `sdist` (m; positive in the ice,
negative outside, at the cell centres), for the `margin` keyword of [`solve`](@ref): on
an ice-free cell with an ice neighbour, `−sdist/h` clamped to `[0, 1 − MARGIN_DMIN]`,
`h = √(dx dy)`; 0 elsewhere. The value is isotropic: exact for margins along a grid axis.
"""
function margin_frac_from_sdist(sdist, mask, dx, dy)
    size(sdist) == size(mask) || throw(DimensionMismatch("sdist and mask must match"))
    nx, ny = size(mask)
    h = sqrt(dx * dy)
    frac = zeros(nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        mask[i, j] && continue
        next_to_ice = mask[max(i - 1, 1), j] || mask[min(i + 1, nx), j] ||
                      mask[i, max(j - 1, 1)] || mask[i, min(j + 1, ny)]
        next_to_ice && (frac[i, j] = clamp(-sdist[i, j] / h, 0.0, 1 - MARGIN_DMIN))
    end
    return frac
end

# --- Godunov update and fast sweeping ----------------------------------------------

# The local update solved at each cell: the eikonal |∇u| = F (`_Eikonal`, u = H²), or the
# plastic surface equation |∇z_s| = K/(z_s − z_b) (`_Surface`, u = z_s, K = τ_b/(ρ_i g)).
struct _Eikonal end
struct _Surface end
@inline _kernel(::_Eikonal, a, da, b, db, F, zb) = godunov_eikonal(a, da, b, db, F)
@inline _kernel(::_Surface, a, da, b, db, K, zb) = godunov_surface(a, da, b, db, zb, K)

# Godunov update at ice cell (i, j) from its four neighbours (clamp-to-edge). The spacing
# to a neighbour is Δ·dfac: < Δ for an ice-free neighbour whose margin lies between the
# centres. With equal factors on both sides of an axis the upwind value is the smaller
# neighbour; otherwise the axis term is max over the sides of (u − a_s)₊/d_s, i.e. the
# update is the minimum over the side combinations (each candidate is monotone, hence so
# is the minimum).
@inline function _update_at(kern, u, Fij, zbij, dfac, i::Int, j::Int, dx, dy)
    nx, ny = size(u)
    im = i > 1  ? i - 1 : 1
    ip = i < nx ? i + 1 : nx
    jm = j > 1  ? j - 1 : 1
    jp = j < ny ? j + 1 : ny
    xsame = dfac[im, j] == dfac[ip, j]
    ysame = dfac[i, jm] == dfac[i, jp]
    a1 = xsame ? min(u[im, j], u[ip, j]) : u[im, j]
    b1 = ysame ? min(u[i, jm], u[i, jp]) : u[i, jm]
    da1 = dx * dfac[im, j]
    db1 = dy * dfac[i, jm]
    unew = _kernel(kern, a1, da1, b1, db1, Fij, zbij)
    xsame && ysame && return unew
    a2, da2 = u[ip, j], dx * dfac[ip, j]
    b2, db2 = u[i, jp], dy * dfac[i, jp]
    xsame || (unew = min(unew, _kernel(kern, a2, da2, b1, db1, Fij, zbij)))
    ysame || (unew = min(unew, _kernel(kern, a1, da1, b2, db2, Fij, zbij)))
    xsame || ysame || (unew = min(unew, _kernel(kern, a2, da2, b2, db2, Fij, zbij)))
    return unew
end

# One Gauss–Seidel sweep in a given (xdir, ydir) ordering. Updates only ice cells,
# keeps the monotone min, and returns the largest decrease seen this sweep.
function _sweep_once!(kern, u, F, zb, mask, dfac, dx, dy, xdir::Int, ydir::Int)
    nx, ny = size(u)
    change = zero(eltype(u))
    irange = xdir > 0 ? (1:nx) : (nx:-1:1)
    jrange = ydir > 0 ? (1:ny) : (ny:-1:1)
    @inbounds for j in jrange, i in irange
        mask[i, j] || continue
        unew = _update_at(kern, u, F[i, j], zb === nothing ? 0.0 : zb[i, j], dfac, i, j, dx, dy)
        if unew < u[i, j]
            change = max(change, u[i, j] - unew)
            u[i, j] = unew
        end
    end
    return change
end

# Fast sweeping for the local update `kern`, from Inf on the ice cells and `bc` elsewhere.
function _sweep(kern, F, zb, mask, bc, dfac, dx, dy; max_sweeps::Int, tol)
    T = promote_type(eltype(F), eltype(bc), typeof(dx), typeof(dy))
    u = Matrix{T}(undef, size(mask)...)
    @inbounds for k in eachindex(u)
        u[k] = mask[k] ? T(Inf) : T(bc[k])
    end
    dirs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    for s in 1:max_sweeps
        xdir, ydir = dirs[mod1(s, 4)]
        change = _sweep_once!(kern, u, F, zb, mask, dfac, dx, dy, xdir, ydir)
        # require a full cycle of four orderings before trusting the early exit
        if s >= 4 && isfinite(change) && change < tol
            break
        end
    end
    return u
end

"""
    sweep_eikonal(F, mask, bc, dx, dy; margin=:face, max_sweeps=200, tol=1e-6)

Solve `|∇u| = F` on the cells where `mask` is true, with Dirichlet values supplied by
`bc` on the ice-free cells, placed at the margin position `margin` (see [`solve`](@ref)).
`F`, `bc` are per-cell arrays; `dx`, `dy` the spacings. Returns the field `u` (ice-free
cells retain their `bc` value).

The four diagonal sweep orderings are alternated so the scheme converges in O(N) for
any characteristic direction. A fixed `max_sweeps` makes the routine reproducible
under AD; `tol` provides an early exit for the forward solve.
"""
sweep_eikonal(F, mask, bc, dx, dy; margin = :face, max_sweeps::Int = 200, tol = 1e-6) =
    _sweep(_Eikonal(), F, nothing, mask, bc, _margin_dfac(mask, margin), dx, dy; max_sweeps, tol)

# --- boundary values --------------------------------------------------------------

# Working variable u and its margin (Dirichlet) value on the ice-free cells:
#   :flat, :full → u = w = H² (non-singular at the margin, unlike the driving slope
#                  τ_b/(ρ_i g H) of the surface equation), bc = H_flt²
#   :surface     → u = z_s, bc = z_b + H_flt (the edge-mean update integrates the margin
#                  singularity, see `godunov_surface`)
# H_flt is the flotation thickness (0 on land); `z_ss` is a scalar or a field.
function _bc(z_b, z_ss, p, mode::Symbol)
    H_flt = flotation_thickness.(z_b, z_ss, Ref(p))
    mode === :surface && return z_b .+ H_flt
    mode in (:flat, :full) || throw(ArgumentError("mode must be :surface, :flat or :full, got :$mode"))
    return H_flt .^ 2
end

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
#   :flat → F = G = 2τ_b/(ρ_i g)        (independent of w)
#   :full   → F = √(G² − 4√w(∇z_b·∇w) − 4w|∇z_b|²)   (depends on w through the gradient term)
# Both the forward sweep and the implicit-diff residual call this, so they cannot
# drift apart.
function _driving_rhs(w, τ_bf, z_b, dx, dy, p::PlasticParams, mode::Symbol)
    G = @. 2 * τ_bf / (p.ρ_i * p.g)
    mode === :flat && return G
    mode === :full || throw(ArgumentError("mode must be :full or :flat, got $mode"))
    z_bx, z_by = _grad(z_b, dx, dy)
    wx, wy = _grad(w, dx, dy)
    # Floor at H_min² so √w stays differentiable at w = 0 (ice-free cells) — sqrt(0)
    # has an infinite derivative that otherwise poisons the implicit-diff VJP.
    sqrtw = @. sqrt(max(w, p.H_min^2))
    grad_zb2 = @. z_bx * z_bx + z_by * z_by
    dot_zbw = @. z_bx * wx + z_by * wy
    # Floor the squared RHS to a small positive fraction of G² so it stays real (a
    # near-flat surface where the bed is too steep for plastic ice to thicken).
    return @. sqrt(max(G * G - 4 * sqrtw * dot_zbw - 4 * w * grad_zb2, (1e-3 * G)^2))
end

# Forward solve for the working variable u (see `_bc`).
#
# :surface — one fast-sweeping solve of the surface equation (proper, monotone scheme).
# :flat    — one fast-sweeping solve of the w eikonal.
# :full    — damped outer fixed point on w, started from :flat. The map "freeze Geff(w),
#   re-solve the eikonal" is NOT contractive for steep beds (Geff carries ∇w, so the
#   undamped iteration oscillates). Under-relaxation by `relax` converges it, where it
#   converges, to the fixed point, i.e. to a vanishing `godunov_residual`, which is also
#   what makes the implicit-diff gradient exact.
function _solve_u(τ_bf, z_b, mask, bc, dfac, dx, dy, p::PlasticParams, mode::Symbol;
                  max_sweeps::Int, tol, n_outer::Int, outer_tol, relax = 0.5)
    if mode === :surface
        K = @. τ_bf / (p.ρ_i * p.g)
        return _sweep(_Surface(), K, z_b, mask, bc, dfac, dx, dy; max_sweeps, tol)
    end
    G = _driving_rhs(τ_bf, τ_bf, z_b, dx, dy, p, :flat)   # flat RHS (w arg unused for :flat)
    w = _sweep(_Eikonal(), G, nothing, mask, bc, dfac, dx, dy; max_sweeps, tol)
    if mode === :full
        for _ in 1:n_outer
            F = _driving_rhs(w, τ_bf, z_b, dx, dy, p, :full)
            wc = _sweep(_Eikonal(), F, nothing, mask, bc, dfac, dx, dy; max_sweeps, tol)
            wnew = @. (1 - relax) * w + relax * wc
            conv = maximum(abs.(sqrt.(max.(wnew, 0)) .- sqrt.(max.(w, 0))))
            w = wnew
            conv < outer_tol && break
        end
    end
    return w
end

"""
    godunov_residual(u, τ_bf, z_b, mask, bc, dfac, dx, dy, params, mode)

Optimality conditions satisfied (≈ 0) by the converged working field `u` (`z_s` in
`:surface` mode, `w = H²` in `:flat`/`:full`), returned with the same shape as `u`:
`u − (Godunov update)` on ice cells, `u − bc` on ice-free cells. `dfac` is the margin
spacing factor (internal `_margin_dfac(mask, margin)`). This is the residual the
implicit-differentiation extension differentiates; it shares the kernels and
`_update_at` with the forward solver so the two cannot diverge.
"""
function godunov_residual(u, τ_bf, z_b, mask, bc, dfac, dx, dy, p::PlasticParams, mode::Symbol)
    kern, F, zb = mode === :surface ? (_Surface(), τ_bf ./ (p.ρ_i * p.g), z_b) :
                  (_Eikonal(), _driving_rhs(u, τ_bf, z_b, dx, dy, p, mode), nothing)
    nx, ny = size(u)
    # Promote so the residual carries duals whether τ_b or u is the differentiation seed.
    T = promote_type(eltype(u), eltype(F), eltype(bc))
    r = Matrix{T}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        r[i, j] = mask[i, j] ?
            u[i, j] - _update_at(kern, u, F[i, j], zb === nothing ? 0.0 : zb[i, j], dfac, i, j, dx, dy) :
            u[i, j] - bc[i, j]
    end
    return r
end

# --- top-level solve --------------------------------------------------------------

"""
    solve(z_b, τ_b, mask; dx, dy, z_ss=0.0, margin=:face, mode=:surface, params=PlasticParams(),
          max_sweeps=200, tol=1e-6, n_outer=100, outer_tol=1e-3, relax=0.5)

Reconstruct a perfectly-plastic, steady-state ice sheet.

Inputs (all matrices share the same `(nx, ny)` shape unless noted):
- `z_b`  : bed elevation (m)
- `τ_b`    : basal shear stress (Pa); a scalar is broadcast to a uniform field
- `mask` : `true` where grounded ice exists; the margin is the mask boundary
- `dx`, `dy` : grid spacings (m)
- `z_ss` : sea level (m), scalar or field; marine margins are grounded at the flotation
           thickness `(ρ_w/ρ_i) max(z_ss − z_b, 0)`

Options:
- `margin`: where the margin lies between an ice cell and an ice-free neighbour —
            `:face` (default, on the cell face), `:centre` (on the ice-free cell centre;
            the behaviour before v0.2), or a field `margin_frac` ∈ [0, 1) on the ice-free
            cells, the distance of the margin from the ice-free cell centre as a fraction
            of the spacing (e.g. from [`margin_frac_from_sdist`](@ref))
- `mode`  : `:surface` (default) solves the plastic surface equation for `z_s` directly,
            bed relief included, in one fast-sweeping solve (no outer iteration);
            `:flat` solves the flat-bed eikonal `|∇(H²)| = 2τ_b/(ρ_i g)` and drapes it on
            the bed (cheapest; equal to `:surface` on a flat bed); `:full` solves the
            bed-corrected `H²` equation by a damped fixed point (kept for comparison;
            does not converge on rough real beds)
- `params`: `PlasticParams`
- `max_sweeps`, `tol`        : fast-sweeping controls (`tol` on `z_s` in m for
  `:surface`, on `H²` in m² otherwise)
- `n_outer`, `outer_tol`, `relax` : `:full` damped fixed-point controls (ignored
  otherwise). `relax` ∈ (0,1] is the under-relaxation factor; lower is more robust on steep
  beds but slower. The undamped map (`relax=1`) can oscillate over strong bed relief.

Returns `(z_s, H)` — ice surface elevation and thickness (m). Ice-free cells have
`H = 0`, `z_s = z_b`. Ice cells in a region with no ice-free cell (unreachable from any
margin) also get `H = 0`, with a warning.

The whole computation is differentiable w.r.t. `τ_b`; see the Algorithms page of the docs (§7).
"""
function solve(z_b::AbstractMatrix, τ_b, mask::AbstractMatrix{Bool};
               dx::Real, dy::Real, z_ss = 0.0, margin = :face, mode::Symbol = :surface,
               params::PlasticParams = PlasticParams(),
               max_sweeps::Int = 200, tol = 1e-6,
               n_outer::Int = 100, outer_tol = 1e-3, relax = 0.5)

    size(z_b) == size(mask) || throw(DimensionMismatch("z_b and mask must match"))
    p = params
    τ_bf = _as_field(τ_b, size(z_b), "τ_b")
    bc = _bc(z_b, _as_field(z_ss, size(z_b), "z_ss"), p, mode)
    dfac = _margin_dfac(mask, margin)
    u = _solve_u(τ_bf, z_b, mask, bc, dfac, dx, dy, p, mode; max_sweeps, tol, n_outer, outer_tol, relax)
    return _geometry(u, z_b, mask, mode)
end

# (z_s, H) from the working variable u (z_s or H², see `_bc`). Ice cells the sweep never
# reached (Inf: an ice region with no ice-free cell) get H = 0, with a warning.
function _geometry(u, z_b, mask, mode::Symbol)
    z0 = zero(eltype(u))   # not inside `@.`, which would dot `zero`/`eltype` and defeat inference
    n_unreached = count(k -> mask[k] && !isfinite(u[k]), eachindex(u))
    n_unreached > 0 &&
        @warn "$n_unreached ice cells not connected to an ice-free cell (H set to 0)"
    H = mode === :surface ? (@. ifelse(mask & isfinite(u), max(u - z_b, z0), z0)) :
                            (@. ifelse(mask & isfinite(u), sqrt(max(u, z0)), z0))
    z_s = @. ifelse(mask, H + z_b, z_b)
    return z_s, H
end

# Reverse-mode differentiable thickness solve w.r.t. a full `τ_b` field. The real method
# is provided by the package extension loaded with `ImplicitDifferentiation` (and an
# AD backend, e.g. ForwardDiff). Without it, this fallback explains what to load.
function differentiable_thickness end
