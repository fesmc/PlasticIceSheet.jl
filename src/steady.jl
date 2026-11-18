# Steady SMB-driven reconstruction, direct formulation: steady mass conservation
#
#     ∇·q = smb on the mask,   q = −H U(τ, H) ∇s/|∇s|,   τ = ρ_i g H |∇s|,
#     U = u_b(τ) + 2A/(n+2) τⁿ H   (the membrane-free DIVA flux of the closures),
#
# with the obstacle H ≥ 0 (complementarity: H ≥ 0, F = ∇·q − smb ≥ 0, H F = 0), solved by
# pseudo-transient continuation with one semismooth (active-set) Newton step per
# pseudo-time step. This is the steady SIA with sliding: a monotone, degenerate-elliptic
# problem with a unique solution, whose flux depends continuously on the local slope
# rather than on a routing partition (the routing iteration of `diva_reconstruct(method =
# :routing)` is only conditionally stable; see the docs).
#
# Finite volumes with staggered face fluxes. Between ice cells the face thickness is the
# mean limited towards the upstream value (`_face_H`), which keeps F non-increasing in the
# neighbours' H (an M-function). From an ice cell to an ice-free neighbour the flux is the
# one-sided SIA flux to the margin, which lies (1 − margin_frac)·Δ from the ice-cell
# centre (the margin position of `solve`). The implied SMB of a geometry,
# `smb_flux_divergence`, is the divergence of the same face fluxes, so a geometry forced
# by its own implied SMB is a root of the residual.

# Fixed context of a steady problem (everything but H and smb).
struct _SteadyProblem{M, L<:SlidingLaw, R<:GlenRheology}
    z_b::Matrix{Float64}
    mask::M
    sm::Matrix{Float64}      # margin surface z_b + H_flt
    hm::Matrix{Float64}      # margin thickness H_flt
    dfac::Matrix{Float64}    # margin spacing factor (see `_margin_dfac`)
    cm::Matrix{Float64}      # cap of the driving stress per cell
    dx::Float64
    dy::Float64
    ρg::Float64
    sliding::L
    rheology::R
end

# Driving-stress cap of the fluxes: 0.999 C for a regularized-Coulomb bed without a speed
# cap (finite speed), none otherwise; at most τ_cap where τ_cap > 0 (a plastic bed carries
# at most its yield stress).
_stress_cap(l::SlidingLaw, k) = Inf
_stress_cap(l::RegularizedCoulomb, k) = isfinite(l.u_max) ? Inf : 0.999 * _coef_at(l.C, k)

function _SteadyProblem(z_b, mask, dx, dy; rheology, sliding, z_ss, margin, params, τ_cap)
    shape = size(z_b)
    size(mask) == shape || throw(DimensionMismatch("z_b and mask must match"))
    _check_law(sliding, shape)
    hm = flotation_thickness.(z_b, _as_field(z_ss, shape, "z_ss"), Ref(params))
    cm = [_stress_cap(sliding, k) for k in CartesianIndices(z_b)]
    if τ_cap !== nothing
        τc = _as_field(τ_cap, shape, "τ_cap")
        @. cm = ifelse(τc > 0, min(cm, τc), cm)
    end
    return _SteadyProblem(Matrix{Float64}(z_b), mask, z_b .+ hm, hm, _margin_dfac(mask, margin),
                          cm, float(dx), float(dy), params.ρ_i * params.g, sliding, rheology)
end

# Normal flux −H_f U(τ, H_f) gₙ/|g| through a face, g = (gₙ, gₜ) the surface slope.
@inline function _face_q(Hf, gn, gt, ρg, τcap, law, rheo)
    g = sqrt(gn * gn + gt * gt)
    (Hf > 0 && g > 0) || return zero(Hf)
    τ = min(ρg * Hf * g, τcap)
    u = basal_velocity(law, τ) + deformational_velocity(rheo, τ, Hf)
    return -Hf * u * gn / g
end

# Face thickness between ice cells A and B: the mean where the surface drop across the
# face is small against the upstream thickness, limited towards the upstream value where it
# is not (downstream weight min(1/2, H_up/(4 Δs))).
@inline function _face_H(HA, HB, sA, sB)
    Hup, Hdn = sA >= sB ? (HA, HB) : (HB, HA)
    ds = abs(sA - sB)
    w = 2ds > Hup ? Hup / (4ds) : 0.5
    return (1 - w) * Hup + w * Hdn
end

# Outflow from ice cell P to the margin at distance dc (one-sided slope, segment-mean
# thickness with the margin thickness hmM), never inflow.
@inline _margin_q(HP, hmM, sP, sM, dc, gt, ρg, τcap, law, rheo) =
    max(_face_q((HP + hmM) / 2, (sM - sP) / dc, gt, ρg, τcap, law, rheo), zero(HP))

# Central slope of s at (i, j) along x (dir = 1) or y (dir = 2), clamp-to-edge.
@inline function _cslope(s, i, j, d, dir)
    nx, ny = size(s)
    if dir == 1
        a, b = max(i - 1, 1), min(i + 1, nx)
        return b > a ? (s[b, j] - s[a, j]) / ((b - a) * d) : zero(eltype(s))
    else
        a, b = max(j - 1, 1), min(j + 1, ny)
        return b > a ? (s[i, b] - s[i, a]) / ((b - a) * d) : zero(eltype(s))
    end
end

# Face fluxes of thickness H: s (surface, margin values off the ice), qx[i+1, j] the flux
# through the face between (i, j) and (i+1, j) (i = 0..nx), qy[i, j+1] likewise in y.
# Faces on the array edge carry no flux.
function _steady_fluxes!(s, qx, qy, H, P::_SteadyProblem)
    nx, ny = size(H)
    m, ρg, rh, sl = P.mask, P.ρg, P.rheology, P.sliding
    @inbounds for k in eachindex(H)
        s[k] = m[k] ? P.z_b[k] + H[k] : P.sm[k]
    end
    fill!(qx, 0); fill!(qy, 0)
    @inbounds for j in 1:ny, i in 1:(nx - 1)
        A, B = CartesianIndex(i, j), CartesianIndex(i + 1, j)
        if m[A] && m[B]
            qx[i + 1, j] = _face_q(_face_H(H[A], H[B], s[A], s[B]), (s[B] - s[A]) / P.dx,
                                   (_cslope(s, i, j, P.dy, 2) + _cslope(s, i + 1, j, P.dy, 2)) / 2,
                                   ρg, (P.cm[A] + P.cm[B]) / 2, _law_at(sl, A, B), rh)
        elseif m[A]
            qx[i + 1, j] = _margin_q(H[A], P.hm[B], s[A], s[B], P.dfac[B] * P.dx,
                                     _cslope(s, i, j, P.dy, 2), ρg, P.cm[A], _law_at(sl, A), rh)
        elseif m[B]
            qx[i + 1, j] = -_margin_q(H[B], P.hm[A], s[B], s[A], P.dfac[A] * P.dx,
                                      _cslope(s, i + 1, j, P.dy, 2), ρg, P.cm[B], _law_at(sl, B), rh)
        end
    end
    @inbounds for j in 1:(ny - 1), i in 1:nx
        A, B = CartesianIndex(i, j), CartesianIndex(i, j + 1)
        if m[A] && m[B]
            qy[i, j + 1] = _face_q(_face_H(H[A], H[B], s[A], s[B]), (s[B] - s[A]) / P.dy,
                                   (_cslope(s, i, j, P.dx, 1) + _cslope(s, i, j + 1, P.dx, 1)) / 2,
                                   ρg, (P.cm[A] + P.cm[B]) / 2, _law_at(sl, A, B), rh)
        elseif m[A]
            qy[i, j + 1] = _margin_q(H[A], P.hm[B], s[A], s[B], P.dfac[B] * P.dy,
                                     _cslope(s, i, j, P.dx, 1), ρg, P.cm[A], _law_at(sl, A), rh)
        elseif m[B]
            qy[i, j + 1] = -_margin_q(H[B], P.hm[A], s[B], s[A], P.dfac[A] * P.dy,
                                      _cslope(s, i, j + 1, P.dx, 1), ρg, P.cm[B], _law_at(sl, B), rh)
        end
    end
    return nothing
end

# Residual F = ∇·q − smb on the ice cells (0 elsewhere), into F (workspace s, qx, qy).
function _steady_residual!(F, s, qx, qy, H, smb, P::_SteadyProblem)
    _steady_fluxes!(s, qx, qy, H, P)
    nx, ny = size(H)
    @inbounds for j in 1:ny, i in 1:nx
        F[i, j] = P.mask[i, j] ?
            (qx[i + 1, j] - qx[i, j]) / P.dx + (qy[i, j + 1] - qy[i, j]) / P.dy - smb[i, j] : 0.0
    end
    return F
end

# Complementarity residuals: max and rms of min(H, F) (|F| with ice, max(−F, 0) at H = 0).
_ncp(h, f) = h > 0 ? f : min(f, zero(f))
_ncp_norm(H, F, mask) = maximum((abs(_ncp(H[k], F[k])) for k in eachindex(H) if mask[k]); init = 0.0)
function _ncp_rms(H, F, mask)
    n = count(mask)
    n == 0 && return 0.0
    return sqrt(sum((_ncp(H[k], F[k])^2 for k in eachindex(H) if mask[k]); init = 0.0) / n)
end

# Steady solve from H (modified in place). Returns (n_steps, residual, converged).
function _steady_solve!(H, smb, P::_SteadyProblem; res_tol, n_steady, dt0, dt_max, verbose)
    nx, ny = size(H)
    mask = P.mask
    s = similar(H); qx = zeros(nx + 1, ny); qy = zeros(nx, ny + 1)
    F = similar(H); Fp = similar(H); Ht = similar(H); dH = similar(H)
    J = zeros(9, nx, ny)                               # ∂F_P/∂H_K over the 3 × 3 block
    act = falses(nx, ny)
    id = zeros(Int, nx, ny)
    resid!(Fx, Hx) = _steady_residual!(Fx, s, qx, qy, Hx, smb, P)

    @. H = ifelse(mask, max(H, 0.0), 0.0)
    resid!(F, H)
    res = _ncp_norm(H, F, mask)
    rms = _ncp_rms(H, F, mask)
    dt = dt0
    ok = res < res_tol
    n_steps = 0
    Is, Js, Vs, b = Int[], Int[], Float64[], Float64[]
    for k in 1:n_steady
        ok && break
        n_steps = k

        # Jacobian by finite differences, 9 colours (every third cell in x and y at once).
        fill!(J, 0.0)
        for c in 0:8
            ci, cj = mod(c, 3), c ÷ 3
            @inbounds for j in 1:ny, i in 1:nx
                Ht[i, j] = H[i, j]
                (mask[i, j] && mod(i, 3) == ci && mod(j, 3) == cj) &&
                    (Ht[i, j] = H[i, j] + 1.0e-7 * max(H[i, j], 10.0))
            end
            resid!(Fp, Ht)
            @inbounds for j in 1:ny, i in 1:nx
                mask[i, j] || continue
                for bb in -1:1, aa in -1:1
                    ii, jj = i + aa, j + bb
                    (1 <= ii <= nx && 1 <= jj <= ny) || continue
                    (mod(ii, 3) == ci && mod(jj, 3) == cj && mask[ii, jj]) || continue
                    J[(bb + 1) * 3 + aa + 2, i, j] = (Fp[i, j] - F[i, j]) / (Ht[ii, jj] - H[ii, jj])
                end
            end
        end

        # Active set: the cell would empty under its own Newton step, H ≤ F/(1/dt + ∂F_P/∂H_P).
        @inbounds for j in 1:ny, i in 1:nx
            d = 1 / dt + J[5, i, j]
            act[i, j] = mask[i, j] && (d > 0 ? H[i, j] * d <= F[i, j] : H[i, j] <= dt * F[i, j])
        end
        n = 0
        @inbounds for j in 1:ny, i in 1:nx
            id[i, j] = (mask[i, j] && !act[i, j]) ? (n += 1) : 0
        end

        # Newton system over the inactive ice cells; active neighbours (d_K = −H_K) move to
        # the right-hand side.
        empty!(Is); empty!(Js); empty!(Vs); resize!(b, n)
        @inbounds for j in 1:ny, i in 1:nx
            row = id[i, j]
            row == 0 && continue
            b[row] = -F[i, j]
            for bb in -1:1, aa in -1:1
                ii, jj = i + aa, j + bb
                (1 <= ii <= nx && 1 <= jj <= ny && mask[ii, jj]) || continue
                v = J[(bb + 1) * 3 + aa + 2, i, j]
                if id[ii, jj] > 0
                    push!(Is, row); push!(Js, id[ii, jj])
                    push!(Vs, aa == 0 && bb == 0 ? v + 1 / dt : v)
                else
                    b[row] += v * H[ii, jj]
                end
            end
        end
        x = n > 0 ? sparse(Is, Js, Vs, n, n) \ b : Float64[]
        @inbounds for kk in eachindex(H)
            dH[kk] = !mask[kk] ? 0.0 : id[kk] == 0 ? -H[kk] : x[id[kk]]
        end

        # Projected backtracking on the rms residual: H(λ) = max(H + λ d, 0).
        λ = 1.0
        rms_new = rms
        for _ in 1:6
            @. Ht = max(H + λ * dH, 0.0)
            resid!(Fp, Ht)
            rms_new = _ncp_rms(Ht, Fp, mask)
            rms_new <= (1 - 1.0e-4 * λ) * rms && break
            λ /= 2
        end
        res_new = _ncp_norm(Ht, Fp, mask)
        verbose && @info "steady $k: dt $(round(dt; sigdigits = 3)) residual $(round(res; sigdigits = 3)) → $(round(res_new; sigdigits = 3)), unknowns $n, active $(count(act)), λ $λ"

        if !(rms_new <= 10 * rms)                    # reject (or NaN): shorter step
            dt /= 2
            dt < 1.0e-12 * dt0 && break
            continue
        end
        H .= Ht
        F .= Fp
        # Switched evolution relaxation on the rms residual, growing at least 1.5× after a
        # step that did not increase it.
        dt = rms_new <= rms ? min(dt * max(rms / max(rms_new, floatmin()), 1.5), dt_max) :
                              dt * rms / rms_new
        dt = max(dt, 1.0e-6 * dt0)
        res, rms = res_new, rms_new
        ok = res < res_tol
    end
    return n_steps, res, ok
end

"""
    smb_flux_divergence(z_b, H, mask, dx, dy; rheology, sliding, z_ss = 0.0, margin = :face,
                        params = PlasticParams(), τ_cap = nothing)

Implied SMB of the geometry `H` (m ice yr⁻¹) for the steady SMB-driven reconstruction
(`diva_reconstruct(method = :steady)`): the divergence of its face fluxes, with the
membrane-free DIVA flux `q = −H (u_b(τ) + 2A/(n+2) τⁿ H) ∇s/|∇s|`, `τ = ρ_i g H |∇s|`, and a
one-sided flux from each ice cell through its margin faces (margin position as in
[`solve`](@ref)). The geometry forced by this SMB is a root of the steady problem.

`τ_cap` (Pa, scalar or field) caps the driving stress of the fluxes where `> 0` (a face
takes the mean of its cells, a margin face its ice cell), e.g. `τ_b`: a plastic bed carries
at most its yield stress.

Returns a named tuple:
- `smb`      : the implied SMB on the ice cells (0 elsewhere)
- `q_land`, `q_marine` : outflow through the margin faces of each ice cell to land and to
  the ocean (ice-free neighbour below sea level), m ice yr⁻¹, ≥ 0;
  `sum(q_land + q_marine) == sum(smb)`
- `us_x`, `us_y` : surface velocity of the same closure at the cell centres,
  `u_b(τ) + 2A/(n+1) τⁿ H` along `−∇s` (central slopes)
"""
function smb_flux_divergence(z_b, H, mask, dx, dy; rheology::GlenRheology, sliding::SlidingLaw,
                             z_ss = 0.0, margin = :face, params::PlasticParams = PlasticParams(),
                             τ_cap = nothing)
    size(H) == size(z_b) || throw(DimensionMismatch("H and z_b must match"))
    P = _SteadyProblem(z_b, mask, dx, dy; rheology, sliding, z_ss, margin, params, τ_cap)
    nx, ny = size(H)
    s = zeros(nx, ny); qx = zeros(nx + 1, ny); qy = zeros(nx, ny + 1)
    smb = _steady_residual!(zeros(nx, ny), s, qx, qy, H, zeros(nx, ny), P)

    # Margin outflow, split by the ice-free neighbour: marine where H_flt > 0.
    q_land = zeros(nx, ny); q_marine = zeros(nx, ny)
    add!(I, r, h) = h > 0 ? (q_marine[I] += r) : (q_land[I] += r)
    @inbounds for j in 1:ny, i in 1:(nx - 1)
        mask[i, j] && !mask[i + 1, j] && add!(CartesianIndex(i, j), qx[i + 1, j] / dx, P.hm[i + 1, j])
        mask[i + 1, j] && !mask[i, j] && add!(CartesianIndex(i + 1, j), -qx[i + 1, j] / dx, P.hm[i, j])
    end
    @inbounds for j in 1:(ny - 1), i in 1:nx
        mask[i, j] && !mask[i, j + 1] && add!(CartesianIndex(i, j), qy[i, j + 1] / dy, P.hm[i, j + 1])
        mask[i, j + 1] && !mask[i, j] && add!(CartesianIndex(i, j + 1), -qy[i, j + 1] / dy, P.hm[i, j])
    end

    # Surface velocity of the same closure.
    us_x = zeros(nx, ny); us_y = zeros(nx, ny)
    fs = (rheology.n + 2) / (rheology.n + 1)
    @inbounds for j in 1:ny, i in 1:nx
        (mask[i, j] && H[i, j] > 0) || continue
        gx, gy = _cslope(s, i, j, dx, 1), _cslope(s, i, j, dy, 2)
        g = hypot(gx, gy)
        g > 0 || continue
        τ = min(P.ρg * H[i, j] * g, P.cm[i, j])
        u = basal_velocity(_law_at(sliding, CartesianIndex(i, j)), τ) +
            deformational_velocity(rheology, τ, H[i, j]) * fs
        us_x[i, j] = -u * gx / g
        us_y[i, j] = -u * gy / g
    end
    return (; smb, q_land, q_marine, us_x, us_y)
end
