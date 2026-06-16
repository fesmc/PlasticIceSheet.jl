# Flux closures on the fixed plastic geometry, built on ONE discrete flux operator so they
# stay mutually consistent.
#
# Everything here routes the vertically-integrated flux down the surface with the same
# finite-volume multiple-flow-direction (MFD) weights: a cell's throughput is partitioned
# to its lower grounded neighbours in proportion to their surface drop. The forward map
# (source → throughput, `balance_flux`) and its adjoint (throughput → divergence,
# `_flux_divergence`) are exact inverses, so:
#
#   * closure B — `smb_from_velocity` is the MFD divergence ∇·(ūH);
#   * closure A — `balance_flux`/`balance_velocity` route a prescribed SMB back to a flux;
#
# and the round trip `balance_flux(smb_from_velocity(vel)) == ū H` holds to machine
# precision. That exactness is what makes the SMB-driven reconstruction
# ([`diva_reconstruct`](@ref)) have the plastic solution as a clean fixed point.

# --- shared MFD core --------------------------------------------------------------

# Downhill surface drops from cell (i,j) to its four grounded neighbours, with their sum.
# Cells off the grid or off the mask, and uphill neighbours, contribute zero — so flux
# leaving a margin cell terminates there (catchment discharge).
@inline function _downhill_drops(z_s, mask, i, j, nx, ny, ::Type{T}) where {T}
    nbrs = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
    drops = ntuple(4) do n
        ii, jj = nbrs[n]
        if 1 <= ii <= nx && 1 <= jj <= ny && mask[ii, jj]
            d = z_s[i, j] - z_s[ii, jj]
            d > 0 ? T(d) : zero(T)
        else
            zero(T)
        end
    end
    return nbrs, drops, sum(drops)
end

# Adjoint of the routing: net volume-flux divergence `S = out − inflow` of a per-cell
# outgoing throughput `out`, using the same MFD weights as `balance_flux`. `S` equals
# `smb · cellarea`. Exact inverse of `balance_flux` (which solves `out` from `S`).
function _flux_divergence(out, z_s, mask, dx, dy)
    nx, ny = size(z_s)
    T = promote_type(eltype(out), eltype(z_s))
    inflow = zeros(T, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        mask[i, j] || continue
        nbrs, drops, wsum = _downhill_drops(z_s, mask, i, j, nx, ny, T)
        wsum > 0 || continue
        oc = out[i, j]
        for n in 1:4
            drops[n] > 0 || continue
            ii, jj = nbrs[n]
            inflow[ii, jj] += oc * drops[n] / wsum
        end
    end
    S = zeros(T, nx, ny)
    @inbounds for k in eachindex(S)
        mask[k] && (S[k] = out[k] - inflow[k])
    end
    return S
end

# Flux width used to turn a volume throughput (m³ time⁻¹) into a per-width flux / velocity.
_flux_width(dx, dy) = sqrt(dx * dy)

# --- closure B: implied SMB -------------------------------------------------------

"""
    smb_from_velocity(vel, H, z_s, mask, dx, dy)

First-order steady-state surface mass balance implied by a velocity field, `ḃ = ∇·(ūH)`,
discretised as the finite-volume MFD divergence of the flux `q = ū H` down the surface
`z_s`. `vel` is the named tuple from [`diva_velocity`](@ref) (only `vel.speed` is read; the
direction is the surface descent, consistent with how `vel` was built). Positive values
are net accumulation, negative net ablation.

This is the exact inverse of [`balance_flux`](@ref): routing this SMB back down `z_s`
recovers `ū H` to machine precision, which is what closes the loop with closure A and with
[`diva_reconstruct`](@ref).
"""
function smb_from_velocity(vel, H, z_s, mask, dx, dy)
    w = _flux_width(dx, dy)
    out = vel.speed .* H .* w
    S = _flux_divergence(out, z_s, mask, dx, dy)
    return S ./ (dx * dy)
end

# --- closure A: balance flux and velocity -----------------------------------------

"""
    balance_flux(z_s, mask, smb, dx, dy)

Vertically-integrated **volume** flux `Q` (m³ time⁻¹) through each grounded cell of the
steady balance `∇·q = ḃ`, by rolling the surface mass balance `smb` downhill on `z_s`
(Budd & Warner 1996) with the shared MFD weights. Cells are processed from highest to
lowest surface, so every upstream contribution has arrived before a cell distributes its
throughput; flux delivered to the margin terminates there. `smb` may be a scalar or a
field; ice-free cells return 0.

Exact inverse of [`smb_from_velocity`](@ref)/[`_flux_divergence`](@ref). Divide by a flux
width and `H` for a velocity — see [`balance_velocity`](@ref).
"""
function balance_flux(z_s, mask, smb, dx, dy)
    nx, ny = size(z_s)
    size(mask) == size(z_s) || throw(DimensionMismatch("mask and z_s must match"))
    smbf = smb isa Number ? fill(float(smb), nx, ny) : smb
    size(smbf) == size(z_s) || throw(DimensionMismatch("smb and z_s must match"))

    T = promote_type(eltype(z_s), eltype(smbf))
    cellarea = dx * dy
    Q = zeros(T, nx, ny)
    idx = [k for k in eachindex(z_s) if mask[k]]
    @inbounds for k in idx
        Q[k] = smbf[k] * cellarea
    end

    order = sortperm([z_s[k] for k in idx]; rev = true)
    cart = CartesianIndices(z_s)
    @inbounds for oi in order
        k = idx[oi]
        c = cart[k]
        i, j = c[1], c[2]
        nbrs, drops, wsum = _downhill_drops(z_s, mask, i, j, nx, ny, T)
        wsum > 0 || continue        # pit or margin: throughput terminates here
        Qk = Q[k]
        for n in 1:4
            drops[n] > 0 || continue
            ii, jj = nbrs[n]
            Q[ii, jj] += Qk * drops[n] / wsum
        end
    end
    return Q
end

"""
    balance_velocity(z_s, H, mask, smb, dx, dy; slope_eps = 1.0e-6, H_min = 1.0)

Depth-averaged **balance velocity** consistent with surface mass balance `smb` on the
fixed geometry: the [`balance_flux`](@ref) `Q` converted to a speed `|ū| = Q / (w H)`
(flux width `w = √(dx dy)`, thickness floored at `H_min`) and directed along
[`flow_direction`](@ref). A net-ablation catchment can give a negative speed (flux
reversal) — the sign rides on the down-surface direction.

Returns a named tuple: `speed`, vector `ux`/`uy`, the volume flux `Q`, and the unit flow
direction `ex`/`ey`.
"""
function balance_velocity(z_s, H, mask, smb, dx, dy; slope_eps = 1.0e-6, H_min = 1.0)
    Q = balance_flux(z_s, mask, smb, dx, dy)
    ex, ey = flow_direction(z_s, dx, dy; slope_eps)
    w = _flux_width(dx, dy)
    T = promote_type(eltype(Q), eltype(H), eltype(ex))
    nx, ny = size(z_s)
    speed = zeros(T, nx, ny)
    ux = zeros(T, nx, ny)
    uy = zeros(T, nx, ny)
    @inbounds for k in eachindex(z_s)
        mask[k] || continue
        s = Q[k] / (w * max(H[k], H_min))
        speed[k] = s
        ux[k] = s * ex[k]
        uy[k] = s * ey[k]
    end
    return (; speed, ux, uy, Q, ex, ey)
end

"""
    implied_basal_velocity(z_s, H, τ_b, mask, smb, dx, dy; rheology,
                           slope_eps = 1.0e-6, H_min = 1.0)

Consistency diagnostic bridging closures A and B: the basal velocity the geometry and
`smb` demand, obtained by stripping the analytical deformation
([`deformational_velocity`](@ref) at `τ = τ_b`) off the [`balance_velocity`](@ref) speed,

    u_basal = u_balance − u_def .

Inverting a sliding law at this `u_basal` and the local `τ_b` gives the basal friction
implied by the reconstruction — a check on whether the plastic geometry plus the assumed
SMB is consistent with a plausible bed. A negative `u_basal` flags a cell where the
balance flux is smaller than internal deformation alone would carry (the assumed SMB or
rheology is too weak there).

Returns a named tuple: `u_balance`, `u_def`, `u_basal`, and the volume flux `Q`.
"""
function implied_basal_velocity(z_s, H, τ_b, mask, smb, dx, dy;
                                rheology::GlenRheology, slope_eps = 1.0e-6, H_min = 1.0)
    bal = balance_velocity(z_s, H, mask, smb, dx, dy; slope_eps, H_min)
    τ_bf = τ_b isa Number ? fill(float(τ_b), size(z_s)) : τ_b
    size(τ_bf) == size(z_s) || throw(DimensionMismatch("τ_b and z_s must match"))
    T = promote_type(eltype(bal.speed), eltype(τ_bf), eltype(H))
    nx, ny = size(z_s)
    u_def = zeros(T, nx, ny)
    u_basal = zeros(T, nx, ny)
    @inbounds for k in eachindex(z_s)
        mask[k] || continue
        udk = deformational_velocity(rheology, τ_bf[k], H[k])
        u_def[k] = udk
        u_basal[k] = bal.speed[k] - udk
    end
    return (; u_balance = bal.speed, u_def, u_basal, Q = bal.Q)
end
