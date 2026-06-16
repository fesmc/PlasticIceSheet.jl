# Closure A — the dual of `velocity.jl`. Instead of assuming a flow law and reading the
# velocity out, prescribe the surface mass balance and route it down the fixed plastic
# surface to get the balance flux, hence the velocity the ice *must* have to be in steady
# state. This is the rigorous reading when τ_b is a basal sliding resistance: a plastic
# bed sets the flow direction (down-surface) but not the speed, which mass continuity then
# fixes.
#
# `implied_basal_velocity` closes the loop with closure B: subtracting the analytical
# deformation from the balance speed leaves the basal velocity the geometry+SMB demands,
# the check on whether a plausible bed is consistent with both.

"""
    balance_flux(z_s, mask, smb, dx, dy)

Vertically-integrated **volume** flux `Q` (m³ time⁻¹) through each grounded cell of the
steady balance `∇·q = ḃ`, by rolling the surface mass balance `smb` downhill on `z_s`
(Budd & Warner 1996). Cells are processed from highest to lowest surface; each passes its
accumulated throughput to lower grounded neighbours, partitioned by their positive surface
drop (a smooth multiple-flow-direction split). Flux delivered to the margin terminates
there (the catchment discharge). `smb` may be a scalar or a field; ice-free cells return 0.

Returns `Q` with the shape of `z_s`. Divide by a flux width and `H` for a velocity — see
[`balance_velocity`](@ref).
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

    # Process grounded cells from highest surface to lowest, so every upstream contribution
    # has already arrived before a cell distributes its throughput.
    order = sortperm([z_s[k] for k in idx]; rev = true)
    cart = CartesianIndices(z_s)
    @inbounds for oi in order
        k = idx[oi]
        c = cart[k]
        i, j = c[1], c[2]
        nbrs = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        drops = ntuple(4) do n
            ii, jj = nbrs[n]
            if 1 <= ii <= nx && 1 <= jj <= ny && mask[ii, jj]
                d = z_s[k] - z_s[ii, jj]
                d > 0 ? d : zero(T)
            else
                zero(T)
            end
        end
        wsum = sum(drops)
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
    w = sqrt(dx * dy)
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
