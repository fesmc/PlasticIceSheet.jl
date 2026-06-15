# Scalar diagnostics of a reconstruction. These are differentiable reductions, handy
# as objectives / observation operators for basal-shear-stress inversion.

"""
    ice_volume(H, dx, dy)

Total ice volume (m³) for a thickness field `H` on a grid with spacings `dx`, `dy`.
"""
ice_volume(H, dx, dy) = sum(H) * dx * dy

"""
    ice_area(mask, dx, dy)

Grounded-ice area (m²).
"""
ice_area(mask, dx, dy) = count(mask) * dx * dy

"""
    surface_misfit(E, E_obs, mask)

Mean-squared surface-elevation misfit over grounded-ice cells — a basic objective for
inverting `τ` against a target surface (e.g. observed topography or a Yelmo field).
"""
function surface_misfit(E, E_obs, mask)
    s = zero(eltype(E))
    n = 0
    @inbounds for k in eachindex(E)
        if mask[k]
            d = E[k] - E_obs[k]
            s += d * d
            n += 1
        end
    end
    return n == 0 ? s : s / n
end
