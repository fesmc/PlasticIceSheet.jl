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
    surface_misfit(z_s, z_s_obs, mask)

Mean-squared surface-elevation misfit over grounded-ice cells — a basic objective for
inverting `τ_b` against a target surface (e.g. observed topography or a Yelmo field).
"""
function surface_misfit(z_s, z_s_obs, mask)
    s = zero(eltype(z_s))
    n = 0
    @inbounds for k in eachindex(z_s)
        if mask[k]
            d = z_s[k] - z_s_obs[k]
            s += d * d
            n += 1
        end
    end
    return n == 0 ? s : s / n
end

"""
    stress_residual(z_s, z_b, τ_b, mask; dx, dy, z_ss=0.0, margin=:face, params=PlasticParams())

Relative plastic-stress residual of a geometry, any mode: on ice cells with `τ_b > 0` and
`H = z_s − z_b > 0`,

    r = | ρ_i g H |∇z_s| − τ_b | / τ_b,

with the Godunov upwind gradient `|∇z_s|² = (max_s (z_s − a_s)₊/d_s)² + (same in y)` over
the two sides of each axis (ice-free neighbours at `z_b + H_flt`, at the margin spacing);
`r = 0` elsewhere. A cell-centred measure: on the exact Nye profile it is O(1) at the first
ice cell (one-sided slope of a √ profile), so read it away from the margin. The
scheme-consistent measure is [`godunov_residual`](@ref).
"""
function stress_residual(z_s, z_b, τ_b, mask; dx, dy, z_ss = 0.0, margin = :face,
                         params::PlasticParams = PlasticParams())
    size(z_s) == size(z_b) == size(mask) || throw(DimensionMismatch("z_s, z_b, mask must match"))
    p = params
    τ_bf = _as_field(τ_b, size(z_b), "τ_b")
    bc = _bc(z_b, _as_field(z_ss, size(z_b), "z_ss"), p, :surface)
    u = @. ifelse(mask, z_s, bc)
    dfac = _margin_dfac(mask, margin)
    ρg = p.ρ_i * p.g
    nx, ny = size(u)
    r = zeros(promote_type(eltype(u), eltype(τ_bf)), nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        H = z_s[i, j] - z_b[i, j]
        (mask[i, j] && τ_bf[i, j] > 0 && H > 0) || continue
        im, ip = max(i - 1, 1), min(i + 1, nx)
        jm, jp = max(j - 1, 1), min(j + 1, ny)
        ui = u[i, j]
        gx = max(max(ui - u[im, j], 0) / (dx * dfac[im, j]), max(ui - u[ip, j], 0) / (dx * dfac[ip, j]))
        gy = max(max(ui - u[i, jm], 0) / (dy * dfac[i, jm]), max(ui - u[i, jp], 0) / (dy * dfac[i, jp]))
        r[i, j] = abs(sqrt(gx^2 + gy^2) * ρg * H - τ_bf[i, j]) / τ_bf[i, j]
    end
    return r
end
