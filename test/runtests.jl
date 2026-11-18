using PlasticIceSheet
using Test
using ForwardDiff
# Trigger packages for the extensions:
using ImplicitDifferentiation, ADTypes, Zygote, NCDatasets

const p = PlasticParams()
const ρg = p.ρ_i * p.g

@testset "godunov kernel" begin
    # One-sided: with one neighbour at Inf the update is the 1-D step.
    @test godunov_eikonal(0.0, 1.0, Inf, 1.0, 2.0) ≈ 2.0
    @test godunov_eikonal(Inf, 1.0, 3.0, 2.0, 1.5) ≈ 3.0 + 1.5 * 2.0
    # Symmetric two-sided: ((u)/h)² + ((u)/h)² = F² ⇒ u = F h / √2.
    @test godunov_eikonal(0.0, 1.0, 0.0, 1.0, 2.0) ≈ 2.0 / sqrt(2)
    # Degenerate F = 0 ⇒ u = min neighbour.
    @test godunov_eikonal(1.0, 1.0, 4.0, 1.0, 0.0) ≈ 1.0
end

@testset "1-D strip vs analytic Nye profile, margin positions" begin
    # Ice strip between land margins at i=1 and i=nx, uniform in y. The eikonal distance
    # d to the nearest margin gives H(d) = sqrt(2τ_b/(ρg) · d); the margin lies f·dx from
    # the ice-free cell centre (`:centre` f = 0, `:face` f = 1/2, or a per-cell field).
    nx, ny = 60, 8
    dx = dy = 1000.0
    τ_b = 1.0e5
    z_b = zeros(nx, ny)
    mask = trues(nx, ny)
    mask[1, :] .= false
    mask[nx, :] .= false
    c = 2τ_b / ρg
    j = ny ÷ 2
    frac(fl, fr) = [i == 1 ? fl : i == nx ? fr : 0.0 for i in 1:nx, _ in 1:ny]
    cases = [(:centre, 0.0, 0.0), (:face, 0.5, 0.5), (frac(0.3, 0.77), 0.3, 0.77),
             (frac(0.95, 0.05), 0.95, 0.05)]
    for (margin, fl, fr) in cases
        _, H = solve(z_b, τ_b, mask; dx, dy, margin, mode = :flat, max_sweeps = 400, tol = 1e-12)
        d = [dx * min(i - 1 - fl, nx - i - fr) for i in 2:(nx - 1)]
        @test H[2:(nx - 1), j] ≈ sqrt.(c .* d) rtol = 1e-10
        @test all(H[1, :] .== 0) && all(H[nx, :] .== 0)
    end
    @test_throws ArgumentError solve(z_b, τ_b, mask; dx, dy, margin = :edge)
    @test_throws ArgumentError solve(z_b, τ_b, mask; dx, dy, margin = frac(1.0, 0.0))
end

@testset "margin_frac from a signed distance" begin
    # Circular cap of radius R, off-grid centre: the sdist margin converges to the exact
    # volume much better than the `:centre` staircase.
    R, dx = 30.0e3, 2.0e3
    n = 40
    xc, yc = 0.37dx + n * dx / 2, 0.61dx + n * dx / 2
    r = [hypot(i * dx - xc, j * dx - yc) for i in 1:n, j in 1:n]
    mask = r .<= R
    sdist = R .- r
    frac = margin_frac_from_sdist(sdist, mask, dx, dx)
    @test all(0 .<= frac .< 1) && all(iszero, frac[mask])
    c = 2 * 1.0e5 / ρg
    Hex = @. sqrt(c * max(R - r, 0)) * mask
    vol(m) = sum(last(solve(zeros(n, n), 1.0e5, mask; dx, dy = dx, margin = m, mode = :flat,
                            max_sweeps = 400, tol = 1e-10)))
    err(m) = abs(vol(m) / sum(Hex) - 1)
    @test err(frac) < err(:centre) / 4
end

@testset "marine margins with a sea-level field" begin
    # Strip on a flat bed below sea level, uniform in y; sea level differs at the two
    # margins, so each side grows from its own flotation thickness: H = √(H_flt² + c d).
    nx, ny = 60, 6
    dx = dy = 1000.0
    τ_b = 1.0e5
    z_b = fill(-300.0, nx, ny)
    z_ss = [i <= nx ÷ 2 ? 0.0 : -100.0 for i in 1:nx, j in 1:ny]
    mask = trues(nx, ny); mask[1, :] .= false; mask[nx, :] .= false
    _, H = solve(z_b, τ_b, mask; dx, dy, z_ss, mode = :flat, max_sweeps = 400, tol = 1e-9)
    c = 2τ_b / ρg
    Hl, Hr = flotation_thickness(-300.0, 0.0, p), flotation_thickness(-300.0, -100.0, p)
    exact(i) = min(sqrt(Hl^2 + c * dx * (i - 1.5)), sqrt(Hr^2 + c * dx * (nx - i - 0.5)))   # :face
    @test all(H[i, 3] ≈ exact(i) for i in 2:(nx - 1))
    @test_throws DimensionMismatch solve(z_b, τ_b, mask; dx, dy, z_ss = zeros(3, 3))
end

@testset "unreachable ice cells get H = 0 with a warning" begin
    mask = trues(10, 10)               # no ice-free cell anywhere
    z_s, H = @test_logs (:warn, r"not connected") solve(zeros(10, 10), 1.0e5, mask;
                                                        dx = 1000.0, dy = 1000.0, mode = :flat)
    @test all(iszero, H) && all(iszero, z_s)
end

@testset "flat and full agree on a flat bed" begin
    nx, ny = 50, 50
    dx = dy = 2000.0
    τ_b = 0.8e5
    z_b = fill(-50.0, nx, ny)          # flat, slightly below 0 but constant
    mask = falses(nx, ny)
    cx, cy, R = 25.5, 25.5, 18.0
    for j in 1:ny, i in 1:nx
        if hypot(i - cx, j - cy) <= R
            mask[i, j] = true
        end
    end
    z_s_f, Hf = solve(z_b, τ_b, mask; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-9)
    z_s_h, Hh = solve(z_b, τ_b, mask; dx, dy, mode = :full, max_sweeps = 400, tol = 1e-9,
                      n_outer = 40, outer_tol = 1e-5)
    # On a constant bed the two formulations are mathematically identical.
    @test maximum(abs.(Hf .- Hh)) < 2.0   # metres
end

@testset "full responds to bed relief (damped fixed point converges)" begin
    nx, ny = 40, 40
    dx = dy = 2000.0
    τ_b = 1.0e5
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 20.5, j - 20.5) <= 15 && (mask[i, j] = true)
    end
    z_b_flat = zeros(nx, ny)
    # broad, smooth bedrock high (~300 m, ~28 km scale) — within the convergence regime
    z_b_bump = [300.0 * exp(-((i - 20.5)^2 + (j - 20.5)^2) / 200) for i in 1:nx, j in 1:ny]
    opts = (; dx, dy, mode = :full, n_outer = 600, outer_tol = 1e-7, relax = 0.4)

    _, H0 = solve(z_b_flat, τ_b, mask; opts...)
    z_s_b, Hb = solve(z_b_bump, τ_b, mask; opts...)
    # A central bedrock high lifts the surface above the flat-bed reconstruction.
    @test z_s_b[20, 20] > H0[20, 20]
    # The damped :full fixed point is genuinely reached (residual ≈ 0).
    bc = flotation_thickness.(z_b_bump, 0.0, Ref(p)) .^ 2
    dfac = PlasticIceSheet._margin_dfac(mask, :face)
    R = PlasticIceSheet.godunov_residual(Hb .^ 2, fill(τ_b, nx, ny), z_b_bump, mask, bc, dfac,
                                         dx, dy, p, :full)
    @test maximum(abs, R) < 1e-2
end

@testset "surface kernel: solves its equation, proper and monotone" begin
    # Deterministic pseudo-random cases (no Random dependency).
    cnt = Ref(0)
    rnd() = (cnt[] += 1; x = sin(cnt[] * 12.9898) * 43758.5453; x - floor(x))
    P(h, α) = max(h - α, 0) * (h + max(α, 0)) / 2
    for _ in 1:20_000
        zb = 2000 * (rnd() - 0.5)
        k = 10 + 200 * rnd()
        da, db = 500 + 4000 * rnd(), 500 + 4000 * rnd()
        a = zb + 1200 * (rnd() - 0.5)
        b = rnd() < 0.1 ? Inf : zb + 1200 * (rnd() - 0.5)
        u = godunov_surface(a, da, b, db, zb, k)
        h = u - zb
        @test h > 0
        @test abs(sqrt((P(h, a - zb) / da)^2 + (isfinite(b) ? (P(h, b - zb) / db)^2 : 0.0)) - k) < 1e-8 * k
        @test godunov_surface(a + 1, da, b, db, zb, k) >= u - 1e-9    # nondecreasing in a
        isfinite(b) && @test godunov_surface(a, da, b + 1, db, zb, k) >= u - 1e-9
    end
end

@testset "surface mode: Nye strip, sloped bed, bed bump" begin
    c = 2 * 1.0e5 / ρg
    # Flat bed: surface ≡ flat (exact Nye profile).
    nx, ny = 60, 4
    mask = trues(nx, ny); mask[1, :] .= false; mask[nx, :] .= false
    _, Hs = solve(zeros(nx, ny), 1.0e5, mask; dx = 1000.0, dy = 1000.0, max_sweeps = 400, tol = 1e-10)
    _, Hf = solve(zeros(nx, ny), 1.0e5, mask; dx = 1000.0, dy = 1000.0, mode = :flat,
                  max_sweeps = 400, tol = 1e-12)
    @test maximum(abs, Hs .- Hf) < 1e-8

    # 1-D bed z_b = s x rising inland from a land margin: reference from the ODE
    # dw/dx = 2k − 2 s √w (w = H², x from the margin), RK4 on a fine grid.
    L, sl, k, hs = 100.0e3, 5.0e-3, c / 2, 0.25
    nref = round(Int, L / hs)
    wref = zeros(nref + 1)
    f(w) = 2k - 2sl * sqrt(max(w, 0.0))
    for i in 1:nref
        w0 = wref[i]
        k1 = f(w0); k2 = f(w0 + hs / 2 * k1); k3 = f(w0 + hs / 2 * k2); k4 = f(w0 + hs * k3)
        wref[i + 1] = w0 + hs * (k1 + 2k2 + 2k3 + k4) / 6
    end
    function slope_err(dx, mode)
        n = round(Int, L / dx) + 1
        z_b = [sl * dx * (i - 1) for i in 1:n, _ in 1:3]
        m = trues(n, 3); m[1, :] .= false
        _, H = solve(z_b, 1.0e5, m; dx, dy = dx, mode, max_sweeps = 1000, tol = 1e-10)
        maximum(abs(H[i, 2] - sqrt(wref[round(Int, dx * (i - 1.5) / hs) + 1]))
                for i in 2:n if dx * (i - 1.5) >= 5000)
    end
    es = [slope_err(dx, :surface) for dx in (4000.0, 2000.0, 1000.0)]
    @test es[3] < es[2] < es[1]
    @test es[3] < 0.02 * slope_err(1000.0, :flat)

    # Bed bump where :full converges: surface → full under refinement, flat does not.
    function bump(dx)
        n = round(Int, 80.0e3 / dx)
        xy = [((i - 0.5) * dx - 40.0e3, (j - 0.5) * dx - 40.0e3) for i in 1:n, j in 1:n]
        z_b = [300 * exp(-(x^2 + y^2) / (2 * 2.0e4^2)) for (x, y) in xy]
        m = [hypot(x, y) <= 30.0e3 for (x, y) in xy]
        inner = [hypot(x, y) <= 24.0e3 for (x, y) in xy]
        o = (; dx, dy = dx, max_sweeps = 1000, tol = 1e-10)
        Hs = last(solve(z_b, 1.0e5, m; o...))
        Hh = last(solve(z_b, 1.0e5, m; o..., mode = :full))
        Hf = last(solve(z_b, 1.0e5, m; o..., mode = :flat))
        return maximum(abs, (Hs .- Hh)[inner]), maximum(abs, (Hf .- Hh)[inner])
    end
    (d2, _), (d1, f1) = bump(2000.0), bump(1000.0)
    @test d1 < d2
    @test d1 < 0.2 * f1
end

@testset "stress residual" begin
    # Bed bump: :surface balances the stress far better than :flat (which ignores the bed).
    n, dx = 40, 2000.0
    xy = [((i - 0.5) * dx - 40.0e3, (j - 0.5) * dx - 40.0e3) for i in 1:n, j in 1:n]
    z_b = [300 * exp(-(x^2 + y^2) / (2 * 2.0e4^2)) for (x, y) in xy]
    m = [hypot(x, y) <= 30.0e3 for (x, y) in xy]
    inner = [hypot(x, y) <= 20.0e3 for (x, y) in xy]
    o = (; dx, dy = dx, max_sweeps = 1000, tol = 1e-10)
    zs_s, _ = solve(z_b, 1.0e5, m; o...)
    zs_f, _ = solve(z_b, 1.0e5, m; o..., mode = :flat)
    rs = stress_residual(zs_s, z_b, 1.0e5, m; dx, dy = dx)
    rf = stress_residual(zs_f, z_b, 1.0e5, m; dx, dy = dx)
    @test all(iszero, rs[.!m])
    med(v) = sort(v)[cld(length(v), 2)]
    @test med(rs[inner]) < 0.1
    @test med(rs[inner]) < med(rf[inner]) / 3
end

@testset "AD: ∂(volume)/∂τ_b via ForwardDiff matches finite difference" begin
    nx, ny = 36, 36
    dx = dy = 2000.0
    z_b = zeros(nx, ny)
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 18.5, j - 18.5) <= 13 && (mask[i, j] = true)
    end

    # Scalar objective: total volume as a function of uniform τ_b.
    function vol_of_τ_b(τ_b)
        _, H = solve(z_b, τ_b, mask; dx, dy, mode = :full, max_sweeps = 300,
                     tol = 1e-8, n_outer = 40, outer_tol = 1e-5)
        return ice_volume(H, dx, dy)
    end

    τ_b0 = 1.0e5
    g_ad = ForwardDiff.derivative(vol_of_τ_b, τ_b0)
    h = 1.0e2
    g_fd = (vol_of_τ_b(τ_b0 + h) - vol_of_τ_b(τ_b0 - h)) / (2h)
    @test isfinite(g_ad)
    @test g_ad ≈ g_fd rtol = 1e-3

    # Gradient w.r.t. a full per-cell τ_b field also flows (a few partials checked).
    τ_bfield = fill(1.0e5, nx, ny)
    function vol_of_field(τ_bv)
        _, H = solve(z_b, τ_bv, mask; dx, dy, mode = :full, max_sweeps = 300,
                     tol = 1e-8, n_outer = 40, outer_tol = 1e-5)
        return ice_volume(H, dx, dy)
    end
    grad = ForwardDiff.gradient(vol_of_field, τ_bfield)
    @test all(isfinite, grad)
    @test grad[18, 18] > 0          # raising local τ_b thickens the ice ⇒ more volume
end

@testset "reverse-mode τ_b-field gradient (implicit diff) vs ForwardDiff" begin
    nx, ny = 24, 24
    dx = dy = 2000.0
    # smooth, moderate bed relief (~300 m) so the damped :full fixed point converges
    z_b = [300.0 * sin(i / 9) * cos(j / 11) for i in 1:nx, j in 1:ny]
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 12.5, j - 12.5) <= 9 && (mask[i, j] = true)
    end
    τ_b0 = fill(1.0e5, nx, ny)
    sk = (; dx, dy, mode = :full, max_sweeps = 400, tol = 1e-10,
          n_outer = 400, outer_tol = 1e-7, relax = 0.5)

    # Reverse mode: Zygote → ImplicitDifferentiation rrule → implicit-function gradient.
    rev_loss(τ_b) = ice_volume(differentiable_thickness(τ_b, z_b, mask; sk...), dx, dy)
    g_rev = Zygote.gradient(rev_loss, τ_b0)[1]

    # Forward-mode oracle through the plain (differentiate-through) solver, which itself
    # matches finite differences; at a converged fixed point the two must coincide.
    fwd_loss(τ_b) = ice_volume(last(solve(z_b, τ_b, mask; sk...)), dx, dy)
    g_fwd = ForwardDiff.gradient(fwd_loss, τ_b0)

    @test all(isfinite, g_rev)
    @test size(g_rev) == size(τ_b0)
    @test g_rev ≈ g_fwd rtol = 1e-4
    @test sum(g_rev) > 0          # raising τ_b everywhere thickens the ice ⇒ more volume

    # :surface (the default): reverse mode equals forward mode through the sweeps.
    sks = (; dx, dy, max_sweeps = 400, tol = 1e-12)
    gs_rev = Zygote.gradient(τ_b -> ice_volume(differentiable_thickness(τ_b, z_b, mask; sks...), dx, dy), τ_b0)[1]
    gs_fwd = ForwardDiff.gradient(τ_b -> ice_volume(last(solve(z_b, τ_b, mask; sks...)), dx, dy), τ_b0)
    @test gs_rev ≈ gs_fwd rtol = 1e-6

    # Also confirm the :flat reverse-mode gradient is exact (clean eikonal residual).
    skf = (; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-10)
    gf_rev = Zygote.gradient(τ_b -> ice_volume(differentiable_thickness(τ_b, z_b, mask; skf...), dx, dy), τ_b0)[1]
    gf_fwd = ForwardDiff.gradient(τ_b -> ice_volume(last(solve(z_b, τ_b, mask; skf...)), dx, dy), τ_b0)
    @test gf_rev ≈ gf_fwd rtol = 1e-8
end

@testset "NetCDF I/O round trip" begin
    nx, ny = 10, 8
    dx = dy = 1000.0
    z_b = zeros(nx, ny); τ_b = fill(1.0e5, nx, ny)
    mask = trues(nx, ny); mask[1, :] .= false; mask[nx, :] .= false
    z_s, H = solve(z_b, τ_b, mask; dx, dy, mode = :flat)

    out = tempname() * ".nc"
    save_reconstruction(out, z_s, H; x = 0.0:dx:(nx - 1) * dx, y = 0.0:dy:(ny - 1) * dy)
    @test isfile(out)
    NCDataset(out) do ds
        @test Array(ds["H"][:, :]) ≈ H
        @test Array(ds["z_s"][:, :]) ≈ z_s
    end

    inp = tempname() * ".nc"
    NCDataset(inp, "c") do ds
        defDim(ds, "x", nx); defDim(ds, "y", ny)
        defVar(ds, "x", collect(0.0:dx:(nx - 1) * dx), ("x",))
        defVar(ds, "y", collect(0.0:dy:(ny - 1) * dy), ("y",))
        defVar(ds, "z_b", z_b, ("x", "y"))
        defVar(ds, "tau", τ_b, ("x", "y"))
        defVar(ds, "mask", Int.(mask), ("x", "y"))
    end
    got = load_plastic_inputs(inp)
    @test got.dx == dx && got.dy == dy
    @test size(got.z_b) == (nx, ny)
    @test got.mask == mask
    @test got.τ_b ≈ τ_b
    rm(out); rm(inp)
end

@testset "constitutive laws: deformation and sliding inversions" begin
    # Glen deformation is the SIA closed form 2A/(n+2)·τⁿ·H.
    r = GlenRheology(A = 1.0e-16, n = 3.0)
    @test deformational_velocity(r, 8.0e4, 2000.0) ≈ (2 * 1.0e-16 / 5) * (8.0e4)^3 * 2000.0
    @test deformational_velocity(r, 0.0, 2000.0) == 0.0   # no driving stress ⇒ no flow

    # Each sliding law inverts its own forward relation at the operating drag.
    τ_b = 8.0e4
    lin = LinearSliding(β = 2.0e3)
    @test lin.β * basal_velocity(lin, τ_b) ≈ τ_b

    wm = WeertmanSliding(β = 1.0e4, m = 3.0)
    @test wm.β * basal_velocity(wm, τ_b)^(1 / wm.m) ≈ τ_b
    # m = 1 Weertman coincides with linear of the same β.
    @test basal_velocity(WeertmanSliding(β = 2.0e3, m = 1.0), τ_b) ≈ basal_velocity(lin, τ_b)

    rc = RegularizedCoulomb(C = 1.0e5, u_0 = 100.0, m = 3.0)
    ub = basal_velocity(rc, τ_b)
    @test rc.C * (ub / (ub + rc.u_0))^(1 / rc.m) ≈ τ_b   # forward relation holds
    # At and above the Coulomb cap the speed is undetermined ⇒ NaN, not a fabricated value.
    @test isnan(basal_velocity(rc, rc.C))
    @test isnan(basal_velocity(rc, 1.2e5))
    # Low-speed limit approaches the matching Weertman power law u_b ∝ τ_bᵐ.
    small = 1.0e3
    @test basal_velocity(rc, small) ≈ rc.u_0 * (small / rc.C)^rc.m rtol = 1e-2
end

@testset "DIVA velocity and implied SMB (closure B)" begin
    # Circular ice cap on a flat bed.
    nx, ny = 61, 61
    dx = dy = 2000.0
    z_b = zeros(nx, ny)
    mask = falses(nx, ny)
    cx = cy = 31.0
    for j in 1:ny, i in 1:nx
        hypot(i - cx, j - cy) <= 24.0 && (mask[i, j] = true)
    end
    τ_b = 8.0e4
    z_s, H = solve(z_b, τ_b, mask; dx, dy, mode = :flat, max_sweeps = 600, tol = 1e-9)

    r = GlenRheology(A = 1.0e-16, n = 3.0)
    sl = WeertmanSliding(β = 1.0e4, m = 3.0)
    vel = diva_velocity(z_s, H, τ_b, mask, dx, dy; rheology = r, sliding = sl)

    # Speed is sliding + deformation at the local driving stress τ = τ_b.
    ci, cj = 31, 31
    @test vel.speed[ci, cj] ≈ basal_velocity(sl, τ_b) + deformational_velocity(r, τ_b, H[ci, cj])
    # Horizontal velocity vanishes at the divide and flows outward (down-gradient).
    @test hypot(vel.ux[ci, cj], vel.uy[ci, cj]) < 1e-8 * maximum(vel.speed)   # sweep-order rounding
    @test vel.ux[ci + 10, cj] > 0
    @test vel.uy[ci, cj + 10] > 0
    # Ice-free cells carry no velocity.
    @test all(vel.speed[.!mask] .== 0)

    # smb_from_velocity is the finite-volume MFD divergence ∇·(ūH), built to be the exact
    # inverse of balance_flux: routing the implied SMB back down the surface recovers ū H
    # to machine precision (the consistency the SMB-driven reconstruction rests on).
    smb = smb_from_velocity(vel, H, z_s, mask, dx, dy)
    Q = balance_flux(z_s, mask, smb, dx, dy)
    out = vel.speed .* H .* sqrt(dx * dy)
    @test maximum(abs.(Q[mask] .- out[mask])) / maximum(out) < 1e-10

    # Differentiable w.r.t. τ_b — the tuple works as an observation operator for inversion.
    obj(tb) = sum(diva_velocity(z_s, H, tb, mask, dx, dy; rheology = r, sliding = sl).speed)
    g = ForwardDiff.derivative(obj, τ_b)
    @test isfinite(g) && g > 0
end

@testset "balance flux/velocity and consistency (closure A)" begin
    med(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

    # 1-D strip, uniform accumulation: balance flux is exact catchment integration.
    nx, ny = 41, 6
    dx = dy = 1000.0
    z_b = zeros(nx, ny)
    mask = trues(nx, ny)
    mask[1, :] .= false
    mask[nx, :] .= false
    z_s, H = solve(z_b, 1.0e5, mask; dx, dy, mode = :flat, max_sweeps = 600, tol = 1e-9)
    ȧ = 0.3
    Q = balance_flux(z_s, mask, ȧ, dx, dy)
    cellarea = dx * dy
    j = 3
    # The divide cell (i=21) sits at the peak and splits its accumulation 50/50 to both
    # flanks; each downstream cell then adds exactly one cell's worth, so the right flank
    # reads 1.5, 2.5, …, 19.5 (in units of ȧ·cellarea) with a constant unit increment.
    @test Q[22:(nx - 1), j] ./ (ȧ * cellarea) ≈ collect(1.5:1.0:19.5)
    @test all(diff(Q[22:(nx - 1), j]) .≈ ȧ * cellarea)
    # Exact mass conservation: all accumulation exits at the two margins.
    @test sum(Q[2, :]) + sum(Q[nx - 1, :]) ≈ ȧ * cellarea * count(mask)

    # Closure A ↔ B self-consistency on a radial cap: because routing is the exact inverse
    # of the implied-SMB divergence, feeding B's SMB back through A recovers B's velocity
    # to machine precision (no stencil scatter — the shared MFD operator).
    nx2 = ny2 = 61
    dx2 = dy2 = 2000.0
    z_b2 = zeros(nx2, ny2)
    mask2 = falses(nx2, ny2)
    for jj in 1:ny2, ii in 1:nx2
        hypot(ii - 31.0, jj - 31.0) <= 24.0 && (mask2[ii, jj] = true)
    end
    τ2 = 8.0e4
    z_s2, H2 = solve(z_b2, τ2, mask2; dx = dx2, dy = dy2, mode = :flat, max_sweeps = 800, tol = 1e-9)
    r = GlenRheology(A = 1.0e-16, n = 3.0)
    sl = WeertmanSliding(β = 1.0e4, m = 3.0)
    vel = diva_velocity(z_s2, H2, τ2, mask2, dx2, dy2; rheology = r, sliding = sl)
    smbB = smb_from_velocity(vel, H2, z_s2, mask2, dx2, dy2)
    bal = balance_velocity(z_s2, H2, mask2, smbB, dx2, dy2)
    sel = [k for k in eachindex(z_s2) if mask2[k] &&
           6 < hypot(CartesianIndices(z_s2)[k][1] - 31.0, CartesianIndices(z_s2)[k][2] - 31.0) < 20]
    @test maximum(abs.(bal.speed[sel] .- vel.speed[sel])) / maximum(vel.speed[sel]) < 1e-9

    # Consistency diagnostic: stripping deformation off the balance speed recovers exactly
    # the sliding velocity that generated the SMB.
    ib = implied_basal_velocity(z_s2, H2, τ2, mask2, smbB, dx2, dy2; rheology = r)
    @test all(ib.u_basal[sel] .> 0)
    @test maximum(abs.(ib.u_basal[sel] .- basal_velocity(sl, τ2))) < 1e-6
end

@testset "SMB-driven reconstruction (feature 2)" begin
    relerr(H, H0, sel) = (d = sort(abs.(H[sel] .- H0[sel]) ./ H0[sel]); d[length(d) ÷ 2])

    nx, ny = 41, 41
    dx = dy = 2500.0
    z_b = zeros(nx, ny)
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 21.0, j - 21.0) <= 16.0 && (mask[i, j] = true)
    end
    τ_b = 8.0e4
    z_s0, H0 = solve(z_b, τ_b, mask; dx, dy, mode = :flat, max_sweeps = 600, tol = 1e-9)
    r = GlenRheology(A = 1.0e-16, n = 3.0)
    sl = WeertmanSliding(β = 1.0e4, m = 3.0)
    vel = diva_velocity(z_s0, H0, τ_b, mask, dx, dy; rheology = r, sliding = sl)
    smb = smb_from_velocity(vel, H0, z_s0, mask, dx, dy)
    sel = [k for k in eachindex(H0) if mask[k] &&
           hypot(CartesianIndices(H0)[k][1] - 21.0, CartesianIndices(H0)[k][2] - 21.0) < 12]

    # The plastic solution is a clean fixed point of the SMB-driven map: started there with
    # its own implied SMB, it does not move (the consistency the operators were built for).
    _, Hfp = diva_reconstruct(z_b, mask, smb, dx, dy; rheology = r, sliding = sl, H_init = H0)
    @test maximum(abs.(Hfp[mask] .- H0[mask])) < 1e-6

    # And it is recovered from a cold (flat) start by the under-relaxed iteration.
    Hg = fill(400.0, nx, ny)
    Hg[.!mask] .= 0
    _, Hc = diva_reconstruct(z_b, mask, smb, dx, dy; rheology = r, sliding = sl,
                             H_init = Hg, relax = 0.05, n_outer = 800)
    @test relerr(Hc, H0, sel) < 0.02
    @test isapprox(maximum(Hc), maximum(H0); rtol = 0.05)
end
