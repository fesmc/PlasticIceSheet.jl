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

@testset "1-D strip vs analytic Nye profile" begin
    # Ice strip between land margins at i=1 and i=nx, uniform in y. The eikonal
    # distance to the nearest margin gives H(d) = sqrt(2τ_b/(ρg) · d).
    nx, ny = 60, 8
    dx = dy = 1000.0
    τ_b = 1.0e5
    z_b = zeros(nx, ny)
    mask = trues(nx, ny)
    mask[1, :] .= false
    mask[nx, :] .= false

    z_s, H = solve(z_b, τ_b, mask; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-9)

    c = 2τ_b / ρg
    j = ny ÷ 2
    @testset "i=$i" for i in 3:(nx - 2)
        d = dx * min(i - 1, nx - i)
        @test H[i, j] ≈ sqrt(c * d) rtol = 1e-3
    end
    # Margins carry zero thickness.
    @test all(H[1, :] .== 0) && all(H[nx, :] .== 0)
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
    bc = PlasticIceSheet.flotation_thickness.(z_b_bump, Ref(p)) .^ 2
    R = PlasticIceSheet.godunov_residual(Hb .^ 2, fill(τ_b, nx, ny), z_b_bump, mask, bc,
                                         dx, dy, p, :full)
    @test maximum(abs, R) < 1e-2
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
    @test hypot(vel.ux[ci, cj], vel.uy[ci, cj]) == 0.0
    @test vel.ux[ci + 10, cj] > 0
    @test vel.uy[ci, cj + 10] > 0
    # Ice-free cells carry no velocity.
    @test all(vel.speed[.!mask] .== 0)

    # smb_from_velocity computes ∇·(ūH): a manufactured ūˣ = a·x, ūʸ = 0 field over
    # uniform H has constant divergence a·H.
    a = 1.0e-4
    Hc = 1500.0
    ux = [a * (i - 1) * dx for i in 1:nx, j in 1:ny]
    smb = smb_from_velocity((; ux = ux, uy = zeros(nx, ny)), fill(Hc, nx, ny), dx, dy)
    @test smb[30, 30] ≈ a * Hc rtol = 1e-6

    # Differentiable w.r.t. τ_b — the tuple works as an observation operator for inversion.
    obj(tb) = sum(diva_velocity(z_s, H, tb, mask, dx, dy; rheology = r, sliding = sl).speed)
    g = ForwardDiff.derivative(obj, τ_b)
    @test isfinite(g) && g > 0
end
