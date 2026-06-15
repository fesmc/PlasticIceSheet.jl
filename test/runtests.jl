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
    # distance to the nearest margin gives H(d) = sqrt(2τ/(ρg) · d).
    nx, ny = 60, 8
    dx = dy = 1000.0
    τ = 1.0e5
    z_b = zeros(nx, ny)
    mask = trues(nx, ny)
    mask[1, :] .= false
    mask[nx, :] .= false

    z_s, H = solve(z_b, τ, mask; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-9)

    c = 2τ / ρg
    j = ny ÷ 2
    @testset "i=$i" for i in 3:(nx - 2)
        d = dx * min(i - 1, nx - i)
        @test H[i, j] ≈ sqrt(c * d) rtol = 1e-3
    end
    # Margins carry zero thickness.
    @test all(H[1, :] .== 0) && all(H[nx, :] .== 0)
end

@testset "flat and hj agree on a flat bed" begin
    nx, ny = 50, 50
    dx = dy = 2000.0
    τ = 0.8e5
    z_b = fill(-50.0, nx, ny)          # flat, slightly below 0 but constant
    mask = falses(nx, ny)
    cx, cy, R = 25.5, 25.5, 18.0
    for j in 1:ny, i in 1:nx
        if hypot(i - cx, j - cy) <= R
            mask[i, j] = true
        end
    end
    z_s_f, Hf = solve(z_b, τ, mask; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-9)
    z_s_h, Hh = solve(z_b, τ, mask; dx, dy, mode = :hj, max_sweeps = 400, tol = 1e-9,
                      n_outer = 40, outer_tol = 1e-5)
    # On a constant bed the two formulations are mathematically identical.
    @test maximum(abs.(Hf .- Hh)) < 2.0   # metres
end

@testset "hj responds to bed relief (damped fixed point converges)" begin
    nx, ny = 40, 40
    dx = dy = 2000.0
    τ = 1.0e5
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 20.5, j - 20.5) <= 15 && (mask[i, j] = true)
    end
    z_b_flat = zeros(nx, ny)
    # broad, smooth bedrock high (~300 m, ~28 km scale) — within the convergence regime
    z_b_bump = [300.0 * exp(-((i - 20.5)^2 + (j - 20.5)^2) / 200) for i in 1:nx, j in 1:ny]
    hj = (; dx, dy, mode = :hj, n_outer = 600, outer_tol = 1e-7, relax = 0.4)

    _, H0 = solve(z_b_flat, τ, mask; hj...)
    z_s_b, Hb = solve(z_b_bump, τ, mask; hj...)
    # A central bedrock high lifts the surface above the flat-bed reconstruction.
    @test z_s_b[20, 20] > H0[20, 20]
    # The damped :hj fixed point is genuinely reached (residual ≈ 0).
    bc = PlasticIceSheet.flotation_thickness.(z_b_bump, Ref(p)) .^ 2
    R = PlasticIceSheet.godunov_residual(Hb .^ 2, fill(τ, nx, ny), z_b_bump, mask, bc,
                                         dx, dy, p, :hj)
    @test maximum(abs, R) < 1e-2
end

@testset "AD: ∂(volume)/∂τ via ForwardDiff matches finite difference" begin
    nx, ny = 36, 36
    dx = dy = 2000.0
    z_b = zeros(nx, ny)
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 18.5, j - 18.5) <= 13 && (mask[i, j] = true)
    end

    # Scalar objective: total volume as a function of uniform τ.
    function vol_of_τ(τ)
        _, H = solve(z_b, τ, mask; dx, dy, mode = :hj, max_sweeps = 300,
                     tol = 1e-8, n_outer = 40, outer_tol = 1e-5)
        return ice_volume(H, dx, dy)
    end

    τ0 = 1.0e5
    g_ad = ForwardDiff.derivative(vol_of_τ, τ0)
    h = 1.0e2
    g_fd = (vol_of_τ(τ0 + h) - vol_of_τ(τ0 - h)) / (2h)
    @test isfinite(g_ad)
    @test g_ad ≈ g_fd rtol = 1e-3

    # Gradient w.r.t. a full per-cell τ field also flows (a few partials checked).
    τfield = fill(1.0e5, nx, ny)
    function vol_of_field(τv)
        _, H = solve(z_b, τv, mask; dx, dy, mode = :hj, max_sweeps = 300,
                     tol = 1e-8, n_outer = 40, outer_tol = 1e-5)
        return ice_volume(H, dx, dy)
    end
    grad = ForwardDiff.gradient(vol_of_field, τfield)
    @test all(isfinite, grad)
    @test grad[18, 18] > 0          # raising local τ thickens the ice ⇒ more volume
end

@testset "reverse-mode τ-field gradient (implicit diff) vs ForwardDiff" begin
    nx, ny = 24, 24
    dx = dy = 2000.0
    # smooth, moderate bed relief (~300 m) so the damped :hj fixed point converges
    z_b = [300.0 * sin(i / 9) * cos(j / 11) for i in 1:nx, j in 1:ny]
    mask = falses(nx, ny)
    for j in 1:ny, i in 1:nx
        hypot(i - 12.5, j - 12.5) <= 9 && (mask[i, j] = true)
    end
    τ0 = fill(1.0e5, nx, ny)
    sk = (; dx, dy, mode = :hj, max_sweeps = 400, tol = 1e-10,
          n_outer = 400, outer_tol = 1e-7, relax = 0.5)

    # Reverse mode: Zygote → ImplicitDifferentiation rrule → implicit-function gradient.
    rev_loss(τ) = ice_volume(differentiable_thickness(τ, z_b, mask; sk...), dx, dy)
    g_rev = Zygote.gradient(rev_loss, τ0)[1]

    # Forward-mode oracle through the plain (differentiate-through) solver, which itself
    # matches finite differences; at a converged fixed point the two must coincide.
    fwd_loss(τ) = ice_volume(last(solve(z_b, τ, mask; sk...)), dx, dy)
    g_fwd = ForwardDiff.gradient(fwd_loss, τ0)

    @test all(isfinite, g_rev)
    @test size(g_rev) == size(τ0)
    @test g_rev ≈ g_fwd rtol = 1e-4
    @test sum(g_rev) > 0          # raising τ everywhere thickens the ice ⇒ more volume

    # Also confirm the :flat reverse-mode gradient is exact (clean eikonal residual).
    skf = (; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-10)
    gf_rev = Zygote.gradient(τ -> ice_volume(differentiable_thickness(τ, z_b, mask; skf...), dx, dy), τ0)[1]
    gf_fwd = ForwardDiff.gradient(τ -> ice_volume(last(solve(z_b, τ, mask; skf...)), dx, dy), τ0)
    @test gf_rev ≈ gf_fwd rtol = 1e-8
end

@testset "NetCDF I/O round trip" begin
    nx, ny = 10, 8
    dx = dy = 1000.0
    z_b = zeros(nx, ny); τ = fill(1.0e5, nx, ny)
    mask = trues(nx, ny); mask[1, :] .= false; mask[nx, :] .= false
    z_s, H = solve(z_b, τ, mask; dx, dy, mode = :flat)

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
        defVar(ds, "tau", τ, ("x", "y"))
        defVar(ds, "mask", Int.(mask), ("x", "y"))
    end
    got = load_plastic_inputs(inp)
    @test got.dx == dx && got.dy == dy
    @test size(got.z_b) == (nx, ny)
    @test got.mask == mask
    @test got.τ ≈ τ
    rm(out); rm(inp)
end
