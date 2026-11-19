# Coulomb yield law in mode :surface, τ_b = μ N_eff(H) (port of the Fortran test_coulomb):
# kernel properties, the exact flat-land profile, the wedge and Nye limits, a marine bed
# against an ODE, continuity of a retrograde trough behind a sill, AD, input checks.
# Uses `p` and `ρg` of runtests.jl.

using PlasticIceSheet: _coulomb_neff

const ρwi = p.ρ_w / p.ρ_i

@testset "Coulomb kernel: solves its equation, at or above flotation, monotone" begin
    law = CoulombYield(0.0; H_N = 1e12)
    @test godunov_coulomb(0.0, 1000.0, Inf, 1000.0, 0.0, 0.01, 0.0, law) ≈ 10.0 rtol = 1e-9
    @test godunov_coulomb(Inf, 1.0, Inf, 1.0, 0.0, 0.01, 0.0, CoulombYield(0.0; H_N = 1e3)) == Inf

    # √(T_a² + T_b²) − 1 at u (the neighbours clamped to the flotation surface).
    function term(u, a, d, zb, μ, hf, law)
        isfinite(a) || return 0.0
        ac = hf > 0 ? max(a, zb + hf) : a
        u > ac || return 0.0
        h = (u - zb + max(ac - zb, 0)) / 2
        s = h > 0 ? μ * _coulomb_neff(h, hf, law) / h : μ
        return ((u - ac) / (d * s))^2
    end
    eq(u, a, da, b, db, zb, μ, hf, law) = sqrt(term(u, a, da, zb, μ, hf, law) + term(u, b, db, zb, μ, hf, law)) - 1

    cnt = Ref(0)
    rnd() = (cnt[] += 1; x = sin(cnt[] * 12.9898) * 43758.5453; x - floor(x))
    nbad_eq = nbad_flt = nbad_inc = nbad_mono = 0
    for _ in 1:20_000
        zb = 2000 * (rnd() - 0.6)
        hf = ρwi * max(-zb, 0)
        a = zb + 3000 * (rnd() - 0.3)
        b = zb + 3000 * (rnd() - 0.3)
        rnd() < 0.1 && (b = Inf)
        μ = 1e-3 + 0.05 * rnd()
        da, db = 500 + 4e4 * rnd(), 500 + 4e4 * rnd()
        pp = rnd() < 0.3 ? 1.0 : rnd()
        H_N = 10 * 10^(3 * rnd())
        δ = rnd() < 0.5 ? 10^(-3 * rnd()) : 1 - 0.999 * rnd()
        law = CoulombYield(μ; H_N, p = pp, δ)
        u = godunov_coulomb(a, da, b, db, zb, μ, hf, law)
        u < zb + hf - 1e-9 * max(1, u - zb) && (nbad_flt += 1)
        lo = hf > 0 ? max(min(max(a, zb + hf), max(b, zb + hf)), zb + hf) : max(min(a, b), zb)
        e = eq(u, a, da, b, db, zb, μ, hf, law)
        dl = 1e-9 * max(1, u - zb)
        if abs(e) <= 1e-9 || (u == lo && e > 0)
        elseif !(eq(u - dl, a, da, b, db, zb, μ, hf, law) <= 0 <= eq(u + dl, a, da, b, db, zb, μ, hf, law))
            nbad_eq += 1
        end
        if hf > 0      # the floor: no zero increment at flotation
            dmin = isfinite(b) ? min(da, db) : da
            smin = μ * δ * H_N / (δ * (u - zb) + H_N)
            u - lo < (1 - 1e-9) * dmin * smin / sqrt(2) && (nbad_inc += 1)
        end
        tolm = 1e-9 * max(1, abs(u))
        godunov_coulomb(a + 1, da, b, db, zb, μ, hf, law) < u - tolm && (nbad_mono += 1)
        isfinite(b) && godunov_coulomb(a, da, b + 1, db, zb, μ, hf, law) < u - tolm && (nbad_mono += 1)
    end
    @test nbad_eq == 0
    @test nbad_flt == 0
    @test nbad_inc == 0
    @test nbad_mono == 0
end

# 1-D strip of 3 rows, ice-free first column (and last, if `both`).
function strip_mask(nx; both = true)
    m = trues(nx, 3); m[1, :] .= false
    both && (m[nx, :] .= false)
    return m
end
csolve(z_b, law, mask, dx; kw...) = solve(z_b, law, mask; dx, dy = dx, max_sweeps = 1000, tol = 1e-10, kw...)

@testset "Coulomb: flat land H²/2 + H_N H = μ H_N x (exact)" begin
    L, μ = 200e3, 0.01
    emax = 0.0
    for dx in (8000.0, 2000.0, 500.0), H_N in (100.0, 1000.0, 5000.0)
        nx = round(Int, L / dx) + 1
        _, H = csolve(zeros(nx, 3), CoulombYield(μ; H_N), strip_mask(nx), dx; z_ss = -1e4)
        for i in 2:nx-1
            x = dx * (min(i - 1, nx - i) - 0.5)
            ex = -H_N + sqrt(H_N^2 + 2μ * H_N * x)
            emax = max(emax, abs(H[i, 2] - ex) / ex)
        end
    end
    @test emax < 1e-10
end

@testset "Coulomb: wedge limit (H ≪ H_N), slope μ on any bed" begin
    dx, μ, nx = 2000.0, 0.01, 101
    for step in (false, true)
        z_b = [0.004 * dx * (i - 1) + (step && i > 40 ? 300.0 : 0.0) for i in 1:nx, _ in 1:3]
        law = CoulombYield(μ; H_N = 1e15)
        mask = strip_mask(nx; both = false)
        z_s, H = csolve(z_b, law, mask, dx; z_ss = -1e4)
        τ = coulomb_taub(H, z_b, mask, law; z_ss = -1e4)
        @test maximum(abs(z_s[i, 2] - (z_b[1, 2] + μ * dx * (i - 1.5))) for i in 2:nx) < 1e-6
        @test maximum(abs(τ[i, 2] / (μ * ρg * H[i, 2]) - 1) for i in 2:nx) < 1e-8
    end
end

@testset "Coulomb: Nye limit (H ≫ H_N) equals the constant-τ_b surface solve" begin
    n, dx, H_N = 80, 1000.0, 1e-5
    xy = [((i - 0.5) * dx - 40e3, (j - 0.5) * dx - 40e3) for i in 1:n, j in 1:n]
    z_b = [300 * exp(-(x^2 + y^2) / (2 * 2e4^2)) for (x, y) in xy]
    τ_b = [1e5 * (1 + 0.3 * cos(j / 9)) for i in 1:n, j in 1:n]
    sd = [30e3 - hypot(x - 700, y + 300) for (x, y) in xy]
    mask = sd .> 0
    frac = margin_frac_from_sdist(sd, mask, dx, dx)
    o = (; dx, dy = dx, margin = frac, z_ss = -1e4, max_sweeps = 1000, tol = 1e-10)
    _, Hs = solve(z_b, τ_b, mask; o...)
    law = CoulombYield(τ_b ./ (ρg * H_N); H_N)
    _, Hc = solve(z_b, law, mask; o...)
    @test maximum(abs.(Hc .- Hs)[mask] ./ max.(Hs[mask], 1)) < 1e-5
    τ = coulomb_taub(Hc, z_b, mask, law; z_ss = -1e4)
    @test maximum(abs.(τ .- τ_b)[mask] ./ τ_b[mask]) < 1e-5
end

@testset "Coulomb: marine bed vs ODE, thins to flotation, floor stress at the margin" begin
    L, s, z0, μ, hs = 300e3, 0.004, -800.0, 0.01, 5.0
    law = CoulombYield(μ; H_N = 1000.0, p = 1.0, δ = 0.05)
    hf0 = ρwi * max(-z0, 0)
    τ_floor = μ * ρg * _coulomb_neff(hf0, hf0, law)
    slope(zb, zs) = (h = zs - zb; h > 0 ? μ * _coulomb_neff(h, ρwi * max(-zb, 0), law) / h : 0.0)
    nref = round(Int, L / hs)
    zref = zeros(nref + 1); zref[1] = z0 + hf0
    for i in 1:nref
        x, zs = (i - 1) * hs, zref[i]
        k1 = slope(z0 + s * x, zs)
        k2 = slope(z0 + s * (x + hs / 2), zs + hs / 2 * k1)
        k3 = slope(z0 + s * (x + hs / 2), zs + hs / 2 * k2)
        k4 = slope(z0 + s * (x + hs), zs + hs * k3)
        zref[i + 1] = zs + hs * (k1 + 2k2 + 2k3 + k4) / 6
    end
    err, gap, τ1, grounded = Float64[], Float64[], Float64[], true
    for dx in (8000.0, 4000.0, 2000.0, 1000.0)
        nx = round(Int, L / dx) + 1
        z_b = [z0 + s * dx * (i - 1.5) for i in 1:nx, _ in 1:3]
        hfl = ρwi .* max.(-z_b, 0)
        mask = strip_mask(nx; both = false)
        z_s, H = csolve(z_b, law, mask, dx)
        τ = coulomb_taub(H, z_b, mask, law)
        grounded &= all(H[2:end, 2] .>= hfl[2:end, 2] .- 1e-9)
        push!(err, maximum(abs(z_s[i, 2] - zref[round(Int, dx * (i - 1.5) / hs) + 1]) for i in 2:nx))
        push!(gap, H[2, 2] - hfl[2, 2])
        push!(τ1, τ[2, 2] / τ_floor)
    end
    @test grounded
    @test issorted(err; rev = true) && allunique(err)
    @test issorted(gap; rev = true) && allunique(gap)
    @test abs(τ1[end] - 1) < abs(τ1[1] - 1) && abs(τ1[end] - 1) < 0.01
end

@testset "Coulomb: retrograde trough behind a sill, continuous in sea level and sill depth" begin
    nx, ny, dx, ns = 70, 40, 5000.0, 41
    mask = trues(nx, ny); mask[[1, nx], :] .= false; mask[:, [1, ny]] .= false
    law = CoulombYield(0.01; H_N = 1000.0, p = 1.0, δ = 0.05)
    function bed(zsill)
        [begin
            x, y = (i - 1.5) * dx, (j - (ny + 1) / 2) * dx
            z = x < 20e3  ? zsill :
                x < 150e3 ? zsill + (-1200 - zsill) * (x - 20e3) / 130e3 :
                            -1200 + 1600 * min((x - 150e3) / 190e3, 1)
            z + (300 - z) * min((abs(y) / 30e3)^4, 1)
         end for i in 1:nx, j in 1:ny]
    end
    for scan in (:sea_level, :sill)
        dz = scan === :sea_level ? 0.1 : 1.0
        Hp, Vp, dhmax, dV = nothing, 0.0, 0.0, Float64[]
        for k in 1:ns
            z_ss = scan === :sea_level ? dz * (k - 1) : 0.0
            zsill = scan === :sill ? -300 - dz * (k - 1) : -300.0
            _, H = csolve(bed(zsill), law, mask, dx; z_ss)
            V = sum(H[mask]) * dx^2
            if Hp !== nothing
                dhmax = max(dhmax, maximum(abs.(H .- Hp)[mask]) / dz)
                push!(dV, abs(V - Vp))
            end
            Hp, Vp = H, V
        end
        med = sort(dV)[(length(dV) + 1) ÷ 2]
        @test dhmax <= 25
        @test maximum(dV) <= 4 * med
    end
end

@testset "Coulomb: forward-mode derivative in μ, inputs" begin
    n, dx = 40, 4000.0
    xy = [((i - 0.5) * dx - n * dx / 2, (j - 0.5) * dx - n * dx / 2) for i in 1:n, j in 1:n]
    z_b = [600 * exp(-(x^2 + y^2) / (2 * 3e4^2)) - 400 + 80 * sin(i / 7) for ((x, y), i) in zip(xy, repeat(1:n, 1, n))]
    mask = [hypot(x + 3000, y - 1700) < 0.37 * n * dx for (x, y) in xy]
    vol(μ) = ice_volume(last(solve(z_b, CoulombYield(μ; H_N = 800.0, p = 0.5), mask; dx, dy = dx,
                                   max_sweeps = 400, tol = 1e-12)), dx, dx)
    g = ForwardDiff.derivative(vol, 0.01)
    h = 1e-6
    @test g ≈ (vol(0.01 + h) - vol(0.01 - h)) / 2h rtol = 1e-4

    @test_throws ArgumentError CoulombYield(0.01; H_N = 1000.0, p = 1.5)
    @test_throws ArgumentError CoulombYield(0.01; H_N = 0.0)
    @test_throws ArgumentError CoulombYield(0.01; H_N = 1000.0, δ = 0.0)
    @test_throws ArgumentError CoulombYield(0.01; H_N = 1000.0, δ = -0.1)
    @test_throws ArgumentError CoulombYield(0.01; H_N = 1000.0, δ = 1.5)
    law = CoulombYield(0.01; H_N = 1000.0)
    @test_throws ArgumentError solve(z_b, law, mask; dx, dy = dx, mode = :flat)
    μ = fill(0.01, n, n); μ[n ÷ 2, n ÷ 2] = -1
    @test_throws ArgumentError solve(z_b, CoulombYield(μ; H_N = 1000.0), mask; dx, dy = dx)
    @test all(isfinite, last(solve(z_b, CoulombYield(0.01; H_N = 1000.0, δ = 1.0), mask; dx, dy = dx)))
end
