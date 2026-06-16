# Basal-shear-stress inversion demo.
#
# Recover a spatially-varying τ_b field from an "observed" ice surface by gradient descent,
# using the reverse-mode gradient from the implicit-differentiation extension. This is
# the modern replacement for ICESHEET's manual shear-stress-domain tuning.
#
# Run with:  julia --project=. examples/invert_tau.jl

using PlasticIceSheet
using ImplicitDifferentiation, ADTypes, ForwardDiff   # activate the reverse-mode ext
using Zygote

# --- synthetic truth --------------------------------------------------------------
nx, ny = 40, 40
dx = dy = 2000.0
z_b = [150.0 * sin(i / 11) * cos(j / 13) for i in 1:nx, j in 1:ny]   # bed, gentle relief
mask = [hypot(i - 20.5, j - 20.5) <= 16 for i in 1:nx, j in 1:ny]

# A "true" τ_b field: higher in one half, lower in the other (kPa → Pa).
τ_b_true = [60.0e3 + 80.0e3 * (i > nx ÷ 2) for i in 1:nx, j in 1:ny]

solver = (; dx, dy, mode = :flat, max_sweeps = 400, tol = 1e-10)
_, H_obs = solve(z_b, τ_b_true, mask; solver...)

# --- inversion --------------------------------------------------------------------
# Objective: mean-squared surface misfit over grounded ice.
function loss(τ_b)
    H = differentiable_thickness(τ_b, z_b, mask; solver...)
    return surface_misfit(H .+ z_b, H_obs .+ z_b, mask)
end

# Adam normalizes by the running gradient magnitude, so the step `lr` is in τ_b-units
# (Pa) and the wild scaling of the raw gradient (loss in m², τ_b in Pa) doesn't matter.
τ_b = fill(100.0e3, nx, ny)          # wrong, uniform first guess
lr = 2.0e3                          # ~2 kPa per step
β1, β2, ϵ = 0.9, 0.999, 1.0e-8
m = zero(τ_b); v = zero(τ_b)
println("iter   loss(m²)      ‖τ_b-τ_b_true‖/‖τ_b_true‖")
for it in 1:300
    L, back = Zygote.pullback(loss, τ_b)
    if it % 30 == 0 || it == 1
        relerr = sqrt(sum(abs2, (τ_b .- τ_b_true)[mask])) / sqrt(sum(abs2, τ_b_true[mask]))
        println(lpad(it, 4), "   ", rpad(round(L, sigdigits = 5), 11), "   ", round(relerr, sigdigits = 4))
    end
    g = back(1.0)[1]
    global m = β1 .* m .+ (1 - β1) .* g
    global v = β2 .* v .+ (1 - β2) .* g .^ 2
    m̂ = m ./ (1 - β1^it); v̂ = v ./ (1 - β2^it)
    global τ_b = clamp.(τ_b .- lr .* m̂ ./ (sqrt.(v̂) .+ ϵ), 1.0e3, 5.0e5)
end

# Report recovery over the two regions.
lo = sum(τ_b[1:nx÷2, :] .* mask[1:nx÷2, :]) / max(count(mask[1:nx÷2, :]), 1)
hi = sum(τ_b[nx÷2+1:end, :] .* mask[nx÷2+1:end, :]) / max(count(mask[nx÷2+1:end, :]), 1)
println("\nrecovered mean τ_b: low half ≈ $(round(lo/1e3,digits=1)) kPa (true 60), " *
        "high half ≈ $(round(hi/1e3,digits=1)) kPa (true 140)")
