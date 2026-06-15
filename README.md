# PlasticIceSheet.jl

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://fesmc.github.io/PlasticIceSheet.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast, AD-friendly reconstruction of perfectly-plastic, steady-state ice sheets.

📖 **Documentation: <https://fesmc.github.io/PlasticIceSheet.jl/>** — including a full
[explanation of the algorithms](https://fesmc.github.io/PlasticIceSheet.jl/algorithms.html).

Given a grounded-ice **margin/mask**, a **bed topography** `B`, and a **basal shear
stress** field `τ`, it returns a glaciologically plausible ice **surface** `E` and
**thickness** `H` — with no time-stepping, no climate, and no ice dynamics. It is meant
as a *complementary first-guess tool* next to a full dynamic model (e.g. Yelmo) and an
independent GIA/sea-level solver: to seed spin-up, force an SMB model, or serve as a
relaxation target.

It is a clean-room simplification and modernization of **ICESHEET 2.0** by Evan Gowan and
colleagues — [source code](https://github.com/evangowan/icesheet), the method's
description paper ([Gowan et al., 2016, *Geosci. Model Dev.*](https://doi.org/10.5194/gmd-9-1673-2016)),
and its global application ([Gowan et al., 2021, *Nat. Commun.*](https://doi.org/10.1038/s41467-021-21469-w)).

The physics is the single plastic-surface equation, for surface elevation `z_s`, bed
elevation `z_b`, and thickness `H = z_s − z_b`:

```
|∇z_s| = τ / (ρ_i g (z_s − z_b)),     H = z_s − z_b
```

solved as a static Hamilton–Jacobi (eikonal) problem with **fast sweeping**. Saddles,
domes, and coalescing ice caps emerge automatically from the upwind viscosity solution
— no flowline tracing, crossover detection, or saddle bookkeeping.

See [`docs/PLAN.md`](docs/PLAN.md) for the full rationale of what was kept, simplified, and
dropped relative to ICESHEET.

## Usage

```julia
using PlasticIceSheet

nx, ny = 100, 100
dx = dy = 2000.0                     # grid spacing (m)
z_b  = zeros(nx, ny)                 # bed elevation (m)
mask = [hypot(i-50.5, j-50.5) <= 40 for i in 1:nx, j in 1:ny]   # grounded ice
τ    = 1.0e5                         # basal shear stress (Pa); scalar or per-cell field

z_s, H = solve(z_b, τ, mask; dx, dy, mode = :hj)   # :hj (default) or :flat
```

### Modes

- `:hj`  — full Hamilton–Jacobi over arbitrary bed relief (default). An outer thickness
  fixed-point holds the driving-stress slope `τ/(ρ_i g H)` and solves the surface
  eikonal, iterating `H = z_s − z_b` to convergence.
- `:flat` — flat-bed eikonal (`|∇H| ≈ |∇z_s|`): solve `|∇H²| = 2τ/(ρ_i g)`, then drape on
  the bed. Exact when bed relief ≪ ice thickness; cheapest.

Marine margins (bed below sea level `z_ss`) are grounded at the flotation thickness `H_flt`.

### Differentiating w.r.t. `τ` (basal-shear-stress inversion)

The solver is a pure function of `τ`, so any scalar objective is differentiable.

**Forward mode** (`ForwardDiff`) — ideal for a scalar or low-dimensional `τ`
parameterization:

```julia
using ForwardDiff
loss(τ) = surface_misfit(first(solve(z_b, τ, mask; dx, dy)), z_s_obs, mask)
g = ForwardDiff.derivative(loss, 1.0e5)
```

**Reverse mode** for a full per-cell `τ` field — implicit differentiation at the
converged Godunov fixed point, one linear solve regardless of iteration count. Load the
extension's triggers, then differentiate `differentiable_thickness` with any reverse-mode
AD (e.g. Zygote):

```julia
using ImplicitDifferentiation, ADTypes, ForwardDiff, Zygote
loss(τ) = surface_misfit(differentiable_thickness(τ, z_b, mask; dx, dy) .+ z_b, z_s_obs, mask)
g = Zygote.gradient(loss, τ)[1]          # gradient w.r.t. the whole τ field
```

See [`examples/invert_tau.jl`](examples/invert_tau.jl) for a full Adam-based recovery of
a spatially-varying `τ` from an observed surface.

### NetCDF I/O

```julia
using NCDatasets
inp = load_plastic_inputs("inputs.nc")             # (; z_b, τ, mask, dx, dy, x, y)
z_s, H = solve(inp.z_b, inp.τ, inp.mask; inp.dx, inp.dy)
save_reconstruction("out.nc", z_s, H; x = inp.x, y = inp.y)
```

## Notes & caveats

- The `:hj` bed-gradient correction is solved by a **damped** fixed-point iteration
  (`relax`, default 0.5); the undamped map oscillates over strong relief. Very steep beds
  (relief approaching the ice thickness over a few cells) may not converge — lower
  `relax`, raise `n_outer`, refine the grid, or fall back to `:flat`.
- The default reverse-mode linear solver is direct (robust for modest grids); pass an
  iterative solver for very large fields (see `differentiable_thickness` docstring).

## Status

v0.1: core solver (`:flat`, `:hj`), diagnostics, analytic (Nye-profile) tests,
ForwardDiff- and reverse-mode-validated `τ` gradients (implicit differentiation),
a NetCDF I/O extension, and a working inversion example.
