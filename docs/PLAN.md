# PlasticIceSheet.jl — design & plan

A small, AD-friendly Julia package that reconstructs a perfectly-plastic, steady-state
ice-sheet surface from three inputs — a **grounded-ice margin/mask**, a **basal shear
stress field** `τ_b(x,y)`, and a **bed elevation** `z_b(x,y)` — by solving the governing
equation as a static Hamilton–Jacobi (eikonal-type) problem on a grid.

It is intended as a *complementary first-guess tool* alongside a full dynamic model
(Yelmo) and an independent 3-D GIA / sea-level solver — e.g. to seed spin-up, to
provide an elevation field for an SMB model, or as a relaxation target. It is **not**
a dynamic model and carries no transient, thermomechanics, or ice-shelf physics.

## Origin

This package is a deliberate simplification and modernization of **ICESHEET 2.0** by
Evan J. Gowan et al.:

- Source code (Fortran): <https://github.com/evangowan/icesheet>
- Local copy analyzed: `~/models/icesheet`
- Description paper: Gowan, E. J., Tregoning, P., Purcell, A., Lea, J., Fransner, O. J.,
  Noormets, R., and Dowdeswell, J. A. (2016). *ICESHEET 1.0: a program to produce
  paleo-ice sheet reconstructions with minimal assumptions.* Geoscientific Model
  Development, 9(5), 1673–1682. <https://doi.org/10.5194/gmd-9-1673-2016>
  (local: `~/models/icesheet/gmd-9-1673-2016.xml`)
- v2.0 / PaleoMIST application: Gowan et al. (2021), Nature Communications 12, 1199.
  <https://doi.org/10.1038/s41467-021-21469-w>

ICESHEET is GPL-3.0. This is a clean-room reimplementation of the *physics* (one
published equation, below); none of the Fortran source is copied. This package is
**MIT-licensed**.

## The physics we keep

The entire scientific content of ICESHEET is one equation. Under the perfectly-plastic,
steady-state assumption the basal shear stress `τ_b` balances the driving stress
(paper Eq. 5):

```
τ_b = ρ_i g H |∇z_s|,    H = z_s − z_b ≥ 0
```

with `z_s` the ice surface elevation, `z_b` the bed elevation, `H` the thickness, `τ_b` the
basal shear stress, `ρ_i` ice density, `g` gravity. With `τ_b` prescribed, rearrange for
the surface slope to get the form we solve:

```
|∇z_s| = τ_b / (ρ_i g H) = τ_b / (ρ_i g (z_s − z_b))
```

Boundary condition: `H = 0` (`z_s = z_b`) at the margin, plus a marine flotation thickness
`H_flt = (ρ_w/ρ_i)(z_ss − z_b)` where the grounded margin sits below sea level `z_ss`.

This is a **static Hamilton–Jacobi equation**. The viscosity solution of an upwind grid
scheme automatically produces the correct surface where flow would converge — so
**saddles, domes, and coalescing ice caps emerge for free**, with no flowline tracing.

### Two solve modes

1. **`:flat` — flat-bed eikonal (simple case).** Neglect bed slope in the flux
   (`|∇z_s| ≈ |∇H|`). Substituting `u = H²` gives a textbook eikonal:

   ```
   |∇u| = 2τ_b / (ρ_i g),     u = 0 at the margin
   ```

   Solve, then `H = √u`, `z_s = H + z_b`. Exact eikonal; ideal when bed relief ≪ ice
   thickness (continental interiors). The paper's own sensitivity tests indicate the
   bed-gradient terms are second-order except in mountainous terrain.

2. **`:full` — full Hamilton–Jacobi (default).** Accounts for the bed slope `∇z_b`. The
   effective right-hand side (in the `w = H²` variable) is

   ```
   |∇w| = √(G² − 4√w(∇z_b·∇w) − 4w|∇z_b|²),    G = 2τ_b / (ρ_i g)
   ```

   which depends on `w` through `∇w`. The frozen right-hand side is re-solved as a
   constant-RHS eikonal and iterated to a fixed point (see Implementation notes — a
   **damped** Picard iteration is required, as the undamped map oscillates). Correct over
   arbitrary bed relief; reduces exactly to `:flat` when `∇z_b = 0`.

### Numerics

- **Fast sweeping** (Zhao 2005): Gauss–Seidel with the Godunov upwind Hamiltonian,
  alternating the four diagonal sweep orderings, iterated to a tolerance on the max
  update. O(N) per sweep, a handful of sweeps to converge.
- Godunov update at node `(i,j)` with spacing `h`, neighbor minima `a, b`:
  - if `|a − b| ≥ F h`: `u = min(a, b) + F h`
  - else: `u = (a + b + √(2 F² h² − (a − b)²)) / 2`
  where `F = G = 2τ_b/(ρ_i g)` (flat mode) or the `w`-dependent effective RHS above (`:full`).

This replaces the ~2,600-line recursive flowline/crossover/saddle machinery of ICESHEET
(`find_flowline_fisher_adaptive_4.f90`) with a kernel on the order of ~150–250 lines.

## What we deliberately drop (vs. ICESHEET)

Given the intended use (first guess; external GIA; Yelmo for dynamics), we drop:

- **Method-of-characteristics flowline tracing** and all of its bookkeeping: crossover
  detection / motorcycle-graph algorithm, saddle & dome detection, polygon splitting,
  recursion, adaptive contour resampling. Subsumed by the grid HJ solver.
- **The GIA / sea-level iteration loop** (CALSEA/SELEN coupling, `global/deform/`,
  `selen_format.sh`, …). The package consumes a bed `z_b`; the user owns GIA.
- **The GMT projection/preprocessing pipeline** (`run.sh`, `prepare_icesheet.sh`,
  `create_ss_grid.f90`, shapefile→τ_b domains, `reduce_dem`, `nearest_int`, `bicubic`,
  `diff_map`) and the hand-rolled GMT binary-record grid reader (`grids.f90`). Replaced
  by in-memory arrays now, NetCDF I/O via a package extension later.
- **The offshore boundary Nye-smoothing pre-pass** — the HJ scheme enforces the maximum
  surface slope intrinsically.
- **The iterative τ_b-tuning shell loop** (`adjust_ss`) — replaced by gradient-based
  optimization of `τ_b` enabled by AD (see below).

## What we keep / add

- Inputs: grounded-ice mask, bed `z_b`, shear-stress field `τ_b` (or a constant / a
  low-dimensional parameterization).
- Plastic physics + marine flotation BC at the grounding line.
- Gridded outputs: surface `z_s`, thickness `H`, plus reductions (volume, area).
- **AD-friendliness**: the solver is written so that a scalar loss on the output
  (e.g. misfit to a target surface, or to Yelmo) can be differentiated w.r.t. `τ_b`,
  enabling gradient-based inversion of basal shear stress — the modern replacement for
  ICESHEET's manual domain tuning.

## Caveats (carried from the physics)

- Steady-state geometry only; biased thick/peaked where the real ice was streaming or
  dynamic (low effective driving stress the plastic assumption can't see).
- `τ_b` is the free field carrying essentially all the uncertainty. For a first guess a
  constant (~50–100 kPa) or a bed-keyed value is fine; don't over-interpret as a
  constrained reconstruction.

## Open questions / decisions

- **AD backend & strategy** (forward-through-solver vs. implicit/adjoint at the fixed
  point; Enzyme vs. ForwardDiff vs. Zygote) — depends on whether `τ_b` is a full per-cell
  field or a low-dimensional parameterization. *To be decided before the core lands.*
- ~~License~~ — decided: MIT (clean-room physics reimplementation; ICESHEET is GPL-3.0).
- NetCDF I/O as a package extension (`ext/`).

## Implementation notes (v0.1)

- **Both modes solve for `w = H²`**, not the surface `z_s` directly: the driving slope
  `τ_b/(ρ_i g H)` blows up at the margin (`H→0`) and biases a direct-`z_s` solve low. `:flat`
  is the plain eikonal `|∇w| = 2τ_b/(ρ_i g)`; `:full` adds the bed-gradient correction.
- **`:full` damping.** The `:full` effective RHS `√(G² − 4√w(∇z_b·∇w) − 4w|∇z_b|²)` carries `∇w`,
  so "freeze RHS, re-solve" is not contractive — undamped it oscillates over real relief.
  It is solved by a **damped (under-relaxed) Picard iteration** (`relax`, default 0.5).
  Very steep beds (relief ≈ ice thickness over a few cells) may still not converge.
- **Reverse-mode AD** (`differentiable_thickness`, extension keyed on
  `ImplicitDifferentiation`): implicit differentiation of the converged Godunov residual
  `godunov_residual(w, τ_b) = 0`. `:flat` gradients are exact to machine precision; `:full`
  gradients are exact once the damped forward solve has driven the residual to ≈ 0 (hence
  the firmer `n_outer`/`outer_tol` defaults on that path). The forward fast sweep is a
  black box; only the residual is differentiated (ForwardDiff), so AD cost is independent
  of sweep/outer-iteration count and scales to a full per-cell `τ_b` field.
- **NaN hygiene**: `√w` is floored at `H_min²` wherever it is differentiated (`sqrt(0)`
  has an infinite derivative that otherwise poisons the reverse pass).

## Status

- [x] Package scaffolding, plan doc
- [x] Core HJ / eikonal fast-sweeping solver (`:flat`, `:full`)
- [x] Forward-mode (ForwardDiff) τ_b-gradients, validated vs finite differences
- [x] Reverse-mode (implicit-diff) full-field τ_b-gradients, validated vs ForwardDiff/FD
- [x] Tests: Nye analytic profile, flat≡hj on flat bed, residual convergence, AD
- [x] NetCDF I/O extension
- [x] τ_b-inversion example (`examples/invert_tau.jl`)
- [ ] Next: implicit-diff iterative linear solver tuning for very large grids; better
      `:full` convergence on steep beds (e.g. Anderson acceleration / a true HJ Godunov for
      the bed-coupled term); coordinate/projection handling for real datasets
