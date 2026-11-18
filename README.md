# PlasticIceSheet.jl

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://fesmc.github.io/PlasticIceSheet.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast, AD-friendly reconstruction of perfectly plastic, steady-state ice sheets.

Given a grounded-ice mask, a bed elevation `z_b` and a basal shear stress `τ_b`, it returns
the ice surface `z_s` and thickness `H`, with no time stepping, no climate and no ice
dynamics. It solves the plastic surface equation

```
|∇z_s| = τ_b / (ρ_i g (z_s − z_b))
```

as a static Hamilton–Jacobi problem by fast sweeping, a simplification of ICESHEET 2.0
(Gowan et al., [2016](https://doi.org/10.5194/gmd-9-1673-2016),
[2021](https://doi.org/10.1038/s41467-021-21469-w)). A DIVA closure adds velocity and
surface mass balance, and an SMB-driven reconstruction solves steady mass conservation on the
same margin. The result is differentiable with respect to `τ_b` for inversion.

📖 **Documentation: <https://fesmc.github.io/PlasticIceSheet.jl/>**

## Quick start

```julia
using Pkg
Pkg.add(url = "https://github.com/fesmc/PlasticIceSheet.jl")

using PlasticIceSheet

nx, ny = 100, 100
dx = dy = 2000.0                     # grid spacing (m)
z_b  = zeros(nx, ny)                 # bed elevation (m)
mask = [hypot(i - 50.5, j - 50.5) <= 40 for i in 1:nx, j in 1:ny]   # grounded ice
τ_b  = 1.0e5                         # basal shear stress (Pa); scalar or field

z_s, H = solve(z_b, τ_b, mask; dx, dy)
```

A dependency-free Fortran port for coupling into Fortran models is available as
[PlasticIceSheet](https://github.com/fesmc/PlasticIceSheet).
