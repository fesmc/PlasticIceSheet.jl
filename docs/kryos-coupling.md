# Calling PlasticIceSheet.jl from kryos (Fortran)

Status: PAUSED (scope change). Forward-solve routes are prototyped and measured; the adjoint at
kryos grid sizes and the tau_b-optimisation designs are not. Prototype: `kryos-embed/`.

Setup: Levante, Julia 1.12.6 (juliaup), ifx 2023.2.1, AMD EPYC 7763, 1 Julia thread, synthetic
case (ellipse mask, dx = 32 km, same formulas in Fortran and Julia). All routes reproduce the
Julia-native result to the printed digits (flat 200x300: Hmax 7080.658, Hmean 3978.163627).

## Routes

| Route | Works? | Startup | First call | Steady 200x300 flat / full | Memory | AD | Verdict |
|---|---|---|---|---|---|---|---|
| native Julia (reference) | yes | - | 4.0 s (JIT) | 4.9 ms / 0.15-0.35 s | 600 MB | yes | baseline |
| a. embed libjulia, C function pointers (`@cfunction`, no Julia objects cross) | yes | default: jl_init 0.15 s + `using` 1.6-3.8 s; sysimage: 0.5-2.2 s total | default 2.8-3.1 s (JIT); sysimage 7 ms | 4.6 ms / 0.5 s (sysimage, noisy) | 360-440 MB | yes (see below) | works; best if AD needed |
| b. `juliac --trim=safe` shared lib, `@ccallable` | yes, forward only | ~0.1 s | 0.05-0.25 s | 4.7 ms / 0.15 s | 64-100 MB | no (Zygote/ImplicitDifferentiation not trim-compatible, not tried) | best for forward-only production: 2.6 MB lib, no Julia install at runtime beyond libjulia, built in 12 s |
| c. persistent worker over FIFOs + /dev/shm files | yes | 1.8 s server warmup (once) | 1.8-2.7 s | 6.6 ms / 0.2-0.65 s round trip | separate process | yes (worker can run anything) | robust, decoupled, simplest to deploy; I/O adds ~2 ms at 200x300 |
| c'. one-shot `julia` per call | measured startup only | 20.6 s default (incl. precompile once), 0.68 s with sysimage | - | - | - | yes | only for rare calls |

400x400: flat 12.4 ms, full 0.5-2.0 s (varies by node load; the full mode time is noisy, repeat before quoting).

Details:
- a. `jl_init` / `jl_init_with_image` once; `using KryosEmbed`; `jl_eval_string("KryosEmbed.entry_solve()")` returns a `@cfunction` pointer, called via `c_f_procpointer`. Arrays are Fortran memory wrapped with `unsafe_wrap` (no copy), exceptions are caught in Julia and returned as a code plus `lasterr()`, so no GC rooting is needed on the Fortran side. Custom sysimage (PackageCompiler): 377 MB, 193 s build, 14 GB peak RAM (needs `--mem=32G`; the default shared allocation was OOM-killed). The sysimage removes the JIT first-call latency (2.8 s to 7 ms).
- b. Needed one fix in the package: `solve` returned `Union{Matrix{Any},Matrix{Float64}}` because `@.` dotted `zero(eltype(w))`, which also allocated temporaries; fixed on this branch (commit on `kryos-embed`), after which `--trim=safe` verifies. The lib initialises Julia automatically on first call (no `jl_init` needed from Fortran). It still links libjulia and libjulia-internal at runtime (rpath).
- Gradient through route a (flat mode, reverse-mode implicit differentiation, Zygote): works from Fortran. 48x60: 0.85 s steady with the dense direct solver; with the matrix-free Krylov solver 5.0 s (slower). 96x120 Krylov was still running when work stopped (see `kryos-embed/grad-*.out` on Levante, if finished). 200x300 untested.

## Key problems for AD at kryos sizes

- The extension default (`MatrixRepresentation` + `DirectLinearSolver`) forms a dense Jacobian of N = nx*ny states: 200x300 is 60000^2 doubles = 29 GB and a dense factorisation. Not usable. The Krylov option is slower than dense even at 48x60 and is unproven at scale.
- The flat-mode Godunov residual is lower-triangular when cells are ordered by increasing w (each update only depends on smaller neighbours). The exact adjoint is then a back-substitution in descending-w order, O(N log N), no linear solver. This is the structural fix I would implement (hand-written `rrule` or custom adjoint) instead of ImplicitDifferentiation for the large grids. Not implemented. `:full` mode (gradient coupling through grad w) is not triangular; needs a separate treatment or an iterative solve.
- AD compile size: Zygote + ImplicitDifferentiation + ForwardDiff make the sysimage 377 MB; trim is not an option for the AD path.

## Threading, instances, robustness (not tested, from reasoning)

- Julia runs one thread here; the solver is serial. The Julia runtime installs signal handlers and starts GC threads; coexistence with OpenMP and MPI in the kryos process was not tested (the route-a program ran standalone). Julia's OpenBLAS uses suffixed ILP64 symbols, so a clash with the Fortran BLAS/MKL is unlikely but unchecked.
- Only one Julia runtime per process, so multiple kryos ice instances share it; calls must be serialised (or use route c with one worker per instance).
- Route b has no Julia code evaluation at runtime and the smallest failure surface; route c isolates crashes.

## tau_b optimisation (design notes, not prototyped)

Gowan et al. (ICESHEET `global/adjust_loop.sh`, `adjust_ss.f90`): tau_b is set per domain polygon (domains_min/max.txt, ~215 domains), with a time-dependent max-to-min ramp from `adjust_*.txt` (columns: domain, time of max, time of min, min value or 0 for domains_min). Present-day: brute-force loop run -> compare with modern thickness -> per-domain mean difference binned to a multiplicative step (up to +-5 %, scaled by 0.5) on both min and max. Paleo: hand tuning against RSL and far-field sea level (readme.txt records the reasoning per iteration), with GIA in the loop.

Recommendation (reasoned, untested):
1. (iii) offline present-day inversion in Julia (AD or even the Gowan rule) for modern sheets, then fix tau_b in kryos. Cheap, no coupling needed; forward model in kryos via route b.
2. (ii) outer-loop calibration of the few per-domain paleo parameters against RSL/modern thickness with GIA in the loop: gradient-free/ensemble (tens of parameters, each run a full VILMA2 transient) since the adjoint does not cover GIA. Forward calls from kryos then need only route b (or a Fortran port).
3. (i) per-step inversion against a target thickness is circular for a reconstruction (the target is the output); skip.
So AD is useful offline, not inside the kryos loop; this weakens the case for route a over b.

## Status: paused. If resumed

1. Check `kryos-embed/grad-*.out` for 96x120 (and 200x300) Krylov timings.
2. Implement the descending-w triangular adjoint for flat mode and compare with the ImplicitDifferentiation gradient on 48x60.
3. Test route a/b inside a real kryos build: MPI/OpenMP coexistence, signal handling, two instances, `-ljulia` link flags in the kryos Makefile on ifx.
4. Rebuild the sysimage with the fixed `solve` (the one measured may predate it) and repeat the full-mode timings several times (noisy).
5. Pin cpu_target for the sysimage to the Levante node types.

Prototype layout and build/run commands: `kryos-embed/README.md`.
