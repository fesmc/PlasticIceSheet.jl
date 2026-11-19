# kryos-embed: prototypes for calling PlasticIceSheet from Fortran

Run everything on Levante (rsync this tree to /work/ba1442/robinson/PlasticIceSheet-julia-embed). Julia 1.12 (`[sources]`).
Mode codes passed by the hosts: 0 = `:flat`, 1 = `:full`, 2 = `:surface` (the package default since v0.2). Since v0.2 the margin defaults to `:face`, so results differ from the v0.1 benchmarks (`margin = :centre` reproduces them).
Packages: KryosEmbed deps (ADTypes, ForwardDiff, ImplicitDifferentiation, Zygote, PlasticIceSheet); env/ adds PackageCompiler.

- `KryosEmbed/`: Julia package with C entry points (`entry_solve`, `entry_grad`). `julia --project=KryosEmbed -e 'using Pkg; Pkg.instantiate()'`
- Native reference: `sbatch run_native.sbatch`
- Route a: `./build.sh` (ifx vs libjulia), `sbatch run_embed.sbatch`; sysimage: `sbatch build_sysimage.sbatch` (needs 32 GB), then `sbatch run_embed_sysimg.sbatch`; gradients: `sbatch run_grad.sbatch`
- Route b: `sbatch build_routeb.sbatch` (juliac, trim), `sbatch run_routeb.sbatch`
- Route c: `sbatch run_routec.sbatch` (server in `routec/server.jl`, client `fortran/routec_client.f90`)
