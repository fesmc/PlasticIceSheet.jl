#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
J=~/.juliaup/bin/julia
$J --project=env -e 'using Pkg; Pkg.instantiate()'
/usr/bin/time -f "sysimage build wall %e s, maxrss %M kB" $J --project=env sysimage/build.jl
