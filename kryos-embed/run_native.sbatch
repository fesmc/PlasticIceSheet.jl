#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
J=~/.juliaup/bin/julia
export JULIA_NUM_THREADS=1
for sz in "200 300" "400 400"; do
  /usr/bin/time -f "wall %e s, maxrss %M kB" $J --project=KryosEmbed julia/bench_native.jl $sz 5
done
