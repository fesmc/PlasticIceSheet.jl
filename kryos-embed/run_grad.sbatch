#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Gradient (reverse-mode, implicit diff) through the embedded libjulia, with the sysimage.
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
export JULIA_PROJECT=$PWD/KryosEmbed JULIA_NUM_THREADS=1
export JULIA_BINDIR=$(~/.juliaup/bin/julia -e 'print(Sys.BINDIR)')
S=$PWD/sysimage/kryos_embed.so
for case in "48 60 1" "48 60 2" "96 120 2" "200 300 2"; do
  set -- $case
  echo "== grad flat ${1}x${2} solver=$([ $3 = 1 ] && echo dense || echo krylov)"
  timeout 1200 /usr/bin/time -f "wall %e s maxrss %M kB" ./embed_main $1 $2 3 0 $3 $S
done
