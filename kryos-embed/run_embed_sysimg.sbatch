#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
export SYSIMG=$PWD/sysimage/kryos_embed.so
export JULIA_PROJECT=$PWD/KryosEmbed JULIA_NUM_THREADS=1
export JULIA_BINDIR=$(~/.juliaup/bin/julia -e 'print(Sys.BINDIR)')
for sz in "200 300" "400 400"; do for mode in 0 1; do
  echo "== ${sz} mode=$mode sysimage"
  /usr/bin/time -f "wall %e s, maxrss %M kB" ./embed_main $sz 6 $mode 0 $SYSIMG
done; done
