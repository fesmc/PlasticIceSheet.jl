#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Route a: Fortran host with embedded libjulia. Optional env SYSIMG=path/to/sysimage.so
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
export JULIA_PROJECT=$PWD/KryosEmbed
export JULIA_NUM_THREADS=1
export JULIA_BINDIR=$(~/.juliaup/bin/julia -e 'print(Sys.BINDIR)')
for sz in "200 300" "400 400"; do
  for mode in 0 1; do
    echo "== ${sz} mode=$mode ${SYSIMG:+sysimage}"
    /usr/bin/time -f "wall %e s, maxrss %M kB" ./embed_main $sz 6 $mode 0 $SYSIMG
  done
done
echo "== gradient, small grid, dense direct"
./embed_main 48 60 2 1 1 $SYSIMG
