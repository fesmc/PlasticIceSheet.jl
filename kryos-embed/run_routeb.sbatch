#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
B=$PWD/routeb/build
ifx -O2 fortran/routeb_main.f90 -o routeb_main -L$B -lplastic -Wl,-rpath,$B
for sz in "200 300" "400 400"; do for mode in 0 1; do
  echo "== $sz mode=$mode"; /usr/bin/time -f "wall %e s maxrss %M kB" ./routeb_main $sz 6 $mode
done; done
