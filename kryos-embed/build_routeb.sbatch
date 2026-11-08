#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed/routeb
J=~/.juliaup/bin/julia
JROOT=$($J -e 'print(dirname(Sys.BINDIR))')
$J --project=. -e 'using Pkg; Pkg.instantiate()'
mkdir -p build
/usr/bin/time -f "juliac wall %e s, maxrss %M kB" $J --project=. $JROOT/share/julia/juliac/juliac.jl \
   --output-lib build/libplastic --experimental --trim=safe --compile-ccallable src/PlasticLib.jl
ls -la build; nm -D build/libplastic.so | grep -E " T (kp_|init_julia|shutdown|jl_init)" 
