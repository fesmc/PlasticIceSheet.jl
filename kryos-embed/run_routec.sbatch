#!/bin/bash
#SBATCH --account=ba1442
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# Route c: persistent worker over FIFOs, plus one-shot julia startup costs.
cd /work/ba1442/robinson/PlasticIceSheet-julia-embed/kryos-embed
J=~/.juliaup/bin/julia
export JULIA_PROJECT=$PWD/KryosEmbed JULIA_NUM_THREADS=1
echo "== one-shot startup (using KryosEmbed; nothing else)"
/usr/bin/time -f "default sysimage: wall %e s" $J -e 'using KryosEmbed'
[ -f sysimage/kryos_embed.so ] && /usr/bin/time -f "custom sysimage: wall %e s" $J -J sysimage/kryos_embed.so -e 'using KryosEmbed'
echo "== persistent worker"
ifx -O2 fortran/routec_client.f90 -o routec_client
D=/dev/shm/kp_$$; mkdir -p $D; cmd=$D/cmd; resp=$D/resp; mkfifo $cmd $resp
for sz in "200 300" "400 400"; do for mode in 0 1; do
  $J ${SYSIMG:+-J $SYSIMG} routec/server.jl $cmd $resp & SP=$!
  echo "-- $sz mode=$mode"; ./routec_client $sz 6 $mode $cmd $resp
  wait $SP
done; done
rm -rf $D
