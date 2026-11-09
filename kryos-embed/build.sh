#!/bin/bash
# Build the Fortran host against libjulia. Run on Levante (login node ok, takes seconds).
set -e
source /etc/profile 2>/dev/null || true
JROOT=${JULIA_ROOT:-$(~/.juliaup/bin/julia -e "print(dirname(Sys.BINDIR))")}
ifx -O2 -fp-model=precise -I$JROOT/include/julia fortran/embed_main.f90 -o embed_main \
    -L$JROOT/lib -ljulia -Wl,-rpath,$JROOT/lib -Wl,-rpath,$JROOT/lib/julia
echo built embed_main against $JROOT
