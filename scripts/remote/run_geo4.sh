#!/bin/bash
# Third geography: certified ladder and table-first rule, held-out evaluation (12 areas, 120 missions).
cd "$HOME/cae_exp/code" || exit 1
export FS_KD_WORKERS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "geo4 $(date +%T)"
FS_BENCH=data/bench_geo4.json FS_POOL=${FS_POOL:-12} python scripts/exp_geo4.py 100 results/geo4.json
echo "REMOTE_DONE $(date +%T)"
