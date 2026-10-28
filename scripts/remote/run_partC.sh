#!/bin/bash
# Disclosure-decision experiment with a Bonferroni split over the declared threats (alpha = 0.05 / 4 per threat) and a mission-level objective.
cd "$HOME/cae_exp/code" || exit 1
export FS_KD_WORKERS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "part C bonferroni $(date +%T)"
FS_BENCH=data/bench.json FS_PER_AREA=20 FS_POOL=40 FS_CERT_T=100 FS_ALPHA=0.0125 python scripts/exp_privacy5.py C results/privacy5_C_bonf.json
echo "REMOTE_DONE $(date +%T)"
