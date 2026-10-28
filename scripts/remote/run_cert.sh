#!/bin/bash
# Certified calibration and held-out evaluation (T = 100) on T2, then the disclosure-decision experiment (part C).
cd "$HOME/cae_exp/code" || exit 1
export FS_KD_WORKERS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "T2 cert $(date +%T)"
FS_BENCH=data/bench_t2.json FS_POOL=44 python scripts/exp_privacy3d.py cert100 100 results/privacy3_cert100_t2.json all
echo "part C $(date +%T)"
FS_BENCH=data/bench.json FS_PER_AREA=20 FS_POOL=44 FS_CERT_T=100 python scripts/exp_privacy5.py C results/privacy5_C.json
echo "REMOTE_DONE $(date +%T)"
