#!/bin/bash
# Re-run the T1 calibration experiment on the remote CPUs (single-threaded KD-tree queries, 40 processes).
cd "$HOME/cae_exp/code" || exit 1
export FS_OUT=privacy3_mata.json FS_KD_WORKERS=1 FS_POOL=40
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/exp_privacy3.py > results/logs_priv_t1.txt 2>&1
tail -6 results/logs_priv_t1.txt
