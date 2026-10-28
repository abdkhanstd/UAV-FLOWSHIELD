#!/bin/bash
# Calibrate the noise scale of every T2 mission on the remote CPUs (single-threaded KD-tree queries, 40 processes).
cd "$HOME/cae_exp/code" || exit 1
export FS_BENCH=data/bench_t2.json FS_OUT=privacy3_t2_calib.json FS_POLICIES=mean10,ucb10 FS_KD_WORKERS=1 FS_POOL=40 FS_ONLY_CALIB=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/exp_privacy3.py > results/logs_calib_t2.txt 2>&1
tail -3 results/logs_calib_t2.txt
