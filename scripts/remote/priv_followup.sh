#!/bin/bash
# After the T1 calibration experiment: side-information attackers (T1), then the held-out evaluation on the fresh geography (T2).
cd "$HOME/cae_exp/code" || exit 1
export FS_KD_WORKERS=1 FS_POOL=40
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
until [ -f results/privacy3_mata.json ]; do sleep 20; done
echo "side-information attackers $(date +%T)"
FS_PRIV3=privacy3_mata.json FS_OUT4=privacy4_mata.json python3 scripts/exp_privacy4.py > results/logs_priv4.txt 2>&1
echo "T2 held-out evaluation $(date +%T)"
FS_BENCH=data/bench_t2.json FS_OUT=privacy3_t2_heldout.json FS_PERCLASS_FROM=privacy3_mata.json FS_POLICIES=mean10,ucb10 python3 scripts/exp_privacy3.py > results/logs_priv_t2.txt 2>&1
echo "PRIV_FOLLOWUP_DONE $(date +%T)"
