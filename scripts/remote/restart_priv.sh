#!/bin/bash
# Stop the running T1 calibration experiment (the 5 % upper-limit rule cannot be met with 30 draws and made it very slow)
# and start it again without that rule. The follow-up script keeps waiting for privacy3_mata.json.
pkill -u "$USER" -f "scripts/exp_privacy3.py"
sleep 3
cd "$HOME/cae_exp/code" || exit 1
export FS_OUT=privacy3_mata.json FS_KD_WORKERS=1 FS_POOL=44
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup nice -n 5 python3 scripts/exp_privacy3.py > results/logs_priv_t1.txt 2>&1 &
sleep 2
pgrep -u "$USER" -f exp_privacy3 | wc -l
