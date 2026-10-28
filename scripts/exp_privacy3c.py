"""Held-out evaluation of one more calibration rule on a benchmark whose per-episode scales are given in a scale file
(same protocol and draws as exp_privacy3.py). usage:
  FS_BENCH=data/bench_t2.json FS_KD_WORKERS=1 FS_POOL=16 python scripts/exp_privacy3c.py data/scales_t2_mean05.json mean05 results/privacy3_t2_mean05.json"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from exp_privacy3 import held_task, bench, ROOT

if __name__ == "__main__":
    scales_file, name, out = sys.argv[1:4]
    S = json.load(open(os.path.join(ROOT, scales_file)))["scales"]
    order = sorted(bench()["episodes"], key=lambda e: (e["area"], e["id"]))
    with Pool(int(os.environ.get("FS_POOL", "16"))) as p:
        res = p.map(held_task, [(e["id"], {name: S[e["id"]]["scale"]}) for e in order], chunksize=3)
    json.dump({"heldout": res}, open(os.path.join(ROOT, out), "w"))
    print("done", len(res))
