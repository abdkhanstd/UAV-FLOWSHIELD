"""Wall-clock time of the noise calibration on one CPU core (mean rule with 10 draws, upper-limit rule with 30 draws)."""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("FS_KD_WORKERS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.reid import reid_draws, calibrate_scale_ucb

PER_CLASS = 8


def main():
    eps = [e for e in bench()["episodes"] if e["split"] == "test"]
    out = {"mean10": {}, "ucb10": {}}
    for cls in ("dense", "medium", "sparse"):
        sel = [e for e in eps if e["area"].startswith(cls)][:PER_CLASS]
        tm, tu = [], []
        for e in sel:
            C, M = matcher(e["area"])
            idx = [s["cand_idx"] for s in e["sites"]]
            H = np.array(e["home"]["xy"])
            rng = np.random.default_rng(seed(e["id"], "time"))
            t0 = time.time()
            for s in GRID:
                r, _ = reid_draws(M, C, idx, H, s, rng, 10)
                if r.mean() <= 0.10:
                    break
            tm.append(time.time() - t0)
            t0 = time.time()
            calibrate_scale_ucb(M, C, idx, H, GRID, 0.10, rng, trials=30)
            tu.append(time.time() - t0)
        out["mean10"][cls] = {"median": float(np.median(tm)), "p95": float(np.percentile(tm, 95))}
        out["ucb10"][cls] = {"median": float(np.median(tu)), "p95": float(np.percentile(tu, 95))}
        print(cls, out["mean10"][cls], out["ucb10"][cls], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "eval", "calib_time.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
