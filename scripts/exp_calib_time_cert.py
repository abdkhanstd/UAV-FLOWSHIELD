"""Wall-clock time of the certified calibration on one CPU core per mission (fixed-sequence ladder, T = 60 and 100 draws per scale, 8 test
missions per registry class; the missions run in parallel processes, each mission on one core, on a machine with idle cores).
usage: python scripts/exp_calib_time_cert.py"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

os.environ.setdefault("FS_KD_WORKERS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.certify import reid_draws_ok, select_scale

PER_CLASS = 8
CAP = {"dense": 300, "medium": 800, "sparse": 800}


def one(args):
    T, cls, eid = args
    e = episode(eid)
    C, M = matcher(e["area"])
    idx = [s["cand_idx"] for s in e["sites"]]
    H = np.array(e["home"]["xy"])
    rng = np.random.default_rng(seed(eid, "timecert", T))
    grid = [s for s in GRID if s <= CAP[cls]]
    t0 = time.time()
    r = select_scale(lambda s, n: reid_draws_ok(M, C, idx, H, s, rng, n), grid, 0.10, "cp_fs", rng, T=T, pilot=10)
    return T, cls, time.time() - t0, r["draws"]


def main():
    eps = [e for e in bench()["episodes"] if e["split"] == "test"]
    tasks = []
    for T in (60, 100):
        for cls in ("dense", "medium", "sparse"):
            for e in [x for x in eps if x["area"].startswith(cls)][:PER_CLASS]:
                tasks.append((T, cls, e["id"]))
    tasks.sort(key=lambda t: 0 if t[1] == "dense" else 1 if t[1] == "medium" else 2)
    with Pool(int(os.environ.get("FS_POOL", "24"))) as p:
        res = p.map(one, tasks, chunksize=1)
    out = {}
    for T in (60, 100):
        out[str(T)] = {}
        for cls in ("dense", "medium", "sparse"):
            r = [x for x in res if x[0] == T and x[1] == cls]
            ts = [x[2] for x in r]
            out[str(T)][cls] = {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95)), "draws": float(np.mean([x[3] for x in r]))}
            print(T, cls, out[str(T)][cls], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "eval", "calib_time_cert.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
