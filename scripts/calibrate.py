"""Attack-calibrated disclosure: for every mission choose the smallest noise scale whose simulated
blind-registration re-identification rate is at most the target (run by the trusted edge, which
holds the same public registry as the adversary)."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa

TARGET = float(os.environ.get("FS_TARGET", "0.10"))
TRIALS = 10


def calib(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    rng = np.random.default_rng(seed(eid, "calib"))
    est = None
    for s in GRID:
        est, _ = reid_risk(M, C, idx, H, s, rng, TRIALS, frame=True)
        if est <= TARGET:
            return eid, s, est
    return eid, GRID[-1], est


if __name__ == "__main__":
    ids = [e["id"] for e in bench()["episodes"]]
    with Pool(12) as p:
        out = p.map(calib, ids, chunksize=2)
    res = {eid: {"scale": s, "est": est} for eid, s, est in out}
    json.dump({"target": TARGET, "grid": GRID, "scales": res}, open(os.path.join(ROOT, "data", "scales.json"), "w"))
    by = {}
    for e in bench()["episodes"]:
        by.setdefault(e["area"], []).append(res[e["id"]]["scale"])
    for a, v in sorted(by.items()):
        print(a, "n", len(v), "median scale", np.median(v), "mean", round(float(np.mean(v)), 1))
