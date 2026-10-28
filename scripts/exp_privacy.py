"""Experiment 2 (privacy): re-identification of inspected sites from released planning geometry.

Mechanisms (all with the same planar Laplace noise scale s, mean displacement s metres):
  frame+noise   secret rigid frame + noise  (FlowShield view)    attacker: blind registration (stronger ICP attacker)
  aligned       exact alignment known (frame compromise, or absolute coordinates + noise): nearest-registry attack
Held-out evaluation of the attack-calibrated scale uses fresh noise/frame draws (different seeds from calibration)."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa

CURVE = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
CURVE_PER_AREA = 40


def task(args):
    eid, do_curve = args
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    out = {"id": eid, "area": ep["area"], "k": len(idx)}
    if do_curve:
        rng = np.random.default_rng(seed(eid, "curve"))
        cur = {}
        for s in CURVE:
            blind, herr = reid_risk(M, C, idx, H, s, rng, 3, frame=True)
            al = float(np.mean([aligned_nearest(M, C, idx, s, rng) for _ in range(10)])) if s > 0 else 1.0
            cur[s] = {"blind": blind, "aligned": al, "home_err": herr}
        out["curve"] = cur
    sc = json.load(open(os.path.join(ROOT, "data", "scales.json")))["scales"][eid]["scale"]
    rng2 = np.random.default_rng(seed(eid, "heldout"))
    blind, herr = reid_risk(M, C, idx, H, sc, rng2, 20, frame=True)
    al = float(np.mean([aligned_nearest(M, C, idx, sc, rng2) for _ in range(40)]))
    out["heldout"] = {"scale": sc, "blind": blind, "aligned": al, "home_err": herr, "calib_est": None}
    return out


if __name__ == "__main__":
    eps = [e for e in bench()["episodes"] if e["split"] == "test"]
    jobs, count = [], {}
    for e in eps:
        c = count.get(e["area"], 0)
        jobs.append((e["id"], c < CURVE_PER_AREA))
        count[e["area"]] = c + 1
    with Pool(12) as p:
        res = p.map(task, jobs, chunksize=1)
    json.dump({"curve_grid": CURVE, "results": res}, open(os.path.join(ROOT, "results", "privacy.json"), "w"))
    for a in sorted(count):
        r = [x for x in res if x["area"] == a]
        print(a, "n", len(r), "heldout blind re-id mean", round(float(np.mean([x["heldout"]["blind"] for x in r])), 3),
              "median scale", np.median([x["heldout"]["scale"] for x in r]),
              "aligned-nearest at that scale", round(float(np.mean([x["heldout"]["aligned"] for x in r])), 3))
