"""Disclosure-policy comparison: held-out re-identification of fixed noise scales on the same missions
as the calibrated policy (same fresh-draw protocol as exp_privacy.py)."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa

FIXED = [25, 50, 100, 150, 300, 600]
PER_AREA = 40
TRIALS = 20


def task(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    rng = np.random.default_rng(seed(eid, "fixed"))
    out = {"id": eid, "area": ep["area"], "k": len(idx), "fixed": {}}
    for s in FIXED:
        r, herr = reid_risk(M, C, idx, H, s, rng, TRIALS, frame=True)
        out["fixed"][s] = r
    return out


if __name__ == "__main__":
    eps = [e for e in bench()["episodes"] if e["split"] == "test"]
    cnt, ids = {}, []
    for e in eps:
        if cnt.get(e["area"], 0) < PER_AREA:
            ids.append(e["id"])
            cnt[e["area"]] = cnt.get(e["area"], 0) + 1
    with Pool(10) as p:
        res = p.map(task, ids, chunksize=1)
    json.dump({"fixed": FIXED, "results": res}, open(os.path.join(ROOT, "results", "privacy_fixed.json"), "w"))
    for a in sorted(cnt):
        r = [x for x in res if x["area"] == a]
        print(a, {s: round(float(np.mean([x["fixed"][s] for x in r])), 3) for s in FIXED})
