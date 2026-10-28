"""Attacker-strength experiment: how much does the choice of registration attacker matter?

weak   : short noisy base pair, one hypothesis, no refinement
strong : longest base pair, top-64 hypotheses, ICP refinement (used for calibration and evaluation)
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.reid import release_view, tau_for_scale, correct_ids

SCALES = [25, 50, 100, 200, 400]
PER_AREA = 24
TRIALS = 3


def task(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    S = C[idx]
    rng = np.random.default_rng(seed(eid, "attacker"))
    out = {"id": eid, "area": ep["area"], "res": {}}
    for s in SCALES:
        w, st = [], []
        for _ in range(TRIALS):
            V, h = release_view(rng, S, H, s, frame=True)
            a, _, _ = M.attack(V, tau_for_scale(s), rng, weak=True)
            w.append(correct_ids(a, idx, C).mean())
            a, _, _ = M.attack(V, tau_for_scale(s), rng)
            st.append(correct_ids(a, idx, C).mean())
        out["res"][s] = {"weak": float(np.mean(w)), "strong": float(np.mean(st))}
    return out


if __name__ == "__main__":
    eps = [e for e in bench()["episodes"] if e["split"] == "test"]
    cnt, ids = {}, []
    for e in eps:
        if cnt.get(e["area"], 0) < PER_AREA:
            ids.append(e["id"])
            cnt[e["area"]] = cnt.get(e["area"], 0) + 1
    with Pool(12) as p:
        res = p.map(task, ids, chunksize=1)
    json.dump({"scales": SCALES, "results": res}, open(os.path.join(ROOT, "results", "attacker.json"), "w"))
    for a in sorted(cnt):
        r = [x for x in res if x["area"] == a]
        print(a, {s: (round(float(np.mean([x["res"][s]["weak"] for x in r])), 2),
                      round(float(np.mean([x["res"][s]["strong"] for x in r])), 2)) for s in SCALES})
