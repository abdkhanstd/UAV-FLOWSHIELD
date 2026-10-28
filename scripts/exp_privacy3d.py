"""Certified calibration (flowshield/certify.py) of every mission of a benchmark and held-out evaluation of the released scales.

For each mission the fixed-sequence ladder (exact test at 5 %, T draws, target rho) is run on the edge-side attack simulation; the released scale
is evaluated on 20 fresh draws (same protocol and seeds as exp_privacy3.py). Missions that cannot be certified get the largest scale of their
ladder and are flagged. Dense registries use a ladder that starts at 300 m (no mission needed more; the cap only saves pre-flight time).
usage: FS_BENCH=data/bench_t2.json FS_KD_WORKERS=1 FS_POOL=48 python scripts/exp_privacy3d.py <name> <T> results/privacy3_cert_t2.json [split]
Output has the layout of exp_privacy3c.py ({"heldout": [...]}) plus "calibration".
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from exp_privacy3 import held_task
from flowshield.certify import reid_draws_ok, select_scale

CAP = {"dense": 300, "medium": 800, "sparse": 800}
NAME = sys.argv[1] if len(sys.argv) > 1 else "cert"          # read at import time: worker processes started by spawn re-import this module
T = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def calib_cert(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    cls = ep["area"].split("_")[0]
    grid = [s for s in GRID if s <= CAP[cls]]
    rng = np.random.default_rng(seed(eid, "cert", NAME, T))
    r = select_scale(lambda s, n: reid_draws_ok(M, C, idx, H, s, rng, n), grid, 0.10, "cp_fs", rng, T=T, pilot=10)
    return {"id": eid, "area": ep["area"], "scale": r["scale"], "certified": bool(r["certified"]), "draws": r["draws"], "est": r["est"]}


if __name__ == "__main__":
    out = sys.argv[3]
    split = sys.argv[4] if len(sys.argv) > 4 else "all"
    order = sorted([e for e in bench()["episodes"] if split == "all" or e["split"] == split], key=lambda e: (e["area"], e["id"]))
    ids = [e["id"] for e in order]
    # expensive dense missions first
    ids.sort(key=lambda i: 0 if episode(i)["area"].startswith("dense") else 1 if episode(i)["area"].startswith("medium") else 2)
    with Pool(int(os.environ.get("FS_POOL", "24"))) as p:
        cal = p.map(calib_cert, ids, chunksize=1)
        cm = {c["id"]: c for c in cal}
        held = p.map(held_task, [(e["id"], {NAME: cm[e["id"]]["scale"]}) for e in order], chunksize=3)
    json.dump({"calibration": cal, "heldout": held}, open(os.path.join(ROOT, out), "w"))
    for cls in ("dense", "medium", "sparse"):
        c = [x for x in cal if x["area"].startswith(cls)]
        h = [x for x in held if x["area"].startswith(cls)]
        print(cls, len(c), "scale", round(float(np.mean([x["scale"] for x in c]))), "uncertified", round(float(np.mean([not x["certified"] for x in c])), 3),
              "held-out", round(float(np.mean([x["pol"][NAME]["site"] for x in h])), 4))
