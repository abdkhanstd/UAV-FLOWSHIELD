"""Uncertainty-aware calibration and held-out comparison of disclosure policies.

Phase 1  calibrate (per mission, on the trusted edge): mean-based (legacy, target 0.10 and 0.05, 10 draws) and
         upper-confidence-limit based (Wilson, 30 draws, per-site and mission-level criterion).
Phase 2  held-out evaluation on the 288 test missions with fresh noise and frame draws (20 per mission and policy):
         per-site and mission-level (at least one site identified) re-identification.
Phase 3  per-class fixed scales chosen on the development missions only.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.reid import reid_draws, calibrate_scale_ucb

FIXED = [25, 50, 100, 150, 300, 600, 800]
HELD = 20
_ALL = {"mean10": ("mean", 0.10), "mean05": ("mean", 0.05), "ucb10": ("ucb", 0.10), "ucb10m": ("ucbm", 0.10)}  # an upper limit at 5 % cannot be met with 30 draws (zero hits already give 8.3 %), so no ucb05 rule
POL = {k: _ALL[k] for k in os.environ.get("FS_POLICIES", ",".join(_ALL)).split(",")}


def calib_task(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    out = {"id": eid, "split": ep["split"], "area": ep["area"], "k": len(idx)}
    for name, (kind, tgt) in POL.items():
        rng = np.random.default_rng(seed(eid, "calib3", name))
        if kind == "mean":
            est = None
            for s in GRID:
                r, _ = reid_draws(M, C, idx, H, s, rng, 10)
                est = float(r.mean())
                if est <= tgt:
                    break
            out[name] = {"scale": s, "est": est}
        else:
            s, m, u = calibrate_scale_ucb(M, C, idx, H, GRID, tgt, rng, trials=30, mission_level=(kind == "ucbm"))
            out[name] = {"scale": s, "est": m, "ucb": u}
    return out


def held_task(args):
    eid, scales = args
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    out = {"id": eid, "area": ep["area"], "k": len(idx), "pol": {}}
    for name, s in scales.items():
        rng = np.random.default_rng(seed(eid, "held3", name, s))
        r, a = reid_draws(M, C, idx, H, s, rng, HELD)
        out["pol"][name] = {"scale": s, "site": float(r.mean()), "any": float(a.mean()), "sd": float(r.std(ddof=1))}
    return out


def dev_curve_task(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    rng = np.random.default_rng(seed(eid, "devcurve"))
    return {"id": eid, "area": ep["area"], "cls": ep["area"].split("_")[0],
            "curve": {s: float(reid_draws(M, C, idx, H, s, rng, 10)[0].mean()) for s in GRID}}


if __name__ == "__main__":
    B = bench()["episodes"]
    order = sorted(B, key=lambda e: (e["area"], e["id"]))
    ids = [e["id"] for e in order]
    OUT = os.environ.get("FS_OUT", "privacy3.json")
    PC_FROM = os.environ.get("FS_PERCLASS_FROM")       # fresh benchmark: per-class scales come from the first benchmark's development split
    legacy_p = os.path.join(ROOT, "data", "scales.json")
    legacy = json.load(open(legacy_p))["scales"] if not PC_FROM else {}
    with Pool(int(os.environ.get("FS_POOL", "11"))) as p:
        print("phase 1: calibration", flush=True)
        cal = p.map(calib_task, ids, chunksize=6)
        calmap = {c["id"]: c for c in cal}
        if os.environ.get("FS_ONLY_CALIB"):
            json.dump({"grid": GRID, "calibration": cal}, open(os.path.join(ROOT, "results", OUT), "w"))
            print("calibration written", flush=True)
            sys.exit(0)
        dev = []
        if PC_FROM:
            perclass = json.load(open(os.path.join(ROOT, "results", PC_FROM)))["perclass"]
            legacy = {c["id"]: {"scale": c["mean10"]["scale"]} for c in cal}
        else:
            print("phase 3: development curves", flush=True)
            dev = p.map(dev_curve_task, [e["id"] for e in order if e["split"] == "dev"], chunksize=4)
            # per-class fixed scales from development missions only
            perclass = {}
            for cls in ("dense", "medium", "sparse"):
                cv = [d["curve"] for d in dev if d["cls"] == cls]
                mean_curve = {s: float(np.mean([c[s] for c in cv])) for s in GRID}
                for tag, tgt in (("pc10", 0.10), ("pc05", 0.05)):
                    perclass.setdefault(tag, {})[cls] = next((s for s in GRID if mean_curve[s] <= tgt), GRID[-1])
        print("per-class fixed scales", perclass, flush=True)
        print("phase 2: held-out evaluation", flush=True)
        tasks = []
        for e in order:
            if e["split"] != "test" and not PC_FROM:
                continue
            cls = e["area"].split("_")[0]
            sc = {n: calmap[e["id"]][n]["scale"] for n in POL}
            sc.update({f"fixed{s}": s for s in FIXED})
            sc["pc10"] = perclass["pc10"][cls]
            sc["pc05"] = perclass["pc05"][cls]
            tasks.append((e["id"], sc))
        held = p.map(held_task, tasks, chunksize=3)
    json.dump({"grid": GRID, "fixed": FIXED, "calibration": cal, "dev_curves": dev, "perclass": perclass,
               "heldout": held}, open(os.path.join(ROOT, "results", OUT), "w"))
    for cls in ("dense", "medium", "sparse"):
        h = [x for x in held if x["area"].startswith(cls)]
        print(cls, {n: (round(float(np.mean([x["pol"][n]["scale"] for x in h])), 0), round(float(np.mean([x["pol"][n]["site"] for x in h])), 3))
                    for n in h[0]["pol"]})
