"""Further privacy analyses (T1 test missions, 40 per registry class).

part A  region-level disclosure: besides the identity of the sites, how well does the registration attacker localise the
        mission? Reported: share of attack draws whose estimated mission centroid (mean of the matched registry points) lies
        within 0.5, 1, 2 and 5 km of the true centroid, at the scales of several policies.
part B  strength of the attacker: identification rate when the attacker keeps more hypotheses, runs more ICP iterations or
        tries several base pairs (flowshield/attacker_plus.py), at the per-class table scales and at 300 m.
part C  disclosure decision: for each mission and each assumed threat (blind attacker, four releases, client portfolio of
        100 or 25 assets) the certified scale (flowshield/certify.py, fixed-sequence exact test, T=100 draws, FS_CERT_T); the mission is
        released to the planner only if the largest certified scale over the assumed threats does not exceed the cap,
        otherwise it is planned on the edge. For each design assumption the risk against every actual attacker is measured at
        the released scale (misspecification matrix).

usage: FS_KD_WORKERS=1 FS_POOL=48 python scripts/exp_privacy5.py A|B|C results/privacy5_<part>.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from exp_privacy4 import portfolio, CLEAN
from flowshield.reid import ShapeMatcher, release_view, tau_for_scale, correct_ids
from flowshield.certify import reid_draws_ok, select_scale
from flowshield.attacker_plus import attack_multi

PER_AREA = 40
CLASS_TABLE = {"dense": 50, "medium": 150, "sparse": 600}
GRID_EXT = GRID + [1200, 1600]
THREATS = ["blind", "x4", "pf100", "pf25"]


def missions(per_area=PER_AREA):
    cnt, ids = {}, []
    for e in bench()["episodes"]:
        if e["split"] == "test" and e["category"] in CLEAN and cnt.get(e["area"], 0) < per_area:
            ids.append(e["id"])
            cnt[e["area"]] = cnt.get(e["area"], 0) + 1
    return ids


def setup(eid):
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    return ep, C, M, idx, H


# --------------------------------------------------------------------------- part A
def task_A(args):
    eid, scales = args
    ep, C, M, idx, H = setup(eid)
    S = C[idx]
    truth = S.mean(0)
    out = {"id": eid, "area": ep["area"], "pol": {}}
    for pol, s in scales.items():
        rng = np.random.default_rng(seed(eid, "A", pol, s))
        errs, ident = [], []
        for _ in range(20):
            V, _ = release_view(rng, S, H, s, frame=True)
            ass, _, _ = M.attack(V, tau_for_scale(s), rng, extra=None)
            ok = ass >= 0
            errs.append(float(np.linalg.norm(C[ass[ok]].mean(0) - truth)) if ok.any() else np.inf)
            ident.append(float(correct_ids(ass, idx, C).mean()))
        errs = np.array(errs)
        out["pol"][pol] = {"scale": s, "site": float(np.mean(ident)),
                           **{f"within{int(r)}": float((errs <= r).mean()) for r in (500, 1000, 2000, 5000)},
                           "median_err": float(np.median(errs))}
    return out


# --------------------------------------------------------------------------- part B
VARIANTS = {"base": dict(ranks=(0,), topk=64, icp_iters=6), "topk256": dict(ranks=(0,), topk=256, icp_iters=6),
            "topk1024": dict(ranks=(0,), topk=1024, icp_iters=6), "icp12": dict(ranks=(0,), topk=64, icp_iters=12),
            "base3": dict(ranks=(0, 1, 2), topk=64, icp_iters=6), "strong": dict(ranks=(0, 1, 2), topk=256, icp_iters=12)}


def task_B(args):
    eid, scales = args
    ep, C, M, idx, H = setup(eid)
    S = C[idx]
    out = {"id": eid, "area": ep["area"], "pol": {}}
    for pol, s in scales.items():
        res = {}
        for name, kw in VARIANTS.items():
            rng = np.random.default_rng(seed(eid, "B", pol, s))      # the same releases for every variant
            r = []
            for _ in range(10):
                V, _ = release_view(rng, S, H, s, frame=True)
                ass = attack_multi(M, V, tau_for_scale(s), rng, **kw)
                r.append(float(correct_ids(ass, idx, C).mean()))
            res[name] = float(np.mean(r))
        out["pol"][pol] = {"scale": s, **res}
    return out


# --------------------------------------------------------------------------- part C
def threat_setup(threat, eid, C, M, idx, H, rng):
    if threat == "blind":
        return M, C, idx, 1
    if threat == "x4":
        return M, C, idx, 4
    m = 100 if threat == "pf100" else 25
    Cp, ip = portfolio(C, idx, m, rng)
    return ShapeMatcher(Cp), Cp, ip, 1


def task_C(eid):
    ep, C, M, idx, H = setup(eid)
    cls = ep["area"].split("_")[0]
    out = {"id": eid, "area": ep["area"], "cls": cls, "k": len(idx), "cert": {}, "risk": {}}
    setups = {}
    for th in THREATS:
        rng = np.random.default_rng(seed(eid, "C-setup", th))
        setups[th] = threat_setup(th, eid, C, M, idx, H, rng)
    for th in THREATS:
        Mt, Ct, it, nrel = setups[th]
        grid = [s for s in (GRID_EXT if th.startswith("pf") else GRID) if not (cls == "dense" and th in ("blind", "x4") and s > 300)]
        rng = np.random.default_rng(seed(eid, "C-cal", th))
        r = select_scale(lambda s, n: reid_draws_ok(Mt, Ct, it, H, s, rng, n, n_release=nrel), grid, 0.10, "cp_fs", rng, T=int(os.environ.get("FS_CERT_T", "100")),
                         alpha=float(os.environ.get("FS_ALPHA", "0.05")), pilot=10)
        out["cert"][th] = {"scale": r["scale"], "certified": r["certified"], "draws": r["draws"]}
    # mission-level objective against the blind attacker (some inspected site identified), certified separately
    Mt, Ct, it, nrel = setups["blind"]
    grid = [s for s in GRID if not (cls == "dense" and s > 300)]
    rng = np.random.default_rng(seed(eid, "C-cal-any"))
    r = select_scale(lambda s, n: reid_draws_ok(Mt, Ct, it, H, s, rng, n), grid, 0.10, "cp_fs", rng, T=int(os.environ.get("FS_CERT_T", "100")),
                     alpha=float(os.environ.get("FS_ALPHA", "0.05")), pilot=10, mission_level=True)
    out["cert"]["blind_any"] = {"scale": r["scale"], "certified": r["certified"], "draws": r["draws"]}
    # misspecification matrix: design assumption (single threat) x actual attacker, 20 draws at the certified scale
    for design in THREATS:
        c = out["cert"][design]
        if not c["certified"]:
            continue
        s = c["scale"]
        out["risk"][design] = {"scale": s}
        for act in THREATS:
            Mt, Ct, it, nrel = setups[act]
            rng = np.random.default_rng(seed(eid, "C-eval", design, act))
            ok = reid_draws_ok(Mt, Ct, it, H, s, rng, 20, n_release=nrel)
            out["risk"][design][act] = float(ok.mean())
    return out


if __name__ == "__main__":
    part, out = sys.argv[1], sys.argv[2]
    ids = missions(int(os.environ.get("FS_PER_AREA", "20")))
    P3 = json.load(open(os.path.join(ROOT, "results", "privacy3_mata.json")))
    cal = {c["id"]: c for c in P3["calibration"]}
    with Pool(int(os.environ.get("FS_POOL", "24"))) as p:
        if part == "A":
            tasks = [(i, {"ucb10": cal[i]["ucb10"]["scale"], "fixed100": 100, "fixed300": 300, "fixed600": 600,
                          "table": CLASS_TABLE[episode(i)["area"].split("_")[0]]}) for i in ids]
            res = p.map(task_A, tasks, chunksize=1)
        elif part == "B":
            tasks = [(i, {"table": CLASS_TABLE[episode(i)["area"].split("_")[0]], "fixed300": 300}) for i in ids]
            res = p.map(task_B, tasks, chunksize=1)
        else:
            res = p.map(task_C, ids, chunksize=1)
    json.dump({"missions": res}, open(os.path.join(ROOT, out), "w"))
    print("done", len(res))
