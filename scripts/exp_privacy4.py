"""Stronger adversaries against the disclosure policies (held-out missions, fresh draws).

  blind      registration attacker of the main evaluation
  fuse2/4    the same sites are released 2 or 4 times (independent frames and noise); the releases are registered onto
             each other through the handles and averaged before the registry attack (composition across releases)
  portfolio  the attacker knows that the client's assets are a subset of m registry entries (m = 25, 100) that contains
             the true sites, so it searches the portfolio instead of the full registry
  anchor     the attacker knows the registry identity of one released point and must identify the others
  budget     the attacker also uses the released energy budget (it follows the 1.4x rule of the benchmark), which
             correlates with the tour length of the true sites
Policies: fixed 300 m, mean-calibrated, upper-limit calibrated (from results/privacy3.json). The upper-limit
calibration is also re-run against the portfolio prior (`ucb10_pf100`) to show that the calibrator can absorb a known
side-information model, and with a release count of 4 (`ucb10_x4`).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.geo import best_tour
from flowshield.reid import (ShapeMatcher, reid_draws, calibrate_scale_ucb, release_view, tau_for_scale, correct_ids,
                             fuse_releases)

PER_AREA = 40
TRIALS = 20
CLEAN = ("benign_plain", "attack", "adaptive")      # budget = 1.4 x full-mission energy, no document extras


def portfolio(C, idx, m, rng):
    others = [i for i in range(len(C)) if i not in set(idx)]
    take = list(rng.choice(others, size=max(0, min(m - len(idx), len(others))), replace=False))
    keep = sorted(set(idx) | set(int(t) for t in take))
    pos = {c: j for j, c in enumerate(keep)}
    return C[keep], [pos[i] for i in idx]


def budget_attack(M, C, S_idx, H, scale, budget, rng):
    """Registration followed by a re-scoring of the refined candidates with the tour length implied by the budget."""
    S = C[S_idx]
    V, h = release_view(rng, S, H, scale, frame=True)
    tau = tau_for_scale(scale)
    ass, _, nh, cands = M.attack(V, tau, rng, extra=None, return_cands=True)
    if not cands:
        return correct_ids(ass, S_idx, C).mean()
    k = len(S_idx)
    L_b = budget / 1.4 - 150.0 * k - 120.0
    best = None
    for fs, a, (refl, R2, t2) in cands:
        hm = ((h * np.array([1.0, -1.0]) if refl else h) @ R2.T + t2)
        P = np.vstack([hm[None], C[a]])
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        _, L_c = best_tour(D, 0, subset=list(range(k + 1)))
        score = fs / max(tau, 1.0) ** 2 + ((L_c - L_b) / (0.05 * L_b + 300.0)) ** 2
        if best is None or score < best[0]:
            best = (score, a)
    return correct_ids(best[1], S_idx, C).mean()


def task(args):
    eid, sc = args
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    k = len(idx)
    out = {"id": eid, "area": ep["area"], "k": k, "res": {}}
    for pol, s in sc.items():
        rng = np.random.default_rng(seed(eid, "p4", pol, s))
        r = {"scale": s}
        r["blind"] = float(reid_draws(M, C, idx, H, s, rng, TRIALS)[0].mean())
        for n in (2, 4):
            r[f"fuse{n}"] = float(reid_draws(M, C, idx, H, s, rng, TRIALS, n_release=n)[0].mean())
        for m in (25, 100):
            Cp, ip = portfolio(C, idx, m, rng)
            Mp = ShapeMatcher(Cp)
            r[f"portfolio{m}"] = float(reid_draws(Mp, Cp, ip, H, s, rng, TRIALS)[0].mean())
        v = []
        for _ in range(TRIALS):
            Sx = C[idx]
            V, h = release_view(rng, Sx, H, s, frame=True)
            p0 = int(rng.integers(k))
            ass, _, _ = M.attack(V, tau_for_scale(s), rng, known=(p0, idx[p0]))
            ok = correct_ids(ass, idx, C)
            others = [j for j in range(k) if j != p0]
            v.append(ok[others].mean())
        r["anchor"] = float(np.mean(v))
        if ep["category"] in CLEAN:
            r["budget"] = float(np.mean([budget_attack(M, C, idx, H, s, ep["budget_m"], rng) for _ in range(TRIALS)]))
        out["res"][pol] = r
    return out


def calib_extra(eid):
    """Scales for the side-information-aware calibrations."""
    ep = episode(eid)
    C, M = matcher(ep["area"])
    idx = [s["cand_idx"] for s in ep["sites"]]
    H = np.array(ep["home"]["xy"])
    rng = np.random.default_rng(seed(eid, "p4calib"))
    Cp, ip = portfolio(C, idx, 100, rng)
    s_pf = calibrate_scale_ucb(ShapeMatcher(Cp), Cp, ip, H, GRID, 0.10, rng, trials=30)[0]
    s_x4 = calibrate_scale_ucb(M, C, idx, H, GRID, 0.10, rng, trials=30, n_release=4)[0]
    return eid, s_pf, s_x4


if __name__ == "__main__":
    P3 = json.load(open(os.path.join(ROOT, "results", os.environ.get("FS_PRIV3", "privacy3.json"))))
    cal = {c["id"]: c for c in P3["calibration"]}
    legacy = json.load(open(os.path.join(ROOT, "data", "scales.json")))["scales"]
    cnt, ids = {}, []
    for e in bench()["episodes"]:
        if e["split"] == "test" and e["category"] in CLEAN and cnt.get(e["area"], 0) < PER_AREA:
            ids.append(e["id"])
            cnt[e["area"]] = cnt.get(e["area"], 0) + 1
    print("missions", len(ids), cnt, flush=True)
    with Pool(int(os.environ.get("FS_POOL", "11"))) as p:
        extra = {eid: (a, b) for eid, a, b in p.map(calib_extra, ids, chunksize=2)}
        tasks = []
        for eid in ids:
            tasks.append((eid, {"fixed300": 300, "mean10": legacy[eid]["scale"], "ucb10": cal[eid]["ucb10"]["scale"],
                                "ucb10_pf100": extra[eid][0], "ucb10_x4": extra[eid][1]}))
        res = p.map(task, tasks, chunksize=2)
    json.dump({"missions": res}, open(os.path.join(ROOT, "results", os.environ.get("FS_OUT4", "privacy4.json")), "w"))
    for cls in ("dense", "medium", "sparse"):
        h = [x for x in res if x["area"].startswith(cls)]
        print(cls)
        for pol in h[0]["res"]:
            row = {a: round(float(np.mean([x["res"][pol][a] for x in h if a in x["res"][pol]])), 3)
                   for a in ("scale", "blind", "fuse2", "fuse4", "portfolio25", "portfolio100", "anchor", "budget")}
            print("  ", pol, row)
