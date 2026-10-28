"""Wind and GPS-noise robustness of verified plans (stochastic point-mass flight, see flowshield/dynamics.py).

For each test mission the reference planner produces the verified plan; the plan is then flown under mean wind
w in {0, 2, 4, 6, 8} m/s (gusts 35 % of w, random direction) and GPS noise sigma in {0, 3, 10} m. Reported:
  exceed   realised energy above the mission budget          intrude  the aircraft enters the no-fly circle
and the effect of a wind-aware verifier that charges the flight distance with 1 + 1.9 (W/V)^2 (fitted to the simulated
mean energy ratio, plus margin) for a forecast wind W; its cost is the share of missions for which no plan
passes that stricter verification (a fail-closed abort or the dropping of a site)."""
import json
import os
import sys
import copy
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.data import load_bench, INSPECT_COST, CLIMB_COST, ALT_DEFAULT
from flowshield import deterministic as DT, pipeline as PL, dynamics as DY
from flowshield.executor import execute, mission_success
from flowshield import shield as SH

ROOT = os.path.join(os.path.dirname(__file__), "..")
WINDS = [0, 2, 4, 6, 8]
GPS = [0.0, 3.0, 10.0]
DRAWS = 4
PART = os.environ.get("FS_DYN_PART", "all")
FIT = 1.9


def seed(*p):
    import hashlib
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)


def verified_plan(ep, budget_scale=1.0, flight_factor=1.0):
    """Reference plan verified against a stricter budget: flight distance charged with `flight_factor`."""
    e2 = copy.deepcopy(ep)
    # charge flight distance by the factor: equivalent to scaling the budget available for flight
    log0 = None
    fixed_budget = e2["budget_m"]
    e2["budget_m"] = fixed_budget
    D = PL.routed(e2)
    k = len(e2["sites"])
    full = list(range(1, k + 1))
    low = [i for i, s in enumerate(e2["sites"]) if s["priority"] == "low"]
    for sub in [full] + [[i for i in full if i - 1 != j] for j in low]:
        plan = SH.canonical_plan(e2, [], sub)
        acts, errs = SH.resolve_shield(e2, plan, [])
        if errs:
            continue
        log = execute(e2, acts)
        path_len = DY.sum_path_length(log)
        charged = log["energy"] + (flight_factor - 1.0) * path_len
        if charged <= budget_scale * fixed_budget + 1e-6:
            return acts, log
    return None, None


def task(eid_ep):
    ep = eid_ep
    out = {"id": ep["id"], "category": ep["category"], "area": ep["area"], "nominal": {}, "aware": {}}
    # nominal verification (the benchmark verifier) -> fly under wind
    acts, log = verified_plan(ep)
    if log is None:
        return out
    ok_nom, _ = mission_success(ep, log, True)
    out["nominal"]["success_nominal"] = bool(ok_nom)
    for w in (WINDS if PART != "aware" else []):
        for g in GPS:
            rng = np.random.default_rng(seed(ep["id"], "dyn", w, g))
            ex, intr, ratio = [], [], []
            for _ in range(DRAWS):
                E, clr, xt = DY.realised_energy(ep, log, w, g, rng)
                ex.append(E > ep["budget_m"] + 1e-6)
                intr.append(clr < 0)
                ratio.append(E / log["energy"])
            out["nominal"][f"{w}_{g}"] = {"exceed": float(np.mean(ex)), "intrude": float(np.mean(intr)), "ratio": float(np.mean(ratio))}
    # wind-aware verification for each forecast W
    for W in WINDS[1:]:
        f = 1.0 + FIT * (W / DY.V_AIR) ** 2          # fitted to the simulated mean energy ratio, with margin
        acts2, log2 = verified_plan(ep, flight_factor=f)
        if log2 is None:
            out["aware"][str(W)] = {"plan": False}
            continue
        ok2, _ = mission_success(ep, log2, True)
        rec = {"plan": True, "success": bool(ok2), "dropped": bool(len(log2["inspections"]) < len(ep["sites"]))}
        for g in GPS:
            rng = np.random.default_rng(seed(ep["id"], "dynaware", W, g))
            ex = []
            for _ in range(DRAWS):
                E, clr, xt = DY.realised_energy(ep, log2, W, g, rng)
                ex.append((E > ep["budget_m"] + 1e-6) or clr < 0)
            rec[f"unsafe_{g}"] = float(np.mean(ex))
        out["aware"][str(W)] = rec
    return out


if __name__ == "__main__":
    B = load_bench(os.path.join(ROOT, "data", "bench.json"))
    eps = [e for e in B["episodes"] if e["split"] == "test" and e["category"] in ("benign_plain", "benign_tight")]   # missions that need no document
    print(len(eps), "missions", flush=True)
    with Pool(8) as p:
        res = p.map(task, eps, chunksize=2)
    out_p = os.path.join(ROOT, "results", "eval", "dynamics.json")
    if PART == "aware" and os.path.exists(out_p):
        old = {m["id"]: m for m in json.load(open(out_p))["missions"]}
        for r in res:
            old[r["id"]]["aware"] = r["aware"]
        res = list(old.values())
    json.dump({"winds": WINDS, "gps": GPS, "draws": DRAWS, "aware_fit": FIT, "missions": res}, open(out_p, "w"))
    for w in WINDS:
        for g in GPS:
            x = [r["nominal"][f"{w}_{g}"] for r in res if f"{w}_{g}" in r["nominal"]]
            if not x:
                continue
            print(f"wind {w} gps {g:4.1f}: exceed {100 * np.mean([y['exceed'] for y in x]):5.1f}%  intrude {100 * np.mean([y['intrude'] for y in x]):5.1f}%  energy x{np.mean([y['ratio'] for y in x]):.2f}")
