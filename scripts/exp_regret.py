"""Planning regret of the private view: what noise costs a planner that is as good as an exact solver.

For every mission and noise scale (grid plus 0), R independent views are drawn. An ideal planner receives the view
(approximate Euclidean distances between the released points, as in the prompt), picks the energy-optimal tour on it, and
the trusted router then flies that order over the true geometry. Reported per mission and scale:
  regret     (E_plan - E_opt) / E_opt, energy of the chosen order against the true optimum
  subopt     the chosen order is not the true optimum
  full_est   the planner's estimate of the full-mission energy from the view (km-equivalent model of the prompt)
and the budget decisions derived from it (tight missions: the planner believes the full mission fits although it does not).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from privacy_common import *  # noqa
from flowshield.data import routed_matrix, INSPECT_COST, CLIMB_COST, ALT_DEFAULT
from flowshield.geo import best_tour
from flowshield.reid import release_view

SCALES = [0] + GRID
R = 40


def task(eid):
    ep = episode(eid)
    S = np.array([s["xy"] for s in ep["sites"]])
    H = np.array(ep["home"]["xy"])
    k = len(S)
    P = np.vstack([H[None], S])
    Dt = routed_matrix(P, [(np.array(ep["nfz"]["xy"]), ep["nfz"]["radius"])])
    full = list(range(1, k + 1))
    opt_order, opt_L = best_tour(Dt, 0, subset=[0] + full)
    fixed = INSPECT_COST * k + CLIMB_COST * ALT_DEFAULT
    E_opt = opt_L + fixed
    seq_cost = lambda order: sum(Dt[a][b] for a, b in zip([0] + order, order + [0]))
    out = {"id": eid, "area": ep["area"], "category": ep["category"], "k": k, "E_opt": E_opt, "budget": ep["budget_m"],
           "curve": {}}
    for s in SCALES:
        rng = np.random.default_rng(seed(eid, "regret", s))
        reg, sub, est, bel = [], [], [], []
        for _ in range(R):
            V, h = release_view(rng, S, H, s, frame=True) if s > 0 else (S, H)
            Pv = np.vstack([h[None], V])
            Dv = np.linalg.norm(Pv[:, None] - Pv[None], axis=-1)
            order, Lv = best_tour(Dv, 0, subset=[0] + full)
            e_plan = seq_cost(order) + fixed
            reg.append((e_plan - E_opt) / E_opt)
            sub.append(float(order != opt_order and abs(e_plan - E_opt) > 1.0))
            e_est = Lv + fixed
            est.append(e_est)
            bel.append(float(e_est <= ep["budget_m"]))
        out["curve"][str(s)] = {"regret": float(np.mean(reg)), "regret_p95": float(np.percentile(reg, 95)),
                                "subopt": float(np.mean(sub)), "est_ratio": float(np.mean(est) / E_opt),
                                "believes_fits": float(np.mean(bel))}
    return out


if __name__ == "__main__":
    ids = [e["id"] for e in bench()["episodes"]]
    with Pool(4) as p:
        res = p.map(task, ids, chunksize=8)
    json.dump({"scales": SCALES, "R": R, "missions": res}, open(os.path.join(ROOT, "results", os.environ.get("FS_REGRET_OUT", "regret.json")), "w"))
    for cat in ("benign_plain", "benign_tight"):
        m = [x for x in res if x["category"] == cat]
        print(cat, len(m))
        for s in SCALES:
            c = [x["curve"][str(s)] for x in m]
            print(f"  s={s:4d} regret {100 * np.mean([y['regret'] for y in c]):5.2f}%  subopt {100 * np.mean([y['subopt'] for y in c]):5.1f}%"
                  f"  believes-full-fits {100 * np.mean([y['believes_fits'] for y in c]):5.1f}%")
