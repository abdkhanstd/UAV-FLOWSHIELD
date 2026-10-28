"""Diversion-energy bound of an admitted spatial request: router-metric envelope versus straight-line envelope.

For stress missions (no-fly zone next to an assigned site) and for the ordinary test missions, random candidate points
are drawn around the assigned sites; each point that the envelope admits is inserted into the cheapest full mission
(loiter right after the inspection of the nearest site, duration 120 s). Reported: the largest increase of the mission energy
and the share of admitted points whose increase exceeds the proved bound 2*rho + 8*T = 2560 metre-equivalents."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import exp_fuzz as F
import flowshield.shield as SH
from flowshield.data import ENVELOPE_R, LOITER_COST
from flowshield.executor import execute

ROOT = F.ROOT
BOUND = 2 * ENVELOPE_R + LOITER_COST * 120


def extra_cost(ep, xy, j_site, seconds=120.0):
    k = len(ep["sites"])
    full = list(range(1, k + 1))
    obs = [{"kind": "inspection_request", "site_name": ep["sites"][j_site]["name"], "value": seconds, "xy": list(map(float, xy)),
            "site": f"W{j_site + 1}", "handle": "O1", "source": "D1"}]

    def energy(ob):
        acts, _ = SH.resolve_shield(ep, SH.canonical_plan(ep, ob, full), ob)
        return execute(ep, acts)["energy"]
    return energy(obs) - energy([])


def main(n_points=40, seed=1):
    rng = np.random.default_rng(seed)
    EP, ST = F.pool(rng, 200)
    out = {}
    for label, pool in (("ordinary missions", EP[:120]), ("stress missions", ST)):
        rows = {"route": [], "euclid": []}
        for ep in pool:
            nfz_c, nfz_r = np.array(ep["nfz"]["xy"]), ep["nfz"]["radius"]
            for _ in range(n_points):
                j = int(rng.integers(len(ep["sites"])))
                a, d = rng.uniform(0, 2 * np.pi), rng.uniform(100, 800)
                xy = np.array(ep["sites"][j]["xy"]) + d * np.array([np.cos(a), np.sin(a)])
                if np.linalg.norm(xy - nfz_c) < nfz_r + SH.NFZ_MARGIN:
                    continue
                for metric in ("route", "euclid"):
                    jj, dj = SH.envelope_distance(ep, xy, metric)
                    if dj <= ENVELOPE_R:
                        rows[metric].append(extra_cost(ep, xy, jj))
        out[label] = {m: {"n": len(v), "max": float(np.max(v)), "p99": float(np.percentile(v, 99)),
                          "share_over_bound": float(np.mean(np.asarray(v) > BOUND + 1e-6))} for m, v in rows.items()}
        print(label, json.dumps(out[label]))
    out["bound"] = BOUND
    json.dump(out, open(os.path.join(ROOT, "results", "eval", "bound.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
