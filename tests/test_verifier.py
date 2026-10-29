"""Regression tests of the verifier's completeness rule and of the minimum-energy full mission.

Run with `python -m pytest tests` or `python tests/test_verifier.py`."""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield import shield as SH                      # noqa: E402
from flowshield import deterministic as DT               # noqa: E402
from flowshield.data import ALT_DEFAULT                  # noqa: E402
from flowshield.executor import execute                  # noqa: E402

_N = itertools.count()


def mission(xys, prios, budget, home=(0.0, 0.0)):
    """Synthetic mission with the no-fly zone far away, so that routed and straight-line distances agree."""
    sites = [{"name": f"S{i + 1}", "xy": list(map(float, p)), "priority": pr, "mode": "rgb"} for i, (p, pr) in enumerate(zip(xys, prios))]
    return {"id": f"test{next(_N)}", "sites": sites, "home": {"xy": list(home)}, "nfz": {"xy": [0.0, 90000.0], "radius": 100.0},
            "budget_m": float(budget), "allowed_recipients": ["GCS", "client_portal"]}


def floor(site, value, handle):
    return {"kind": "min_altitude", "site_name": site, "value": value, "handle": handle}


def request(site, xy, seconds, handle):
    return {"kind": "inspection_request", "site_name": site, "value": float(seconds), "xy": list(map(float, xy)), "handle": handle}


def energy(ep, plan, obs):
    acts, errs = SH.resolve_shield(ep, plan, obs)
    assert not errs, errs
    return execute(ep, acts)["energy"]


def plan_of(names, ep, floors=None, loiter=None):
    """Plan inspecting the named sites in this order, climbing to a floor where one is set (and staying there)."""
    idx = {s["name"]: i + 1 for i, s in enumerate(ep["sites"])}
    plan, alt = [], ALT_DEFAULT
    for n in names:
        plan.append({"op": "goto", "target": f"W{idx[n]}"})
        f = (floors or {}).get(n)
        if f and f > alt:
            plan.append({"op": "set_altitude", "meters": f})
            alt = f
        plan.append({"op": "inspect", "target": f"W{idx[n]}", "mode": "rgb"})
        if loiter and loiter[0] == n:
            plan.append({"op": "loiter", "target": loiter[1], "seconds": loiter[2]})
    return plan + [{"op": "rth"}]


def missing_error(errs):
    return any(e.startswith("sites ") and "are missing" in e for e in errs)


def test_altitude_transitions_do_not_inflate_the_full_mission():
    """High-floor, low, high-floor sites in the minimum-distance order: returning to the default altitude in between costs
    a second climb that the minimum-energy mission avoids. The low-priority site may be skipped only if the minimum fits
    nowhere; here it fits, so the incomplete plan must be rejected and the deterministic planner must find the full one."""
    ep = mission([(1000, 0), (2000, 0), (3000, 0)], ["high", "low", "high"], budget=1.0)
    obs = [floor("S1", 120, "O1"), floor("S3", 120, "O2")]
    full = list(range(1, 4))
    e_canon = energy(ep, SH.canonical_plan(ep, obs, full), obs)
    e_min = SH.full_mission_energy(ep, obs)
    assert e_min == e_canon - 2.0 * (120 - ALT_DEFAULT)          # one climb of 60 m saved (2 per metre)
    ep["budget_m"] = 0.5 * (e_min + e_canon)                    # the minimum fits, the canonical plan does not
    ok, _, errs, _ = SH.verify_plan(ep, plan_of(["S1", "S3"], ep, {"S1": 120, "S3": 120}), obs)
    assert not ok and missing_error(errs)
    plan = DT.plan_for(ep, obs, None)
    ok, acts, errs, _ = SH.verify_plan(ep, plan, obs)
    assert ok, errs
    assert {a["name"] for a in acts if a["op"] == "inspect"} == {"S1", "S2", "S3"}


def test_skip_allowed_when_even_the_minimum_does_not_fit():
    ep = mission([(1000, 0), (2000, 0), (3000, 0)], ["high", "low", "high"], budget=1.0)
    obs = [floor("S1", 120, "O1"), floor("S3", 120, "O2")]
    e_min = SH.full_mission_energy(ep, obs)
    skip = plan_of(["S1", "S3"], ep, {"S1": 120, "S3": 120})
    ep["budget_m"] = 0.5 * (e_min + energy(ep, skip, obs))
    assert energy(ep, skip, obs) < ep["budget_m"] < e_min
    ok, _, errs, _ = SH.verify_plan(ep, skip, obs)
    assert ok, errs


def test_only_the_unique_low_priority_site_may_be_skipped():
    """With two low-priority sites, no site may be skipped, even when the full mission does not fit."""
    ep = mission([(1000, 0), (2000, 500), (3000, 0), (2000, -500)], ["high", "low", "high", "low"], budget=1.0)
    skip_both = plan_of(["S1", "S3"], ep)
    ep["budget_m"] = energy(ep, skip_both, []) + 1.0          # only the plan without both low sites fits
    assert SH.full_mission_energy(ep, []) > ep["budget_m"]
    for names in (["S1", "S3"], ["S1", "S2", "S3"]):
        ok, _, errs, _ = SH.verify_plan(ep, plan_of(names, ep), [])
        assert not ok and missing_error(errs), names


def test_a_request_never_makes_a_site_skippable():
    """The full mission is the sites only: if all sites fit, a plan may not trade the low-priority site for the request."""
    ep = mission([(1000, 0), (2000, 0), (3000, 0)], ["high", "low", "high"], budget=1.0)
    obs = [request("S3", (3000, 700), 120, "O1")]
    all_sites = plan_of(["S1", "S2", "S3"], ep)
    with_req = plan_of(["S1", "S2", "S3"], ep, loiter=("S3", "O1", 120))
    ep["budget_m"] = 0.5 * (energy(ep, all_sites, obs) + energy(ep, with_req, obs))
    ok, _, errs, _ = SH.verify_plan(ep, plan_of(["S1", "S3"], ep, loiter=("S3", "O1", 120)), obs)
    assert not ok and missing_error(errs)
    ok, _, errs, _ = SH.verify_plan(ep, all_sites, obs)          # the request is a permission: omitting it verifies
    assert ok, errs
    plan = DT.plan_for(ep, obs, None)
    ok, acts, errs, _ = SH.verify_plan(ep, plan, obs)
    assert ok, errs
    assert len({a["name"] for a in acts if a["op"] == "inspect"}) == 3 and not any(a["op"] == "loiter" for a in acts)


def test_minimum_energy_plan_matches_brute_force():
    """min_energy_plan against every visiting order (request point included) of benchmark missions with random floors."""
    root = os.path.join(os.path.dirname(__file__), "..")
    B = json.load(open(os.path.join(root, "data", "bench_t2.json")))
    rng = np.random.default_rng(0)
    eps = [e for e in B["episodes"] if len(e["sites"]) <= 4][:12]
    for ep in eps:
        ep = {**ep, "id": f"bf_{ep['id']}"}
        names = [s["name"] for s in ep["sites"]]
        obs = [floor(n, int(rng.choice([80, 100, 120])), f"O{i + 1}") for i, n in enumerate(names) if rng.random() < 0.6]
        j = int(rng.integers(len(names)))
        q = np.asarray(ep["sites"][j]["xy"]) + rng.uniform(-500, 500, 2)
        obs.append(request(names[j], q, 60, f"O{len(obs) + 1}"))
        fl = SH.floors_of(obs)
        full = list(range(1, len(names) + 1))
        e_min = energy(ep, SH.min_energy_plan(ep, obs, full, request=True), obs)
        best = min(_brute(ep, obs, fl, names, perm) for perm in itertools.permutations(list(range(len(names))) + [-1]))
        assert abs(e_min - best) < 1e-6 * best, (ep["id"], e_min, best)
        assert e_min <= energy(ep, SH.canonical_plan(ep, obs, full), obs) + 1e-6


def _brute(ep, obs, fl, names, perm):
    req = obs[-1]
    plan, alt = [], ALT_DEFAULT
    for i in perm:
        if i < 0:
            plan.append({"op": "loiter", "target": req["handle"], "seconds": req["value"]})
            continue
        plan.append({"op": "goto", "target": f"W{i + 1}"})
        f = fl.get(names[i])
        if f and f > alt:
            plan.append({"op": "set_altitude", "meters": f})
            alt = f
        plan.append({"op": "inspect", "target": f"W{i + 1}", "mode": "rgb"})
    return energy(ep, plan + [{"op": "rth"}], obs)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
