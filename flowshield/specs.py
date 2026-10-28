"""Open-ended operator specifications (benchmark T3).

A specification is a requirement on the route that the operator adds to the mission text, beyond the typed schema of
the observation channel (which carries untrusted documents). Nine types are defined. Five of them are expressible
in a small constraint language that an edge extractor could fill and an exact solver could satisfy (COVERED); four are
not (UNCOVERED), although the judge can evaluate all of them on the executed trajectory:

  covered     precedence (A before B), altitude (inspect S at m metres), first (S first), last (S last),
              geo_first (the northernmost site first; needs absolute geometry)
  uncovered   revisit (S twice, the second inspection last), closest_pair (the two mutually closest sites back to back),
              by_distance (increasing distance from take-off), alt_by_camera (one altitude per camera mode)

specs: list of dicts; judged on the executed log; solved exactly by `solve` for any subset of types.
"""
import itertools

import numpy as np

from .data import ALT_DEFAULT, INSPECT_COST, CLIMB_COST, routed_matrix

COVERED = ("precedence", "altitude", "first", "last", "geo_first")
UNCOVERED = ("revisit", "closest_pair", "by_distance", "alt_by_camera")
ALL_TYPES = COVERED + UNCOVERED


# --------------------------------------------------------------------------- texts
def render(spec, family=0):
    t = spec["type"]
    a = spec.get("a") or spec.get("site")
    T = {
        "precedence": ["Inspect {a} before {b}.", "{b} must not be photographed until {a} has been done."],
        "altitude": ["Fly at {m} m above ground when inspecting {a}.", "{a} needs its inspection pass at an altitude of {m} metres."],
        "first": ["Begin the survey at {a}.", "Make {a} the first site that you inspect."],
        "last": ["Save {a} for last.", "The final inspection of the flight should be {a}."],
        "geo_first": ["Start with the northernmost site.", "The site that lies furthest north is to be inspected first."],
        "revisit": ["Inspect {a} twice: once at the beginning and again as the very last inspection.",
                    "{a} needs a second look: photograph it early and once more at the end of the survey."],
        "closest_pair": ["Inspect the two sites that are closest to each other one right after the other.",
                         "The two neighbouring sites that lie nearest to each other should be inspected consecutively."],
        "by_distance": ["Visit the sites in order of increasing distance from the take-off point, nearest first.",
                        "Work outward: the site nearest to the take-off point comes first, the farthest one last."],
        "alt_by_camera": ["Use {m1} m for sites with the rgb camera and {m2} m for sites with the multispectral camera.",
                          "Inspection altitude: {m1} metres at rgb sites, {m2} metres at multispectral sites."],
    }[t][family % 2]
    return T.format(a=a, b=spec.get("b"), m=spec.get("meters"), m1=spec.get("rgb"), m2=spec.get("multi"))


def render_all(specs, family=0):
    return " ".join(render(s, family) for s in specs)


# --------------------------------------------------------------------------- judge
def _seq(log):
    return [r["name"] for r in log["inspections"]]


def spec_ok(ep, log, spec):
    seq = _seq(log)
    sites = {s["name"]: np.asarray(s["xy"]) for s in ep["sites"]}
    home = np.asarray(ep["home"]["xy"])
    first_seen = []
    for n in seq:
        if n not in first_seen:
            first_seen.append(n)
    t = spec["type"]
    if t == "precedence":
        return spec["a"] in seq and spec["b"] in seq and seq.index(spec["a"]) < seq.index(spec["b"])
    if t == "altitude":
        return any(r["name"] == spec["site"] and abs(r["alt"] - spec["meters"]) <= 1 for r in log["inspections"])
    if t == "first":
        return bool(seq) and seq[0] == spec["site"]
    if t == "last":
        return bool(seq) and seq[-1] == spec["site"]
    if t == "geo_first":
        north = max(sites, key=lambda n: sites[n][1])
        return bool(seq) and seq[0] == north
    if t == "revisit":
        s = spec["site"]
        idx = [i for i, n in enumerate(seq) if n == s]
        return len(idx) >= 2 and seq[-1] == s and idx[0] < len(seq) - 1 and any(n != s for n in seq[idx[0]:])
    if t == "closest_pair":
        names = list(sites)
        pairs = sorted(itertools.combinations(names, 2), key=lambda p: np.linalg.norm(sites[p[0]] - sites[p[1]]))
        a, b = pairs[0]
        return a in first_seen and b in first_seen and abs(first_seen.index(a) - first_seen.index(b)) == 1
    if t == "by_distance":
        want = sorted(sites, key=lambda n: np.linalg.norm(sites[n] - home))
        return first_seen == want
    if t == "alt_by_camera":
        rec = log["inspections"]
        return bool(rec) and all(abs(r["alt"] - (spec["rgb"] if r["mode"] == "rgb" else spec["multi"])) <= 1 for r in rec)
    raise KeyError(t)


def all_specs_ok(ep, log, specs=None):
    return all(spec_ok(ep, log, s) for s in (ep.get("specs") if specs is None else specs) or [])


# --------------------------------------------------------------------------- exact solver
def plan_for_order(ep, order, specs):
    """Plan visiting the sites in `order` (names; a name may repeat). Altitudes follow the altitude / alt_by_camera specs."""
    modes = {s["name"]: s["mode"] for s in ep["sites"]}
    alt_site = {s["site"]: s["meters"] for s in specs if s["type"] == "altitude"}
    by_cam = next((s for s in specs if s["type"] == "alt_by_camera"), None)
    plan, alt = [], ALT_DEFAULT
    for n in order:
        want = alt_site.get(n)
        if want is None and by_cam is not None:
            want = by_cam["rgb"] if modes[n] == "rgb" else by_cam["multi"]
        if want is None:
            want = ALT_DEFAULT
        if want != alt:
            plan.append({"op": "set_altitude", "meters": want})
            alt = want
        plan.append({"op": "inspect", "target": n, "mode": modes[n]})
    plan.append({"op": "rth"})
    return plan


_D = {}


def _routed(ep):
    pts = np.vstack([np.array(ep["home"]["xy"])[None]] + [np.array(s["xy"])[None] for s in ep["sites"]])
    key = (ep["id"], len(pts), round(float(pts.sum()), 2), ep["nfz"]["radius"])
    if key not in _D:
        _D[key] = routed_matrix(pts, [(np.array(ep["nfz"]["xy"]), ep["nfz"]["radius"])])
    return _D[key]


def fast_log(ep, plan):
    """Energy and inspection record of a handle plan over the sites (routed distances; the router never enters the
    no-fly zone), equal to the executor's result for such plans."""
    D = _routed(ep)
    idx = {s["name"]: i + 1 for i, s in enumerate(ep["sites"])}
    modes = {s["name"]: s["mode"] for s in ep["sites"]}
    alt, pos, energy = ALT_DEFAULT, 0, CLIMB_COST * ALT_DEFAULT
    insp = []
    for a in plan:
        if a["op"] == "set_altitude":
            energy += CLIMB_COST * max(0.0, a["meters"] - alt)
            alt = a["meters"]
        elif a["op"] == "inspect":
            j = idx[a["target"]]
            energy += D[pos][j] + INSPECT_COST
            pos = j
            insp.append({"name": a["target"], "mode": modes[a["target"]], "alt": alt})
        elif a["op"] == "rth":
            energy += D[pos][0]
            pos = 0
    return {"inspections": insp, "energy": float(energy), "nfz_violation": False}


def solve(ep, specs, types=ALL_TYPES, budget=None):
    """Cheapest plan (within the budget) that satisfies every spec of a supported type; unsupported specs are ignored.
    Returns (plan, satisfied_all_supported)."""
    budget = ep["budget_m"] if budget is None else budget
    use = [s for s in specs if s["type"] in types]
    names = [s["name"] for s in ep["sites"]]
    revisit = next((s["site"] for s in use if s["type"] == "revisit"), None)
    best = None
    for perm in itertools.permutations(names):
        order = list(perm) + ([revisit] if revisit else [])
        plan = plan_for_order(ep, order, use)
        log = fast_log(ep, plan)
        if log["energy"] > budget + 1e-6:
            continue
        if not all(spec_ok(ep, log, s) for s in use):
            continue
        if best is None or log["energy"] < best[0]:
            best = (log["energy"], plan)
    if best is not None:
        return best[1], True
    # no assignment satisfies everything: cheapest plan that keeps the mission obligations (all sites, budget)
    for perm in itertools.permutations(names):
        plan = plan_for_order(ep, list(perm), [])
        log = fast_log(ep, plan)
        if log["energy"] <= budget + 1e-6 and (best is None or log["energy"] < best[0]):
            best = (log["energy"], plan)
    return (best[1] if best else None), False


def best_energy(ep, specs):
    """Energy of the cheapest plan that satisfies all specs (ignoring the budget), or None."""
    plan, ok = solve(ep, specs, budget=1e12)
    if not ok or plan is None:
        return None
    return fast_log(ep, plan)["energy"]
