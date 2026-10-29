"""UAV-FlowShield trusted-edge components.

1. build_view      : private planning view (secret rigid frame + calibrated planar Laplace noise)
2. admit_observations : typed-observation admission under a mission-derived spatial authority envelope
3. verify_plan     : resolution of planner handles to exact local coordinates and verification of the
                     resolved plan against the authenticated mission (completeness, bounds, budget)

Security properties (under an uncompromised edge and a correct data boundary):
  P1  no absolute map-grid coordinate and no operator identifier reaches the planner or the outbound text;
  P2  every executed spatial target is an assigned site, HOME, or an admitted observation point that lies
      within ENVELOPE_R of an assigned site; at most one such point exists, with at most MAX_LOITER seconds.
"""
import math
import re

import numpy as np

from . import auth as AUTH
from .data import ALT_DEFAULT, ALT_MAX, ALT_MIN, ENVELOPE_R, MAX_LOITER, routed_matrix
from .executor import execute
from .geo import best_tour, route
from .methods import grid_to_xy, in_nfz, get_plan, _num, _coord
from .reid import release_view

AUTH_SPATIAL_CHANNELS = {"portal"}      # only client-originated tickets may add a spatial task
# Channel authority: which source channels may supply which observation kind. Peer messages carry no authority.
# min_altitude and crop_stress are monotone (they can only raise a floor or require the better camera); the spatial task
# is the only kind that adds a target, which is why it additionally requires an authenticated origin.
AUTHORITY = {"min_altitude": {"feed", "portal"}, "crop_stress": {"portal"}, "inspection_request": AUTH_SPATIAL_CHANNELS}
NFZ_MARGIN = 30.0                       # the router keeps this clearance; admitted points must lie outside it
ALLOWED_OPS = {"goto", "inspect", "loiter", "set_altitude", "send", "rth"}
_BIGNUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def floors_of(obs):
    """Altitude floor per site; conflicting floors resolve to the highest one (monotone, safe direction)."""
    f = {}
    for o in obs:
        if o["kind"] == "min_altitude":
            f[o["site_name"]] = max(f.get(o["site_name"], 0.0), o["value"])
    return f


def build_view(ep, rng, scale):
    S = np.array([s["xy"] for s in ep["sites"]])
    H = np.array(ep["home"]["xy"])
    V, h = release_view(rng, S, H, scale, frame=True)
    return {"W": V.tolist(), "H": h.tolist(), "scale": float(scale)}


def redact_numbers(text):
    """Outbound text may not carry numeric values with three or more digits."""
    return _BIGNUM.sub(lambda m: "[n]" if sum(c.isdigit() for c in m.group()) >= 3 else m.group(), text)


# --------------------------------------------------------------------------- observation admission
def envelope_distance(ep, xy, metric="route"):
    """Distance from a point to the nearest assigned site, in the router's metric (default) or straight-line."""
    nfz = (np.asarray(ep["nfz"]["xy"]), ep["nfz"]["radius"])
    if metric == "route":
        d = [route(np.asarray(s["xy"]), xy, [nfz])[1] for s in ep["sites"]]
    else:
        d = [float(np.linalg.norm(xy - np.asarray(s["xy"]))) for s in ep["sites"]]
    j = int(np.argmin(d))
    return j, float(d[j])


def admit_observations(ep, extracted, envelope=True, auth="signed", metric="route", authority=True):
    """extracted: list of (doc, parsed extractor JSON). Returns (admitted, rejected).

    auth      'signed' : a spatial task needs a verified origin tag (client portal signature) in addition to the channel
              'channel': the channel alone decides (earlier design, kept for comparison)
    metric    'route'  : the envelope is measured with the trusted router, so the cost of a diversion is bounded
              'euclid' : straight-line radius (earlier design, kept for comparison)
    authority apply the channel-authority table (False: any channel may supply min_altitude and crop_stress)"""
    sites = {s["name"]: i for i, s in enumerate(ep["sites"])}
    admitted, rejected, seen = [], [], set()
    n_spatial = 0
    for doc, res in extracted:
        items = res.get("observations") if isinstance(res, dict) else None
        if not isinstance(items, list):
            continue
        origin_ok = AUTH.verify(ep, doc) if auth == "signed" else True
        for o in items:
            if not isinstance(o, dict):
                continue
            kind = str(o.get("kind", "other"))
            site = o.get("site").upper() if isinstance(o.get("site"), str) else None
            rec = None
            chan_ok = (not authority) or doc["channel"] in AUTHORITY.get(kind, set())
            if kind == "min_altitude":
                v = _num(o.get("min_altitude_m"))
                v = math.ceil(v - 1e-6) if v is not None else None      # unit conversions round up (safe direction)
                if chan_ok and site in sites and v is not None and ALT_DEFAULT <= v <= ALT_MAX:
                    rec = {"kind": kind, "site_name": site, "value": v}
            elif kind == "crop_stress":
                if chan_ok and site in sites:
                    rec = {"kind": kind, "site_name": site, "value": 0.0}
            elif kind == "inspection_request" and envelope:
                e, n = _num(o.get("e")), _num(o.get("n"))
                if doc["channel"] in AUTH_SPATIAL_CHANNELS and origin_ok and e is not None and n is not None and n_spatial == 0:
                    xy = np.array(grid_to_xy(ep, e, n))
                    j, dj = envelope_distance(ep, xy, metric)
                    if dj <= ENVELOPE_R and np.linalg.norm(xy - np.asarray(ep["nfz"]["xy"])) >= ep["nfz"]["radius"] + NFZ_MARGIN:
                        sec = _num(o.get("loiter_seconds"), 60) or 60
                        rec = {"kind": kind, "site_name": ep["sites"][j]["name"], "value": float(min(max(sec, 0.0), MAX_LOITER)),
                               "xy": xy.tolist(), "dist": dj}
                        n_spatial += 1
            if rec is None:
                rejected.append({"doc": doc["id"], "kind": kind})
                continue
            key = (rec["kind"], rec["site_name"], rec["value"])
            if key in seen:
                continue
            seen.add(key)
            rec["site"] = f"W{sites[rec['site_name']] + 1}"
            rec["handle"] = f"O{len(admitted) + 1}"
            rec["source"] = doc["id"]
            admitted.append(rec)
    return admitted, rejected


# --------------------------------------------------------------------------- canonical full mission
_DCACHE = {}


def routed_D(ep):
    if ep["id"] not in _DCACHE:
        pts = np.vstack([np.array(ep["home"]["xy"])[None]] + [np.array(x["xy"])[None] for x in ep["sites"]])
        _DCACHE[ep["id"]] = routed_matrix(pts, [(np.array(ep["nfz"]["xy"]), ep["nfz"]["radius"])])
    return _DCACHE[ep["id"]]


def canonical_plan(ep, obs, subset):
    """Handle-space plan visiting `subset` (node ids 1..k) in the order of minimum routed distance and honouring the
    admitted observations (altitude floors, camera requirement, the admitted loiter right after its qualifying site).
    Its energy can exceed the minimum (it returns to the default altitude after a floor and does not place the loiter
    freely); min_energy_plan gives the minimum, and the completeness rule of verify_plan uses that."""
    k = len(ep["sites"])
    names = [f"W{i + 1}" for i in range(k)]
    floors = floors_of(obs)
    msi = {o["site_name"] for o in obs if o["kind"] == "crop_stress"}
    spatial = [o for o in obs if o["kind"] == "inspection_request"]
    order, _ = best_tour(routed_D(ep), 0, subset=[0] + list(subset))
    plan, alt, loiter_done = [], ALT_DEFAULT, not spatial
    for idx in order:
        s, h = ep["sites"][idx - 1], names[idx - 1]
        plan.append({"op": "goto", "target": h})
        want = floors.get(s["name"], ALT_DEFAULT)
        if want != alt:
            plan.append({"op": "set_altitude", "meters": want})
            alt = want
        plan.append({"op": "inspect", "target": h, "mode": "multispectral" if s["name"] in msi else s["mode"]})
        if not loiter_done and spatial[0]["site_name"] == s["name"]:
            plan.append({"op": "loiter", "target": spatial[0]["handle"], "seconds": spatial[0]["value"]})
            loiter_done = True
    plan.append({"op": "rth"})
    return plan


def min_energy_plan(ep, obs, subset, request=True):
    """Minimum-energy handle-space plan that inspects the sites in `subset` (node ids 1..k) with the required cameras and
    above the admitted floors and, if `request` is set and its qualifying site is in `subset`, serves the admitted spatial
    request (a loiter of its admitted duration at its point, at any position in the route).

    Under the executor's energy model (routed flight, 150 per inspection, 8 per loiter second, 2 per metre of climb,
    descents free) the climb charge of a route is at least (highest floor - default altitude) for every visiting order,
    and a route that only climbs, when a floor requires it, attains this; the remaining charges other than routed flight
    do not depend on the order. The minimum is therefore an exact tour (Held-Karp) over home, the sites and the request
    point under the router's metric, flown with that altitude policy."""
    floors = floors_of(obs)
    msi = {o["site_name"] for o in obs if o["kind"] == "crop_stress"}
    spatial = [o for o in obs if o["kind"] == "inspection_request"]
    sub = list(subset)
    req = None
    if request and spatial and any(ep["sites"][i - 1]["name"] == spatial[0]["site_name"] for i in sub):
        req = spatial[0]
    D0 = routed_D(ep)
    idx = [0] + sub
    n = len(idx) + (1 if req is not None else 0)
    M = np.zeros((n, n))
    for a in range(len(idx)):
        for b in range(len(idx)):
            M[a, b] = D0[idx[a]][idx[b]]
    if req is not None:
        pts = [np.asarray(ep["home"]["xy"])] + [np.asarray(s["xy"]) for s in ep["sites"]]
        nfz = [(np.asarray(ep["nfz"]["xy"]), ep["nfz"]["radius"])]
        q = np.asarray(req["xy"])
        for a in range(len(idx)):
            M[a, n - 1] = route(pts[idx[a]], q, nfz)[1]
            M[n - 1, a] = route(q, pts[idx[a]], nfz)[1]
    order, _ = best_tour(M, 0, subset=list(range(n)))
    plan, alt = [], ALT_DEFAULT
    for j in order:
        if req is not None and j == n - 1:
            plan.append({"op": "loiter", "target": req["handle"], "seconds": req["value"]})
            continue
        i = idx[j]
        s, h = ep["sites"][i - 1], f"W{i}"
        plan.append({"op": "goto", "target": h})
        want = floors.get(s["name"], ALT_DEFAULT)
        if want > alt:
            plan.append({"op": "set_altitude", "meters": want})
            alt = want
        plan.append({"op": "inspect", "target": h, "mode": "multispectral" if s["name"] in msi else s["mode"]})
    plan.append({"op": "rth"})
    return plan


def full_mission_energy(ep, obs):
    """Minimum energy of a mission that inspects every site with the required cameras above the admitted floors. An admitted
    spatial request is a permission (verify_plan allows one loiter at its handle but does not require it), so it is not part
    of the full mission: a request can never make a site skippable."""
    acts, _ = resolve_shield(ep, min_energy_plan(ep, obs, list(range(1, len(ep["sites"]) + 1)), request=False), obs)
    return execute(ep, acts)["energy"]


# --------------------------------------------------------------------------- plan resolution and verification
def resolve_shield(ep, plan, obs, allow_points=False):
    """Resolve handles to exact coordinates. Returns (actions, errors)."""
    sites = {}
    for i, s in enumerate(ep["sites"]):
        sites[f"W{i + 1}"] = s
        sites[s["name"].upper()] = s
    oh = {o["handle"]: o for o in obs}
    acts, errors = [], []
    for a in plan:
        op = str(a.get("op", "")).lower()
        if op not in ALLOWED_OPS:
            errors.append(f"operation '{op}' is not permitted")
            continue
        r = {"op": op}
        tgt = a.get("target").upper() if isinstance(a.get("target"), str) else None
        if op in ("goto", "inspect", "loiter"):
            e, n = _coord(a)
            if tgt in sites:
                s = sites[tgt]
                r.update(xy=s["xy"], name=s["name"], kind="site")
            elif tgt in ("H", "HOME"):
                r.update(xy=ep["home"]["xy"], kind="home")
            elif tgt in oh and oh[tgt]["kind"] == "inspection_request":
                r.update(xy=oh[tgt]["xy"], kind="obs", obs=tgt, cap=oh[tgt]["value"])
            elif allow_points and e is not None and n is not None:
                r.update(xy=grid_to_xy(ep, e, n), kind="point")
            else:
                errors.append(f"target '{a.get('target')}' is not an allowed handle")
                continue
        if op == "inspect":
            if r["kind"] != "site":
                errors.append("inspect is permitted at assigned sites only")
                continue
            r["mode"] = str(a.get("mode", "rgb")).lower()
        if op == "loiter":
            if r["kind"] not in ("obs", "point"):
                errors.append("loiter is permitted only at an admitted observation")
                continue
            r["seconds"] = _num(a.get("seconds"), 60) or 60
        if op == "set_altitude":
            r["meters"] = _num(a.get("meters"), ALT_DEFAULT)
        if op == "send":
            r.update(to=str(a.get("to", "")), text=str(a.get("text", "")))
        acts.append(r)
    return acts, errors


def verify_plan(ep, plan, obs, allow_points=False, typed=True):
    """Returns (ok, actions, errors, clamped). Actions are modified only by clamping and redaction.

    Completeness: every site is inspected, except that a plan may omit the unique low-priority site when even the
    minimum-energy mission over all sites (full_mission_energy) exceeds the budget. An admitted spatial request is a
    permission, not an obligation: at most one loiter, only at its handle, clamped to its admitted duration; whether the
    request was served is scored by the mission-success judge, not enforced here."""
    acts, errors = resolve_shield(ep, plan, obs, allow_points)
    sites = ep["sites"]
    floors = floors_of(obs)
    modes = {s["name"]: s["mode"] for s in sites}
    modes.update({o["site_name"]: "multispectral" for o in obs if o["kind"] == "crop_stress"})
    n_loiter, alt, clamped = 0, ALT_DEFAULT, 0
    for a in acts:
        op = a["op"]
        if op == "set_altitude":
            if a["meters"] is None or not (ALT_MIN <= a["meters"] <= ALT_MAX):
                errors.append(f"altitude must be between {ALT_MIN} and {ALT_MAX} m")
            else:
                alt = a["meters"]
        elif op == "inspect":
            if typed and a["mode"] != modes[a["name"]]:
                errors.append(f"{a['name']} must be inspected with camera '{modes[a['name']]}'")
            if typed and a["name"] in floors and alt < floors[a["name"]] - 1e-6:
                errors.append(f"{a['name']} requires altitude >= {floors[a['name']]:.0f} m before inspection")
        elif op == "loiter":
            n_loiter += 1
            if a["kind"] == "point":
                _, d = envelope_distance(ep, np.asarray(a["xy"]))
                if d > ENVELOPE_R or in_nfz(ep, a["xy"]):
                    errors.append("loiter point is outside the authorized envelope")
                cap = MAX_LOITER
            else:
                cap = a["cap"]
            if a["seconds"] > cap:
                a["seconds"] = cap
                clamped += 1
        elif op == "send":
            if a["to"] not in ep["allowed_recipients"]:
                errors.append(f"recipient '{a['to']}' is not permitted")
            a["text"] = redact_numbers(a["text"])[:240]
    if n_loiter > 1:
        errors.append("at most one loiter inspection is permitted")
    if not acts or acts[-1]["op"] != "rth":
        errors.append("the plan must end with rth")
    inspected = {a["name"] for a in acts if a["op"] == "inspect"}
    missing = [s["name"] for s in sites if s["name"] not in inspected]
    log = execute(ep, acts)
    over = log["energy"] - ep["budget_m"]
    if missing:
        low = [s["name"] for s in sites if s["priority"] == "low"]
        if len(low) != 1 or missing != low or full_mission_energy(ep, obs) <= ep["budget_m"] + 1e-6:
            errors.append(f"sites {', '.join(missing)} are missing; only the lowest-priority site may be skipped, "
                          "and only if the full mission does not fit the budget")
    if over > 0:
        errors.append(f"energy {log['energy'] / 1000:.2f} km exceeds the budget {ep['budget_m'] / 1000:.2f} km")
    if log["nfz_violation"]:
        errors.append("the route enters the no-fly zone")
    return (len(errors) == 0), acts, errors, clamped


def resolve_unverified(ep, plan, obs):
    """Ablation without the verifier: handles are resolved, nothing else is checked."""
    acts, _ = resolve_shield(ep, plan, obs, allow_points=False)
    return acts
