"""Runtime supervision of a verified plan under mid-flight updates and degraded conditions (simulation).

The verifier of UAV-FlowShield checks a plan before take-off under the nominal energy model. This module adds a
trusted runtime layer and measures its effect in a time-stepped simulation (not
PX4 software-in-the-loop): a point-mass aircraft with constant airspeed, a crab-angle follower fed by noisy position
fixes, Ornstein-Uhlenbeck gusts, a battery of `capacity` metre-equivalents and scheduled events:

  wind_step     the mean wind rises mid-flight (the forecast feed announces it after a delay)
  nfz_grow      the no-fly zone grows by 60 % (announced after a delay)
  floor_update  the feed publishes a higher minimum altitude for a site that has not been inspected yet
  gps_degraded  the positioning noise rises from 3 m to 25 m (detected after 5 s)
  combined      wind step, zone growth and floor update together

Supervisors
  open_loop     executes the verified plan unchanged and ignores announcements
  reverify      on every announcement and every 10 s: re-runs the trusted checks on the remaining plan (wind-aware energy charge
                1 + 1.9 (W/V)^2 on the flight distance, router with the current zone, current floors); if the remaining plan no
                longer fits it drops the lowest-priority remaining site, then any remaining site, then returns home; a battery guard
                returns home when the remaining energy is below the energy needed to get home (with margin)

Outcomes per flight: unsafe (zone entered, battery exhausted before landing, or a site inspected below a floor that was
already published), completed (all sites of the verified plan inspected and landed at home), degraded (landed at home with a site dropped).
"""
import numpy as np

from .data import INSPECT_COST, LOITER_COST, CLIMB_COST, ALT_DEFAULT
from .geo import route

V_AIR = 12.0
DT = 1.0
TAU_GUST = 6.0
FIT = 1.9
MARGIN0 = 30.0
EVENTS = ("none", "wind_step", "nfz_grow", "floor_update", "gps_degraded", "combined")


def _wind_factor(w):
    return 1.0 + FIT * (w / V_AIR) ** 2


def _plan_energy(pos, tasks, R, nfz_c, factor, alt):
    """Charged energy of the remaining tasks from `pos`: routed distance x wind factor + fixed costs."""
    e, p = 0.0, np.asarray(pos, float)
    for t in tasks:
        op = t["op"]
        if op == "set_altitude":
            e += CLIMB_COST * max(0.0, t["meters"] - alt)
            alt = t["meters"]
        elif op in ("inspect", "goto", "loiter", "rth", "land") and t.get("xy") is not None:
            q = np.asarray(t["xy"], float)
            if np.linalg.norm(q - p) > 1.0:
                e += factor * route(p, q, [(nfz_c, R)], MARGIN0)[1]
            p = q
            if op == "inspect":
                e += INSPECT_COST
            if op == "loiter":
                e += LOITER_COST * t.get("seconds", 0)
    return e


def simulate(ep, acts, supervisor, event, rng, w0=3.0, gps0=3.0, reserve=0.10, t_max=9000.0):
    home = np.asarray(ep["home"]["xy"], float)
    nfz_c = np.asarray(ep["nfz"]["xy"], float)
    R_true = R_known = float(ep["nfz"]["radius"])
    capacity = ep["budget_m"] * (1.0 + reserve)
    sites = {s["name"]: s for s in ep["sites"]}
    low = {s["name"] for s in ep["sites"] if s["priority"] == "low"}
    tasks = [dict(a) for a in acts]
    planned = {a["name"] for a in tasks if a["op"] == "inspect"}
    for a in tasks:
        if a["op"] in ("rth", "land") and a.get("xy") is None:
            a["xy"] = home.tolist()
    # nominal flight time for scheduling the events
    nominal = sum(np.linalg.norm(np.asarray(a["xy"]) - np.asarray(b["xy"])) for a, b in zip(
        [{"xy": home}] + [t for t in tasks if t.get("xy") is not None], [t for t in tasks if t.get("xy") is not None])) / V_AIR
    ang = rng.uniform(0, 2 * np.pi)
    wdir = np.array([np.cos(ang), np.sin(ang)])
    w_mean = w0
    gps = gps0
    ev = []
    t_ev = float(rng.uniform(0.15, 0.55) * max(nominal, 60.0))
    remaining_names = [t["name"] for t in tasks if t["op"] == "inspect"]
    if event in ("wind_step", "combined"):
        ev.append({"t": t_ev, "type": "wind_step", "w": float(rng.choice([5.0, 7.0])), "lat": 60.0})
    if event in ("nfz_grow", "combined"):
        ev.append({"t": t_ev * 1.2, "type": "nfz_grow", "R": 1.6 * R_true, "lat": 30.0})
    if event in ("floor_update", "combined") and remaining_names:
        ev.append({"t": t_ev * 0.8, "type": "floor", "site": remaining_names[int(rng.integers(len(remaining_names)))], "floor": 90, "lat": 30.0})
    if event == "gps_degraded":
        ev.append({"t": t_ev, "type": "gps", "sigma": 25.0, "lat": 5.0})
    floors_pub, floors_known = {}, {}
    state = {"w_est": w0, "R_known": R_known, "floors": floors_known, "gps_est": gps0}
    pos, alt, t, used = home.copy(), ALT_DEFAULT, 0.0, CLIMB_COST * ALT_DEFAULT
    gust = np.zeros(2)
    out = {"inside_at_ann": False, "nfz": False, "nfz_post": False, "depleted": False, "floor": False, "floor_post": False, "inspected": [], "landed": False, "unsafe": False}
    poly, tgt_i = None, 1
    last_check = 0.0
    announced = []
    while tasks and t < t_max:
        # --- events: physical effect at t, announcement to the supervisor after the latency
        for e in ev:
            if not e.get("fired") and t >= e["t"]:
                e["fired"] = True
                if e["type"] == "wind_step":
                    w_mean = e["w"]
                elif e["type"] == "nfz_grow":
                    R_true = e["R"]
                elif e["type"] == "floor":
                    floors_pub[e["site"]] = e["floor"]
                elif e["type"] == "gps":
                    gps = e["sigma"]
            if e.get("fired") and not e.get("announced") and t >= e["t"] + e["lat"]:
                e["announced"] = True
                e["t_ann"] = t
                announced.append(e)
        changed = False
        for e in announced:
            if e.get("applied"):
                continue
            e["applied"] = True
            changed = True
            if e["type"] == "wind_step":
                state["w_est"] = e["w"]
            elif e["type"] == "nfz_grow":
                state["R_known"] = e["R"]
                out["inside_at_ann"] = bool(np.linalg.norm(pos - nfz_c) < e["R"] - 1.0)   # already inside the grown zone when the update arrives
            elif e["type"] == "floor":
                floors_known[e["site"]] = e["floor"]
            elif e["type"] == "gps":
                state["gps_est"] = e["sigma"]
        if supervisor == "reverify" and (changed or t - last_check >= 10.0):
            last_check = t
            tasks, poly_reset = _reverify(ep, tasks, pos, alt, capacity - used, state, nfz_c, sites, low)
            if poly_reset or changed:
                poly = None
        task = tasks[0]
        op = task["op"]
        # --- one step of motion or action
        if op in ("inspect", "goto", "loiter", "rth", "land") and task.get("xy") is not None and np.linalg.norm(np.asarray(task["xy"]) - pos) > 12.0:
            if poly is None:
                margin = MARGIN0 + (3.0 * state["gps_est"] if supervisor == "reverify" else 0.0)
                pts, _ = route(pos, np.asarray(task["xy"], float), [(nfz_c, state["R_known"])], margin)
                poly, tgt_i = pts, 1
            while tgt_i < len(poly) - 1 and np.linalg.norm(poly[tgt_i] - pos) < 12.0:
                tgt_i += 1
            aim = np.asarray(poly[min(tgt_i, len(poly) - 1)], float)
            fix = pos + rng.normal(0, gps, 2)
            des = aim - fix
            des = des / (np.linalg.norm(des) + 1e-9)
            w = w_mean * wdir + gust
            wperp = w @ np.array([-des[1], des[0]])
            u = (np.sqrt(max(V_AIR ** 2 - wperp ** 2, 1.0)) * des - wperp * np.array([-des[1], des[0]])) if abs(wperp) < V_AIR else V_AIR * des
            pos = pos + (u + w) * DT
            used += V_AIR * DT
            t += DT
            gust = gust + (-gust / TAU_GUST) * DT + 0.35 * w_mean * np.sqrt(2 * DT / TAU_GUST) * rng.normal(size=2)
        else:
            poly = None
            if task.get("xy") is not None:
                pos = np.asarray(task["xy"], float).copy()
            if op == "set_altitude":
                used += CLIMB_COST * max(0.0, task["meters"] - alt)
                alt = task["meters"]
            elif op == "inspect":
                used += INSPECT_COST
                out["inspected"].append(task["name"])
                if task["name"] in floors_pub and alt < floors_pub[task["name"]] - 1e-6:
                    out["floor"] = True
                    if task["name"] in floors_known:
                        out["floor_post"] = True
            elif op == "loiter":
                used += LOITER_COST * task.get("seconds", 0)
                t += task.get("seconds", 0)
            elif op in ("rth", "land"):
                out["landed"] = bool(np.linalg.norm(pos - home) < 60.0)
                tasks = []
                break
            tasks = tasks[1:]
            t += DT
        if np.linalg.norm(pos - nfz_c) < R_true - 1.0:
            out["nfz"] = True
            if R_true == R_known or state["R_known"] == R_true:
                out["nfz_post"] = True      # the zone was already known (or never changed): the entry is the supervisor's responsibility
        if used > capacity:
            out["depleted"] = True
            break
        if supervisor == "reverify" and t - last_check >= 5.0 and tasks and tasks[0]["op"] != "rth":
            # battery guard: return home when the remaining energy is below the energy needed to get home (with margin)
            home_e = _plan_energy(pos, [{"op": "rth", "xy": home}], state["R_known"], nfz_c, _wind_factor(state["w_est"]), alt)
            if capacity - used < 1.15 * home_e + 100.0:
                tasks = [{"op": "rth", "xy": home.tolist()}]
                poly = None
    done = set(out["inspected"])
    out["completed"] = bool(out["landed"] and not out["depleted"] and all(s in done for s in planned))
    out["degraded"] = bool(out["landed"] and not out["depleted"] and not out["completed"])
    out["unsafe"] = bool(out["nfz_post"] or out["depleted"] or out["floor_post"])        # violations after the update reached the supervisor, or exhaustion
    out["latency_unsafe"] = bool((out["nfz"] and not out["nfz_post"]) or (out["floor"] and not out["floor_post"]))   # inside the announcement delay
    out["energy_ratio"] = float(used / capacity)
    return out


def _reverify(ep, tasks, pos, alt, remaining, state, nfz_c, sites, low):
    """Trusted checks on the remaining plan; returns (tasks, reset_polyline)."""
    factor = _wind_factor(state["w_est"])
    names = [t["name"] for t in tasks if t["op"] == "inspect"]
    home = ep["home"]["xy"]
    modes = {s["name"]: s["mode"] for s in ep["sites"]}

    def build(order):
        out, a = [], alt
        for n in order:
            fl = state["floors"].get(n)
            if fl is not None and a < fl:
                out.append({"op": "set_altitude", "meters": fl})
                a = fl
            out.append({"op": "inspect", "name": n, "mode": modes[n], "xy": sites[n]["xy"]})
        out.append({"op": "rth", "xy": home})
        return out

    # keep an altitude change already in progress as the first task
    head = [tasks[0]] if tasks and tasks[0]["op"] == "set_altitude" else []
    new = build(names)
    cand = head + new if head and new and new[0]["op"] != "set_altitude" else new
    order = list(names)
    ok = _plan_energy(pos, cand, state["R_known"], nfz_c, factor, alt) <= remaining * 0.97
    while not ok and order:
        drop = next((n for n in order if n in low), order[-1])
        order.remove(drop)
        cand = build(order)
        ok = _plan_energy(pos, cand, state["R_known"], nfz_c, factor, alt) <= remaining * 0.97
    if not ok:
        cand = [{"op": "rth", "xy": home}]
    same = [(t["op"], t.get("name"), t.get("meters")) for t in cand] == [(t["op"], t.get("name"), t.get("meters")) for t in tasks]
    return (tasks if same else cand), (not same)
