"""Stochastic flight-dynamics check of verified plans (wind, gusts, GPS noise).

This is not PX4 software-in-the-loop. It is a point-mass aircraft with a constant airspeed, a crab-angle path follower
fed with noisy position fixes, and an Ornstein-Uhlenbeck gust model. It answers one question the nominal verifier cannot:
how much of the verified budget and no-fly clearance survive when the executed flight differs from the planned one?

Realised energy keeps the metres-equivalent accounting of the benchmark: flight energy is the air distance flown
(airspeed x time, which equals the path length in still air), inspections and loiter seconds cost what they cost in the
nominal model, climb is unchanged."""
import numpy as np

from .data import INSPECT_COST, LOITER_COST, CLIMB_COST, ALT_DEFAULT

V_AIR = 12.0         # m/s
DT = 0.5
TAU_GUST = 6.0       # s


def fly_polyline(poly, wind_mean, gust_sigma, gps_sigma, rng, nfz=None, v_air=V_AIR):
    """Follow a polyline; returns (flight time s, air distance m, minimum clearance to the no-fly circle m, max cross-track m)."""
    pts = [np.asarray(p, float) for p in poly]
    pos = pts[0].copy()
    gust = np.zeros(2)
    t, air, clearance, xtrack = 0.0, 0.0, np.inf, 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-6:
            continue
        d = seg / L
        n_ = np.array([-d[1], d[0]])
        for _ in range(int(L / 1.0 / max(v_air * DT / 1.0, 1e-6)) * 40 + 4000):
            fix = pos + rng.normal(0, gps_sigma, 2) if gps_sigma > 0 else pos
            if (b - fix) @ d <= 6.0:        # reached (or passed) the waypoint
                break
            # aim at a point 40 m ahead on the segment, measured from the (noisy) fix
            s = np.clip((fix - a) @ d + 40.0, 0.0, L)
            tgt = a + s * d
            des = tgt - fix
            des = des / (np.linalg.norm(des) + 1e-9)
            w = wind_mean + gust
            # airspeed vector whose sum with the wind points along `des` (crab solution); fall back to heading along des
            wperp = w @ np.array([-des[1], des[0]])
            if abs(wperp) < v_air:
                u = np.sqrt(v_air ** 2 - wperp ** 2) * des - wperp * np.array([-des[1], des[0]])
            else:
                u = v_air * des
            pos = pos + (u + w) * DT
            t += DT
            air += v_air * DT
            gust = gust + (-gust / TAU_GUST) * DT + gust_sigma * np.sqrt(2 * DT / TAU_GUST) * rng.normal(size=2)
            xtrack = max(xtrack, abs((pos - a) @ n_))
            if nfz is not None:
                clearance = min(clearance, float(np.linalg.norm(pos - nfz[0]) - nfz[1]))
    return t, air, clearance, xtrack


def realised_energy(ep, log, wind_speed, gps_sigma, rng, gust_frac=0.35):
    """Energy of a nominal-verified flight under wind. `log` is the executor log of the verified plan."""
    ang = rng.uniform(0, 2 * np.pi)
    wind = wind_speed * np.array([np.cos(ang), np.sin(ang)])
    nfz = (np.asarray(ep["nfz"]["xy"]), ep["nfz"]["radius"])
    t, air, clr, xt = fly_polyline(log["path"], wind, gust_frac * wind_speed, gps_sigma, rng, nfz)
    fixed = INSPECT_COST * len(log["inspections"]) + CLIMB_COST * ALT_DEFAULT
    loiter = sum(LOITER_COST * l["seconds"] for l in log["loiters"])
    climb = log["energy"] - sum_path_length(log) - fixed - loiter    # climb above the default altitude
    return air + fixed + loiter + max(climb, 0.0), clr, xt


def sum_path_length(log):
    p = np.asarray(log["path"])
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
