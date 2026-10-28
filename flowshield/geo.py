"""Geometry utilities: local projection, secret mission frames, location noise,
NFZ-aware routing and the energy model used by the trusted executor."""
import math
import numpy as np

EARTH_R = 6371008.8


class LocalProjection:
    """Equirectangular ENU projection around a region centre (metres)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = EARTH_R * math.cos(math.radians(lat0)) * math.pi / 180.0
        self.ky = EARTH_R * math.pi / 180.0

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        return np.stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], axis=-1)

    def to_latlon(self, xy):
        xy = np.asarray(xy, float)
        return xy[..., 1] / self.ky + self.lat0, xy[..., 0] / self.kx + self.lon0


# ---------------------------------------------------------------- mission frames
class SecretFrame:
    """Secret rigid transform v = R x + t (optionally with reflection).

    Distances are preserved, so the planner can still reason about route length
    and battery, but absolute position and heading are hidden."""

    def __init__(self, rng, reflect=None, offset_scale=50000.0):
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        if reflect is None:
            reflect = rng.random() < 0.5
        if reflect:
            R = R @ np.array([[1, 0], [0, -1]])
        self.R = R
        self.t = rng.uniform(-offset_scale, offset_scale, size=2)

    def fwd(self, x):
        return np.asarray(x, float) @ self.R.T + self.t

    def inv(self, v):
        return (np.asarray(v, float) - self.t) @ self.R


def planar_laplace(rng, n, eps):
    """Planar Laplace noise (geo-indistinguishability) with parameter eps [1/m].

    Radius ~ Gamma(2, 1/eps); mean radius 2/eps."""
    if eps is None or eps <= 0 or not np.isfinite(eps):
        return np.zeros((n, 2))
    r = rng.gamma(2.0, 1.0 / eps, size=n)
    th = rng.uniform(0, 2 * np.pi, size=n)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)


def gaussian_noise(rng, n, sigma):
    if sigma <= 0:
        return np.zeros((n, 2))
    return rng.normal(0, sigma, size=(n, 2))


# ---------------------------------------------------------------- routing
def seg_circle_hit(p, q, c, r):
    """True if segment pq passes strictly inside circle (c, r)."""
    p, q, c = map(lambda a: np.asarray(a, float), (p, q, c))
    d = q - p
    L2 = d @ d
    if L2 < 1e-9:
        return np.linalg.norm(p - c) < r
    t = np.clip(((c - p) @ d) / L2, 0, 1)
    return np.linalg.norm(p + t * d - c) < r - 1e-6


def _tangent_points(p, c, r):
    v = p - c
    d = np.linalg.norm(v)
    if d <= r:
        return []
    a = math.acos(r / d)
    b = math.atan2(v[1], v[0])
    return [c + r * np.array([math.cos(b + a), math.sin(b + a)]),
            c + r * np.array([math.cos(b - a), math.sin(b - a)])]


def route(p, q, nfzs, margin=30.0):
    """Shortest path p->q around circular NFZs (single-circle detour model).

    Returns (polyline as list of points, length). If an endpoint lies inside an NFZ
    the straight segment is returned (the executor then reports a violation)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    pts = [p]
    cur = p
    length = 0.0
    for (c, r) in nfzs:
        c = np.asarray(c, float)
        rr = r + margin
        if np.linalg.norm(cur - c) <= rr or np.linalg.norm(q - c) <= rr:
            continue
        if not seg_circle_hit(cur, q, c, rr):
            continue
        best = None
        for tp in _tangent_points(cur, c, rr):
            for tq in _tangent_points(q, c, rr):
                a1 = math.atan2(*(tp - c)[::-1])
                a2 = math.atan2(*(tq - c)[::-1])
                da = abs((a1 - a2 + math.pi) % (2 * math.pi) - math.pi)
                L = np.linalg.norm(tp - cur) + rr * da + np.linalg.norm(q - tq)
                if best is None or L < best[0]:
                    best = (L, tp, tq, a1, a2)
        L, tp, tq, a1, a2 = best
        # sample the arc for plotting / violation checks
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        arc = [c + rr * np.array([math.cos(a1 + da * s), math.sin(a1 + da * s)]) for s in np.linspace(0, 1, 9)]
        pts.extend(arc)
        length += np.linalg.norm(tp - cur) + rr * abs(da)
        cur = tq
    pts.append(q)
    length += np.linalg.norm(q - cur)
    return pts, length


def tour_length(points, order, start, nfzs=()):
    seq = [start] + [points[i] for i in order] + [start]
    return sum(route(seq[i], seq[i + 1], nfzs)[1] for i in range(len(seq) - 1))


def dist_matrix(P):
    P = np.asarray(P, float)
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)


def best_tour(D, start_idx=0, subset=None):
    """Exact Held-Karp TSP over `subset` (indices into D, excluding start)."""
    nodes = [i for i in (subset if subset is not None else range(len(D))) if i != start_idx]
    n = len(nodes)
    if n == 0:
        return [], 0.0
    INF = float("inf")
    dp = {}
    for j in range(n):
        dp[(1 << j, j)] = (D[start_idx][nodes[j]], [j])
    for mask in range(1, 1 << n):
        for j in range(n):
            if (mask, j) not in dp:
                continue
            cost, path = dp[(mask, j)]
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                nc = cost + D[nodes[j]][nodes[k]]
                if (nm, k) not in dp or nc < dp[(nm, k)][0]:
                    dp[(nm, k)] = (nc, path + [k])
    full = (1 << n) - 1
    best = (INF, None)
    for j in range(n):
        cost, path = dp[(full, j)]
        c = cost + D[nodes[j]][start_idx]
        if c < best[0]:
            best = (c, path)
    return [nodes[j] for j in best[1]], best[0]
