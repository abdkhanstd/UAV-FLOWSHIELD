"""Re-identification attacks on released mission geometry, and the defender's
attack-calibrated disclosure (the core privacy mechanism of UAV-FlowShield).

Threat: an honest-but-curious cloud planner provider (optionally colluding with
an author of external content) observes the planning view and holds the public
infrastructure map (all candidate sites of the inspected type, from OSM).
"""
import os

import numpy as np
from scipy.spatial import cKDTree

from .geo import SecretFrame, planar_laplace


def _rigid_fit(A, B):
    """Least-squares rotation/translation (proper or improper) mapping A->B."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:      # keep a proper rotation (reflection is handled by the caller)
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cb - ca @ R.T


KD_WORKERS = int(os.environ.get("FS_KD_WORKERS", "-1"))     # set to 1 when many processes run in parallel


class ShapeMatcher:
    """Rigid point-set registration of the released waypoints against the public map.

    Hypotheses are generated from one base pair of released points matched to all
    candidate pairs with a compatible distance; each hypothesis is scored by the
    robust residual of the remaining points; the best one is refined by Procrustes."""

    def __init__(self, cand_xy, max_hyp=3_000_000, chunk=400_000, max_pair_dist=25_000.0):
        self.C = np.asarray(cand_xy, float)
        self.tree = cKDTree(self.C)
        self.max_hyp = max_hyp
        self.chunk = chunk
        # all candidate pairs sorted by distance (both orientations are used later)
        C = self.C
        ia, ib, dd = [], [], []
        step = max(1, int(3_000_000 // max(1, len(C))))
        for s in range(0, len(C), step):
            D = np.linalg.norm(C[s:s + step, None, :] - C[None, :, :], axis=-1).astype(np.float32)
            r, c = np.nonzero((D <= max_pair_dist) & (np.arange(len(C))[None, :] > (np.arange(s, min(s + step, len(C)))[:, None])))
            ia.append((r + s).astype(np.int32))
            ib.append(c.astype(np.int32))
            dd.append(D[r, c])
        ia, ib, dd = np.concatenate(ia), np.concatenate(ib), np.concatenate(dd)
        o = np.argsort(dd, kind="stable")
        self.pa, self.pb, self.pd = ia[o], ib[o], dd[o]

    def _pairs(self, d, tau):
        lo = np.searchsorted(self.pd, d - tau, "left")
        hi = np.searchsorted(self.pd, d + tau, "right")
        a, b = self.pa[lo:hi].astype(np.int64), self.pb[lo:hi].astype(np.int64)
        return np.concatenate([a, b]), np.concatenate([b, a])

    def attack(self, V, tau, rng, extra=None, anchored=False, topk=64, icp_iters=6, weak=False, known=None,
               return_cands=False):
        """V: (k,2) released site positions (unknown rigid frame unless anchored).
        tau: residual tolerance (the attacker knows the noise scale).
        Stage 1: hypotheses from the longest released pair matched to every compatible registry pair,
                 scored by clipped residual of the remaining points.
        Stage 2: top-K hypotheses refined by iterative closest point; best final residual wins.
        known=(p, c): side information, released point p is registry entry c (the base pair then contains p).
        return_cands: also return every refined candidate as (score, assignment) for re-scoring by side information.
        Returns (assigned registry index per released point, mapped extra points, n hypotheses)."""
        V = np.asarray(V, float)
        k = len(V)
        D = np.linalg.norm(V[:, None] - V[None], axis=-1)
        iu = np.triu_indices(k, 1)
        if known is not None:
            p0 = int(known[0])
            q0 = int(np.argmax(np.where(np.arange(k) == p0, -1.0, D[p0])))
            i, j = p0, q0
            d = D[i, j]
            a, b = self._pairs(d, 1.5 * tau + 1.0)
            keep = a == int(known[1])
            a, b = a[keep], b[keep]
        else:
            if weak:   # ablation: short noisy base pair, single hypothesis, no refinement
                topk, icp_iters = 1, 0
                ds = D[iu]
                ok = ds >= max(4 * tau, 250.0)
                idx = int(np.argmin(np.where(ok, ds, np.inf))) if ok.any() else int(np.argmax(ds))
            else:
                idx = int(np.argmax(D[iu]))
            i, j = iu[0][idx], iu[1][idx]
            d = D[i, j]
            a, b = self._pairs(d, 1.5 * tau + 1.0)
        if anchored:
            R = 3.0 * tau + 100.0
            keep = (np.linalg.norm(self.C[a] - V[i], axis=1) < R) & (np.linalg.norm(self.C[b] - V[j], axis=1) < R)
            a, b = a[keep], b[keep]
        nh = len(a)
        if nh == 0:
            return (np.full(k, -1), None, 0) if not return_cands else (np.full(k, -1), None, 0, [])
        if nh > self.max_hyp:
            sel = rng.choice(nh, self.max_hyp, replace=False)
            a, b = a[sel], b[sel]
        rest = [m for m in range(k) if m not in (i, j)]
        cap = 2.0 * tau if weak else 3.5 * tau
        cands = []
        for refl in ((False,) if anchored else (False, True)):
            W = V * np.array([1.0, -1.0]) if refl else V
            vij = W[j] - W[i]
            ang_v = np.arctan2(vij[1], vij[0])
            mid_w = 0.5 * (W[i] + W[j])
            sc_all, th_all, mc_all = [], [], []
            for s in range(0, len(a), self.chunk):
                aa, bb = a[s:s + self.chunk], b[s:s + self.chunk]
                cab = self.C[bb] - self.C[aa]
                th = np.arctan2(cab[:, 1], cab[:, 0]) - ang_v
                c, sn = np.cos(th), np.sin(th)
                mid_c = 0.5 * (self.C[aa] + self.C[bb])
                score = np.minimum((np.linalg.norm(cab, axis=1) - d) ** 2, cap ** 2)
                for m in rest:
                    rel = W[m] - mid_w
                    px = mid_c[:, 0] + c * rel[0] - sn * rel[1]
                    py = mid_c[:, 1] + sn * rel[0] + c * rel[1]
                    dist, _ = self.tree.query(np.stack([px, py], 1), workers=KD_WORKERS)
                    score += np.minimum(dist, cap) ** 2
                sc_all.append(score)
                th_all.append(th)
                mc_all.append(mid_c)
            score, th, mid_c = np.concatenate(sc_all), np.concatenate(th_all), np.concatenate(mc_all)
            top = np.argpartition(score, min(topk, len(score) - 1))[:topk]
            for h in top:
                cands.append((refl, W, th[h], mid_c[h], mid_w))
        best = (np.inf, None)
        refined = []
        for refl, W, th, mid_c, mid_w in cands:
            c, sn = np.cos(th), np.sin(th)
            Rm = np.array([[c, -sn], [sn, c]])
            R2, t2 = Rm, mid_c - mid_w @ Rm.T
            for _ in range(icp_iters):
                P = W @ R2.T + t2
                dist, ass = self.tree.query(P)
                g = dist < 2.5 * tau + 1.0
                if g.sum() < 3:
                    break
                R2, t2 = _rigid_fit(W[g], self.C[ass[g]])
            P = W @ R2.T + t2
            dist, ass = self.tree.query(P)
            fs = float(np.sum(np.minimum(dist, cap) ** 2))
            if return_cands:
                refined.append((fs, ass.copy(), (refl, R2, t2)))
            if fs < best[0]:
                best = (fs, (refl, R2, t2, ass))
        refl, R2, t2, ass = best[1]
        mapped = None
        if extra is not None:
            E = np.asarray(extra, float)
            E = E * np.array([1.0, -1.0]) if refl else E
            mapped = E @ R2.T + t2
        if return_cands:
            return ass, mapped, nh, refined
        return ass, mapped, nh


def tau_for_scale(s):
    """Attacker tolerance for planar-Laplace noise with mean radius s (95% radius ~2.4 s)."""
    return max(2.4 * s, 5.0)


def release_view(rng, sites_xy, home_xy, scale, frame=True):
    """Release mechanism: optional secret rigid frame + planar Laplace noise (mean radius `scale`)."""
    eps = 2.0 / scale if scale > 0 else None
    P = np.vstack([sites_xy, home_xy[None]])
    P = P + planar_laplace(rng, len(P), eps)
    if frame:
        P = SecretFrame(rng).fwd(P)
    return P[:-1], P[-1]


def correct_ids(assigned, true_idx, C, tol=25.0):
    """Per-site correctness (assigned candidate within tol metres of the true site)."""
    ok = np.zeros(len(true_idx), bool)
    for m, (a, t) in enumerate(zip(assigned, true_idx)):
        if a >= 0:
            ok[m] = np.linalg.norm(C[a] - C[t]) <= tol
    return ok


def reid_risk(matcher, C, true_idx, home_xy, scale, rng, trials, frame=True):
    """Monte-Carlo estimate of the per-site identification rate and home localisation error.

    frame=True : secret rigid frame + noise (attacker must register the pattern against the registry)
    frame=False: absolute coordinates + noise (attacker registers the pattern near the released positions)"""
    rates, herr = [], []
    S = C[true_idx]
    for _ in range(trials):
        V, h = release_view(rng, S, home_xy, scale, frame=frame)
        ass, mapped, _ = matcher.attack(V, tau_for_scale(scale), rng, extra=h[None], anchored=not frame)
        herr.append(np.linalg.norm(mapped[0] - home_xy) if mapped is not None else np.inf)
        rates.append(correct_ids(ass, true_idx, C).mean())
    return float(np.mean(rates)), float(np.median(herr))


def calibrate_scale(matcher, C, true_idx, home_xy, grid, target, rng, trials=8):
    """Attack-calibrated disclosure: smallest noise scale whose simulated per-site
    re-identification rate is at most `target` (run on the trusted edge)."""
    for s in grid:
        r, _ = reid_risk(matcher, C, true_idx, home_xy, s, rng, trials, frame=True)
        if r <= target:
            return s, r
    return grid[-1], r


# --------------------------------------------------------------------------- risk with uncertainty
def reid_draws(matcher, C, true_idx, home_xy, scale, rng, n, frame=True, n_release=1):
    """n independent simulated attacks. Returns per-draw per-site identification rate and the per-draw
    indicator that at least one inspected site was identified (mission-level event)."""
    S = C[true_idx]
    rates, anyhit = [], []
    for _ in range(n):
        if n_release == 1:
            V, h = release_view(rng, S, home_xy, scale, frame=frame)
            tau = tau_for_scale(scale)
        else:
            V, tau = fuse_releases(rng, S, home_xy, scale, n_release)
        ass, _, _ = matcher.attack(V, tau, rng, extra=None, anchored=not frame)
        ok = correct_ids(ass, true_idx, C)
        rates.append(float(ok.mean()))
        anyhit.append(float(ok.any()))
    return np.array(rates), np.array(anyhit)


def fuse_releases(rng, S, home_xy, scale, n_release):
    """Averaging attacker: the same sites are released n times with independent frames and noise (the handles give
    the correspondence). The releases are registered onto the first one and averaged, which shrinks the noise
    by roughly sqrt(n); the registry attack then runs on the fused points."""
    views = [release_view(rng, S, home_xy, scale, frame=True)[0] for _ in range(n_release)]
    ref = views[0]
    acc = [ref]
    for V in views[1:]:
        best = None
        for refl in (False, True):
            W = V * np.array([1.0, -1.0]) if refl else V
            R, t = _rigid_fit(W, ref)
            err = float(np.sum((W @ R.T + t - ref) ** 2))
            if best is None or err < best[0]:
                best = (err, W @ R.T + t)
        acc.append(best[1])
    fused = np.mean(acc, axis=0)
    return fused, tau_for_scale(scale / np.sqrt(n_release))


def wilson_upper(p_hat, n, z=1.645):
    """One-sided Wilson upper confidence limit for a proportion estimated from n independent draws."""
    p_hat = float(np.clip(p_hat, 0.0, 1.0))
    return float((p_hat + z * z / (2 * n) + z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def calibrate_scale_ucb(matcher, C, true_idx, home_xy, grid, target, rng, trials=30, pilot=8, z=1.645, n_release=1,
                        mission_level=False):
    """Uncertainty-aware calibration: the smallest scale whose one-sided upper confidence limit (Wilson, with the
    simulated attack draws as the independent units) on the mean re-identification rate is at most `target`.
    A pilot batch rejects clearly unsafe scales early. mission_level=True bounds the probability that at least one
    inspected site is identified. Returns (scale, mean estimate, upper limit)."""
    last = None
    for s in grid:
        r, a = reid_draws(matcher, C, true_idx, home_xy, s, rng, pilot, n_release=n_release)
        x = a if mission_level else r
        if x.mean() > 2.5 * target + 0.05:
            last = (s, float(x.mean()), 1.0)
            continue
        r2, a2 = reid_draws(matcher, C, true_idx, home_xy, s, rng, trials - pilot, n_release=n_release)
        x = np.concatenate([x, (a2 if mission_level else r2)])
        ucb = wilson_upper(x.mean(), len(x), z)
        last = (s, float(x.mean()), ucb)
        if ucb <= target:
            return last
    return last
