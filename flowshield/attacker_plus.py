"""Stronger variants of the registration attacker, to test whether the privacy numbers saturate.

The attacker of the paper (reid.ShapeMatcher.attack) generates hypotheses from the longest released pair, keeps the 64 best
and refines them with 6 ICP iterations. Variants here: more retained hypotheses (topk), more ICP iterations, and several
base pairs (the `ranks` longest pairs; the alignment with the smallest final residual wins)."""
import numpy as np

from .reid import _rigid_fit


def attack_core(M, V, tau, rng, rank=0, topk=64, icp_iters=6):
    """One run of the attack from the rank-th longest released pair. Returns (final clipped residual, assignment)."""
    V = np.asarray(V, float)
    k = len(V)
    D = np.linalg.norm(V[:, None] - V[None], axis=-1)
    iu = np.triu_indices(k, 1)
    order = np.argsort(D[iu])[::-1]
    idx = int(order[min(rank, len(order) - 1)])
    i, j = iu[0][idx], iu[1][idx]
    d = D[i, j]
    a, b = M._pairs(d, 1.5 * tau + 1.0)
    nh = len(a)
    if nh == 0:
        return np.inf, np.full(k, -1)
    if nh > M.max_hyp:
        sel = rng.choice(nh, M.max_hyp, replace=False)
        a, b = a[sel], b[sel]
    rest = [m for m in range(k) if m not in (i, j)]
    cap = 3.5 * tau
    cands = []
    for refl in (False, True):
        W = V * np.array([1.0, -1.0]) if refl else V
        vij = W[j] - W[i]
        ang_v = np.arctan2(vij[1], vij[0])
        mid_w = 0.5 * (W[i] + W[j])
        sc_all, th_all, mc_all = [], [], []
        for s in range(0, len(a), M.chunk):
            aa, bb = a[s:s + M.chunk], b[s:s + M.chunk]
            cab = M.C[bb] - M.C[aa]
            th = np.arctan2(cab[:, 1], cab[:, 0]) - ang_v
            c, sn = np.cos(th), np.sin(th)
            mid_c = 0.5 * (M.C[aa] + M.C[bb])
            score = np.minimum((np.linalg.norm(cab, axis=1) - d) ** 2, cap ** 2)
            for m in rest:
                rel = W[m] - mid_w
                px = mid_c[:, 0] + c * rel[0] - sn * rel[1]
                py = mid_c[:, 1] + sn * rel[0] + c * rel[1]
                dist, _ = M.tree.query(np.stack([px, py], 1), workers=1)
                score += np.minimum(dist, cap) ** 2
            sc_all.append(score)
            th_all.append(th)
            mc_all.append(mid_c)
        score, th, mid_c = np.concatenate(sc_all), np.concatenate(th_all), np.concatenate(mc_all)
        top = np.argpartition(score, min(topk, len(score) - 1))[:topk]
        for h in top:
            cands.append((W, th[h], mid_c[h], mid_w))
    best = (np.inf, None)
    for W, th, mid_c, mid_w in cands:
        c, sn = np.cos(th), np.sin(th)
        Rm = np.array([[c, -sn], [sn, c]])
        R2, t2 = Rm, mid_c - mid_w @ Rm.T
        for _ in range(icp_iters):
            P = W @ R2.T + t2
            dist, ass = M.tree.query(P)
            g = dist < 2.5 * tau + 1.0
            if g.sum() < 3:
                break
            R2, t2 = _rigid_fit(W[g], M.C[ass[g]])
        P = W @ R2.T + t2
        dist, ass = M.tree.query(P)
        fs = float(np.sum(np.minimum(dist, cap) ** 2))
        if fs < best[0]:
            best = (fs, ass)
    return best


def attack_multi(M, V, tau, rng, ranks=(0,), topk=64, icp_iters=6):
    best = (np.inf, None)
    for r in ranks:
        fs, ass = attack_core(M, V, tau, rng, rank=r, topk=topk, icp_iters=icp_iters)
        if ass is not None and fs < best[0]:
            best = (fs, ass)
    return best[1] if best[1] is not None else np.full(len(V), -1)
