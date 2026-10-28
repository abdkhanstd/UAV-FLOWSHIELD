"""Certified attack-calibrated disclosure.

A simpler calibration compares a Wilson upper limit of the mean of per-draw identification rates with the
target and takes the first grid scale that passes. This module provides instead a procedure whose risk statement
holds exactly under independence of the simulated attack draws:

* every simulated attack is scored on ONE uniformly chosen inspected site (the audited site), so the outcome Y_t is a
  Bernoulli variable whose mean is the per-site risk mu(s) of the mission at scale s (mission-level variant: Y_t is
  the event that some site is identified);
* a scale is certified when the exact binomial (Clopper-Pearson) test rejects H0: mu(s) >= rho at level alpha, i.e. when
  the number of hits among T audited sites is at most c_max(T, rho, alpha);
* the scales are tested in a fixed order from the largest to the smallest (fixed-sequence testing) and the ladder
  stops at the first scale that is not certified; the family-wise error of the whole ladder is then at most alpha
  without any correction for the number of scales, and the smallest certified scale is released.

If the first scale of the ladder cannot be certified the mission is returned as uncertified (no scale is claimed to
be safe).
"""
import numpy as np
from scipy.stats import binom

from .reid import (correct_ids, fuse_releases, release_view, tau_for_scale, wilson_upper)


def cp_max_hits(T, rho, alpha):
    """Largest hit count c with P(Bin(T, rho) <= c) <= alpha (-1 if even zero hits cannot be certified)."""
    c = -1
    while c + 1 <= T and binom.cdf(c + 1, T, rho) <= alpha:
        c += 1
    return c


def reid_draws_ok(matcher, C, true_idx, home_xy, scale, rng, n, n_release=1):
    """n independent simulated attacks; returns the (n, k) boolean matrix of per-site identification."""
    S = C[true_idx]
    out = np.zeros((n, len(true_idx)), bool)
    for t in range(n):
        if n_release == 1:
            V, _ = release_view(rng, S, home_xy, scale, frame=True)
            tau = tau_for_scale(scale)
        else:
            V, tau = fuse_releases(rng, S, home_xy, scale, n_release)
        ass, _, _ = matcher.attack(V, tau, rng, extra=None, anchored=False)
        out[t] = correct_ids(ass, true_idx, C)
    return out


def audited(ok, rng, mission_level=False):
    """Bernoulli outcomes of the draws: identification of one uniformly chosen site (or of any site)."""
    if mission_level:
        return ok.any(axis=1).astype(float)
    k = ok.shape[1]
    return ok[np.arange(len(ok)), rng.integers(k, size=len(ok))].astype(float)


def select_scale(draw, grid, rho, rule, rng, T=30, alpha=0.05, pilot=8, mission_level=False, start=None):
    """Select a noise scale from the ascending list `grid`.

    draw(scale, n) must return an (n, k) boolean identification matrix from n fresh independent attack simulations.
    rules
      mean     ascending; first scale whose mean per-site rate over 10 draws is at most rho
      wilson   ascending; pilot of `pilot` draws, then the first scale whose Wilson upper limit of the mean rate over T
               draws is at most rho (upper-limit rule)
      cp_asc   ascending; first scale certified by the exact test at level alpha (no correction for the scan)
      cp_bonf  ascending; the same at level alpha / len(grid)
      cp_fs    descending fixed sequence at level alpha, stopping at the first scale that fails (certified rule)
      cp_tab   table first: one exact test of the public table scale `start` at level alpha / 2; if it passes, that scale is
               released (a data-independent scale, T attacks); otherwise the full cp_fs ladder runs at level alpha / 2. The
               release of a scale with risk at least rho has probability at most alpha / 2 + alpha / 2 = alpha
    Returns {'scale', 'certified', 'draws', 'est'}; for an uncertified outcome the scale is the largest of the grid."""
    used = 0
    if rule == "mean":
        est = None
        for s in grid:
            ok = draw(s, 10)
            used += 10
            est = float(ok.mean())
            if est <= rho:
                return {"scale": s, "certified": False, "passed": True, "draws": used, "est": est}
        return {"scale": grid[-1], "certified": False, "passed": False, "draws": used, "est": est}
    if rule == "wilson":
        est = None
        for s in grid:
            ok = draw(s, pilot)
            used += pilot
            x = ok.any(1).astype(float) if mission_level else ok.mean(1)
            if x.mean() > 2.5 * rho + 0.05:
                est = float(x.mean())
                continue
            ok2 = draw(s, T - pilot)
            used += T - pilot
            x = np.concatenate([x, ok2.any(1).astype(float) if mission_level else ok2.mean(1)])
            est = float(x.mean())
            if wilson_upper(x.mean(), len(x)) <= rho:
                return {"scale": s, "certified": False, "passed": True, "draws": used, "est": est}
        return {"scale": grid[-1], "certified": False, "passed": False, "draws": used, "est": est}
    if rule in ("cp_asc", "cp_bonf"):
        a = alpha / len(grid) if rule == "cp_bonf" else alpha
        cmax = cp_max_hits(T, rho, a)
        for s in grid:
            ok = draw(s, pilot)
            used += pilot
            y = audited(ok, rng, mission_level)
            if y.sum() > cmax:
                continue                      # the pilot alone already excludes certification of this scale
            ok2 = draw(s, T - pilot)
            used += T - pilot
            y = np.concatenate([y, audited(ok2, rng, mission_level)])
            if y.sum() <= cmax:
                return {"scale": s, "certified": True, "passed": True, "draws": used, "est": float(y.mean())}
        return {"scale": grid[-1], "certified": False, "passed": False, "draws": used, "est": None}
    if rule == "cp_tab":
        c_tab = cp_max_hits(T, rho, alpha / 2)
        ok = draw(start, T)
        used += T
        y = audited(ok, rng, mission_level)
        if y.sum() <= c_tab:
            return {"scale": start, "certified": True, "passed": True, "draws": used, "est": float(y.mean()), "table": True}
        r = select_scale(draw, grid, rho, "cp_fs", rng, T=T, alpha=alpha / 2, pilot=pilot, mission_level=mission_level)
        r["draws"] += used
        r["table"] = False
        return r
    if rule == "cp_fs":
        cmax = cp_max_hits(T, rho, alpha)
        last = None
        for s in sorted(grid, reverse=True):
            ok = draw(s, pilot)
            used += pilot
            y = audited(ok, rng, mission_level)
            if y.sum() > cmax:
                break                         # fixed-sequence testing stops at the first failure
            ok2 = draw(s, T - pilot)
            used += T - pilot
            y = np.concatenate([y, audited(ok2, rng, mission_level)])
            if y.sum() > cmax:
                break
            last = (s, float(y.mean()))
        if last is None:
            return {"scale": max(grid), "certified": False, "passed": False, "draws": used, "est": None}
        return {"scale": last[0], "certified": True, "passed": True, "draws": used, "est": last[1]}
    raise ValueError(rule)
