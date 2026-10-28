"""Small statistics helpers: Wilson intervals, exact McNemar, bootstrap, formatting."""
import numpy as np
from scipy import stats


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a, b):
    """a, b: boolean arrays (paired). Returns (n01, n10, two-sided exact p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return n01, n10, 1.0
    return n01, n10, float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)


def boot_ci(x, fn=np.mean, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(fn(x)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pct(p, d=1):
    return f"{100 * p:.{d}f}"


def pct_ci(k, n, d=1):
    p, lo, hi = wilson(k, n)
    return f"{100 * p:.{d}f}", f"[{100 * lo:.{d}f}, {100 * hi:.{d}f}]"
