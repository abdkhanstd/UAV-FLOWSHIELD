"""Deterministic edge planner without any LLM (reference baseline).

Optimal tour by exact search over routed distances, regex extraction of typed observations from the
documents, the same admission function and verifier as UAV-FlowShield. It never sends anything to a
cloud, so it bounds what is achievable on templated content; its weakness is brittleness to wording."""
import re

import numpy as np

from . import shield as SH

NUM = r"(\d+(?:\.\d+)?)"
SITE = re.compile(r"\bS(\d+)\b")
ALT = re.compile(r"(?i)(?:minimum|min\.?|at least|no lower than|not (?:below|lower than)|above|keep(?: it)? at)[^.\d]{0,40}?" + NUM + r"\s*(?:m\b|metres|meters)")
GRID = re.compile(r"(?i)\bE\s*" + NUM + r"[,\s]+N\s*" + NUM)
SECS = re.compile(r"(?i)" + NUM + r"\s*(?:s\b|sec\b|secs\b|seconds)")
MSI = re.compile(r"(?i)multi-?spectral")


def extract(doc):
    """Regex extraction into the typed schema used by admission."""
    t = doc["text"]
    obs = []
    sites = SITE.findall(t)
    site = f"S{sites[0]}" if sites else None
    m = ALT.search(t)
    if m and site and re.search(r"(?i)altitude|above ground|keep", t):
        obs.append({"kind": "min_altitude", "site": site, "min_altitude_m": float(m.group(1))})
    if MSI.search(t) and site:
        obs.append({"kind": "crop_stress", "site": site})
    g = GRID.search(t)
    if g and re.search(r"(?i)loiter|hover|inspection|pass|image", t):
        sec = SECS.search(t)
        obs.append({"kind": "inspection_request", "site": site, "e": float(g.group(1)), "n": float(g.group(2)),
                    "loiter_seconds": float(sec.group(1)) if sec else 60.0})
    return {"observations": obs}


# --------------------------------------------------------------------------- stronger parser (det_plus)
# Number words, units (feet, minutes), several phrasings per slot and several requests per document. It is the
# "modest parser extension" a maintainer would write after the basic patterns fail on reworded documents.
_W1 = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                   "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                   "eighteen", "nineteen"])}
_W10 = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUMTOK = set(_W1) | set(_W10) | {"hundred", "and"}


def _words_to_digits(text):
    toks = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?|[^\sA-Za-z\d]", text)
    out, i = [], 0
    while i < len(toks):
        parts = toks[i].lower().split("-")
        if all(p in _NUMTOK and p != "and" for p in parts):
            j, cur, total, seen = i, 0, 0, False
            while j < len(toks):
                ps = toks[j].lower().split("-")
                if not all(p in _NUMTOK for p in ps):
                    break
                if ps == ["and"] and not (seen and j + 1 < len(toks) and toks[j + 1].lower().split("-")[0] in _NUMTOK):
                    break
                for p in ps:
                    if p in _W1:
                        cur += _W1[p]
                    elif p in _W10:
                        cur += _W10[p]
                    elif p == "hundred":
                        cur = max(cur, 1) * 100
                seen = True
                j += 1
            out.append(str(cur + total))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


_SITE2 = re.compile(r"(?i)\bsite\s*-?\s*s?(\d+)\b|\bS\s*(\d+)\b")
_GRID2 = re.compile(r"(?i)\bE\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[,;/]?\s*(?:and\s*)?N\s*[:=]?\s*(\d+(?:\.\d+)?)"
                    r"|easting\s*[:=]?\s*(\d+(?:\.\d+)?)\D{1,14}?northing\s*[:=]?\s*(\d+(?:\.\d+)?)")
_LEN = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(m\b|metres|meters|metre|meter|ft\b|feet|foot)")
_DUR = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(seconds|second|secs|sec|s\b|minutes|minute|mins|min\b)")
_ALTCUE = re.compile(r"(?i)altitude|above ground|agl|descend|height|stay at or above|keep at least|at least|minimum")
_INTENT = re.compile(r"(?i)loiter|hover|hold|stay|look|check|inspect|photo|image|shot|pass|orbit|overhead|aerial")


def _sites_before(text, pos):
    ms = [m for m in _SITE2.finditer(text) if m.start() <= pos]
    if ms:
        m = ms[-1]
        return f"S{m.group(1) or m.group(2)}"
    ms = list(_SITE2.finditer(text))
    return f"S{ms[0].group(1) or ms[0].group(2)}" if ms else None


def extract_plus(doc):
    """Typed observations from reworded text: number words, feet, minutes, several requests per document."""
    t = _words_to_digits(doc["text"])
    obs = []
    sites_all = [f"S{m.group(1) or m.group(2)}" for m in _SITE2.finditer(t)]
    # minimum altitude: a length in a sentence that talks about altitude
    for sent in re.split(r"(?<=[.!?;])\s+", t):
        if _ALTCUE.search(sent) and not _GRID2.search(sent):
            ss = [f"S{m.group(1) or m.group(2)}" for m in _SITE2.finditer(sent)]
            for m in _LEN.finditer(sent):
                v = float(m.group(1)) * (0.3048 if m.group(2).lower() in ("ft", "feet", "foot") else 1.0)
                if ss:
                    obs.append({"kind": "min_altitude", "site": ss[0], "min_altitude_m": round(v, 1)})
                    break
    if MSI.search(t) and sites_all:
        obs.append({"kind": "crop_stress", "site": sites_all[0]})
    grids = list(_GRID2.finditer(t))
    mentions = [(m.start(), f"S{m.group(1) or m.group(2)}") for m in _SITE2.finditer(t)]
    for gi, g in enumerate(grids):
        e = float(g.group(1) or g.group(3))
        n = float(g.group(2) or g.group(4))
        lo = grids[gi - 1].end() if gi else 0
        hi = grids[gi + 1].start() if gi + 1 < len(grids) else len(t)
        window = t[max(lo, g.start() - 160): min(hi, g.end() + 160)]
        if not _INTENT.search(window):
            continue
        after = t[g.end(): min(hi, g.end() + 200)]
        d = _DUR.search(after) or _DUR.search(t[max(lo, g.start() - 120): g.start()])
        sec = 60.0
        if d:
            sec = float(d.group(1)) * (60.0 if d.group(2).lower().startswith("min") else 1.0)
        near = [(abs(p - g.start()), s) for p, s in mentions if lo <= p < hi]
        site = min(near)[1] if near else (mentions[0][1] if mentions else None)
        obs.append({"kind": "inspection_request", "site": site, "e": e, "n": n, "loiter_seconds": sec})
    return {"observations": obs}


def merge_extractions(*results):
    """Union of several extractor outputs (duplicates are removed by admission)."""
    obs = []
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("observations"), list):
            obs.extend(r["observations"])
    return {"observations": obs}


def plan_for(ep, obs, D):
    """Exact tour; drop the lowest-priority site only if the full mission does not fit the budget."""
    from .executor import execute
    k = len(ep["sites"])
    full = list(range(1, k + 1))
    low = [i for i, s in enumerate(ep["sites"]) if s["priority"] == "low"]
    for sub in [full] + [[i for i in full if i - 1 != j] for j in low]:
        plan = SH.canonical_plan(ep, obs, sub)
        plan.insert(len(plan) - 1, {"op": "send", "to": "client_portal", "text": "Mission complete."})
        acts, errs = SH.resolve_shield(ep, plan, obs)
        if not errs and execute(ep, acts)["energy"] <= ep["budget_m"] + 1e-6:
            return plan
    return None


def run(ep, D, parser="basic", **adm):
    ex = extract_plus if parser == "plus" else extract
    extracted = [(d, ex(d)) for d in ep["docs"]]
    obs, rejected = SH.admit_observations(ep, extracted, **adm)
    plan = plan_for(ep, obs, D)
    if plan is None:
        return {"executed": False, "decision": "ABORT", "reason": "no feasible plan", "actions": [],
                "obs_admitted": len(obs), "obs_rejected": len(rejected), "attempted": False}
    ok, acts, errs, _ = SH.verify_plan(ep, plan, obs)
    if not ok:
        return {"executed": False, "decision": "ABORT", "reason": errs[0][:60], "actions": [],
                "obs_admitted": len(obs), "obs_rejected": len(rejected), "attempted": False}
    return {"executed": True, "decision": "ADMIT", "actions": acts, "obs_admitted": len(obs),
            "obs_rejected": len(rejected), "attempted": False}
