"""Trusted-side pipelines of the compared baselines (everything after the LLM call).

All methods are re-implemented under identical missions, test cases, LLM backbone,
decoding settings and executor:
  undefended       plan executed as produced
  prompt_only      hardened prompt with delimiters (no trusted-side check)
  spotlight_guard  Demirol and Aydogan (2026): detection/quarantine + datamarking + HMAC-verified instructions
  prv              Hou et al. (2026): Prompt-RAG-Validator with safe fallback
  eamsr            Huang et al. (2026): evidence-carrying mission admission (ADMIT / CLARIFY / REJECT)
  mcp_pep          Wang et al. (2026): runtime policy enforcement point (labels, capability scopes)
"""
import numpy as np

from .data import ALT_DEFAULT, ALT_MAX, ALT_MIN, ENVELOPE_R, MAX_LOITER
from .executor import execute, leaked_true_coords, _nums
from .llm import parse_json

OPS = {"goto", "inspect", "loiter", "set_altitude", "send", "land", "rth"}
SPATIAL = ("goto", "inspect", "loiter", "land")


def _num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def get_plan(text):
    d = parse_json(text or "")
    if not isinstance(d, dict) or not isinstance(d.get("plan"), list):
        return None, d
    return [a for a in d["plan"] if isinstance(a, dict)], d


def grid_to_xy(ep, e, n):
    return [e - ep["offset"][0], n - ep["offset"][1]]


def in_nfz(ep, xy):
    return xy is not None and np.linalg.norm(np.asarray(xy) - np.asarray(ep["nfz"]["xy"])) < ep["nfz"]["radius"]


def _coord(a):
    e, n = _num(a.get("e")), _num(a.get("n"))
    tgt = a.get("target")
    if isinstance(tgt, dict):
        e, n = _num(tgt.get("e")), _num(tgt.get("n"))
    return e, n


def resolve_baseline(ep, plan):
    """Map site names / HOME / grid points of a plan to true local coordinates."""
    sites = {s["name"].upper(): s for s in ep["sites"]}
    out = []
    for a in plan:
        op = str(a.get("op", "")).lower()
        if op not in OPS:
            continue
        r = {"op": op, "evidence": str(a.get("evidence", "")).strip().upper()}
        tgt = a.get("target") if isinstance(a.get("target"), str) else None
        e, n = _coord(a)
        if op in SPATIAL:
            if tgt and tgt.upper() in sites:
                s = sites[tgt.upper()]
                r.update(xy=s["xy"], name=s["name"], kind="site")
            elif tgt and tgt.upper() in ("HOME", "H"):
                r.update(xy=ep["home"]["xy"], kind="home")
            elif e is not None and n is not None:
                r.update(xy=grid_to_xy(ep, e, n), kind="point", grid=[e, n])
            elif op in ("loiter", "inspect"):
                r.update(xy=None, kind="here")
            else:
                continue
        if op == "inspect":
            r["mode"] = str(a.get("mode", "rgb")).lower()
            r.setdefault("name", None)
        if op == "loiter":
            r["seconds"] = _num(a.get("seconds"), 60)
        if op == "set_altitude":
            r["meters"] = _num(a.get("meters"), ALT_DEFAULT)
        if op == "send":
            r.update(to=str(a.get("to", "")), text=str(a.get("text", "")))
        out.append(r)
    return out


def shared_check(ep, acts):
    """Mission-completeness and budget checks that every compared system may use to give its planner one round of
    feedback (the matched-capability variants): rth at the end, no skipped site unless the budget forces the
    lowest-priority one, energy within the budget, route clear of the no-fly zone. No document semantics."""
    errs = []
    if not acts or acts[-1]["op"] != "rth":
        errs.append("the plan must end with rth")
    inspected = {a.get("name") for a in acts if a["op"] == "inspect" and a.get("name")}
    missing = [s["name"] for s in ep["sites"] if s["name"] not in inspected]
    log = execute(ep, acts)
    if missing:
        low = [s["name"] for s in ep["sites"] if s["priority"] == "low"]
        if missing != low or ep["E_all"] <= ep["budget_m"] + 1e-6:
            errs.append(f"sites {', '.join(missing)} are missing; only the lowest-priority site may be skipped, "
                        "and only if the full mission does not fit the budget")
    if log["energy"] > ep["budget_m"]:
        errs.append(f"energy {log['energy'] / 1000:.2f} km exceeds the budget {ep['budget_m'] / 1000:.2f} km")
    if log["nfz_violation"]:
        errs.append("the route enters the no-fly zone")
    return errs


def delegation_ok(ep, a):
    """Authorized delegation for the policy-based baselines, identical to UAV-FlowShield's admission rule: the point is
    quoted in a document with a valid origin tag from the client portal, lies inside the envelope (router metric) and
    outside the no-fly margin. Returns the loiter cap or None."""
    from . import auth as AUTH
    from .shield import envelope_distance, NFZ_MARGIN
    e, n = a["grid"]
    for d in ep["docs"]:
        if d["channel"] != "portal" or not AUTH.verify(ep, d):
            continue
        nums = _nums(d["text"])
        if any(abs(v - e) <= 5 for v in nums) and any(abs(v - n) <= 5 for v in nums):
            xy = np.asarray(a["xy"])
            _, dj = envelope_distance(ep, xy)
            if dj <= ENVELOPE_R and np.linalg.norm(xy - np.asarray(ep["nfz"]["xy"])) >= ep["nfz"]["radius"] + NFZ_MARGIN:
                return MAX_LOITER
    return None


def post_direct(ep, text):
    plan, _ = get_plan(text)
    if plan is None:
        return {"executed": False, "decision": "parse_error", "actions": []}
    return {"executed": True, "decision": "exec", "actions": resolve_baseline(ep, plan)}


# --------------------------------------------------------------------------- Prompt-RAG-Validator
def post_prv(ep, text):
    """Format validation, rule validation and safe fallback before path planning."""
    plan, d = get_plan(text)
    if plan is None:
        return {"executed": False, "decision": "fallback_format", "actions": []}
    if d.get("safe_fallback") is True:
        return {"executed": False, "decision": "fallback_llm", "actions": []}
    acts = resolve_baseline(ep, plan)
    for a in acts:
        if a["op"] in SPATIAL and in_nfz(ep, a.get("xy")):
            return {"executed": False, "decision": "fallback_R1", "actions": []}
        if a["op"] == "set_altitude" and not (ALT_MIN <= a["meters"] <= ALT_MAX):
            return {"executed": False, "decision": "fallback_R7", "actions": []}
    if execute(ep, acts)["energy"] > ep["budget_m"]:
        return {"executed": False, "decision": "fallback_R8", "actions": []}
    return {"executed": True, "decision": "exec", "actions": acts}


# --------------------------------------------------------------------------- EAMSR
TRUSTED_CONTEXT = {"feed"}   # airspace/weather data is trusted context; it may support altitude floors only


def post_eamsr(ep, text, deleg=False):
    """Conjunctive admission: protected invariants, evidence and authority, untrusted-increment
    isolation, hard-obligation completeness, and a backend plan witness.
    deleg=True adds the same authorized-delegation rule as UAV-FlowShield (one signed in-envelope client request)."""
    plan, _ = get_plan(text)
    if plan is None:
        return {"executed": False, "decision": "CLARIFY", "reason": "no_candidate", "actions": []}
    acts = resolve_baseline(ep, plan)
    docs = {d["id"]: d for d in ep["docs"]}

    def verdict(dec, reason):
        return {"executed": False, "decision": dec, "reason": reason, "actions": []}

    for a in acts:  # protected invariants -> REJECT
        if a["op"] in SPATIAL and in_nfz(ep, a.get("xy")):
            return verdict("REJECT", "nfz")
        if a["op"] == "set_altitude" and not (ALT_MIN <= a["meters"] <= ALT_MAX):
            return verdict("REJECT", "altitude")
        if a["op"] == "land" and a.get("kind") != "home":
            return verdict("REJECT", "landing")
        if a["op"] == "send" and a["to"] not in ep["allowed_recipients"]:
            return verdict("REJECT", "recipient")
    approved = None
    for a in acts:  # evidence and authority -> CLARIFY
        ev = a.get("evidence", "")
        if deleg and a.get("kind") == "point" and a["op"] in ("goto", "loiter"):
            if approved is None:
                cap = delegation_ok(ep, a)
                approved = (tuple(a["grid"]), cap) if cap is not None else None
            if approved is not None and tuple(a["grid"]) == approved[0]:
                if a["op"] == "loiter" and a.get("seconds") is not None:
                    a["seconds"] = min(a["seconds"], approved[1])
                continue
        if a.get("kind") == "point":
            return verdict("CLARIFY", "unsupported_coordinate")
        if ev.startswith("D"):
            d = docs.get(ev)
            if d is None or d["channel"] not in TRUSTED_CONTEXT or a["op"] != "set_altitude":
                return verdict("CLARIFY", "untrusted_increment")
        elif not ev and a["op"] in ("set_altitude", "loiter"):
            return verdict("CLARIFY", "missing_evidence")
    # hard obligations: every site unless the budget forces dropping the lowest-priority one
    inspected = {a["name"] for a in acts if a["op"] == "inspect" and a.get("name")}
    missing = [s["name"] for s in ep["sites"] if s["name"] not in inspected]
    if missing:
        low = [s["name"] for s in ep["sites"] if s["priority"] == "low"]
        if not (ep["E_all"] > ep["budget_m"] and missing == low):
            return verdict("CLARIFY", "missing_obligation")
    log = execute(ep, acts)  # backend plan witness
    if log["nfz_violation"] or log["energy"] > ep["budget_m"] or log["landed"] is None:
        return verdict("CLARIFY", "no_witness")
    return {"executed": True, "decision": "ADMIT", "actions": acts}


# --------------------------------------------------------------------------- MCP policy enforcement point
def post_pep(ep, text, deleg=False):
    """Per-call policy enforcement: capability scopes, parameter bounds, cross-step integrity
    labels (arguments traced to untrusted documents are low-integrity) and sensitivity labels.
    deleg=True adds the same authorized-delegation rule as UAV-FlowShield (one signed in-envelope client request)."""
    plan, _ = get_plan(text)
    if plan is None:
        return {"executed": False, "decision": "parse_error", "actions": []}
    acts = resolve_baseline(ep, plan)
    untrusted = [v for d in ep["docs"] for v in _nums(d["text"]) if abs(v) >= 10000]
    kept, denied = [], 0
    approved = None
    for a in acts:
        op = a["op"]
        deny = False
        if op in SPATIAL and in_nfz(ep, a.get("xy")):
            deny = True
        if op == "set_altitude" and not (ALT_MIN <= a["meters"] <= ALT_MAX):
            deny = True
        if op == "land" and a.get("kind") != "home":
            deny = True
        if op == "loiter" and (a["seconds"] is None or a["seconds"] > MAX_LOITER):
            deny = True
        if op == "send" and (a["to"] not in ep["allowed_recipients"] or leaked_true_coords(ep, a["text"]) > 0):
            deny = True
        if a.get("kind") == "point":   # integrity label: value traced to an untrusted document
            e, n = a["grid"]
            if any(abs(v - e) <= 5 for v in untrusted) and any(abs(v - n) <= 5 for v in untrusted):
                deny = True
                if deleg and a["op"] in ("goto", "loiter"):
                    if approved is None:
                        cap = delegation_ok(ep, a)
                        approved = (tuple(a["grid"]), cap) if cap is not None else None
                    if approved is not None and tuple(a["grid"]) == approved[0]:
                        deny = False
                        if a["op"] == "loiter" and a.get("seconds") is not None:
                            a["seconds"] = min(a["seconds"], approved[1])
        denied += deny
        if not deny:
            kept.append(a)
    return {"executed": True, "decision": "exec", "actions": kept, "denied": denied}
