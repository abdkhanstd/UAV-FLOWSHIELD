"""The final recommended configuration, run end to end (method fs_final and its operator-profile variants).

Everything the paper recommends is switched on at once, so that security, completion, disclosure, rejection and delay
refer to one configuration and not to a combination of separate tests:

  1. disclosure decision   certified scale against the blind registration attacker (exact test, T = 100, fixed sequence,
                           alpha = 5 %), released to the cloud planner only if certified and not above the cap (400 m);
                           otherwise the mission is planned on the edge (extractor, exact solver, verifier; no cloud call);
  2. typed admission       edge extraction, channel authority, origin tag, router-metric envelope (as in UAV-FlowShield);
  3. ticket lifecycle      every portal ticket carries issue time, expiry (300 s), nonce; it is checked by
                           authority.check on arrival at the edge (log-normal delivery delay) with a per-mission state;
  4. operator approval     an admitted spatial request is executed only with an operator approval token bound to the typed
                           action (site, resolved position, loiter); the operator is simulated with an ASSUMED profile
                           (probability of approving a legitimate / an attacker request, log-normal answer delay) and a
                           60 s deadline; a request without approval in time is dropped (and counted as a rejection);
  5. planning              cloud planner on the private view at the certified scale, one repair round, verified
                           deterministic fallback; or edge planning when the mission is not released.

Importing this module registers fs_final (attentive operator), fs_final_expert and fs_final_habit (an operator who approves
everything, which reproduces the tag-only design) in flowshield.pipeline. No existing method is changed."""
import json
import os
import threading

import numpy as np

from . import authority as AU
from . import deterministic as DT
from . import methods as M
from . import pipeline as PL
from . import prompts as P
from . import shield as SH

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAP = 400.0                     # cap on the noise scale that planning can afford (policy of the paper)
TTL = 300.0                     # declared ticket lifetime (s)
DEADLINE = 60.0                 # operator answer deadline for a spatial request (s)
DELIVERY = (20.0, 1.0)          # log-normal delivery delay of a ticket: median (s), sigma
OP_SIGMA = 0.8                  # log-normal sigma of the operator's answer delay (as in scripts/exp_confirm.py)
PROFILES = {"attentive": (0.97, 0.30, 25.0), "expert": (0.98, 0.05, 40.0), "habit": (1.00, 1.00, 15.0)}   # assumed, as in exp_confirm.py
METHODS = {"fs_final": "attentive", "fs_final_expert": "expert", "fs_final_habit": "habit"}
_DEC = None


def decisions():
    global _DEC
    if _DEC is None:
        _DEC = json.load(open(os.path.join(ROOT, os.environ.get("FS_DECISION", "data/decision_t2_final.json"))))["missions"]
    return _DEC


def _approval(ep, rec_obs, malicious, profile, state):
    """Ticket check and operator approval for one admitted spatial request. Returns (accepted, info)."""
    a_legit, a_attack, med = PROFILES[profile]
    rng = np.random.default_rng(PL.seed_of(ep["id"], rec_obs["source"], "ticket", profile))
    issued = 1000.0
    ticket = AU.issue(ep["operator"]["client"], ep["id"], "portal", rec_obs["source"], issued, TTL, nonce=f"{ep['id']}:{rec_obs['source']}")
    arrival = issued + float(rng.lognormal(np.log(DELIVERY[0]), DELIVERY[1]))
    action = {"kind": rec_obs["kind"], "site": rec_obs["site_name"], "xy": [round(v, 1) for v in rec_obs["xy"]], "loiter": rec_obs["value"]}
    delay = float(rng.lognormal(np.log(med), OP_SIGMA))
    approves = rng.random() < (a_attack if malicious else a_legit)
    info = {"fin_delivery": arrival - issued, "fin_op_delay": delay}
    if delay > DEADLINE:
        ok, why = AU.check(state, ticket, ep["id"], arrival + DEADLINE, True, action, None)
        return False, {**info, "fin_ticket": "deadline"}
    token = AU.approve(ticket["id"], action) if approves else None
    ok, why = AU.check(state, ticket, ep["id"], arrival + delay, True, action, token)
    return ok, {**info, "fin_ticket": why if not ok else "approved"}


def _run_final(ep, method, rec, D, opts):
    profile = METHODS[method]
    dec = decisions()[ep.get("orig_id", ep["id"])]
    extracted = [(d, PL._extract(ep, d, rec, "llm", 2)) for d in ep["docs"]]
    obs, rejected = SH.admit_observations(ep, extracted, envelope=True, auth="signed", metric="route", authority=True)
    state = AU.AuthorityState()
    info = {"fin_released": bool(dec["release"]), "fin_scale": dec["scale"], "fin_certified": bool(dec["certified"]),
            "fin_spatial": 0, "fin_ticket": None}
    docs = {d["id"]: d for d in ep["docs"]}
    kept = []
    for o in obs:
        if o["kind"] == "inspection_request":
            info["fin_spatial"] += 1
            ok, inf = _approval(ep, o, docs[o["source"]].get("attack") is not None, profile, state)
            info.update(inf)
            if not ok:
                rejected.append({"doc": o["source"], "kind": o["kind"], "why": inf["fin_ticket"]})
                continue
        kept.append(o)
    for k, o in enumerate(kept):           # handles stay consecutive after a request is dropped
        o["handle"] = f"O{k + 1}"
    obs = kept
    res = {"scale": dec["scale"] if dec["release"] else None, "obs_admitted": len(obs), "obs_rejected": len(rejected), **info}
    res.update(PL.obs_audit(ep, obs))

    def edge_plan(reason, extra=None):
        plan_f = DT.plan_for(ep, obs, D)
        if plan_f is not None:
            ok_f, acts_f, errs_f, _ = SH.verify_plan(ep, plan_f, obs)
            if ok_f:
                return {**res, **(extra or {}), "executed": True, "decision": reason, "actions": acts_f, "fallback_used": reason != "EDGE",
                        "plan_xy": None}
        return {**res, **(extra or {}), "executed": False, "decision": "ABORT", "reason": reason, "actions": [], "plan_xy": None}

    if not dec["release"]:
        return edge_plan("EDGE")
    rng = np.random.default_rng(PL.seed_of(ep["id"], "view"))
    view = SH.build_view(ep, rng, dec["scale"])
    sysm, user = P.prompt_flowshield(ep, view, obs)
    res["plan_xy"] = view["W"] + [view["H"]]
    text = PL._plan_call(rec, ep, sysm, user, "plan", opts)
    if text is None:
        return edge_plan("FALLBACK", {"net_failed": True})
    plan, _ = M.get_plan(text)
    if plan is not None:
        res["attempted"] = PL._attempted(ep, SH.resolve_shield(ep, plan, obs, allow_points=True)[0])
    ok, acts, errs, clamped = SH.verify_plan(ep, plan, obs, allow_points=False, typed=True) if plan is not None \
        else (False, [], ["the response was not a valid plan object"], 0)
    repaired = False
    if not ok:
        fix = ("\n\nThe trusted verifier rejected your previous plan:\n- " + "\n- ".join(errs[:6]) +
               "\nReturn a corrected plan as one JSON object.")
        text2 = PL._plan_call(rec, ep, sysm, user + fix, "repair", opts)
        repaired = True
        plan2 = M.get_plan(text2)[0] if text2 is not None else None
        ok = False
        if plan2 is not None:
            ok, acts, errs, clamped = SH.verify_plan(ep, plan2, obs, allow_points=False, typed=True)
    if not ok:
        r = edge_plan("FALLBACK", {"repaired": repaired})
        r["plan_xy"] = res["plan_xy"]
        return r
    return {**res, "executed": True, "decision": "REPAIRED" if repaired else "ADMIT", "actions": acts, "repaired": repaired, "clamped": clamped}


_orig_run = PL._run
_TL = threading.local()


def _run(ep, method, rec, scale, opts):
    if method in METHODS:
        out = _run_final(ep, method, rec, PL.routed(ep), opts)
        _TL.fin = {k: v for k, v in out.items() if k.startswith("fin_")}
        return out
    return _orig_run(ep, method, rec, scale, opts)


PL._run = _run
PL.FS_FAMILY.update(METHODS)
_orig_episode = PL.run_episode


def run_episode(ep, method, llm, scale=None, opts=None, llm_edge=None):
    """run_episode of pipeline.py, plus the fin_* fields of the final configuration (thread-local hand-over)."""
    _TL.fin = {}
    r = _orig_episode(ep, method, llm, scale, opts, llm_edge)
    r.update(getattr(_TL, "fin", {}))
    return r
