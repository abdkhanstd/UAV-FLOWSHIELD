"""Common-boundary variants of the baselines (method names <baseline>_cb).

The question: how much of the security of UAV-FlowShield comes from the trusted boundary around the planner, and how
much from keeping the document and the exact geometry away from it? In a common-boundary variant the baseline keeps its
own prompt, its own language-processing mechanism (hardened prompt, spotlighting and detector, rule retrieval, ...) and
its own checks, and the plan it produces is additionally passed through the SAME trusted boundary as UAV-FlowShield:
the edge extractor and the admission function (channel authority, origin tag, router-metric envelope) produce the typed
observations, and the FlowShield verifier decides whether the plan may be executed. One round of repair feedback
(the baseline's own withholding reason and the verifier's errors) is given, as for the matched variants. The planner
still sees the raw documents and the absolute coordinates, which is what separates these variants from UAV-FlowShield.

Importing this module registers the variants in flowshield.pipeline (no existing method is changed).
"""
import numpy as np

from . import methods as M
from . import pipeline as PL
from . import prompts as P
from . import shield as SH
from .llm import parse_json

CB = [b + "_cb" for b in PL.BASELINES]
ENV_MATCH_M = 100.0       # a literal coordinate within 100 m of an admitted request position is mapped to its handle


def _to_handles(ep, plan, obs):
    """Plan of a baseline (site names, HOME, literal grid coordinates) -> plan over the handles of the trusted boundary:
    a literal coordinate that coincides with an admitted inspection request becomes that request's handle; any other
    literal coordinate is left in place and rejected by the verifier."""
    reqs = [o for o in obs if o["kind"] == "inspection_request"]
    out = []
    for a in plan:
        b = dict(a)
        if str(a.get("op", "")).lower() in ("goto", "loiter"):
            e, n = M._coord(a)
            if e is not None and n is not None and not (isinstance(a.get("target"), str) and a["target"]):
                xy = np.asarray(M.grid_to_xy(ep, e, n))
                for o in reqs:
                    if np.linalg.norm(xy - np.asarray(o["xy"])) <= ENV_MATCH_M:
                        b = {**a, "target": o["handle"]}
                        break
        out.append(b)
    return out


def _run_cb(ep, method, rec, scale, D, opts):
    base = method[:-3]
    deleg = base in ("eamsr", "mcp_pep")
    extracted = [(d, PL._extract(ep, d, rec, "llm", opts.get("extract_version", 2))) for d in ep["docs"]]
    obs, rejected = SH.admit_observations(ep, extracted)
    quarantined = []
    if base == "spotlight_guard":
        for d in ep["docs"]:
            s, u = P.prompt_detector(d)
            r = parse_json(rec.call(s, u, f"detect:{d['id']}"))
            if isinstance(r, dict) and r.get("injection") is True:
                quarantined.append(d["id"])
        sysm, user = P.prompt_spotlight(ep, D, quarantined)
    else:
        sysm, user = PL._PROMPT[base](ep, D)

    def attempt(u, tag):
        text = rec.call(sysm, u, tag)
        plan, _ = M.get_plan(text)
        raw = M.resolve_baseline(ep, plan) if plan else []
        post = PL._baseline_post(base, ep, text, deleg)
        if plan is None:
            return raw, post, False, [], ["the response was not a valid plan object"], 0
        ok, acts, errs, clamped = SH.verify_plan(ep, _to_handles(ep, plan, obs), obs, allow_points=False, typed=True)
        return raw, post, ok, acts, errs, clamped

    raw, post, ok, acts, errs, clamped = attempt(user, "plan")
    attempted = PL._attempted(ep, raw)
    repaired = False
    if not (post["executed"] and ok):
        fb = list(errs)
        if not post["executed"] and post.get("reason"):
            fb.append(f"the baseline's own checks withheld the plan ({post.get('decision')}: {post.get('reason')})")
        elif not post["executed"]:
            fb.append("the baseline's own checks withheld the plan")
        fix = ("\n\nThe trusted checker rejected your previous plan:\n- " + "\n- ".join(fb[:6]) +
               "\nReturn a corrected plan as one JSON object.")
        raw2, post2, ok2, acts2, errs2, clamped2 = attempt(user + fix, "repair")
        repaired = True
        raw, post, ok, acts, errs, clamped = raw2, post2, ok2, acts2, errs2, clamped2
        attempted = attempted or PL._attempted(ep, raw2)
    res = {"attempted": attempted, "repaired": repaired, "obs_admitted": len(obs), "obs_rejected": len(rejected)}
    res.update(PL.obs_audit(ep, obs))
    if quarantined:
        res["quarantined"] = quarantined
    if post["executed"] and ok:
        return {**res, "executed": True, "decision": "CB_ADMIT", "actions": acts, "clamped": clamped}
    why = (errs or [post.get("reason") or post.get("decision") or "withheld"])[0][:60]
    return {**res, "executed": False, "decision": "CB_ABORT", "reason": why, "actions": []}


_orig_run = PL._run


def _run_with_cb(ep, method, rec, scale, opts):
    if method.endswith("_cb") and method[:-3] in PL.BASELINES:
        return _run_cb(ep, method, rec, scale, PL.routed(ep), opts)
    return _orig_run(ep, method, rec, scale, opts)


PL._run = _run_with_cb
