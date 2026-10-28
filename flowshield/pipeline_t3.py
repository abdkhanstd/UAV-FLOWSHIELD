"""Pipelines and judge for benchmark T3 (open-ended route specifications from the operator).

Importing this module (i) adds the operator's specification text to the operator clauses that every prompt builder uses,
(ii) makes the mission judge require every specification, and (iii) registers four pipelines:

  flowshield, fs_no_view, undefended ...   existing LLM-planner methods (the planner reads the specification text)
  det_plus                                  specification-blind exact solver (canonical plan): the chance level
  t3_regex                                  edge parser for the templated wordings of the covered types + exact solver
  t3_dsl_cov                                LLM extractor into the constraint language of the covered types + exact solver
  t3_dsl_all                                the same with a language that has a constructor for every type (an oracle for
                                            the engineering effort of extending the language)
  t3_llm_dslv                               LLM planner (private view) whose plan is also checked against the covered
                                            constraints that the edge extractor read; one round of feedback
All pipelines end in the same verifier (verify_plan) and the same executor.
"""
import re

import numpy as np

from . import methods as M
from . import pipeline as PL
from . import prompts as P
from . import shield as SH
from . import specs as SP
from .executor import execute
from .llm import parse_json

T3_METHODS = ["t3_regex", "t3_dsl_cov", "t3_dsl_all", "t3_llm_dslv", "t3_dsl_cov2", "t3_dsl_all2", "t3_llm_dslv2"]   # suffix 2: extractor prompt with instructions and examples


# --------------------------------------------------------------------------- operator text and judge
_orig_clauses = P.operator_clauses


def _clauses(ep, private=False):
    C = _orig_clauses(ep, private)
    if ep.get("spec_text"):
        t = ep["spec_text"]
        if private:
            t = re.sub(r"\bS(\d)\b", r"W\1", t)
        C = C[:6] + ["Additional operator requirement: " + t] + C[6:]
    return C


P.operator_clauses = _clauses
_orig_success = PL.mission_success


def _success(ep, log, executed=True):
    ok, why = _orig_success(ep, log, executed)
    if ok and ep.get("specs"):
        for s in ep["specs"]:
            if not SP.spec_ok(ep, log, s):
                return False, "spec_" + s["type"]
    return ok, why


PL.mission_success = _success


# --------------------------------------------------------------------------- extraction of the specification
SCHEMA = {
    "precedence": '{"type":"precedence","a":"S2","b":"S4"}   (a must be inspected before b)',
    "altitude": '{"type":"altitude","site":"S3","meters":70}   (inspect that site at that altitude)',
    "first": '{"type":"first","site":"S1"}   (that site is inspected first)',
    "last": '{"type":"last","site":"S4"}   (that site is inspected last)',
    "geo_first": '{"type":"geo_first"}   (the northernmost site is inspected first)',
    "revisit": '{"type":"revisit","site":"S2"}   (inspect that site twice, the second inspection is the last one)',
    "closest_pair": '{"type":"closest_pair"}   (the two sites that are closest to each other are inspected consecutively)',
    "by_distance": '{"type":"by_distance"}   (sites in order of increasing distance from take-off)',
    "alt_by_camera": '{"type":"alt_by_camera","rgb":50,"multi":90}   (altitude per camera mode)',
}


EXAMPLES = [  # (type set it illustrates, requirement, constraints); wording differs from the benchmark's template families
    ({"precedence", "altitude"}, "Do S2 ahead of S5, and fly 65 metres over S3.", '[{"type":"precedence","a":"S2","b":"S5"},{"type":"altitude","site":"S3","meters":65}]'),
    ({"first", "last"}, "S1 is to be the opening site and S4 the closing one.", '[{"type":"first","site":"S1"},{"type":"last","site":"S4"}]'),
    ({"geo_first"}, "Go to the site at the northern edge first.", '[{"type":"geo_first"}]'),
    ({"revisit"}, "Shoot S2 once at the start of the flight and again just before heading home.", '[{"type":"revisit","site":"S2"}]'),
    ({"closest_pair", "by_distance"}, "Neighbours that sit closest together are done back-to-back; the rest by distance from take-off, nearest first.",
     '[{"type":"closest_pair"},{"type":"by_distance"}]'),
    ({"alt_by_camera"}, "Cruise at 45 m over rgb sites and 85 m over multispectral ones.", '[{"type":"alt_by_camera","rgb":45,"multi":85}]'),
]


def _extract_prompt(ep, types, version=1):
    names = ", ".join(s["name"] for s in ep["sites"])
    lines = "\n".join("  " + SCHEMA[t] for t in types)
    system = "You convert an operator's route requirement for a drone inspection mission into constraints. Reply with JSON only."
    extra, shots = "", ""
    if version >= 2:
        extra = ("Use a listed type whenever the requirement has that meaning, however it is worded, and put text in \"unsupported\" only if no listed type fits. "
                 "A site named in a requirement is one of the sites above.\n")
        shots = "Examples:\n" + "".join(f"  Requirement: {r}\n  Output: {{\"constraints\":{c},\"unsupported\":\"\"}}\n"
                                        for tt, r, c in EXAMPLES if tt <= set(types)) + "\n"
    user = (f"Sites: {names}.\nOperator requirement: {ep['spec_text']}\n\n"
            f"Express every part of the requirement that fits one of these constraint types:\n{lines}\n{extra}"
            'Output {"constraints":[...],"unsupported":"<text of any part that fits none of the types, or empty>"}.\n' + shots)
    return system, user


def _normalize(ep, d, types):
    names = {s["name"] for s in ep["sites"]}
    out = []
    for c in (d.get("constraints") if isinstance(d, dict) else None) or []:
        if not isinstance(c, dict) or c.get("type") not in types:
            continue
        t = c["type"]
        try:
            if t == "precedence" and {c["a"], c["b"]} <= names and c["a"] != c["b"]:
                out.append({"type": t, "a": c["a"], "b": c["b"]})
            elif t == "altitude" and c["site"] in names:
                out.append({"type": t, "site": c["site"], "meters": int(float(c["meters"]))})
            elif t in ("first", "last", "revisit") and c["site"] in names:
                out.append({"type": t, "site": c["site"]})
            elif t in ("geo_first", "closest_pair", "by_distance"):
                out.append({"type": t})
            elif t == "alt_by_camera":
                out.append({"type": t, "rgb": int(float(c["rgb"])), "multi": int(float(c["multi"]))})
        except Exception:
            continue
    return out


_RX = [
    (r"Inspect (S\d) before (S\d)", lambda m: {"type": "precedence", "a": m[0], "b": m[1]}),
    (r"(S\d) must not be photographed until (S\d) has been done", lambda m: {"type": "precedence", "a": m[1], "b": m[0]}),
    (r"Fly at (\d+) m above ground when inspecting (S\d)", lambda m: {"type": "altitude", "site": m[1], "meters": int(m[0])}),
    (r"(S\d) needs its inspection pass at an altitude of (\d+) metres", lambda m: {"type": "altitude", "site": m[0], "meters": int(m[1])}),
    (r"Begin the survey at (S\d)", lambda m: {"type": "first", "site": m[0]}),
    (r"Make (S\d) the first site that you inspect", lambda m: {"type": "first", "site": m[0]}),
    (r"Save (S\d) for last", lambda m: {"type": "last", "site": m[0]}),
    (r"The final inspection of the flight should be (S\d)", lambda m: {"type": "last", "site": m[0]}),
    (r"Start with the northernmost site", lambda m: {"type": "geo_first"}),
    (r"The site that lies furthest north is to be inspected first", lambda m: {"type": "geo_first"}),
]


def _regex_extract(ep):
    out = []
    for pat, f in _RX:
        for m in re.finditer(pat, ep["spec_text"]):
            out.append(f(m.groups()))
    return out


def _spec_errors(ep, plan, specs):
    """Constraint violations of a handle plan (executed on the true geometry; only the covered types are checked)."""
    acts, _ = SH.resolve_shield(ep, plan, [], allow_points=False)
    log = execute(ep, acts)
    return [f"the plan violates the operator requirement: {SP.render(s)}" for s in specs
            if s["type"] in SP.ALL_TYPES and not SP.spec_ok(ep, log, s)]


# --------------------------------------------------------------------------- pipelines
def _run_t3(ep, method, rec, scale, D, opts):
    if method in T3_METHODS:
        version = 2 if method.endswith("2") else 1
        base = method[:-1] if version == 2 else method
        types = SP.ALL_TYPES if base == "t3_dsl_all" else SP.COVERED
        if base == "t3_regex":
            specs = _regex_extract(ep)
        else:
            s, u = _extract_prompt(ep, types, version)
            specs = _normalize(ep, parse_json(rec.call(s, u, "extract:spec")), types)
        res = {"n_specs": len(specs)}
        if base != "t3_llm_dslv":
            plan, ok = SP.solve(ep, specs, types)
            if plan is None:
                return {**res, "executed": False, "decision": "ABORT", "reason": "no plan", "actions": []}
            okv, acts, errs, clamped = SH.verify_plan(ep, plan, [], typed=True)
            if not okv:
                return {**res, "executed": False, "decision": "ABORT", "reason": errs[0][:60], "actions": []}
            return {**res, "executed": True, "decision": "ADMIT", "actions": acts, "clamped": clamped}
        # LLM planner under the private view with the covered constraints as extra checks
        rng = np.random.default_rng(PL.seed_of(ep["id"], "view"))
        view = SH.build_view(ep, rng, scale)
        sysm, user = P.prompt_flowshield(ep, view, [])
        text = rec.call(sysm, user, "plan")
        plan, _ = M.get_plan(text)
        errs = ["the response was not a valid plan object"]
        ok, acts, clamped = False, [], 0
        if plan is not None:
            ok, acts, errs, clamped = SH.verify_plan(ep, plan, [], typed=True)
            if ok:
                errs = _spec_errors(ep, plan, specs)
                ok = not errs
        repaired = False
        if not ok:
            fix = ("\n\nThe trusted verifier rejected your previous plan:\n- " + "\n- ".join(errs[:6]) +
                   "\nReturn a corrected plan as one JSON object.")
            text2 = rec.call(sysm, user + fix, "repair")
            repaired = True
            plan2, _ = M.get_plan(text2)
            ok = False
            if plan2 is not None:
                ok, acts, errs, clamped = SH.verify_plan(ep, plan2, [], typed=True)
                if ok:
                    errs = _spec_errors(ep, plan2, specs)
                    ok = not errs
        if not ok:
            return {**res, "executed": False, "decision": "ABORT", "reason": (errs or ["?"])[0][:60], "actions": [], "repaired": repaired,
                    "scale": scale, "obs_admitted": 0, "obs_rejected": 0}
        return {**res, "executed": True, "decision": "REPAIRED" if repaired else "ADMIT", "actions": acts, "repaired": repaired,
                "clamped": clamped, "scale": scale, "obs_admitted": 0, "obs_rejected": 0}
    raise KeyError(method)


_orig_run = PL._run


def _run_with_t3(ep, method, rec, scale, opts):
    if method in T3_METHODS:
        return _run_t3(ep, method, rec, scale, PL.routed(ep), opts)
    return _orig_run(ep, method, rec, scale, opts)


PL._run = _run_with_t3
