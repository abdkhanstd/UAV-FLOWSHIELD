"""Tables and macros for benchmark T3 (open-ended route specifications).
usage: python scripts/analyze_t3.py results/eval/t3_gemma.jsonl <tag>   -> ../paper/auto/table_t3_<tag>.tex, table_t3type_<tag>.tex, numbers_t3_<tag>.tex"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stats_util import wilson  # noqa
from flowshield.specs import COVERED, UNCOVERED  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
METHODS = [("det_plus", "Specification-blind solver", "Blind"), ("t3_regex", "Edge parser + solver (templated wordings)", "Regex"),
           ("t3_dsl_cov2", "LLM extractor + solver, language of the covered types", "DslCov"),
           ("t3_dsl_all2", "LLM extractor + solver, language extended to all nine types", "DslAll"),
           ("t3_dsl_cov", "LLM extractor + solver, covered types (first extractor prompt)", "DslCovOne"),
           ("t3_dsl_all", "LLM extractor + solver, all nine types (first extractor prompt)", "DslAllOne"),
           ("flowshield", "LLM planner, private view (UAV-FlowShield)", "Llm"), ("fs_no_view", "LLM planner, exact coordinates, same verifier", "LlmAbs"),
           ("t3_llm_dslv2", "LLM planner, private view, covered constraints as verifier feedback", "LlmDslv"),
           ("undefended", "LLM planner, exact coordinates, no checks", "Undef")]
GROUPS = [("cov", "Covered type", lambda e: e["kind"] == "single" and e["types"][0] in COVERED),
          ("unc", "Uncovered type", lambda e: e["kind"] == "single" and e["types"][0] in UNCOVERED),
          ("two", "Two specifications", lambda e: e["kind"] != "single"),
          ("all", "All episodes", lambda e: True)]


def main(path, tag, benchfile="bench_t3.json", screened=False):
    T = tag[0].upper() + tag[1:]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path))]
    bench = json.load(open(os.path.join(ROOT, "data", benchfile)))["episodes"]
    n_all = len(bench)
    if screened:
        bench = [e for e in bench if e.get("review_ok", True)]
    info = {e["id"]: {"kind": e["spec_kind"], "types": [s["type"] for s in e["specs"]], "author": e["spec_author"]} for e in bench}
    rows = [r for r in rows if r["id"] in info]
    n_g = {g: sum(f(v) for v in info.values()) for g, _, f in GROUPS}
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["id"]] = r
    macros, L = [], ["\\begin{tabular}{lcccc}", "\\toprule",
                     f"Pipeline & Covered type ({n_g['cov']}) & Uncovered type ({n_g['unc']}) & Two specs ({n_g['two']}) & All ({n_g['all']}) \\\\", "\\midrule"]
    for m, name, code in METHODS:
        if m not in by:
            continue
        cells = []
        for g, gname, f in GROUPS:
            ids = [i for i in by[m] if f(info[i])]
            k = sum(by[m][i]["success"] for i in ids)
            p, lo, hi = wilson(k, len(ids))
            cells.append(f"{100 * p:.1f}\\,\\scriptsize[{100 * lo:.1f}, {100 * hi:.1f}]\\normalsize")
            macros.append(f"\\newcommand{{\\tth{g.capitalize()}{code}{T}}}{{{100 * p:.1f}}}")
            macros.append(f"\\newcommand{{\\tth{g.capitalize()}{code}{T}N}}{{{len(ids)}}}")
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if m in ("det_plus", "t3_dsl_all2", "t3_dsl_all"):
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_t3_{tag}.tex"), "w").write("\n".join(L) + "\n")
    # per type (single-specification episodes)
    types = list(COVERED) + list(UNCOVERED)
    short = {"precedence": "before", "altitude": "alt.", "first": "first", "last": "last", "geo_first": "north", "revisit": "revisit",
             "closest_pair": "pair", "by_distance": "by dist.", "alt_by_camera": "alt./cam."}
    L = ["\\begin{tabular}{l" + "c" * len(types) + "}", "\\toprule", "Pipeline & " + " & ".join(short[t] for t in types) + " \\\\",
         " & \\multicolumn{5}{c}{covered} & \\multicolumn{4}{c}{uncovered} \\\\", "\\midrule"]
    for m, name, code in METHODS:
        if m not in by:
            continue
        cells = []
        for t in types:
            ids = [i for i in by[m] if info[i]["kind"] == "single" and info[i]["types"][0] == t]
            k = sum(by[m][i]["success"] for i in ids)
            cells.append(f"{k}/{len(ids)}")
            macros.append(f"\\newcommand{{\\tty{t.replace('_', '').capitalize()}{code}{T}}}{{{100 * k / max(len(ids), 1):.0f}}}")
        short_name = {"Blind": "Specification-blind solver", "Regex": "Edge parser + solver", "DslCov": "Extractor + solver, covered language",
                      "DslAll": "Extractor + solver, all nine types", "DslCovOne": "Extractor + solver, covered (first prompt)",
                      "DslAllOne": "Extractor + solver, all nine (first prompt)", "Llm": "Planner, private view", "LlmAbs": "Planner, exact coordinates, verifier",
                      "LlmDslv": "Planner, private view, constraint feedback", "Undef": "Planner, exact coordinates, no checks"}[code]
        L.append(f"{short_name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_t3type_{tag}.tex"), "w").write("\n".join(L) + "\n")
    macros.append(f"\\newcommand{{\\{tag}Kept}}{{{len(info)}}}")
    macros.append(f"\\newcommand{{\\{tag}Removed}}{{{n_all - len(info)}}}")
    open(os.path.join(AUTO, f"numbers_t3_{tag}.tex"), "w").write("\n".join(macros) + "\n")
    # failure analysis of the LLM planner
    for m in ("flowshield", "t3_llm_dslv2"):
        if m in by:
            fr = {}
            for r in by[m].values():
                fr[r["fail_reason"]] = fr.get(r["fail_reason"], 0) + 1
            print(m, fr)
    for m, name, code in METHODS:
        if m in by:
            print(f"{m:14s}", {g: round(100 * np.mean([by[m][i]['success'] for i in by[m] if f(info[i])]), 1) for g, _, f in GROUPS}, len(by[m]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["bench_t3.json"]), screened=len(sys.argv) > 4 and sys.argv[4] == "screened")
