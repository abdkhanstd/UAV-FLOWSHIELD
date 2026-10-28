"""Per-goal tables (G1-G6 and adaptive goals) and paired-test tables from results/eval/summary_<tag>.json.
usage: python scripts/analyze_goal.py <tag>"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import LABEL, CITE, BASE

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
GOALS = ["G1", "G2", "G3", "G4", "G5", "G6"]


def main(tag):
    D = json.load(open(os.path.join(ROOT, "results", "eval", f"summary_{tag}.json")))
    S, P = D["summary"], D["paired"]
    ads = sorted({g for r in S.values() for g in r.get("adapt_goal", {})})
    methods = [m for m in BASE + ["det_plus", "undefended_rep", "prompt_only_rep", "spotlight_guard_rep", "prv_rep", "eamsr_rep", "eamsr_repdel", "mcp_pep_rep", "mcp_pep_repdel", "flowshield", "fs_chan"] if m in S]
    cols = GOALS + ads
    L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "Method & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for c in cols:
            d = S[m].get("asr_goal", {}) if c in GOALS else S[m].get("adapt_goal", {})
            cells.append(f"{100 * d[c][0]:.0f}" if c in d else "--")
        L.append(f"{LABEL[m]}{(' ' + CITE[m]) if m in CITE else ''} & " + " & ".join(cells) + " \\\\")
        if m in ("mcp_pep", "det_plus", "eamsr_repdel"):
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_goal_{tag}.tex"), "w").write("\n".join(L) + "\n")
    # paired comparison with FlowShield
    pv = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "Compared with \\sys{} & \\multicolumn{3}{c}{Attack success (injection suite)} & \\multicolumn{3}{c}{Mission completion (benign)} \\\\",
         " & FS better & FS worse & $p$ & FS better & FS worse & $p$ \\\\", "\\midrule"]
    for m in methods:
        if m == "flowshield":
            continue
        a, c = P["asr"].get(m), P["mcr"].get(m)
        if not a or not c:
            continue
        L.append(f"{LABEL[m]} & {a['other_only']} & {a['ref_only']} & {pv(a['p'])} & {c['ref_only']} & {c['other_only']} & {pv(c['p'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, f"table_paired_{tag}.tex"), "w").write("\n".join(L) + "\n")
    print("written for", tag)


if __name__ == "__main__":
    main(sys.argv[1])
