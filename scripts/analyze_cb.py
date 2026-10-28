"""Common-boundary baselines on T2 (flowshield/pipeline_cb.py): table and macros.
usage: python scripts/analyze_cb.py   -> ../paper/auto/table_cb_tbg.tex, numbers_cb_tbg.tex"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_eval import load, summarize, paired, fmt, rate  # noqa
from stats_util import mcnemar_exact  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
R = os.path.join(ROOT, "results", "eval")
BASES = [("undefended", "Undefended", "Undef"), ("prompt_only", "Prompt-only", "Prompt"), ("spotlight_guard", "Spotlight-Guard", "Spot"),
         ("prv", "Prompt-RAG-Validator", "Prv"), ("eamsr", "EAMSR", "Eamsr"), ("mcp_pep", "MCP-PEP", "Pep")]


def main():
    df = load([os.path.join(R, f) for f in ("t2_gemma_base.jsonl", "t2_gemma_fs.jsonl", "t2_gemma_cb.jsonl")])
    meths = []
    for b, _, _ in BASES:
        meths += [b, b + "_rep", b + "_cb"]
    meths += ["flowshield"]
    S = summarize(df, meths)
    cols = [("Attempted", "attempt", False), ("ASR", "asr", True), ("Adaptive ASR", "adapt", True), ("Without AD1t", "adapt_nt", True), ("MCR", "mcr", True), ("Exposure", "exposure", False)]
    L = ["\\begin{tabular}{llcccccc}", "\\toprule", "Baseline & Variant & " + " & ".join(c[0] for c in cols) + " \\\\", "\\midrule"]
    macros = []
    var = {"": "as published", "_rep": "+ repair", "_cb": "+ repair + common boundary"}
    for b, name, code in BASES:
        for sfx in ("", "_rep", "_cb"):
            m = b + sfx
            if m not in S:
                continue
            r = S[m]
            cells = [fmt(r[k], ci=ci) if r.get(k) else "--" for _, k, ci in cols]
            L.append((f"{name}" if sfx == "" else "") + f" & {var[sfx]} & " + " & ".join(cells) + " \\\\")
            if sfx == "_cb":
                for key, mk in (("asr", "asr"), ("adapt", "adapt"), ("adapt_nt", "adaptnt"), ("mcr", "mcr"), ("attempt", "attempt")):
                    if r.get(key):
                        macros.append(f"\\newcommand{{\\{mk}CbTbg{code}}}{{{100 * r[key][0]:.1f}}}")
                if r.get("asr"):
                    macros.append(f"\\newcommand{{\\asrCbTbg{code}Hi}}{{{100 * r['asr'][2]:.1f}}}")
        L.append("\\midrule")
    r = S["flowshield"]
    L.append("UAV-FlowShield & private view, typed channel, verifier & " + " & ".join(fmt(r[k], ci=ci) if r.get(k) else "--" for _, k, ci in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_cb_tbg.tex"), "w").write("\n".join(L) + "\n")
    # paired comparisons with UAV-FlowShield: "better" = FlowShield resists the attack / completes the mission and the other does not
    from scipy.stats import binomtest
    for key, field, filt, good in (("asr", "attack_success", lambda d: d.category == "attack", False),
                                   ("adapt", "attack_success", lambda d: (d.category == "adaptive") & (~d.goal.isin(["AD4", "AD6b"])), False),
                                   ("mcr", "success", lambda d: d.category.str.startswith("benign"), True)):
        sub = df[filt(df)]
        piv = sub.pivot_table(index="id", columns="method", values=field, aggfunc="first")
        for b, _, code in BASES:
            o = b + "_cb"
            if "flowshield" not in piv or o not in piv:
                continue
            ok = piv[["flowshield", o]].dropna().astype(bool)
            fs, ot = ok["flowshield"].values, ok[o].values
            better = int((fs & ~ot).sum() if good else (~fs & ot).sum())
            worse = int((~fs & ot).sum() if good else (fs & ~ot).sum())
            p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
            macros += [f"\\newcommand{{\\pair{key.capitalize()}CbTbg{code}Better}}{{{better}}}", f"\\newcommand{{\\pair{key.capitalize()}CbTbg{code}Worse}}{{{worse}}}",
                       f"\\newcommand{{\\pair{key.capitalize()}CbTbg{code}P}}{{{p:.3f}}}"]
    open(os.path.join(AUTO, "numbers_cb_tbg.tex"), "w").write("\n".join(macros) + "\n")
    for m in meths:
        if m in S:
            r = S[m]
            print(f"{m:24s} ASR {100 * r['asr'][0]:5.1f} adapt {100 * r['adapt'][0] if r.get('adapt') else float('nan'):5.1f} MCR {100 * r['mcr'][0]:5.1f} expo {100 * r['exposure'][0]:5.1f}")


if __name__ == "__main__":
    main()
