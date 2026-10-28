"""FlowShield on T2 under the three calibration rules (mean10, mean05 = default selected on T1, ucb10), and on T1 when run.
usage: python scripts/analyze_scalepol.py"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import rate, fmt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "eval")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
macros = []


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def rows(path, method="flowshield"):
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    return df[df.method == method]


def main():
    sources = [("mean rule, 10\\,\\% target", "Mean", os.path.join(RES, "t2_gemma_alt.jsonl")),
               ("mean rule, 5\\,\\% target (default)", "Meanfive", os.path.join(RES, "t2_gemma_mean05.jsonl")),
               ("upper limit, 10\\,\\% target (family)", "Ucb", os.path.join(RES, "t2_gemma_fs.jsonl")),
               ("certified rule, $T$=100 (run added after the frozen evaluation)", "Cert", os.path.join(RES, "t2_gemma_cert.jsonl"))]
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "Calibration rule & Mean scale (m) & ASR (\\%) & Adaptive ASR (\\%) & Without AD1t (\\%) & MCR (\\%) \\\\", "\\midrule"]
    for lab, code, path in sources:
        if not os.path.exists(path):
            continue
        d = rows(path)
        atk = d[d.category == "attack"]
        ad = d[(d.category == "adaptive") & (~d.goal.isin(["AD4", "AD6b"]))]
        ben = d[d.category.str.startswith("benign")]
        a, b, c, m = rate(atk.attack_success), rate(ad.attack_success), rate(ad[ad.goal != "AD1t"].attack_success), rate(ben.success)
        L.append(f"{lab} & {d.scale.mean():.0f} & {fmt(a)} & {fmt(b, ci=False)} & {fmt(c)} & {fmt(m)} \\\\")
        for k, v in (("Asr", a), ("Adapt", b), ("AdaptNT", c), ("Mcr", m)):
            mac(f"scp{k}{code}", f"{100 * v[0]:.1f}")
        mac(f"scpScale{code}", f"{d.scale.mean():.0f}")
        mac(f"scpAsr{code}Hi", f"{100 * a[2]:.1f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_scalepol_tb.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_scalepol_tb.tex"), "w").write("\n".join(macros) + "\n")
    print(open(os.path.join(AUTO, "table_scalepol_tb.tex")).read())


def t1():
    src = [("gemma4:12b", "Gem", [("mean rule 10\\,\\%", "Mean", "t1_gemma_fs_mean10.jsonl"), ("mean rule 5\\,\\% (default)", "Meanfive", "t1_gemma_fs_mean05.jsonl"),
                                  ("upper limit 10\\,\\%", "Ucb", "t1_gemma_fs_ucb.jsonl")]),
           ("qwen3:8b", "Qwe", [("mean rule 10\\,\\%", "Mean", "t1_qwen_fs_mean10.jsonl"), ("mean rule 5\\,\\% (default)", "Meanfive", "t1_qwen_fs_mean05.jsonl"),
                                ("upper limit 10\\,\\%", "Ucb", "t1_qwen_fs_ucb.jsonl")])]
    L = ["\\begin{tabular}{llccccc}", "\\toprule", "Backbone & Calibration rule & Mean scale (m) & ASR (\\%) & Adaptive ASR (\\%) & MCR (\\%) & MCR with fallback (\\%) \\\\", "\\midrule"]
    for model, mc, rules in src:
        for lab, code, f in rules:
            p = os.path.join(RES, f)
            if not os.path.exists(p):
                continue
            d, fb = rows(p), rows(p, "fs_fallback")
            atk = d[d.category == "attack"]
            ad = d[(d.category == "adaptive") & (d.goal != "AD4")]
            ben = d[d.category.str.startswith("benign")]
            a, b, m = rate(atk.attack_success), rate(ad.attack_success), rate(ben.success)
            mf = rate(fb[fb.category.str.startswith("benign")].success) if len(fb) else None
            L.append(f"\\texttt{{{model}}} & {lab} & {d.scale.mean():.0f} & {fmt(a, ci=False)} & {fmt(b, ci=False)} & {fmt(m, ci=False)} & {fmt(mf, ci=False) if mf else '--'} \\\\")
            mac(f"scpT{mc}Mcr{code}", f"{100 * m[0]:.1f}")
            mac(f"scpT{mc}Asr{code}", f"{100 * a[0]:.1f}")
            mac(f"scpT{mc}Adapt{code}", f"{100 * b[0]:.1f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_scalepol_ta.tex"), "w").write("\n".join(L) + "\n")
    open(os.path.join(AUTO, "numbers_scalepol_tb.tex"), "w").write("\n".join(macros) + "\n")
    print(open(os.path.join(AUTO, "table_scalepol_ta.tex")).read())


if __name__ == "__main__":
    main()
    t1()
