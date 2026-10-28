"""Network-loss, edge-extractor and smaller-planner tables for T2.
usage: python scripts/analyze_net_edge.py   (reads results/eval/t2_gemma_fs.jsonl, t2_gemma_net*.jsonl, t2_gemma_edge_*.jsonl, t2_<model>.jsonl)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import load, rate, fmt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "eval")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
macros = []
EDGE_NAME = {"granite4.1_3b": "granite4.1:3b", "nemotron-3-nano_4b": "nemotron-3-nano:4b", "qwen3_1.7b": "qwen3:1.7b", "qwen3_8b": "qwen3:8b"}
EDGE_CODE = {"granite4.1_3b": "Granite", "nemotron-3-nano_4b": "Nemo", "qwen3_1.7b": "QwenSmall", "qwen3_8b": "QwenEight"}
WORD = {"0.1": "OnePct", "0.3": "ThreePct", "0.5": "FivePct", "1.0": "All"}


def mac(n, v):
    macros.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def benign(df):
    return df[df.category.str.startswith("benign")]


def net():
    ref = os.path.join(RES, "t2_gemma_fs.jsonl")
    files = sorted(glob.glob(os.path.join(RES, "t2_gemma_net*.jsonl")))
    if not files or not os.path.exists(ref):
        return
    base = load([ref])
    rows = {"0.0": benign(base)}
    for f in files:
        p = os.path.basename(f).replace("t2_gemma_net", "").replace(".jsonl", "")
        rows[p] = benign(load([f]))
    L = ["\\begin{tabular}{lcccccc}", "\\toprule", "Loss per attempt & \\multicolumn{3}{c}{Abort (hold) on failure} & \\multicolumn{3}{c}{Verified fallback on failure} \\\\",
         " & MCR (\\%) & Unreachable (\\%) & Executed unverified & MCR (\\%) & Fallback used (\\%) & Executed unverified \\\\", "\\midrule"]
    for p in sorted(rows, key=float):
        d = rows[p]
        a, b = d[d.method == "flowshield"], d[d.method == "fs_fallback"]
        if not len(a) or not len(b):
            continue
        ra, rb = rate(a.success), rate(b.success)
        L.append(f"{float(p) * 100:.0f}\\,\\% & {fmt(ra, ci=False)} & {100 * a.net_failed.mean():.1f} & 0 & {fmt(rb, ci=False)} & {100 * b.fallback_used.mean():.1f} & 0 \\\\")
        key = WORD.get(p, "Zero")
        mac(f"netMcrHold{key}", f"{100 * ra[0]:.1f}")
        mac(f"netMcrFall{key}", f"{100 * rb[0]:.1f}")
        mac(f"netFailed{key}", f"{100 * a.net_failed.mean():.1f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_net_tb.tex"), "w").write("\n".join(L) + "\n")


def edge():
    ref = os.path.join(RES, "t2_gemma_fs.jsonl")
    files = sorted(glob.glob(os.path.join(RES, "t2_gemma_edge_*.jsonl")))
    if not files or not os.path.exists(ref):
        return
    L = ["\\begin{tabular}{lccccccc}", "\\toprule", "Edge extractor & Effects recovered (\\%) & Audit false admissions (\\%) & MCR (\\%) & ASR (\\%) & Adaptive ASR (\\%) & of which AD6 (\\%) & Edge calls \\\\", "\\midrule"]
    srcs = [("gemma4:12b (same as planner)", load([ref]), "Same")] + [(EDGE_NAME.get(os.path.basename(f)[len("t2_gemma_edge_"):-6], os.path.basename(f)), load([f]), EDGE_CODE.get(os.path.basename(f)[len("t2_gemma_edge_"):-6])) for f in files]
    for name, df, code in srcs:
        d = df[df.method == "flowshield"]
        if not len(d):
            continue
        need = d[d.ext_need > 0]
        rec = rate(need.ext_hit == need.ext_need)
        fa = rate(d[d.category.str.startswith("benign")].ext_extra > 0)
        mcr = rate(benign(d).success)
        asr = rate(d[d.category == "attack"].attack_success)
        ad = rate(d[(d.category == "adaptive") & (~d.goal.isin(["AD4", "AD6b"]))].attack_success)
        ad6 = rate(d[(d.category == "adaptive") & (d.goal == "AD6")].attack_success)
        calls = d.edge_calls.mean()
        L.append(f"\\texttt{{{name}}} & {fmt(rec, ci=False)} & {fmt(fa, ci=False)} & {fmt(mcr, ci=False)} & {fmt(asr)} & {fmt(ad, ci=False)} & {fmt(ad6, ci=False)} & {calls:.2f} \\\\")
        c = code
        for k, v in (("EdgeRecall", rec), ("EdgeFalse", fa), ("EdgeMcr", mcr), ("EdgeAsr", asr), ("EdgeAdSix", ad6)):
            mac(f"{k}{c}", f"{100 * v[0]:.1f}")
        mac(f"EdgeAsr{c}Hi", f"{100 * asr[2]:.1f}")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_edge_tb.tex"), "w").write("\n".join(L) + "\n")


def planners():
    gem = os.path.join(RES, "t2_gemma_base.jsonl")
    pats = [("gemma4:12b", [os.path.join(RES, "t2_gemma_base.jsonl"), os.path.join(RES, "t2_gemma_fs.jsonl")], "Gemma"),
            ("qwen3:8b", [os.path.join(RES, "t2_qwen_base.jsonl"), os.path.join(RES, "t2_qwen_fs.jsonl")], "Qwen"),
            ("nemotron-3-nano:4b", [os.path.join(RES, "t2_nemotron-3-nano_4b.jsonl")], "Nemo"),
            ("granite4.1:3b", [os.path.join(RES, "t2_granite4.1_3b.jsonl")], "Granite"),
            ("qwen3:1.7b", [os.path.join(RES, "t2_qwen3_1.7b.jsonl")], "QwenSmall")]
    meths = [("undefended", "Undefended", "Undef"), ("prv", "PRV", "Prv"), ("mcp_pep", "MCP-PEP", "Pep"), ("mcp_pep_repdel", "MCP-PEP +rep+del", "PepRepDel"),
             ("flowshield", "FlowShield", "Fs"), ("fs_fallback", "FlowShield + fallback", "FsFall")]
    L = ["\\begin{tabular}{l" + "cc" * len(meths) + "}", "\\toprule", "Planner & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m[1]}}}" for m in meths) + " \\\\",
         " & " + " & ".join("ASR & MCR" for _ in meths) + " \\\\", "\\midrule"]
    any_row = False
    for name, fs, code in pats:
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            continue
        df = load(fs)
        cells = []
        for m, _, c in meths:
            d = df[df.method == m]
            if not len(d):
                cells += ["--", "--"]
                continue
            a, b = rate(d[d.category == "attack"].attack_success), rate(benign(d).success)
            cells += [fmt(a, ci=False), fmt(b, ci=False)]
            mac(f"plAsr{code}{c}", f"{100 * a[0]:.1f}")
            mac(f"plAsr{code}{c}Hi", f"{100 * a[2]:.1f}")
            mac(f"plMcr{code}{c}", f"{100 * b[0]:.1f}")
        L.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + " \\\\")
        any_row = True
    L += ["\\bottomrule", "\\end{tabular}"]
    if any_row:
        open(os.path.join(AUTO, "table_planners_tb.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    net()
    edge()
    planners()
    open(os.path.join(AUTO, "numbers_netedge_tb.tex"), "w").write("\n".join(macros) + "\n")
    print(len(macros), "macros")
