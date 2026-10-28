"""Analysis of the follow-up experiments: cross-machine consistency, paraphrase generalization,
privacy-utility trade-off, sampled-decoding runs, cross-backbone comparison.
Writes LaTeX fragments and number macros into ../paper/auto and figures into ../paper/figs."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats_util import wilson, boot_ci
from analyze_main import load, summarize, LABEL, MAIN, rate

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
FIG = os.path.join(ROOT, "..", "paper", "figs")
CLS = ["dense", "medium", "sparse"]
macros = []
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def W(n):
    return "".join(WORDS[int(c)] for c in str(n))


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def exists(*p):
    return all(os.path.exists(os.path.join(RES, x)) for x in p)


def fmt(t, d=1):
    p, lo, hi = t
    return f"{100 * p:.{d}f}\\,\\scriptsize[{100 * lo:.{d}f}, {100 * hi:.{d}f}]\\normalsize"


# ----------------------------------------------------------------------------- consistency
def consistency():
    if not exists("consist_mata.jsonl", "main_gemma4-12b.jsonl"):
        return
    a = load([os.path.join(RES, "consist_mata.jsonl")])
    b = load([os.path.join(RES, "main_gemma4-12b.jsonl")])
    m = a.merge(b, on=["id", "method"], suffixes=("_m", "_l"))
    n = len(m)
    out = {}
    for f in ("decision", "executed", "success", "attack_success", "attempted"):
        out[f] = float((m[f + "_m"] == m[f + "_l"]).mean())
    mac("consN", n)
    for f, v in out.items():
        mac("cons" + f.replace("_", " ").title().replace(" ", ""), f"{100 * v:.1f}")
    # metric-level agreement (aggregate rates per machine)
    for k, d in (("Mata", a), ("Local", b.merge(a[["id", "method"]], on=["id", "method"]))):
        atk = d[d.category.isin(["attack", "adaptive"])]
        ben = d[d.category.str.startswith("benign")]
        mac(f"consAsr{k}", f"{100 * atk.attack_success.mean():.1f}")
        mac(f"consMcr{k}", f"{100 * ben.success.mean():.1f}")
    print("consistency", n, out)


# ----------------------------------------------------------------------------- paraphrase + backbone tables
def load_summary(tag):
    p = os.path.join(RES, f"summary_{tag}.json")
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def table_pair(tag_a, tag_b, name_a, name_b, out_name):
    A, B = load_summary(tag_a), load_summary(tag_b)
    if not A or not B:
        return
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         f"Method & \\multicolumn{{2}}{{c}}{{ASR (\\%)}} & \\multicolumn{{2}}{{c}}{{Adaptive ASR (\\%)}} & \\multicolumn{{2}}{{c}}{{MCR (\\%)}} \\\\",
         f" & {name_a} & {name_b} & {name_a} & {name_b} & {name_a} & {name_b} \\\\", "\\midrule"]
    for m in MAIN:
        if m in A and m in B:
            cell = lambda S, k: f"{100 * S[m][k][0]:.1f}" if k in S[m] else "--"
            L.append(f"{LABEL[m]} & {cell(A, 'asr')} & {cell(B, 'asr')} & {cell(A, 'asr_adapt')} & {cell(B, 'asr_adapt')} & {cell(A, 'mcr')} & {cell(B, 'mcr')} \\\\")
            if m == "mcp_pep":
                L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, out_name), "w").write("\n".join(L) + "\n")


def paraphrase_extra():
    A, B = load_summary("gemma"), load_summary("para")
    if not A or not B:
        return
    for m, c in (("det_template", "Det"), ("flowshield", "Fs"), ("undefended", "Undef"), ("mcp_pep", "Pep")):
        if m in A and m in B:
            for cat in ("benign_constraint", "benign_dynamic"):
                if cat in A[m].get("mcr_cat", {}) and cat in B[m].get("mcr_cat", {}):
                    k = cat.replace("benign_", "").capitalize()
                    mac(f"paraOrig{k}{c}", f"{100 * A[m]['mcr_cat'][cat][0]:.0f}")
                    mac(f"paraNew{k}{c}", f"{100 * B[m]['mcr_cat'][cat][0]:.0f}")
    if os.path.exists(os.path.join(ROOT, "data", "bench_para.json")):
        P = json.load(open(os.path.join(ROOT, "data", "bench_para.json")))["episodes"]
        mac("paraRewritten", sum(1 for e in P if e.get("paraphrased")))
        mac("paraTotal", len(P))


# ----------------------------------------------------------------------------- trade-off
def tradeoff():
    if not exists("trade_gemma4-12b.jsonl", "privacy.json"):
        return
    df = load([os.path.join(RES, "trade_gemma4-12b.jsonl")])
    df["cls"] = df.area.str.split("_").str[0]
    P = json.load(open(os.path.join(RES, "privacy.json")))
    grid = P["curve_grid"]
    scales = [0, 100, 300, 600, 800]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    rows = []
    for ax, c in zip(axs, CLS):
        d = df[df.cls == c]
        mc, lo, hi = [], [], []
        for s in scales:
            x = d[d.method == f"fs_s{s}"]
            p, l, h = rate(x.success) if len(x) else (np.nan,) * 3
            mc.append(p); lo.append(l); hi.append(h)
            rows.append((c, s, p, l, h, len(x)))
            mac(f"trade{c.capitalize()}S{W(s)}", f"{100 * p:.0f}" if len(x) else "--")
        xs = [max(s, 6) for s in scales]
        ax.errorbar(xs, mc, yerr=[np.array(mc) - np.array(lo), np.array(hi) - np.array(mc)], fmt="o-", color="#2c7a3f", lw=1.3,
                    ms=3.5, capsize=2, label="Mission completion (fixed scale)")
        f = d[d.method == "flowshield"]
        if len(f):
            p, l, h = rate(f.success)
            med = float(np.median(f.scale.dropna()))
            ax.errorbar([med], [p], yerr=[[p - l], [h - p]], fmt="D", color="#e08a1e", ms=5, capsize=2, label="Mission completion (calibrated)")
            ax.axvline(med, color="#e08a1e", lw=0.8, ls="-.")
            mac(f"tradeCal{c.capitalize()}", f"{100 * p:.0f}")
        cur = [x for x in P["results"] if x["area"].split("_")[0] == c and "curve" in x]
        gs = [g for g in grid if g in scales]
        ax.plot([max(g, 6) for g in gs], [np.mean([x["curve"][str(g)]["blind"] for x in cur]) for g in gs], "--", color="#1f5fa8", lw=1.2,
                label="Re-identified (blind attacker)")
        ax.plot([max(g, 6) for g in gs], [np.mean([x["curve"][str(g)]["aligned"] for x in cur]) for g in gs], ":", color="#c0392b", lw=1.4,
                label="Re-identified (frame compromised)")
        ax.set_xscale("log")
        ax.set_xticks([6, 100, 300, 600, 800])
        ax.set_xticklabels(["0", "100", "300", "600", "800"], fontsize=6.5)
        ax.minorticks_off()
        ax.set_title(f"{c.capitalize()} registry", fontsize=8)
        ax.set_xlabel("Noise scale (m)")
        ax.set_ylim(-0.02, 1.05)
    axs[0].set_ylabel("Fraction")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=6.3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(FIG, "tradeoff.pdf"))
    plt.close(fig)
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "Registry & 0\\,m & 100\\,m & 300\\,m & 600\\,m & 800\\,m \\\\", "\\midrule"]
    for c in CLS:
        r = {s: (p, l, h) for cc, s, p, l, h, n in rows if cc == c}
        L.append(c.capitalize() + " & " + " & ".join(f"{100 * r[s][0]:.0f}" for s in scales) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_trade.tex"), "w").write("\n".join(L) + "\n")
    print("tradeoff rows", len(rows))


# ----------------------------------------------------------------------------- sampled decoding
def sampling():
    files = [os.path.join(RES, f"samp_gemma4-12b_s{s}.jsonl") for s in (1, 2, 3)]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 2:
        return
    meths = ["undefended", "prv", "mcp_pep", "flowshield"]
    per = []
    for f in files:
        S = summarize(load([f]), meths)
        per.append(S)
    L = ["\\begin{tabular}{lccc}", "\\toprule", f"Method & ASR (\\%) & Adaptive ASR (\\%) & MCR (\\%) \\\\", "\\midrule"]
    for m in meths:
        cells = []
        for k in ("asr", "asr_adapt", "mcr"):
            v = [100 * S[m][k][0] for S in per if m in S and k in S[m]]
            cells.append(f"{np.mean(v):.1f} $\\pm$ {np.std(v, ddof=1):.1f}" if len(v) > 1 else "--")
            if v:
                mac(f"samp{k.replace('_', '').capitalize()}Mean{m.replace('_', '').capitalize()[:6]}", f"{np.mean(v):.1f}")
                mac(f"samp{k.replace('_', '').capitalize()}Sd{m.replace('_', '').capitalize()[:6]}", f"{np.std(v, ddof=1) if len(v) > 1 else 0:.1f}")
        L.append(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_sampling.tex"), "w").write("\n".join(L) + "\n")
    mac("sampRuns", len(files))
    print("sampling runs", len(files))


def dev_extraction():
    p = os.path.join(RES, "dev_extraction.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    v = {"1": "One", "2": "Two"}
    for ver, name in v.items():
        r = d[ver]
        mac(f"devExtDyn{name}", f"{100 * r['benign_dynamic']['rate']:.0f}")
        mac(f"devExtCon{name}", f"{100 * r['benign_constraint']['rate']:.0f}")
        mac(f"devExtFalse{name}", f"{100 * r['false_admit']['rate']:.1f}")
    mac("devExtDynN", d["2"]["benign_dynamic"]["n"])
    mac("devExtConN", d["2"]["benign_constraint"]["n"])
    mac("devExtFalseN", d["2"]["false_admit"]["n"])
    mac("devExtAdOneN", d["2"]["ad1_portal"]["n"])
    mac("devExtAdOneTwo", f"{100 * d['2']['ad1_portal']['rate']:.0f}")
    mac("devExtAdOneOne", f"{100 * d['1']['ad1_portal']['rate']:.0f}")


# ----------------------------------------------------------------------------- trade-off summary tables (both backbones)
def trade_summary():
    from stats_util import mcnemar_exact
    specs = [("gemma", "trade_gemma4-12b.jsonl", None, [0, 100, 300, 600, 800]),
             ("qwen", "trade_qwen3-8b.jsonl", "main_qwen3-8b.jsonl", [0, 300, 800])]
    for tag, f, mainf, scales in specs:
        if not exists(f):
            continue
        df = load([os.path.join(RES, f)])
        if mainf and exists(mainf):
            m = load([os.path.join(RES, mainf)])
            m = m[(m.method == "flowshield") & m.category.str.startswith("benign")]
            df = pd.concat([df, m], ignore_index=True)
        df["cls"] = df.area.str.split("_").str[0]
        meths = [f"fs_s{s}" for s in scales] + ["flowshield"]
        cols = [f"{s}" + r"\,m" for s in scales] + ["calibrated"]
        T = tag.capitalize()
        L = [r"\begin{tabular}{l" + "c" * len(meths) + "}", r"\toprule", "Registry & " + " & ".join(cols) + r" \\", r"\midrule"]
        for name, sub in [("All", df)] + [(c.capitalize(), df[df.cls == c]) for c in CLS]:
            cells = []
            for me in meths:
                x = sub[sub.method == me]
                p = rate(x.success)[0] if len(x) else np.nan
                cells.append(f"{100 * p:.0f}" if len(x) else "--")
                mac(f"tr{T}{name}{'Cal' if me == 'flowshield' else 'S' + W(me[4:])}", f"{100 * p:.1f}" if len(x) else "--")
            L.append(f"{name} & " + " & ".join(cells) + r" \\")
            if name == "All":
                L.append(r"\midrule")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(AUTO, f"table_trade_{tag}.tex"), "w").write("\n".join(L) + "\n")
        piv = df.pivot_table(index="id", columns="method", values="success", aggfunc="first")
        for a, b, nm in (("fs_s0", f"fs_s{scales[-1]}", "ZeroMax"), ("fs_s0", "flowshield", "ZeroCal")):
            if a in piv and b in piv:
                ok = piv[[a, b]].dropna()
                n01, n10, p = mcnemar_exact(ok[a].astype(bool), ok[b].astype(bool))
                mac(f"tr{T}Pair{nm}A", n10)   # completed with a, not with b
                mac(f"tr{T}Pair{nm}B", n01)
                mac(f"tr{T}Pair{nm}P", f"{p:.2f}")
        mac(f"tr{T}N", int(df[df.method == 'fs_s0'].id.nunique()))
    print("trade summary done")


if __name__ == "__main__":
    dev_extraction()
    consistency()
    table_pair("gemma", "qwen", "gemma4", "qwen3", "table_backbone.tex")
    table_pair("gemma", "para", "orig.", "para.", "table_para.tex")
    paraphrase_extra()
    tradeoff()
    sampling()
    trade_summary()
    open(os.path.join(AUTO, "numbers_extra.tex"), "w").write("\n".join(macros) + "\n")
    print("extra analysis written")


