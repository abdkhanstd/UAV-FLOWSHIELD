"""Aggregate LLM-experiment results into tables (LaTeX), number macros and a JSON summary.

Usage: python scripts/analyze_main.py --tag gemma --files results/main_gemma4-12b.jsonl [--out-dir ../paper/auto]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats_util import wilson, mcnemar_exact, pct

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAIN = ["undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep", "flowshield", "det_template"]
ABL = ["flowshield", "fs_no_view", "fs_no_view_euclid", "fs_no_typed", "fs_no_envelope", "fs_no_verifier", "fs_raw_noverifier"]
LABEL = {"det_template": "Deterministic planner (no LLM)", "undefended": "Undefended", "prompt_only": "Prompt-only", "spotlight_guard": "Spotlight-Guard",
         "prv": "Prompt-RAG-Validator", "eamsr": "EAMSR", "mcp_pep": "MCP-PEP", "flowshield": "UAV-FlowShield",
         "fs_no_view": "FlowShield w/o private view", "fs_no_typed": "FlowShield w/o typed observations",
         "fs_no_envelope": "FlowShield w/o envelope", "fs_no_verifier": "FlowShield w/o verifier",
         "fs_raw_noverifier": "FlowShield w/o typed observations and verifier",
         "fs_no_view_euclid": "FlowShield w/o private view, straight-line distances"}
CODE = {"det_template": "Det", "undefended": "Undef", "prompt_only": "Prompt", "spotlight_guard": "Spot", "prv": "Prv", "eamsr": "Eamsr",
        "mcp_pep": "Pep", "flowshield": "Fs", "fs_no_view": "FsNoView", "fs_no_typed": "FsNoTyped",
        "fs_no_envelope": "FsNoEnv", "fs_no_verifier": "FsNoVer", "fs_raw_noverifier": "FsRawNoVer", "fs_no_view_euclid": "FsNoViewEu"}
GOALS = ["G1", "G2", "G3", "G4", "G5", "G6"]
AD = ["AD1", "AD2", "AD3", "AD4"]
CITE = {"prv": "\\citep{hou2026prv}", "eamsr": "\\citep{huang2026eamsr}", "mcp_pep": "\\citep{wang2026pep}",
        "spotlight_guard": "\\citep{demirol2026spotlight}"}


def load(files):
    df = pd.concat([pd.DataFrame([json.loads(l) for l in open(f, encoding="utf-8")]) for f in files], ignore_index=True)
    df = df.drop_duplicates(["id", "method", "seed", "temperature"], keep="last")
    return df


def rate(x):
    x = np.asarray(x, bool)
    return wilson(int(x.sum()), len(x))


def fmt(p, lo=None, hi=None, d=1, ci=True):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if lo is None or not ci:
        return f"{100 * p:.{d}f}"
    return f"{100 * p:.{d}f} \\;\\scriptsize[{100 * lo:.{d}f}, {100 * hi:.{d}f}]\\normalsize"


def blocks(df):
    atk = df[df.category == "attack"]
    adp = df[(df.category == "adaptive") & df.goal.isin(["AD1", "AD2", "AD3"])]
    ad4 = df[(df.category == "adaptive") & (df.goal == "AD4")]
    ben = df[df.category.str.startswith("benign")]
    return atk, adp, ad4, ben


def summarize(df, methods):
    atk, adp, ad4, ben = blocks(df)
    out = {}
    for m in methods:
        a, d, d4, b, al = [x[x.method == m] for x in (atk, adp, ad4, ben, df)]
        r = {"n_attack": len(a), "n_adapt": len(d), "n_benign": len(b), "n_all": len(al)}
        if len(a):
            r["asr"] = rate(a.attack_success)
            r["attempt"] = rate(a.attempted)
            r["asr_goal"] = {g: rate(a[a.goal == g].attack_success) for g in GOALS if (a.goal == g).any()}
            r["asr_style"] = {g: rate(a[a.style == g].attack_success) for g in a.style.dropna().unique()}
            r["asr_channel"] = {g: rate(a[a.channel == g].attack_success) for g in a.channel.dropna().unique()}
        if len(d):
            r["asr_adapt"] = rate(d.attack_success)
            r["attempt_adapt"] = rate(d.attempted)
            r["asr_ad"] = {g: rate(d[d.goal == g].attack_success) for g in ["AD1", "AD2", "AD3"] if (d.goal == g).any()}
        if len(d4) and m != "fs_no_view":     # planning-frame disclosure needs a private view
            r["asr_ad4"] = rate(d4.attack_success)
        if len(b):
            r["mcr"] = rate(b.success)
            r["mcr_cat"] = {c: rate(b[b.category == c].success) for c in sorted(b.category.unique())}
            blocked = (~b.executed) | (b.denied > 0) | (b.quarantined > 0)
            r["frr"] = rate(blocked)
            r["abort"] = rate(~b.executed)
        if len(al) and "obs_admitted" in al and al.obs_admitted.notna().any() and m in ("flowshield", "fs_no_envelope", "fs_no_view"):
            oa = al.obs_admitted.fillna(0)
            ext = {}
            for c in ("benign_constraint", "benign_dynamic"):
                x = al[al.category == c]
                if len(x):
                    ext[c] = rate(oa[x.index] >= 1)
            for c in ("benign_plain",):
                x = al[al.category == c]
                if len(x):
                    ext[c] = rate(oa[x.index] == 0)
            x = al[al.category == "attack"]
            if len(x):
                ext["attack"] = rate(oa[x.index] == 0)
            for g in ("AD1", "AD2", "AD3"):
                x = al[(al.category == "adaptive") & (al.goal == g)]
                if len(x):
                    ext[g] = rate(oa[x.index] >= 1) if g == "AD1" else rate(oa[x.index] == 0)
            r["ext"] = ext
        if len(al):
            r["svr"] = rate(al.nfz_violation | al.alt_violation)
            r["exposure"] = rate(al.leak_abs_prompt > 0)
            r["ident"] = rate(al.leak_ident_prompt > 0)
            g4 = al[(al.goal == "G4")]
            if len(g4):
                r["leak_out_g4"] = rate(g4.leak_out > 0)
            r["calls"] = float(al.n_calls.mean())
            r["pt"] = float(al.pt.mean())
            r["ot"] = float(al.ot.mean())
            r["repair"] = float(al.repaired.mean()) if "repaired" in al else 0.0
        out[m] = r
    return out


def paired(df, ref, others, field, cat_filter):
    res = {}
    sub = df[cat_filter(df)]
    piv = sub.pivot_table(index="id", columns="method", values=field, aggfunc="first")
    for o in others:
        if ref in piv and o in piv:
            ok = piv[[ref, o]].dropna()
            n01, n10, p = mcnemar_exact(ok[ref].astype(bool), ok[o].astype(bool))
            res[o] = {"n": len(ok), "ref_only": n10, "other_only": n01, "p": p}
    return res


# --------------------------------------------------------------------------- LaTeX
def tex_main(S, methods, tag):
    L = []
    L.append("\\begin{tabular}{lcccccc}")
    L.append("\\toprule")
    L.append("Method & ASR $\\downarrow$ & Adaptive ASR $\\downarrow$ & MCR $\\uparrow$ & FRR $\\downarrow$ & SVR $\\downarrow$ & Coord.\\ exposure $\\downarrow$ \\\\")
    L.append("\\midrule")
    for m in methods:
        r = S[m]
        cell = lambda k, ci=True: fmt(*r[k], ci=ci) if k in r else "--"
        name = LABEL[m] + (" " + CITE[m] if m in CITE else "")
        L.append(f"{name} & {cell('asr')} & {cell('asr_adapt')} & {cell('mcr')} & {cell('frr', False)} & {cell('svr', False)} & {cell('exposure', False)} \\\\")
        if m in ("mcp_pep", "flowshield") and m != methods[-1]:
            L.append("\\midrule")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    return "\n".join(L)


def tex_goal(S, methods):
    cols = GOALS + AD
    L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "Method & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for m in methods:
        r = S[m]
        cells = []
        for g in GOALS:
            cells.append(fmt(*r["asr_goal"][g], ci=False, d=0) if g in r.get("asr_goal", {}) else "--")
        for g in ["AD1", "AD2", "AD3"]:
            cells.append(fmt(*r["asr_ad"][g], ci=False, d=0) if g in r.get("asr_ad", {}) else "--")
        cells.append(fmt(*r["asr_ad4"], ci=False, d=0) if "asr_ad4" in r else "--")
        L.append(LABEL[m] + " & " + " & ".join(cells) + " \\\\")
        if m in ("mcp_pep", "flowshield") and m != methods[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def tex_cat(S, methods):
    cats = ["benign_plain", "benign_constraint", "benign_dynamic", "benign_tight"]
    names = ["Plain", "Constraint", "Dynamic task", "Tight budget"]
    L = ["\\begin{tabular}{l" + "c" * 4 + "}", "\\toprule", "Method & " + " & ".join(names) + " \\\\", "\\midrule"]
    for m in methods:
        r = S[m]
        cells = [fmt(*r["mcr_cat"][c], ci=False) if c in r.get("mcr_cat", {}) else "--" for c in cats]
        L.append(LABEL[m] + " & " + " & ".join(cells) + " \\\\")
        if m in ("mcp_pep", "flowshield") and m != methods[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def tex_cost(S, methods):
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Method & LLM calls & Prompt tokens & Output tokens \\\\", "\\midrule"]
    for m in methods:
        r = S[m]
        L.append(f"{LABEL[m]} & {r['calls']:.2f} & {r['pt']:.0f} & {r['ot']:.0f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def tex_paired(P, methods):
    """FlowShield versus each baseline on the same episodes (exact McNemar): counts where FlowShield is better / worse."""
    def pv(x):
        return "$<$0.001" if x < 0.001 else f"{x:.3f}"
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Baseline & \multicolumn{3}{c}{Attack success (108 episodes)} & \multicolumn{3}{c}{Mission completion (120 episodes)} \\",
         r" & FS better & FS worse & $p$ & FS better & FS worse & $p$ \\", r"\midrule"]
    for m in methods:
        a = P.get("asr_attack", {}).get(m)
        c = P.get("mcr", {}).get(m)
        if a is None or c is None:
            continue
        L.append(f"{LABEL[m]} & {a['other_only']} & {a['ref_only']} & {pv(a['p'])} & {c['ref_only']} & {c['other_only']} & {pv(c['p'])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def numbers_tex(S, tag):
    T = tag[0].upper() + tag[1:]
    L = []

    def add(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for m, r in S.items():
        c = CODE[m]
        for k in ["asr", "attempt", "asr_adapt", "mcr", "frr", "svr", "exposure", "ident", "asr_ad4", "leak_out_g4", "abort"]:
            if k in r:
                kk = k.replace("_", "").replace("ad4", "Adfour").replace("g4", "Gfour")
                p, lo, hi = r[k]
                nm = f"{kk}{T}{c}"
                add(nm, f"{100 * p:.1f}")
                add(nm + "Lo", f"{100 * lo:.1f}")
                add(nm + "Hi", f"{100 * hi:.1f}")
        for k, v in r.get("ext", {}).items():
            kk = k.replace("benign_", "").capitalize().replace("1", "One").replace("2", "Two").replace("3", "Three")
            add(f"ext{kk}{T}{c}", f"{100 * v[0]:.0f}")
        for k, v in r.get("mcr_cat", {}).items():
            add(f"mcrCat{k.replace('benign_', '').capitalize()}{T}{c}", f"{100 * v[0]:.1f}")
        for k in ["calls", "pt", "ot", "repair"]:
            if k in r:
                add(f"{k}{T}{c}", f"{r[k]:.2f}" if k in ("calls", "repair") else f"{r[k]:.0f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "..", "paper", "auto"))
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = load([os.path.join(ROOT, f) if not os.path.isabs(f) else f for f in args.files])
    methods = [m for m in MAIN + ABL[1:] if m in set(df.method)]
    S = summarize(df, methods)
    main_m = [m for m in MAIN if m in S]
    abl_m = [m for m in ABL if m in S]
    P = {}
    base = [m for m in main_m if m != "flowshield"]
    if "flowshield" in S:
        P["asr_attack"] = paired(df, "flowshield", base, "attack_success", lambda d: d.category == "attack")
        P["asr_adapt"] = paired(df, "flowshield", base, "attack_success",
                                lambda d: (d.category == "adaptive") & d.goal.isin(["AD1", "AD2", "AD3"]))
        P["mcr"] = paired(df, "flowshield", base, "success", lambda d: d.category.str.startswith("benign"))
    json.dump({"summary": S, "paired": P}, open(os.path.join(ROOT, "results", f"summary_{args.tag}.json"), "w"), indent=1, default=float)
    w = lambda name, s: open(os.path.join(args.out_dir, f"{name}_{args.tag}.tex"), "w", encoding="utf-8").write(s + "\n")
    if P:
        w("table_paired", tex_paired(P, [m for m in main_m if m != "flowshield"]))
    w("table_main", tex_main(S, main_m, args.tag))
    w("table_goal", tex_goal(S, main_m))
    w("table_cat", tex_cat(S, main_m))
    w("table_cost", tex_cost(S, main_m))
    extra_numbers = ""
    if len(abl_m) > 1:
        # like-for-like: every ablation row (including the full system) on the episodes the ablations ran on
        abl_ids = set(df[df.method == "fs_no_view"].id) if (df.method == "fs_no_view").any() else set(df[df.method == abl_m[1]].id)
        S_sub = summarize(df[df.id.isin(abl_ids)], abl_m)
        w("table_ablation", tex_main(S_sub, abl_m, args.tag))
        w("table_ablation_goal", tex_goal(S_sub, abl_m))
        extra_numbers = numbers_tex(S_sub, args.tag + "Sub")
        json.dump({"summary": S_sub, "n_episodes": len(abl_ids)}, open(os.path.join(ROOT, "results", f"summary_{args.tag}_sub.json"), "w"), indent=1, default=float)
    w("numbers", numbers_tex(S, args.tag) + "\n" + extra_numbers)
    print(f"{args.tag}: {len(df)} rows; methods {methods}")
    for m in methods:
        r = S[m]
        g = lambda k: (f"{100 * r[k][0]:.1f}" if k in r else "--")
        print(f"{m:16s} ASR {g('asr'):>5} att {g('attempt'):>5} adapt {g('asr_adapt'):>5} MCR {g('mcr'):>5} FRR {g('frr'):>5} "
              f"SVR {g('svr'):>5} expo {g('exposure'):>5} calls {r.get('calls', 0):.2f}")
    for k, v in P.items():
        print(k, {o: (d["ref_only"], d["other_only"], round(d["p"], 4)) for o, d in v.items()})


if __name__ == "__main__":
    main()
