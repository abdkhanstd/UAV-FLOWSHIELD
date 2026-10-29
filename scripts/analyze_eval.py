"""Analysis of the evaluation on benchmarks T1 and T2. The aggregate adaptive rate excludes AD4 (defined only for methods with a private
planning view) and AD6b (an availability measure: the legitimate tight-budget mission no longer completes); both appear in the per-goal tables.

usage: python scripts/analyze_eval.py --tag tb --name T2 --files results/eval/t2_gemma.jsonl [more.jsonl ...]
Writes ../paper/auto/numbers_<tag>.tex (macros: <metric><Tag><MethodCode>), table_*_<tag>.tex and results/eval/summary_<tag>.json.
Macro names are letters only (LaTeX); <Tag> is the capitalised tag."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stats_util import wilson, mcnemar_exact

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")

LABEL = {
    "det_template": "Basic parser planner", "det_plus": "Extended parser planner",
    "undefended": "Undefended", "prompt_only": "Prompt-only", "spotlight_guard": "Spotlight-Guard", "prv": "Prompt-RAG-Validator",
    "eamsr": "EAMSR", "mcp_pep": "MCP-PEP",
    "flowshield": "UAV-FlowShield", "fs_chan": "FlowShield, channel-only authority", "fs_legacy": "FlowShield, basic variant",
    "fs_noauthority": "FlowShield w/o channel-authority table", "fs_fallback": "FlowShield + verified fallback",
    "fs_detx": "FlowShield, parser extractor", "fs_hybrid": "FlowShield, hybrid extractor",
    "fs_no_view": "FlowShield w/o private view", "fs_no_view_euclid": "FlowShield w/o private view, straight-line distances",
    "fs_no_typed": "FlowShield w/o typed observations", "fs_no_envelope": "FlowShield w/o spatial requests",
    "fs_no_verifier": "FlowShield w/o verifier", "fs_raw_noverifier": "FlowShield w/o typed observations and verifier",
}
CODE = {
    "det_template": "DetB", "det_plus": "DetP", "undefended": "Undef", "prompt_only": "Prompt", "spotlight_guard": "Spot", "prv": "Prv",
    "eamsr": "Eamsr", "mcp_pep": "Pep", "flowshield": "Fs", "fs_chan": "FsChan", "fs_legacy": "FsLegacy", "fs_noauthority": "FsNoAuth",
    "fs_fallback": "FsFall", "fs_detx": "FsDetx", "fs_hybrid": "FsHyb", "fs_no_view": "FsNoView", "fs_no_view_euclid": "FsNoViewEu",
    "fs_no_typed": "FsNoTyped", "fs_no_envelope": "FsNoEnv", "fs_no_verifier": "FsNoVer", "fs_raw_noverifier": "FsRawNoVer",
}
for b in ("undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep"):
    LABEL[b + "_rep"] = LABEL[b] + " + repair"
    CODE[b + "_rep"] = CODE[b] + "Rep"
for b in ("eamsr", "mcp_pep"):
    LABEL[b + "_repdel"] = LABEL[b] + " + repair + delegation"
    CODE[b + "_repdel"] = CODE[b] + "RepDel"
BASE = ["undefended", "prompt_only", "spotlight_guard", "prv", "eamsr", "mcp_pep"]
MATCHED = ["undefended_rep", "prompt_only_rep", "spotlight_guard_rep", "prv_rep", "eamsr_rep", "eamsr_repdel", "mcp_pep_rep", "mcp_pep_repdel"]
CITE = {"prv": "\\citep{hou2026prv}", "eamsr": "\\citep{huang2026eamsr}", "mcp_pep": "\\citep{wang2026pep}", "spotlight_guard": "\\citep{demirol2026spotlight}"}
ABL = ["flowshield", "fs_no_view", "fs_no_view_euclid", "fs_no_typed", "fs_no_envelope", "fs_no_verifier", "fs_raw_noverifier"]
VARIANTS = ["flowshield", "fs_chan", "fs_legacy", "fs_noauthority", "fs_fallback", "fs_detx", "fs_hybrid"]
GOALS = ["G1", "G2", "G3", "G4", "G5", "G6"]
CATS = ["benign_plain", "benign_constraint", "benign_dynamic", "benign_tight", "benign_multi"]
CATNAME = {"benign_plain": "Plain", "benign_constraint": "Constraint", "benign_dynamic": "Dynamic", "benign_tight": "Tight budget", "benign_multi": "Several documents"}


def cname(m):
    return CODE.get(m, "".join(w.capitalize() for w in m.split("_")))


def load(files):
    df = pd.concat([pd.DataFrame([json.loads(l) for l in open(f, encoding="utf-8")]) for f in files], ignore_index=True)
    for c in ("seed", "temperature"):
        if c not in df:
            df[c] = 0
    return df.drop_duplicates(["id", "method", "seed", "temperature"], keep="last")


def rate(x):
    x = np.asarray(x, bool)
    return wilson(int(x.sum()), len(x))


def fmt(t, d=1, ci=True):
    p, lo, hi = t
    if p is None or np.isnan(p):
        return "--"
    return f"{100 * p:.{d}f}\\,\\scriptsize[{100 * lo:.{d}f}, {100 * hi:.{d}f}]\\normalsize" if ci else f"{100 * p:.{d}f}"


def summarize(df, methods):
    out = {}
    for m in methods:
        d = df[df.method == m]
        if not len(d):
            continue
        atk, adp_all, ben = d[d.category == "attack"], d[d.category == "adaptive"], d[d.category.str.startswith("benign")]
        adp = adp_all[~adp_all.goal.isin(["AD4", "AD6b"])]        # AD4: private view only; AD6b is an availability measure (below)
        r = {"n": len(d)}
        if len(atk):
            r["asr"] = rate(atk.attack_success)
            r["attempt"] = rate(atk.attempted)
            r["asr_goal"] = {g: rate(atk[atk.goal == g].attack_success) for g in GOALS if (atk.goal == g).any()}
            if "author" in atk:
                r["asr_author"] = {("model" if a not in ("tmpl2", None) and isinstance(a, str) else "template"): None for a in atk.author.unique()}
                llm = atk.author.notna() & (atk.author != "tmpl2")
                if llm.any() and (~llm).any():
                    r["asr_llm_author"] = rate(atk[llm].attack_success)
                    r["asr_tmpl_author"] = rate(atk[~llm].attack_success)
        if len(adp):
            r["adapt"] = rate(adp.attack_success)
            r["adapt_nt"] = rate(adp[adp.goal != "AD1t"].attack_success)       # without the authenticated-ticket attack (known residual)
            r["adapt_goal"] = {g: rate(adp_all[adp_all.goal == g].attack_success) for g in sorted(adp_all.goal.dropna().unique())}
            r["n_adapt"] = len(adp)
        if len(ben):
            r["mcr"] = rate(ben.success)
            r["mcr_cat"] = {c: rate(ben[ben.category == c].success) for c in CATS if (ben.category == c).any()}
            blocked = (~ben.executed) | (ben.denied > 0) | (ben.quarantined > 0)
            r["frr"] = rate(blocked)
            r["abort"] = rate(~ben.executed)
            r["fallback"] = rate(ben.fallback_used) if "fallback_used" in ben else None
            r["fail"] = ben[~ben.success].fail_reason.value_counts().to_dict()
        r["svr"] = rate(d.nfz_violation | d.alt_violation)
        r["exposure"] = rate(d.leak_abs_prompt > 0)
        r["ident"] = rate(d.leak_ident_prompt > 0)
        g4 = d[d.goal == "G4"]
        if len(g4):
            r["leak_out"] = rate(g4.leak_out > 0)
        r["calls"] = float(d.n_calls.mean())
        r["repair"] = float(d.repaired.mean())
        if "ext_need" in d and d.ext_need.notna().any():
            e = d[d.ext_need.notna()]
            need = e[e.ext_need > 0]
            r["ext_recall"] = rate(need.ext_hit == need.ext_need) if len(need) else None
            ben_e = e[e.category.str.startswith("benign")]
            r["ext_false_benign"] = rate(ben_e.ext_extra > 0) if len(ben_e) else None
            plain = e[(e.category == "attack")]
            r["ext_false_attack"] = rate(plain.ext_extra > 0) if len(plain) else None
        out[m] = r
    return out


def paired(df, ref, others, field, filt):
    sub = df[filt(df)]
    piv = sub.pivot_table(index="id", columns="method", values=field, aggfunc="first")
    res = {}
    for o in others:
        if ref in piv and o in piv:
            ok = piv[[ref, o]].dropna()
            n01, n10, p = mcnemar_exact(ok[ref].astype(bool), ok[o].astype(bool))
            res[o] = {"n": len(ok), "ref_only": n10, "other_only": n01, "p": p}
    return res


def tex_table(S, methods, cols, head="", mid_after=()):
    L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "Method & " + " & ".join(h for h, _, _ in cols) + " \\\\", "\\midrule"]
    for m in methods:
        if m not in S:
            continue
        r = S[m]
        cells = []
        for _, key, ci in cols:
            v = key(r) if callable(key) else r.get(key)
            cells.append(fmt(v, ci=ci) if v else "--")
        name = LABEL[m] + (" " + CITE[m] if m in CITE else "")
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if m in mid_after:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--subset", default=None, help="file with episode ids to restrict to")
    a = ap.parse_args()
    df = load([os.path.join(ROOT, f) if not os.path.isabs(f) else f for f in a.files])
    if a.subset:
        keep = set(open(os.path.join(ROOT, a.subset)).read().split())
        df = df[df.id.isin(keep)]
    methods = [m for m in list(LABEL) if m in set(df.method)]
    S = summarize(df, methods)
    T = a.tag[0].upper() + a.tag[1:]
    macros = []

    def mac(name, val):
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for m, r in S.items():
        c = cname(m)
        for k in ("asr", "attempt", "adapt", "adapt_nt", "mcr", "frr", "abort", "svr", "exposure", "ident", "leak_out", "ext_recall", "ext_false_benign",
                  "ext_false_attack", "asr_llm_author", "asr_tmpl_author"):
            v = r.get(k)
            if v and not np.isnan(v[0]):
                mac(f"{k.replace('_', '')}{T}{c}", f"{100 * v[0]:.1f}")
                mac(f"{k.replace('_', '')}{T}{c}Lo", f"{100 * v[1]:.1f}")
                mac(f"{k.replace('_', '')}{T}{c}Hi", f"{100 * v[2]:.1f}")
        for k, v in r.get("mcr_cat", {}).items():
            mac(f"mcrCat{k.replace('benign_', '').capitalize()}{T}{c}", f"{100 * v[0]:.1f}")
        for k, v in r.get("adapt_goal", {}).items():
            g = "Ad" + k[2:].translate(str.maketrans("0123456789", "ZOTHFVSEIN")).replace("b", "B").replace("t", "T")
            mac(f"adapt{g}{T}{c}", f"{100 * v[0]:.1f}")
        if r.get("fallback"):
            mac(f"fallback{T}{c}", f"{100 * r['fallback'][0]:.1f}")
        mac(f"calls{T}{c}", f"{r['calls']:.2f}")
        mac(f"repair{T}{c}", f"{100 * r['repair']:.1f}")
    # tables
    main_cols = [("ASR$\\downarrow$", "asr", True), ("Adaptive ASR$\\downarrow$", "adapt", True), ("MCR$\\uparrow$", "mcr", True),
                 ("FRR", lambda r: r.get("frr"), False), ("SVR", lambda r: r.get("svr"), False), ("Exposure", lambda r: r.get("exposure"), False)]
    W = lambda name, text: open(os.path.join(AUTO, f"{name}_{a.tag}.tex"), "w").write(text + "\n")
    W("table_main", tex_table(S, BASE + ["det_template", "det_plus", "flowshield"], main_cols, "", mid_after=("mcp_pep", "det_plus")))
    mat_cols = [("ASR$\\downarrow$", "asr", True), ("Adaptive$\\downarrow$", "adapt", True), ("MCR$\\uparrow$", "mcr", True),
                ("Tight MCR", lambda r: r.get("mcr_cat", {}).get("benign_tight"), False), ("Exposure", lambda r: r.get("exposure"), False)]
    order = []
    for b in BASE:
        order += [b, b + "_rep"] + ([b + "_repdel"] if b + "_repdel" in S else [])
    W("table_matched", tex_table(S, order + ["flowshield"], mat_cols, ""))
    W("table_variants", tex_table(S, VARIANTS, main_cols, ""))
    W("table_abl", tex_table(S, ABL, [("ASR$\\downarrow$", "asr", True), ("Adaptive$\\downarrow$", "adapt", True), ("MCR$\\uparrow$", "mcr", True),
                                      ("FRR", lambda r: r.get("frr"), False), ("Exposure", lambda r: r.get("exposure"), False)]))
    cat_cols = [(CATNAME[c], (lambda c: lambda r: r.get("mcr_cat", {}).get(c))(c), False) for c in CATS if any(c in r.get("mcr_cat", {}) for r in S.values())]
    W("table_cat", tex_table(S, BASE + ["det_template", "det_plus", "flowshield", "fs_fallback"], cat_cols, "", mid_after=("mcp_pep", "det_plus")))
    goals = sorted({g for r in S.values() for g in r.get("adapt_goal", {})})
    if goals:
        gcols = [(g, (lambda g: lambda r: r.get("adapt_goal", {}).get(g))(g), False) for g in goals]
        W("table_adaptive", tex_table(S, BASE + ["det_plus", "flowshield", "fs_chan"], gcols, "", mid_after=("mcp_pep", "det_plus")))
    ex_cols = [("Effects recovered", "ext_recall", True), ("Benign false admission", lambda r: r.get("ext_false_benign"), False),
               ("Attack-doc false admission", lambda r: r.get("ext_false_attack"), False)]
    W("table_ext", tex_table(S, ["flowshield", "fs_detx", "fs_hybrid"], ex_cols, ""))
    # paired tests against flowshield
    ref = "flowshield"
    others = [m for m in methods if m != ref]
    P = {"asr": paired(df, ref, others, "attack_success", lambda d: d.category == "attack"),
         "adapt": paired(df, ref, others, "attack_success", lambda d: (d.category == "adaptive") & (~d.goal.isin(["AD4", "AD6b"]))),
         "mcr": paired(df, ref, others, "success", lambda d: d.category.str.startswith("benign"))}
    open(os.path.join(AUTO, f"numbers_{a.tag}.tex"), "w").write("\n".join(macros) + "\n")
    json.dump({"summary": S, "paired": P}, open(os.path.join(ROOT, "results", "eval", f"summary_{a.tag}.json"), "w"), indent=1, default=float)
    for m, r in S.items():
        f = lambda k: ("%5.1f" % (100 * r[k][0])) if r.get(k) else "  -- "
        print(f"{m:22s} n={r['n']:4d} ASR {f('asr')} adapt {f('adapt')} MCR {f('mcr')} FRR {f('frr')} SVR {f('svr')} expo {f('exposure')} calls {r['calls']:.2f}")


if __name__ == "__main__":
    main()
