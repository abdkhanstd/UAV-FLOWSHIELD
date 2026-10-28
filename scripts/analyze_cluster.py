"""Cluster-aware uncertainty on T2: the episodes come from six synthetic areas (two per registry class) and share generators, so
episode-level Wilson intervals describe benchmark sampling only. Reported here:
  * per-area values of the headline metrics for UAV-FlowShield and the main matched baselines (spread across areas),
  * leave-one-area-out range,
  * a cluster bootstrap over areas (10 000 resamples of the six areas with replacement) of the difference in mission completion
    and in adaptive attack success between UAV-FlowShield and each matched baseline (clusters with the same index drawn twice count twice).
usage: python scripts/analyze_cluster.py   -> ../paper/auto/table_cluster_tbg.tex, numbers_cluster_tbg.tex"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_eval import load  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
AUTO = os.path.join(ROOT, "..", "paper", "auto")
R = os.path.join(ROOT, "results", "eval")
METHODS = [("flowshield", "UAV-FlowShield", "Fs"), ("undefended_rep", "Undefended + repair", "UndefRep"), ("spotlight_guard_rep", "Spotlight-Guard + repair", "SpotRep"),
           ("prv_rep", "Prompt-RAG-Validator + repair", "PrvRep"), ("eamsr_repdel", "EAMSR + repair + delegation", "EamsrRepDel"),
           ("mcp_pep_repdel", "MCP-PEP + repair + delegation", "PepRepDel")]


def metrics(d):
    atk = d[d.category == "attack"]
    adp = d[(d.category == "adaptive") & (~d.goal.isin(["AD4", "AD6b"]))]
    ben = d[d.category.str.startswith("benign")]
    f = lambda x, c: 100 * float(x[c].mean()) if len(x) else float("nan")
    return {"asr": f(atk, "attack_success"), "adapt": f(adp, "attack_success"), "mcr": f(ben, "success")}


def main():
    df = load([os.path.join(R, f) for f in ("t2_gemma_base.jsonl", "t2_gemma_fs.jsonl", "t2_gemma_fixed.jsonl")])
    areas = sorted(df.area.unique())
    macros = []
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Method & ASR: per-area range (\\%) & Adaptive ASR: per-area range (\\%) & MCR: per-area range (\\%) \\\\", "\\midrule"]
    per = {}
    for m, name, code in METHODS:
        d = df[df.method == m]
        per[m] = {a: metrics(d[d.area == a]) for a in areas}
        cells = []
        for k in ("asr", "adapt", "mcr"):
            v = [per[m][a][k] for a in areas]
            cells.append(f"{np.nanmin(v):.1f}--{np.nanmax(v):.1f}")
            macros += [f"\\newcommand{{\\area{k.capitalize()}Min{code}}}{{{np.nanmin(v):.1f}}}", f"\\newcommand{{\\area{k.capitalize()}Max{code}}}{{{np.nanmax(v):.1f}}}"]
            lo = [metrics(d[d.area != a])[k] for a in areas]
            macros += [f"\\newcommand{{\\loao{k.capitalize()}Min{code}}}{{{np.nanmin(lo):.1f}}}", f"\\newcommand{{\\loao{k.capitalize()}Max{code}}}{{{np.nanmax(lo):.1f}}}"]
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(AUTO, "table_cluster_tbg.tex"), "w").write("\n".join(L) + "\n")
    # cluster bootstrap of differences to UAV-FlowShield (areas as clusters); per-area sums and counts make a resample cheap
    rng = np.random.default_rng(0)

    def counts(d, k):
        out = []
        for a in areas:
            x = d[d.area == a]
            if k == "mcr":
                x = x[x.category.str.startswith("benign")]
                out.append((float(x.success.sum()), len(x)))
            else:
                x = x[(x.category == "adaptive") & (~x.goal.isin(["AD4", "AD6b"]))]
                out.append((float(x.attack_success.sum()), len(x)))
        return np.array(out)

    picks = rng.integers(len(areas), size=(10000, len(areas)))
    for m, name, code in METHODS[1:]:
        for k, lab in (("mcr", "Mcr"), ("adapt", "Adapt")):
            cf, co = counts(df[df.method == "flowshield"], k), counts(df[df.method == m], k)
            num_f, den_f = cf[picks, 0].sum(1), cf[picks, 1].sum(1)
            num_o, den_o = co[picks, 0].sum(1), co[picks, 1].sum(1)
            ok = (den_f > 0) & (den_o > 0)
            diffs = 100 * (num_f[ok] / den_f[ok] - num_o[ok] / den_o[ok])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            macros += [f"\\newcommand{{\\clu{lab}Diff{code}Lo}}{{{lo:.1f}}}", f"\\newcommand{{\\clu{lab}Diff{code}Hi}}{{{hi:.1f}}}"]
            print(f"{m:22s} {k}: FS - other = {100 * (cf[:, 0].sum() / cf[:, 1].sum() - co[:, 0].sum() / co[:, 1].sum()):6.1f}  cluster bootstrap [{lo:.1f}, {hi:.1f}]")
    open(os.path.join(AUTO, "numbers_cluster_tbg.tex"), "w").write("\n".join(macros) + "\n")
    for m, name, code in METHODS:
        print(name, {k: [round(per[m][a][k], 1) for a in areas] for k in ("asr", "adapt", "mcr")})


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    main()
