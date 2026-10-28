"""Build the submission folder paper_clean/: one self-contained main.tex (short paper) and one supplementary.tex
(the full material), with every generated number written out, every generated table inlined, numbered citations and the
figures in paper_clean/figs.

  python scripts/make_clean.py supp     -> paper_clean/supplementary.tex (from the full paper sources)
  python scripts/make_clean.py main     -> paper_clean/main.tex (from paper/short/main_src.tex; \\supp{label} resolved from
                                           paper_clean/supplementary.aux, so compile the supplement first)
"""
import os
import re
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "paper_clean")
EXCLUDED_FILES = ["sections/abstract", "sections/intro", "sections/related", "sections/appendix_history"]


def macros():
    """The numbers that the paper actually loads: the files listed in auto/numbers_all.tex, in that order (first definition wins)."""
    m, dup = {}, set()
    listing = open(os.path.join(PAPER, "auto", "numbers_all.tex"), encoding="utf-8").read()
    for rel in re.findall(r"\\input\{(auto/numbers_[^}]+)\}", listing):
        p = os.path.join(PAPER, rel + ".tex")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line.strip())
            if r:
                if r.group(1) in m:
                    dup.add(r.group(1))
                    continue
                m[r.group(1)] = r.group(2)
    if dup:
        print("duplicate number macros (first kept):", sorted(dup)[:20])
    return m


def read(rel):
    p = os.path.join(PAPER, rel if rel.endswith(".tex") else rel + ".tex")
    return open(p, encoding="utf-8").read()


def inline(text):
    def sub(mo):
        if mo.group(1).startswith("auto/"):
            return "%AUTOFILE " + mo.group(1) + "\n" + inline(read(mo.group(1)))
        return inline(read(mo.group(1)))
    return re.sub(r"\\input\{([^}]+)\}", sub, text)


def expand(text, M):
    def sub(mo):
        name = mo.group(1)
        if name in M:
            return M[name]
        return mo.group(0)
    return re.sub(r"\\([A-Za-z]+)(?:\{\})?", lambda mo: sub(mo) if mo.group(1) in M else mo.group(0), text)


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "%", text)


def labels_in(text):
    return set(re.findall(r"\\label(?:\[[^]]*\])?\{([^}]+)\}", text))


def fix_refs(text, excluded, phrase="the main paper"):
    """\\cref{a,b} with labels that live only in the main paper: drop them, or replace the whole reference by a phrase."""
    def sub(mo):
        cmd, keys = mo.group(1), [k.strip() for k in mo.group(2).split(",")]
        keep = [k for k in keys if k not in excluded]
        if len(keep) == len(keys):
            return mo.group(0)
        if not keep:
            return phrase
        return f"\\{cmd}{{{','.join(keep)}}}"
    return re.sub(r"\\(cref|Cref|ref)\{([^}]*)\}", sub, text)


def bib_authors():
    """First-author surnames (and the second for two-author papers) from refs.bib, for numbered textual citations."""
    txt = open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()
    out = {}
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", txt, flags=re.S):
        a = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", body, flags=re.S)
        if not a:
            continue
        names = [n.strip() for n in re.split(r"\s+and\s+", a.group(1).replace("\n", " "))]
        sur = [n.split(",")[0].strip().strip("{}") for n in names if n.lower() != "others"]
        if len(names) == 1:
            out[key] = sur[0]
        elif len(names) == 2 and "others" not in a.group(1):
            out[key] = f"{sur[0]} and {sur[1]}"
        else:
            out[key] = f"{sur[0]} et al."
    return out


def numbered_citet(text):
    """\\citet{a,b} is undefined with a numbered style: write the names and a numbered citation."""
    A = bib_authors()

    def sub(mo):
        keys = [k.strip() for k in mo.group(1).split(",")]
        return ", ".join(f"{A.get(k, k)}~\\cite{{{k}}}" for k in keys)
    return re.sub(r"\\citet\{([^}]+)\}", sub, text)


def main_floats():
    """Floats of the short paper: {auto table or figure file: (main label, kind)} and the main paper's numbers (main.aux)."""
    src = inline(open(os.path.join(PAPER, "short", "main_cas_src.tex"), encoding="utf-8").read())
    owners = {}
    for env in re.findall(r"\\begin\{(?:table|figure|inlinetable)\*?\}.*?\\end\{(?:table|figure|inlinetable)\*?\}", src, flags=re.S):
        lab = re.search(r"\\label\{([^}]+)\}", env)
        if not lab:
            continue
        kind = "Figure" if env.startswith("\\begin{figure") else "Table"
        for f in re.findall(r"%AUTOFILE (auto/\S+)", env) + re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", env):
            owners[f.replace(".pdf", "")] = (lab.group(1), kind)
    nums = {}
    aux = os.path.join(OUT, "main.aux")
    if os.path.exists(aux):
        nums = dict(re.findall(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}", open(aux, encoding="utf-8").read()))
    return owners, nums


SAME_LABEL = {"tab:ladder", "tab:params", "tab:admission", "tab:t2", "tab:t3h", "tab:costmodel"}

# supplement figures that redraw what the main paper shows: supplement label -> (main label, panel)
SAME_CONTENT = {"fig:privacy-curves": ("fig:priv", "a"), "fig:frontier": ("fig:priv", "c"), "fig:matched": ("tab:matched", ""), "fig:heat": ("tab:goal", "")}


def drop_duplicates(body):
    """Remove from the supplement every float that the main paper already shows, and point its references to the main paper."""
    owners, nums = main_floats()
    gone = {}
    for env in re.findall(r"\\begin\{(table|figure)(\*?)\}.*?\\end\{\1\2\}", body, flags=re.S):
        pass
    for m in list(re.finditer(r"\\begin\{(table|figure)(\*?)\}.*?\\end\{\1\2\}\n?", body, flags=re.S)):
        env = m.group(0)
        files = [f.replace(".pdf", "") for f in re.findall(r"%AUTOFILE (auto/\S+)", env) + re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", env)]
        hit = [owners[f] for f in files if f in owners]
        lab = re.search(r"\\label\{([^}]+)\}", env)
        if hit and lab:
            main_lab, kind = hit[0]
            gone[lab.group(1)] = f"{kind}~{nums.get(main_lab, '?')} of the main paper"
        elif lab and lab.group(1) in SAME_LABEL and lab.group(1) in nums:
            kind = "Figure" if env.startswith("\\begin{figure") else "Table"
            gone[lab.group(1)] = f"{kind}~{nums[lab.group(1)]} of the main paper"
        elif lab and lab.group(1) in SAME_CONTENT and SAME_CONTENT[lab.group(1)][0] in nums:
            main_lab, panel = SAME_CONTENT[lab.group(1)]
            kind = "Figure" if main_lab.startswith("fig:") else "Table"
            gone[lab.group(1)] = f"{kind}~{nums[main_lab]}{panel} of the main paper"
    for lab in gone:
        body = re.sub(r"\\begin\{(table|figure)(\*?)\}(?:(?!\\end\{\1\2\}).)*?\\label\{" + re.escape(lab) + r"\}.*?\\end\{\1\2\}\n?", "", body, flags=re.S)

    def sub(mo):
        cmd, keys = mo.group(1), [k.strip() for k in mo.group(2).split(",")]
        keep = [k for k in keys if k not in gone]
        out = [gone[k] for k in keys if k in gone]
        if keep:
            out.insert(0, f"\\{cmd}{{{','.join(keep)}}}")
        return ", ".join(out) if len(keep) != len(keys) else mo.group(0)
    body = re.sub(r"\\(cref|Cref|ref)\{([^}]*)\}", sub, body)
    # "Table 8 of the main paper and Table 8 of the main paper run ..." once a figure and its table both point to one main table
    body = re.sub(r"(\w+~\w+ of the main paper) and \1 (run|give|show|put)\b", r"\1 \2s", body)
    return body, gone


PREAMBLE = r"""\documentclass[preprint,12pt]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[export]{adjustbox}
\usepackage{array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\usepackage{xcolor}
\usepackage{placeins}
\usepackage{microtype}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage[capitalise,noabbrev]{cleveref}
\biboptions{sort&compress}

\graphicspath{{figs/}}
\journal{Computers and Electrical Engineering}

\newtheorem{property}{Property}
\newtheorem{proposition}{Proposition}
% Foundations that predate the 2024 review window (geo-indistinguishability, adversary-aware location privacy, exact binomial
% limits, fixed-sequence testing). Set \olderfalse to remove every such citation from the text and the reference list.
\newif\ifolder
\olderfalse
\newcommand{\sys}{\textsc{UAV-FlowShield}}
\newcommand{\bench}{\textsc{FlowShield-Bench}}
\newcommand{\E}{\mathcal{E}}
\newcommand{\reg}{\mathcal{R}}
"""
SUPP_PREAMBLE = r"""\documentclass[a4paper,fleqn]{cas-sc}

\usepackage[numbers]{natbib}
\usepackage[scaled=0.9]{helvet}
\usepackage[scaled=.85]{inconsolata}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[export]{adjustbox}
\usepackage{array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\usepackage{placeins}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[capitalise,noabbrev]{cleveref}

\graphicspath{{figs/}}
\newtheorem{property}{Property}
\newtheorem{proposition}{Proposition}
\newif\ifolder
\olderfalse
\newcommand{\sys}{UAV-FlowShield}
\newcommand{\bench}{FlowShield-Bench}
\newcommand{\E}{\mathcal{E}}
\newcommand{\reg}{\mathcal{R}}
"""
TITLE = r"\sys: A Trusted Execution Boundary for LLM-Assisted UAV Mission Planning with Attack-Calibrated Private Views and Authority-Bounded Observations"


def build_supp(M):
    main = read("main.tex")
    body_files = re.findall(r"\\input\{(sections/[^}]+)\}", main)
    excluded_labels = set()
    for f in EXCLUDED_FILES:
        excluded_labels |= labels_in(read(f))
    parts = []
    for f in body_files:
        if f in EXCLUDED_FILES or f == "sections/results_final":
            continue
        if f == "sections/availability":
            parts.append("\\section{Data and code availability}\n\\label{sec:avail}\n" + inline(read(f)).replace("\\label{sec:avail}", ""))
            continue
        parts.append(inline(read(f)))
    body = "\n\n".join(parts)
    body = body.replace("\\label[appendix]{", "\\label{").replace("\\FloatBarrier", "")
    excluded_labels |= labels_in(read("sections/results_final"))
    body = fix_refs(body, excluded_labels)
    body, gone = drop_duplicates(body)
    print("floats shown in the main paper and dropped here:", len(gone))
    body = numbered_citet(expand(strip_comments(body), M))
    body = re.sub(r"(\\includegraphics(?:\[[^]]*\])?\{)figs/", r"\1", body)
    tex = SUPP_PREAMBLE + r"""
\let\oldsubsection\subsection
\renewcommand{\subsection}{\FloatBarrier\oldsubsection}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}

\begin{document}
\let\WriteBookmarks\relax
\shorttitle{Supplementary material: UAV-FlowShield}
\shortauthors{A.A. Khan et al.}
\title[mode=title]{Supplementary material for: """ + TITLE + r"""}
\author[1]{Abdullah Aman Khan}
\cormark[1]
\ead{abdkhan@stbu.edu.cn}
\affiliation[1]{organization={Sichuan Technology and Business University}, city={Chengdu}, state={Sichuan}, country={China}}
\author[2]{Waqas Amin}
\affiliation[2]{organization={University of Management and Technology}, city={Lahore}, country={Pakistan}}
\author[3]{Muhammad Asim Saleem}
\affiliation[3]{organization={Riphah International University, Faisalabad Campus}, city={Faisalabad}, country={Pakistan}}
\cortext[1]{Corresponding author}

\begin{abstract}
This supplement contains the complete description of the method, the experimental setup, every experiment and the appendices of the study. Sections, tables, figures and equations are numbered with the prefix S; Sections 1 and 2 (introduction and related work) are those of the main paper, and tables and figures that the main paper shows are not repeated here but cited by their number in the main paper. Every number is generated by the released scripts from the raw results.
\end{abstract}
\maketitle

""" + body + r"""

\bibliographystyle{elsarticle-num}
\bibliography{refs}

\end{document}
"""
    tex = tex.replace("\\resizebox{\\textwidth}{!}{", "\\resizebox{\\ifdim\\width>\\textwidth \\textwidth\\else\\width\\fi}{!}{")
    open(os.path.join(OUT, "supplementary.tex"), "w", encoding="utf-8").write(tex)
    figs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))
    return figs


def supp_labels():
    p = os.path.join(OUT, "supplementary.aux")
    L = {}
    if os.path.exists(p):
        for name, num in re.findall(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}", open(p, encoding="utf-8").read()):
            L[name] = num
    return L


def build_main(M, name="main_src.tex"):
    src = open(os.path.join(PAPER, "short", name), encoding="utf-8").read()
    src = inline(src)
    L = supp_labels()
    missing = set()

    def sub(mo):
        k = mo.group(1)
        if k not in L:
            missing.add(k)
            return "S??"
        return L[k]
    src = re.sub(r"\\supp\{([^}]+)\}", sub, src)
    src = numbered_citet(expand(strip_comments(src), M))
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(src)
    if missing:
        print("unresolved supplement labels:", sorted(missing))
    figs = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", src))
    for a, b, c in re.findall(r"\\threefig\{([^}]+)\}\{([^}]+)\}\{([^}]+)\}", src):
        figs |= {a, b, c}
    return figs


def copy_figs(figs):
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    for f in [g for g in figs if "#" not in g]:
        name = os.path.basename(f)
        if not name.endswith(".pdf") and not name.endswith(".png"):
            name += ".pdf"
        shutil.copy(os.path.join(PAPER, "figs", name), os.path.join(OUT, "figs", name))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    M = macros()
    shutil.copy(os.path.join(PAPER, "refs.bib"), os.path.join(OUT, "refs.bib"))
    what = sys.argv[1]
    if what == "supp":
        figs = build_supp(M)
    elif what == "cas":
        figs = build_main(M, "main_cas_src.tex")
        tpl = os.environ.get("CAS_TEMPLATE", os.path.join(PAPER, "cas"))
        for f in ("cas-dc.cls", "cas-common.sty"):
            shutil.copy(os.path.join(tpl, f), os.path.join(OUT, f))
        import subprocess
        bst = subprocess.run(["kpsewhich", "elsarticle-num.bst"], capture_output=True, text=True).stdout.strip()
        if bst:
            shutil.copy(bst, os.path.join(OUT, "elsarticle-num.bst"))
    else:
        figs = build_main(M)
    copy_figs(figs)
    print(what, "written;", len(M), "numbers; figures:", sorted(figs))
