"""Architecture figure for the paper (paper/figs/arch.pdf)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")

BLUE, RED, AMBER = "#2b5d9b", "#b03a2e", "#9a6b00"


def zone(x, y, w, h, fc, ec, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 1.3, label, ha="center", va="top", fontsize=7.6, fontweight="bold", color=ec)


def box(x, y, w, h, text, fc="#ffffff", ec="#444", fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def line(pts, color="#222", ls="-", arrow=True, text=None, tpos=None, tfs=6.2, ha="center", va="bottom"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if arrow else xs, ys[:-1] if arrow else ys, color=color, ls=ls, lw=1.0, solid_capstyle="butt")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, ls=ls,
                                                                    mutation_scale=8, shrinkA=0, shrinkB=0))
    if text:
        ax.text(*tpos, text, color=color, fontsize=tfs, ha=ha, va=va, linespacing=1.15)


zone(0.5, 12, 19, 40, "#fff7e0", AMBER, "Untrusted\ncontent")
zone(22.5, 2, 52, 56, "#e8f1fb", BLUE, "Trusted edge")
zone(81, 22, 18.5, 36, "#fdecec", RED, "Untrusted cloud")

box(2, 22, 16, 17, "Portal tickets\n(origin tag),\nairspace feed,\npeer messages\n\n(may carry\ninjections)", ec=AMBER)
box(25, 44, 22, 10, "Authenticated mission\nsites, home, NFZ, budget,\nlicence, envelope $\\mathcal{E}$")
box(51, 42.5, 22, 12.5, "Private view builder\nsecret frame + planar noise;\nscale certified against a\nmodelled registration attack;\nelse: plan on the edge", fc="#dbe9f9", fs=5.9)
box(25, 23.5, 22, 13.5, "Observation extractor\n(edge LLM, schema only);\nadmission: channel, ticket\n(tag, fresh, single use,\napproval), envelope $\\mathcal{E}$", fc="#dbe9f9", fs=5.7)
box(51, 25, 22, 11, "Handle resolver and\nplan verifier: bounds,\ncompleteness, budget,\nredaction; fail closed\nor verified fallback", fc="#dbe9f9", fs=6.0)
box(51, 6, 22, 11, "Executor and runtime\nsupervisor: exact\ncoordinates, NFZ router,\nre-verification")
box(82.2, 26, 16.6, 27, "Planner LLM\n\nreceives:\nhandles W, O, H;\nnoisy planning-frame\ncoordinates;\ntyped observations\n\nreturns:\nplan over handles\nonly", ec=RED, fs=6.0)

# mission -> view builder, mission -> extractor
line([(47, 49), (51, 49)])
line([(36, 44), (36, 36)], text="mission,\nenvelope", tpos=(34.6, 40), ha="right", va="center")
# documents -> extractor (they never leave the edge)
line([(18, 30.5), (25, 30.5)], color=AMBER, ls="--", text="raw text", tpos=(21.5, 31.2), tfs=5.8)
# private view -> planner
line([(73, 49), (82.2, 49)], color=BLUE, text="private view", tpos=(78, 49.8))
# typed observations -> planner (route through the gap between rows)
line([(47, 33), (49, 33), (49, 40.0), (79, 40.0), (79, 38), (82.2, 38)], color=BLUE,
     text="typed observations only", tpos=(63, 40.5))
# extractor -> verifier
line([(47, 29), (51, 29)], text="admitted", tpos=(49, 26.2), tfs=5.8)
# plan from planner -> verifier ; errors back
line([(82.2, 29), (73, 29)], color=RED, text="plan", tpos=(78, 29.6))
line([(73, 33), (82.2, 33)], color=RED, ls="--", text="typed errors", tpos=(78, 33.6))
# verifier -> executor
line([(62, 25), (62, 17)], text="verified plan", tpos=(63, 21), ha="left", va="center")


fig.savefig(os.path.join(OUT, "arch.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "arch.png"), dpi=170, bbox_inches="tight")
print("saved")
