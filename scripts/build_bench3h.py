"""Held-out wording for benchmark T3: the same 144 missions and specifications as data/bench_t3.json, with every operator
requirement re-written by a model of another family than the extractor (qwen3:8b) in a register drawn at random
(radio message, e-mail, chat message, checklist line, free choice). The extractor prompt, its examples and the regular
expressions of the edge parser were written before these texts existed, so this is the held-out wording test.

A rewrite is accepted when every site name and number of the template survives, no other site name or number appears, and
the rewrite contains a word of the specification's type (the keyword lists are broader than in build_bench3.py). Accepted
rewrites are then read; the file data/bench_t3h_screening.json lists the ones removed because the meaning changed.

usage: python scripts/build_bench3h.py [--author-model qwen3:8b] --out data/bench_t3h.json
"""
import argparse
import copy
import difflib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield import specs as SP
from flowshield.llm import LLM, parse_json

ROOT = os.path.join(os.path.dirname(__file__), "..")
REGISTERS = ["a terse radio message from the operations desk", "a polite e-mail sentence from the safety officer",
             "a casual chat message from a field technician", "one line of a pre-flight checklist",
             "the way an experienced pilot would say it out loud"]
KW = {"altitude": r"altitude|above ground|height|high|metres|meters|\bm\b", "alt_by_camera": r"altitude|above ground|height|high|metres|meters|\bm\b",
      "geo_first": r"north", "by_distance": r"distance|nearest|closest|closer|outward|farth|furth|out from|away",
      "closest_pair": r"closest|nearest|neighbo|adjacent|next to|together|back.to.back|pair",
      "revisit": r"twice|second|again|once more|two|revisit|re-?shoot|return", "first": r"first|begin|start|open|kick",
      "last": r"last|final|end|clos|save|finish|wrap", "precedence": r"before|until|prior|ahead|precede|first|after|then"}


def ok(text, specs):
    tmpl = SP.render_all(specs, 0) + " " + SP.render_all(specs, 1)
    want = set(re.findall(r"\bS\d\b", tmpl)) | set(re.findall(r"\d+", SP.render_all(specs, 0)))
    got = set(re.findall(r"\bS\d\b", text)) | set(re.findall(r"\d+", text))
    if not want <= got or got - want:
        return False
    if any(difflib.SequenceMatcher(None, text.lower(), SP.render_all(specs, f).lower()).ratio() > 0.9 for f in (0, 1)):
        return False                                    # a copy of a template wording is not a held-out wording
    return all(re.search(KW[s["type"]], text, re.I) for s in specs) and 15 <= len(text) <= 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author-model", default="qwen3:8b")
    ap.add_argument("--src", default="data/bench_t3.json")
    ap.add_argument("--out", default="data/bench_t3h.json")
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    B = json.load(open(os.path.join(ROOT, a.src)))
    llms = [LLM(a.author_model, os.path.join(ROOT, "results", "cache_author_t3h.jsonl"), temperature=0.8, seed=3 + k, num_predict=300) for k in range(8)]
    out, stats = [], {"accepted": 0, "failed": 0}
    for ep in B["episodes"]:
        specs = ep["specs"]
        tmpl = SP.render_all(specs, 0)
        reg = str(rng.choice(REGISTERS))
        text = None
        for tries in range(8):
            try:
                anchors = bool(re.search(r"S\d|\d", tmpl))
                r = llms[tries].chat('You rewrite operator requirements for a drone inspection company. Reply with JSON {"text": "..."} only.',
                             f"Rewrite the requirement below as {reg}. Use your own wording, not the original sentence structure. "
                             "Keep every site name (S1, S2, ...) and every number exactly as written, keep the meaning, and do not add requirements "
                             "or other numbers." + ("" if anchors else " This requirement names no site and gives no number, so your text must not name any site or give any number.")
                             + f"\nRequirement: {tmpl}" + (f"\n(variant {tries})" if tries else ""), tag=f"{ep['id']}:h:{tries}")
                d = parse_json(r["content"])
                t2 = d.get("text") if isinstance(d, dict) else None
                if isinstance(t2, str) and ok(t2, specs):
                    text = re.sub(r"\s*[–—]\s*", ", ", t2.strip())
                    break
            except Exception:
                pass
        e2 = copy.deepcopy(ep)
        e2["spec_template_text"] = ep["spec_text"]
        if text is None:
            stats["failed"] += 1
            e2["spec_ok"] = False
            stats.setdefault("failed_types", []).append("+".join(x["type"] for x in specs))
            e2["spec_text"] = ep["spec_text"]
        else:
            stats["accepted"] += 1
            e2["spec_ok"], e2["spec_text"] = True, text
        e2["spec_author"], e2["spec_register"] = a.author_model, reg
        out.append(e2)
    B2 = dict(B)
    B2["episodes"] = out
    B2["stats"] = stats
    json.dump(B2, open(os.path.join(ROOT, a.out), "w"))
    print(stats)
    for e in out[:8]:
        print(e["id"], e["spec_kind"], "|", e["spec_text"])


if __name__ == "__main__":
    main()
