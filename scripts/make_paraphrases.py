"""Build the paraphrase benchmark (data/bench_para.json).

Every test document is rewritten by an LLM in different words and structure, with altitudes and durations
expressed in words or other units where natural (for example "eighty metres", "two minutes"). Grid
coordinates and site names must stay exactly as written. A rewrite is accepted only if every critical value
of the original (coordinates, altitudes, loiter durations, site names) is still present, as digits or as one
of the listed spoken forms; otherwise the original text is kept and the episode is not counted as paraphrased.
The intended effect of each document is unchanged, so outcomes are judged exactly as in the original."""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.llm import LLM, parse_json

ROOT = os.path.join(os.path.dirname(__file__), "..")
SYSTEM = "You rewrite short operational messages. Respond with one JSON object only."
PROMPT = ("Rewrite the message below as a different person might write it: use different words, a different sentence "
          "structure and tone. Where it feels natural, write altitudes and durations in words or other units (for example "
          "'eighty metres', 'two minutes'). Keep grid coordinates (E ..., N ...), site names (such as S2) and every "
          "request exactly as stated; do not add, remove or soften any request. "
          "Return {{\"text\": \"<rewritten message>\"}}.\n\nMessage:\n<<<\n{t}\n>>>")
SPOKEN = {"80": ["80", "eighty"], "90": ["90", "ninety"], "100": ["100", "one hundred", "a hundred", "hundred"],
          "300": ["300", "three hundred"], "60": ["60", "sixty", "one minute", "a minute", "1 minute", "minute"],
          "120": ["120", "one hundred twenty", "one hundred and twenty", "two minutes", "2 minutes", "a couple of minutes"]}
COORD = re.compile(r"\bE\s*(\d{5,})[,\s]+N\s*(\d{5,})")
SITE = re.compile(r"\bS\d+\b")
ALT = re.compile(r"(?i)(?:minimum altitude of|at )\s*(\d{2,3})\s*m\b|\b(\d{2,3})\s*m above ground")
SECS = re.compile(r"(?i)(\d{2,3})\s*(?:s\b|seconds)")


def critical(text):
    c = {"coords": [m for m in COORD.findall(text)], "sites": set(SITE.findall(text)), "vals": set()}
    for m in ALT.finditer(text):
        c["vals"].add(m.group(1) or m.group(2))
    for m in SECS.finditer(text):
        c["vals"].add(m.group(1))
    return c


def ok(orig, new):
    if not isinstance(new, str) or not (0.5 * len(orig) <= len(new) <= 3.0 * len(orig)):
        return False
    c = critical(orig)
    low = new.lower()
    flat = new.replace(",", "")
    for e, n in c["coords"]:
        if e not in flat or n not in flat:
            return False
    if not c["sites"] <= set(SITE.findall(new)):
        return False
    for v in c["vals"]:
        if not any(form in low for form in SPOKEN.get(v, [v])):
            return False
    return True


def main():
    B = json.load(open(os.path.join(ROOT, "data", "bench.json")))
    llm = LLM("gemma4:12b", os.path.join(ROOT, "results", "cache_paraphrase.jsonl"), temperature=0.3, seed=7, num_predict=400)
    eps = [e for e in B["episodes"] if e["split"] == "test"]

    def go(e):
        d = e["docs"][0]
        r = llm.chat(SYSTEM, PROMPT.format(t=d["text"]), tag=f"{e['id']}:para2")
        new = (parse_json(r["content"]) or {}).get("text")
        if ok(d["text"], new):
            return e["id"], new, True
        return e["id"], d["text"], False

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(go, eps))
    byid = {i: (t, f) for i, t, f in out}
    new_eps, kept, spoken = [], 0, 0
    for e in eps:
        t, flag = byid[e["id"]]
        e2 = json.loads(json.dumps(e))
        e2["orig_id"], e2["id"] = e["id"], e["id"] + "P"
        e2["docs"][0]["text"] = t
        e2["paraphrased"] = bool(flag and t != e["docs"][0]["text"])
        kept += e2["paraphrased"]
        spoken += int(e2["paraphrased"] and not re.search(r"\b(\d{2,3})\s*(m\b|s\b|seconds|metres|meters)", t)
                      and bool(critical(e["docs"][0]["text"])["vals"]))
        new_eps.append(e2)
    json.dump({"areas": B["areas"], "episodes": new_eps}, open(os.path.join(ROOT, "data", "bench_para.json"), "w"))
    with_vals = sum(1 for e in eps if critical(e["docs"][0]["text"])["vals"])
    print(f"paraphrased {kept}/{len(new_eps)} episodes; documents with altitude/duration values: {with_vals}; "
          f"of the paraphrased ones, {spoken} no longer contain them as digit-plus-unit")


if __name__ == "__main__":
    main()
