"""Run methods over a benchmark with a given LLM backbone (resumable, load-balanced over Ollama servers).

Example:
  python scripts/run_llm.py --model gemma4:12b --methods undefended,flowshield --out results/main_gemma.jsonl
Options: --bench (benchmark file), --scales (per-episode calibrated noise scales), --opts '{"drop_p":0.3}' (pipeline
options), --edge-model (separate model for the edge-side extraction calls).
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flowshield.data import load_bench
from flowshield.llm import LLM
from flowshield.pipeline import run_episode, is_fs

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default=None, help="file with episode ids (one per line) to restrict to")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--bench", default="data/bench.json")
    ap.add_argument("--scales", default=None, help="json with {'scales': {episode id: {'scale': metres}}}")
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--edge-model", default=None)
    ap.add_argument("--tag", default="", help="suffix for the LLM cache file")
    args = ap.parse_args()

    B = load_bench(os.path.join(ROOT, args.bench))
    sp = os.path.join(ROOT, args.scales) if args.scales else os.path.join(ROOT, "data", "scales.json")
    scales = json.load(open(sp))["scales"] if os.path.exists(sp) else {}
    eps = [e for e in B["episodes"] if args.split == "all" or e["split"] == args.split]
    if args.ids:
        keep = set(open(args.ids).read().split())
        eps = [e for e in eps if e["id"] in keep]
    if args.categories:
        cats = set(args.categories.split(","))
        eps = [e for e in eps if e["category"] in cats]
    methods = args.methods.split(",")
    opts = json.loads(args.opts)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["id"], r["method"]))
            except Exception:
                pass
    jobs = []
    for e in eps:
        for m in methods:
            if (e["attack"] or {}).get("goal") == "AD4" and not is_fs(m):
                continue   # planning-frame disclosure is only defined for private-view methods
            if (e["id"], m) not in done:
                jobs.append((e, m))
    random.Random(0).shuffle(jobs)
    tag = args.model.replace(":", "-").replace("/", "-")
    llm = LLM(args.model, os.path.join(ROOT, "results", f"cache_{tag}_t{args.temperature}_s{args.seed}{args.tag}.jsonl"),
              temperature=args.temperature, seed=args.seed)
    llm_edge = None
    if args.edge_model:
        etag = args.edge_model.replace(":", "-").replace("/", "-")
        llm_edge = LLM(args.edge_model, os.path.join(ROOT, "results", f"cache_{etag}_t{args.temperature}_s{args.seed}{args.tag}.jsonl"),
                       temperature=args.temperature, seed=args.seed)
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]

    def go(job):
        e, m = job
        try:
            r = run_episode(e, m, llm, scale=scales.get(e.get("orig_id", e["id"]), {}).get("scale"), opts=opts, llm_edge=llm_edge)
        except Exception as ex:
            print("ERROR", e["id"], m, repr(ex)[:200], flush=True)
            return
        r["model"], r["temperature"], r["seed"] = args.model, args.temperature, args.seed
        if args.edge_model:
            r["edge_model"] = args.edge_model
        if opts:
            r["opts"] = opts
        with lock:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            n_done[0] += 1
            if n_done[0] % 25 == 0:
                el = time.time() - t0
                print(f"{n_done[0]}/{len(jobs)} done, {el / 60:.1f} min, eta {el / n_done[0] * (len(jobs) - n_done[0]) / 60:.1f} min",
                      flush=True)

    print(f"{len(jobs)} jobs ({len(done)} already done)", flush=True)
    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(go, jobs))
    print("finished", time.time() - t0)


if __name__ == "__main__":
    main()
