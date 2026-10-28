"""Record the state of the pipeline before the fresh test benchmark is evaluated.

Writes results/eval/FROZEN.json with the SHA-256 of every source file of the system and of the prompts, the benchmark
files and the scale files. The paper reports the result of one pass over the fresh benchmark with this exact state.
usage: python scripts/freeze.py [extra files ...]"""
import glob
import hashlib
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


files = sorted(glob.glob(os.path.join(ROOT, "flowshield", "*.py"))) + sys.argv[1:]
rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {os.path.relpath(f, ROOT).replace("\\", "/"): sha(f) for f in files}}
for extra in ("data/bench.json", "data/bench_t2.json", "data/scales_ucb.json", "data/scales_t2.json"):
    p = os.path.join(ROOT, extra)
    if os.path.exists(p):
        rec["files"][extra] = sha(p)
os.makedirs(os.path.join(ROOT, "results", "eval"), exist_ok=True)
out = os.path.join(ROOT, "results", "eval", "FROZEN.json")
if os.path.exists(out):
    print("FROZEN.json already exists; refusing to overwrite", out)
    sys.exit(1)
json.dump(rec, open(out, "w"), indent=1)
print(len(rec["files"]), "files recorded at", rec["time"])
