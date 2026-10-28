"""Check that the code on this machine has the hashes recorded in results/eval/FROZEN.json."""
import hashlib
import json
import os
import sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
rec = json.load(open(os.path.join(root, "results", "eval", "FROZEN.json")))
bad = 0
for f, h in rec["files"].items():
    p = os.path.join(root, f)
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    if got != h:
        bad += 1
        print("MISMATCH", f)
print("frozen at", rec["time"], "-", len(rec["files"]), "files,", bad, "mismatches")
sys.exit(1 if bad else 0)
