import json, os, sys
p = os.path.expanduser("~/cae_exp/code/results/cache_gemma4-12b_t0.0_s0.jsonl")
rows = [json.loads(l) for l in open(p)]
print(len(rows), "cached calls")
r = rows[-200:]
print("last 200: mean wall %.1fs, mean prompt tokens %.0f, mean output tokens %.0f" % (sum(x["wall_s"] for x in r) / len(r), sum(x["prompt_tokens"] for x in r) / len(r), sum(x["output_tokens"] for x in r) / len(r)))
from collections import Counter
print(Counter(x["endpoint"] for x in r))
