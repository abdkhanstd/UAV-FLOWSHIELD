"""Latency of single model calls in the system-cost run (results/cost_<model>.json, written by exp_cost.py).

UAV-FlowShield makes its edge extraction call before any planner call (flowshield/pipeline.py), so in every
UAV-FlowShield row with one edge call the first entry of call_walls is the extraction call. Prints the extraction-call
median and 95th percentile and the share of missions with a repair call, as reported in the cost section of the paper."""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main(model="gemma4:12b"):
    D = json.load(open(os.path.join(ROOT, "results", f"cost_{model.replace(':', '-')}.json")))
    fs = [r for r in D["rows"] if r["method"] == "flowshield"]
    assert fs and all(r["edge_calls"] == 1 for r in fs), "expected one edge (extraction) call per mission"
    ext = np.array([r["call_walls"][0] for r in fs])
    repaired = np.mean([r["cloud_calls"] > 1 for r in fs])
    print(f"extraction call: median {np.median(ext):.1f} s, 95th percentile {np.percentile(ext, 95):.1f} s (n = {len(ext)})")
    print(f"missions with a repair call: {100 * repaired:.1f} % (mean planner calls {np.mean([r['cloud_calls'] for r in fs]):.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemma4:12b")
