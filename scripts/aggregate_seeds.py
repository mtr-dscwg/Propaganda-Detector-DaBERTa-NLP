"""Mean ± std over seeds (D8: >= 5 seeds before a number leaves the notebook).

    python scripts/aggregate_seeds.py runs/
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs_dir", nargs="?", default="runs")
    args = p.parse_args()
    groups = defaultdict(list)
    for f in Path(args.runs_dir).glob("*/metrics.json"):
        name = re.sub(r"-s\d+$", "", f.parent.name)
        groups[name].append(json.loads(f.read_text(encoding="utf-8"))["best"])
    for name, runs in sorted(groups.items()):
        keys = [k for k in runs[0] if k.endswith(("f1", "precision", "recall"))]
        stats = []
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            sd = stdev(vals) if len(vals) > 1 else 0.0
            stats.append(f"{k} {mean(vals):.4f} ± {sd:.4f}")
        warn = "" if len(runs) >= 5 else "   (fewer than 5 seeds)"
        print(f"{name:<28} n={len(runs)}  " + "  ".join(stats) + warn)
