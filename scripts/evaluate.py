"""D8: SI, gold-span TC and FLC end-to-end (+ FP report, + per-outlet with --outlets).

    python scripts/evaluate.py --si-run runs/si-base-s13 --tc-run runs/tc-base-s13 --split dev
"""
import argparse
from pathlib import Path

from ptc.config import add_config_args, config_from_args
from ptc.evaluation import evaluate, format_report
from ptc.utils import setup_logging

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--si-run", required=True)
    p.add_argument("--tc-run", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--out", help="default: runs/eval-<si>__<tc>")
    p.add_argument("--outlets", help="CSV from scripts/join_outlets.py")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    out = args.out or Path(cfg.runtime.out_dir) / f"eval-{Path(args.si_run).name}__{Path(args.tc_run).name}"
    results = evaluate(cfg.data.processed, args.si_run, args.tc_run, args.split, out,
                       args.outlets, device=cfg.runtime.device, precision=cfg.runtime.precision)
    print(format_report(results))
    print(f"\nfull results -> {out}")
