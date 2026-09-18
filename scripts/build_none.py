"""D4: stage-1 false positives on train -> NONE examples for stage 2.

    python scripts/build_none.py --si-run runs/si-base-s13
    python scripts/train_tc.py --set tc.use_none=true
"""
import argparse

from ptc.config import add_config_args, config_from_args
from ptc.training.none import build_none
from ptc.utils import setup_logging

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--si-run", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()
    build_none(cfg, args.si_run, args.split, args.out)
