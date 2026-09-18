"""Score a submission file against gold. Use this to reproduce the published SI baseline
(0.31 on dev) before trusting anything else.

    python scripts/score.py --task si --pred baseline-output-SI.txt --split dev
"""
import argparse
import json

from ptc.config import add_config_args, config_from_args
from ptc.data.corpus import load_split, parse_label_file
from ptc.scoring import score_flc, score_si, score_tc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--task", choices=["si", "tc", "flc"], required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--split", default="dev", help="processed split to use as gold")
    p.add_argument("--gold", help="gold label file instead of a processed split")
    args = p.parse_args()
    cfg = config_from_args(args)

    pred = parse_label_file(args.pred)
    if args.gold:
        gold = parse_label_file(args.gold)
    else:
        data = load_split(cfg.data.processed, args.split)
        gold = data.si if args.task == "si" else data.tc
    result = {"si": score_si, "tc": score_tc, "flc": score_flc}[args.task](pred, gold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
