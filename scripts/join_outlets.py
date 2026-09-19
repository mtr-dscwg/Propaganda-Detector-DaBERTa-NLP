"""Q1 / D9: recover outlets for PTC articles from proppy 1.0.

    python scripts/join_outlets.py --proppy data/raw/proppy/proppy_1.0.*.tsv
    python scripts/evaluate.py ... --outlets data/processed/outlets.csv
"""
import argparse
import glob

import pandas as pd

from ptc.config import add_config_args, config_from_args
from ptc.data.corpus import load_split
from ptc.outlets import join_outlets, load_proppy
from ptc.utils import setup_logging

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--proppy", nargs="+", required=True, help="proppy TSV files (globs ok)")
    p.add_argument("--source-col", type=int, help="use this proppy column as outlet instead of URL domain")
    p.add_argument("--out", default="processed-dataset/outlets.csv")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()

    paths = [f for pat in args.proppy for f in glob.glob(pat)] or args.proppy
    proppy = load_proppy(paths, args.source_col)
    frames = []
    for split in ("train", "dev", "test"):
        try:
            data = load_split(cfg.data.processed, split)
        except FileNotFoundError:
            continue
        j = join_outlets(data.articles, proppy)
        j.insert(1, "split", split)
        frames.append(j)
        print(f"{split:<6} {j['outlet'].notna().mean():6.1%} matched   "
              f"(by id {(j['match'] == 'id').sum()}, by text {(j['match'] == 'text').sum()}, of {len(j)})")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print("\ntop outlets (sanity-check against EMNLP 2019 Table 3):")
    print(out["outlet"].value_counts().head(15).to_string())
    print(f"\n-> {args.out}")
