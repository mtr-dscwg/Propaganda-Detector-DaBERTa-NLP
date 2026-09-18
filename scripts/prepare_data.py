"""Phase 0: official PTC release -> Parquet (+ stats to compare with 371 / 5,468 / 6,128)."""
import argparse

from ptc.config import add_config_args, config_from_args
from ptc.data.corpus import prepare
from ptc.utils import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    setup_logging()
    meta = prepare(cfg.data)
    print(f"\nwritten to {cfg.data.processed}   (dev is holdout: {meta['dev_is_holdout']})")
    print(f"{'split':<6} {'articles':>8} {'SI spans':>9} {'TC spans':>9}")
    for split in ("train", "dev", "test"):
        if split in meta:
            m = meta[split]
            print(f"{split:<6} {m['articles']:>8} {m['si_spans']:>9} {m['tc_spans']:>9}")
    if meta["train"]["techniques"]:
        print("\ntrain technique counts:")
        for t, n in sorted(meta["train"]["techniques"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5}  {t}")


if __name__ == "__main__":
    main()
