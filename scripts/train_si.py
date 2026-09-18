"""Train stage 1 (span identification): python scripts/train_si.py [--backbone base] [--seed 13]"""
import argparse

from ptc.config import add_config_args, config_from_args
from ptc.training.si import train_si
from ptc.utils import setup_logging

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    setup_logging()
    train_si(cfg)
