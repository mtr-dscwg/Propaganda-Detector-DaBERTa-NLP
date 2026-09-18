"""Train stage 2 (technique classification): python scripts/train_tc.py [--set tc.use_none=true]"""
import argparse

from ptc.config import add_config_args, config_from_args
from ptc.training.tc import train_tc
from ptc.utils import setup_logging

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    setup_logging()
    train_tc(cfg)
