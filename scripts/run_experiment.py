"""Run a full experiment: train SI, train TC, evaluate, all into one folder."""
import argparse
from pathlib import Path

from ptc.config import add_config_args, config_from_args
from ptc.evaluation import evaluate, format_report
from ptc.training.si import train_si
from ptc.training.tc import train_tc
from ptc.utils import experiment_dir, save_json, setup_logging

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    setup_logging()

    out = experiment_dir(Path(cfg.runtime.out_dir), cfg.group_name())
    out.mkdir(parents=True)

    si_metrics = train_si(cfg, out_dir=out / "si")
    tc_metrics = train_tc(cfg, out_dir=out / "tc")
    results = evaluate(cfg.data.processed, out / "si", out / "tc", "dev", out / "eval")

    # TODO: yours. Save config, si_metrics, tc_metrics and results into out/"experiment.json"
    save_json({
        "config": cfg.to_dict(),
        "si_metrics": si_metrics,
        "tc_metrics": tc_metrics,
        "results": results
    }, out / "experiment.json")
    print(format_report(results))