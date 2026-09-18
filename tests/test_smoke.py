"""End-to-end: every script, synthetic PTC-format data, tiny local backbone, CPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synthetic import make_tiny_backbone, write_corpus

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_end_to_end(tmp_path):
    raw, backbone = tmp_path / "raw", tmp_path / "tiny-backbone"
    write_corpus(raw)
    make_tiny_backbone(backbone, raw)
    cfg = {
        "backbone": str(backbone), "seed": 1,
        "data": {"root": str(raw), "processed": str(tmp_path / "processed")},
        "si": {"max_length": 64, "stride": 16, "epochs": 12, "batch_size": 8, "grad_accum": 1,
               "lr": 1e-3, "head_lr": 5e-3, "patience": 12},
        "tc": {"max_length": 64, "context_chars": 60, "epochs": 12, "batch_size": 8, "grad_accum": 1,
               "lr": 1e-3, "head_lr": 1e-3, "patience": 12,
               "none_path": str(tmp_path / "processed" / "none_train.parquet")},
        "runtime": {"device": "cpu", "out_dir": str(tmp_path / "runs"), "log_every": 1000},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def run(script, *args):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--config", str(cfg_path), *args],
                           capture_output=True, text=True, encoding="utf-8", cwd=tmp_path)
        assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
        return r.stdout

    print(run("prepare_data.py"))
    print(run("check_roundtrip.py", "--report", str(tmp_path / "rt.jsonl")))
    run("train_si.py")
    run("train_tc.py")
    runs = tmp_path / "runs"
    si_run, tc_run = runs / "si-tiny-backbone-s1", runs / "tc-tiny-backbone-s1"

    si_best = json.loads((si_run / "metrics.json").read_text())["best"]
    tc_best = json.loads((tc_run / "metrics.json").read_text())["best"]
    assert si_best["si_f1"] > 0.6, si_best
    assert tc_best["tc_micro_f1"] > 0.6, tc_best

    report = run("evaluate.py", "--si-run", str(si_run), "--tc-run", str(tc_run), "--split", "dev")
    print(report)
    assert "FLC F1 (end-to-end)" in report
    run("evaluate.py", "--si-run", str(si_run), "--tc-run", str(tc_run), "--split", "test")

    run("build_none.py", "--si-run", str(si_run))
    run("train_tc.py", "--set", "tc.use_none=true", "--set", "tc.epochs=2")
    assert (runs / "tc-none-tiny-backbone-s1" / "model.pt").exists()

    article = next((raw / "dev-articles").glob("*.txt"))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "predict.py"), str(article),
                          "--si-run", str(si_run), "--tc-run", str(tc_run),
                          "--json", str(tmp_path / "p.jsonl"), "--html", str(tmp_path / "html"), "--device", "cpu"],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    print(out.stdout[-1500:])
    assert (tmp_path / "p.jsonl").exists()
    print(subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate_seeds.py"), str(runs)],
                         capture_output=True, text=True).stdout)
