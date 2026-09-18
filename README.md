# Propaganda detector — PTC-SemEval20, two stages

Stage 1 finds propagandistic spans (DeBERTa-v3 + BIO + CRF). Stage 2 labels each span with one of
14 techniques (DeBERTa-v3 over `context [BOP] span [EOP] context`). Implements plan-of-record rev. 3.

## Setup (Windows, PowerShell, RTX 5060)

Install PyTorch **before** the project. On Windows, PyPI's `torch` is CPU-only, and RTX 50-series
(Blackwell) needs a CUDA 12.8+ build.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -m "not slow"                 # unit tests, seconds
pytest -m slow -s                    # full pipeline on synthetic data + tiny model, ~1-2 min
```

If any dependency has no Python 3.14 wheel yet, recreate the venv with `py -3.12`.

## Data

Download datasets-v2.tgz from https://zenodo.org/records/3952415 (CC BY 4.0, no registration) and put it on
`datasets/train-articles/` exists (or point `data.root` elsewhere). The loader finds label
files by pattern. If dev gold is missing, set `data.dev_holdout_articles: 75` to carve dev from train.


## Workflow (phases from the plan)

```powershell
# Phase 0 - data + scorer gates
python scripts/prepare_data.py              # expect train 371 articles / 5,468 SI / 6,128 TC
python scripts/check_roundtrip.py           # exit 0 = no offset bugs; prints the BIO ceiling
python scripts/score.py --task si --pred <baseline_output.txt> --split dev   # must give 0.31

# Phase 1 - MVP
python scripts/train_si.py                  # -> runs/si-base-s13
python scripts/train_tc.py                  # -> runs/tc-base-s13
python scripts/evaluate.py --si-run runs/si-base-s13 --tc-run runs/tc-base-s13 --split dev
python scripts/predict.py --si-run runs/si-base-s13 --tc-run runs/tc-base-s13 article.txt

# Phase 2 - seeds (>= 5 before quoting any number)
foreach ($s in 1..5) { python scripts/train_si.py --seed $s; python scripts/train_tc.py --seed $s }
python scripts/aggregate_seeds.py runs
python scripts/train_si.py --set si.use_crf=false   # CRF ablation

# Phase 3 - NONE class (D4) and outlet control (D9)
python scripts/build_none.py --si-run runs/si-base-s13
python scripts/train_tc.py --set tc.use_none=true
python scripts/join_outlets.py --proppy "data/raw/proppy/*.tsv"
python scripts/evaluate.py ... --outlets data/processed/outlets.csv

# Test-set submission files (official formats) land in runs/eval-*/test_*.txt
python scripts/evaluate.py --si-run ... --tc-run ... --split test
```

Any setting can be overridden: `--backbone xsmall`, `--seed 3`, `--set si.lr=2e-5` (repeatable).

## Hardware notes (8 GB VRAM)

`base` runs at batch 8 × grad-accum 2 in bf16. On CUDA OOM, set `si.gradient_checkpointing: true`
(slower, much less memory) or drop `max_length` to 192. `large` does not fit for full fine-tuning
on 8 GB; that and parallel seed sweeps are what a rented GPU is for.

## What each number means (D8)

`evaluate.py` always reports: **SI F1** (char-level partial match, official definition),
**TC micro/macro-F1 on gold spans** (leaderboard-comparable), **FLC F1** end-to-end on predicted
spans (the honest number), plus the **false-positive rate** overall and per technique.

## Layout

```
src/ptc/
  config.py            YAML config, backbone flag, CLI overrides
  techniques.py        the 14 labels (+ NONE)
  data/corpus.py       raw release -> Parquet, validation, SI derivation
  data/bio.py          offsets <-> BIO, windows, round-trip diagnostics
  data/tokenization.py whole-article tokenization with stripped offsets
  data/si_dataset.py   stage-1 windows/batching
  data/tc_dataset.py   stage-2 marker inputs, class weights
  models/crf.py        CRF: NLL, Viterbi, forward-backward marginals
  models/span_tagger.py, span_classifier.py, checkpoint.py
  training/loop.py     AdamW, warmup, AMP, accumulation, early stopping
  training/si.py, tc.py, none.py
  inference.py         window merging, confidences, Pipeline, sentence view
  scoring.py           official SI/FLC/TC scorers, FP report, submission I/O
  evaluation.py        three-number report, per-outlet split
  outlets.py           proppy join (ID, then text fingerprint)
  render.py            terminal UI + HTML
scripts/               thin CLIs for each step
tests/                 scorer, CRF-vs-brute-force, BIO/offsets, end-to-end smoke
```

## Policies worth knowing

- **Overlaps:** stage 1 trains on merged SI gold, so BIO never has to represent nesting.
  Stage 2 sees every TC span separately. A gold span with k techniques gets the top-k labels.
- **Round-trip:** spans whose boundaries fall mid-token cannot survive BIO by construction; they
  are counted and costed, not asserted. Token-aligned spans that fail are bugs and fail the check.
- **Confidence:** span confidence = mean CRF marginal P(inside) over the span; label confidence =
  softmax. The UI shows both; treat highlights as suggestions (the SI state of the art is ~0.52).
