# Changelog

## [Unreleased]

### Fixed
- **Encoder weights are now always loaded as float32** (`models/checkpoint.py`).
  transformers 5 keeps a checkpoint's stored dtype, and the DeBERTa-v3-base safetensors
  are float16, so the whole encoder, and therefore AdamW's update, ran in float16.
  
  AdamW's eps (1e-8) underflows to 0 in float16, so the first optimizer update divided by zero and turned the encoder weights into NaN. It surfaced as a non-finite loss at step 3, the first forward pass after the first update (grad_accum = 2 in `default.yaml`).

  The `models/checkpoint.py` was given a _as_fp32 helper to convert the float16 to float32 in order to fix this issue, put in load_encoder and load_bundle

  bf16 autocast still applies on top of fp32 master weights.
  Regression test: `tests/test_checkpoint.py`.



### Added
- Training loop detects non-finite gradients, names the offending parameters, and skips
  the update instead of writing NaN into the weights; aborts after `max_skipped_steps`.
  On a non-finite loss it reports whether the weights themselves are non-finite.
  Test: `tests/test_loop.py`.
- `scripts/diagnose_nan.py`: isolates loading / forward / backward / optimizer step,
  in fp32 and bf16, with default and non-foreach AdamW.

### Changed
- Default config targets the Zenodo release (record 3952415), which has gold for train
  only: `data.dev_holdout_articles: 75`. Dev scores are for model selection, not
  leaderboard comparison.