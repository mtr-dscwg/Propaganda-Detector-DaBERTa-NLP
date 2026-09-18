"""D4: build NONE-class training examples from stage-1 false positives."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Config
from ..data.corpus import load_split
from ..data.spans import spans_by_article
from ..inference import predict_spans
from ..models.checkpoint import load_bundle
from ..utils import get_device, resolve_precision

log = logging.getLogger(__name__)


def build_none(cfg: Config, si_run: str | Path, split: str = "train", out_path: str | Path | None = None) -> pd.DataFrame:
    """Predicted spans on `split` that overlap no gold TC span -> NONE examples.

    Caveat: stage 1 was trained on train, so it makes fewer mistakes there than on unseen
    text. If the FP count is small relative to dev, train stage 1 on half of train and
    predict the other half (cross-fitting) for more realistic negatives.
    """
    device = get_device(cfg.runtime.device)
    amp = resolve_precision(cfg.runtime.precision, device)
    bundle = load_bundle(si_run, device)
    data = load_split(cfg.data.processed, split)
    m = bundle.meta
    pred = predict_spans(bundle.model, bundle.tokenizer, data.articles, m["max_length"], m["stride"],
                         cfg.si.eval_batch_size, device, amp)
    gold = spans_by_article(data.tc)
    keep = [not any(s < ge and gs < e for gs, ge in gold.get(a, []))
            for a, s, e in zip(pred["article_id"], pred["start"], pred["end"])]
    none = pred[keep].reset_index(drop=True)[["article_id", "start", "end"]]
    out = Path(out_path or cfg.tc.none_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    none.to_parquet(out, index=False)
    log.info("%s: %d predicted spans, %d false positives -> %s", split, len(pred), len(none), out)
    return none
