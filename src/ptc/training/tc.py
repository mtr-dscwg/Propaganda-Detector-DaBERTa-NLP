"""Train stage 2 (technique classification). Model selection on gold-span micro-F1 on dev."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Config
from ..data.corpus import load_split
from ..data.tc_dataset import SpanEncoder, TCCollator, TCDataset, add_marker_tokens, class_weights
from ..inference import assign_gold_span_labels, classify_spans
from ..models.checkpoint import load_encoder, load_tokenizer, save_bundle
from ..models.span_classifier import SpanClassifier
from ..scoring import score_tc, write_submission
from ..techniques import NONE_LABEL, label_list
from ..utils import describe_device, get_device, resolve_precision, save_json, set_seed
from .loop import fit

log = logging.getLogger(__name__)


def training_rows(cfg: Config, train_tc: pd.DataFrame) -> pd.DataFrame:
    rows = train_tc[["article_id", "start", "end", "technique"]]
    if cfg.tc.use_none:
        path = Path(cfg.tc.none_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run scripts/build_none.py first (tc.use_none=true)")
        none = pd.read_parquet(path)
        none["technique"] = NONE_LABEL
        log.info("adding %d NONE examples from %s", len(none), path)
        rows = pd.concat([rows, none[["article_id", "start", "end", "technique"]]], ignore_index=True)
    return rows.reset_index(drop=True)


def train_tc(cfg: Config) -> dict:
    set_seed(cfg.seed)
    hp = cfg.tc
    device = get_device(cfg.runtime.device)
    amp = resolve_precision(cfg.runtime.precision, device)
    out = Path(cfg.runtime.out_dir) / cfg.run_name("tc")
    log.info("TC run %s on %s (amp=%s) backbone=%s", out.name, describe_device(device), amp, cfg.backbone_id)

    train = load_split(cfg.data.processed, "train")
    dev = load_split(cfg.data.processed, "dev")
    if dev.tc.empty:
        raise RuntimeError("dev split has no TC gold. Set data.dev_holdout_articles and re-run prepare_data.")

    labels = label_list(hp.use_none)
    label2id = {t: i for i, t in enumerate(labels)}
    tokenizer = load_tokenizer(cfg.backbone_id)
    add_marker_tokens(tokenizer)
    encoder_in = SpanEncoder(tokenizer, hp.max_length, hp.context_chars)

    rows = training_rows(cfg, train.tc)
    dataset = TCDataset.build(train.articles, rows, encoder_in, label2id)
    weights = class_weights([x["labels"] for x in dataset.items], len(labels), hp.class_weight_power)
    log.info("train: %d spans, %d labels; class weights %s", len(dataset), len(labels),
             {t: round(float(w), 2) for t, w in zip(labels, weights)})

    encoder = load_encoder(cfg.backbone_id, tokenizer, hp.gradient_checkpointing)
    model = SpanClassifier(encoder, len(labels), hp.dropout, weights).to(device)
    dev_rows = dev.tc[["article_id", "start", "end", "technique"]].reset_index(drop=True)
    state = {}

    def evaluate() -> dict:
        probs = classify_spans(model, tokenizer, dev.articles, dev_rows, hp.max_length,
                               hp.context_chars, hp.eval_batch_size, device, amp)
        pred = assign_gold_span_labels(dev_rows.drop(columns="technique"), probs, labels)
        state["pred"] = pred
        m = score_tc(pred, dev_rows)
        return {"tc_micro_f1": m["micro_f1"], "tc_macro_f1": m["macro_f1"]}

    meta = {"backbone": cfg.backbone_id, "labels": labels, "max_length": hp.max_length,
            "context_chars": hp.context_chars, "use_none": hp.use_none, "seed": cfg.seed}

    def save(metrics: dict) -> None:
        save_bundle(model, tokenizer, out, {**meta, "dev_metrics": metrics})
        write_submission(state["pred"], out / "dev_tc_pred.txt", "tc")

    best = fit(model, dataset, TCCollator(encoder_in.pad), hp, device, amp, evaluate,
               "tc_micro_f1", save, out / "train_log.jsonl", cfg.seed, cfg.runtime.log_every)
    save_json({"best": best, "config": cfg.to_dict()}, out / "metrics.json")
    log.info("best dev gold-span TC micro-F1 %.4f -> %s", best.get("tc_micro_f1", float("nan")), out)
    return best
