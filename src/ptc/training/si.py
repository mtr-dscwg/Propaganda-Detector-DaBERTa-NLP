"""Train stage 1 (span identification). Model selection on official SI F1 on dev."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import Config
from ..data.corpus import load_split
from ..data.si_dataset import SICollator, SIDataset
from ..data.tokenization import special_ids
from ..inference import predict_spans
from ..models.checkpoint import load_encoder, load_tokenizer, save_bundle
from ..models.span_tagger import SpanTagger
from ..scoring import score_si, write_submission
from ..utils import describe_device, get_device, resolve_precision, save_json, set_seed
from .loop import fit

log = logging.getLogger(__name__)


def train_si(cfg: Config) -> dict:
    set_seed(cfg.seed)
    hp = cfg.si
    device = get_device(cfg.runtime.device)
    amp = resolve_precision(cfg.runtime.precision, device)
    out = Path(cfg.runtime.out_dir) / cfg.run_name("si")
    log.info("SI run %s on %s (amp=%s) backbone=%s", out.name, describe_device(device), amp, cfg.backbone_id)

    train = load_split(cfg.data.processed, "train")
    dev = load_split(cfg.data.processed, "dev")
    if dev.si.empty:
        raise RuntimeError("dev split has no SI gold. Set data.dev_holdout_articles and re-run prepare_data.")

    tokenizer = load_tokenizer(cfg.backbone_id)
    dataset = SIDataset.from_split(train, tokenizer, hp.max_length, hp.stride)
    log.info("train: %d articles -> %d windows", len(train.articles), len(dataset))

    encoder = load_encoder(cfg.backbone_id, tokenizer, hp.gradient_checkpointing)
    model = SpanTagger(encoder, use_crf=hp.use_crf, dropout=hp.dropout).to(device)

    state = {}

    def evaluate() -> dict:
        pred = predict_spans(model, tokenizer, dev.articles, hp.max_length, hp.stride,
                             hp.eval_batch_size, device, amp, hp.min_span_chars, hp.confidence_threshold)
        state["pred"] = pred
        m = score_si(pred, dev.si)
        return {"si_f1": m["f1"], "si_precision": m["precision"], "si_recall": m["recall"],
                "n_pred": m["n_pred"]}

    meta = {"backbone": cfg.backbone_id, "use_crf": hp.use_crf, "max_length": hp.max_length,
            "stride": hp.stride, "min_span_chars": hp.min_span_chars,
            "confidence_threshold": hp.confidence_threshold, "seed": cfg.seed}

    def save(metrics: dict) -> None:
        save_bundle(model, tokenizer, out, {**meta, "dev_metrics": metrics})
        write_submission(state["pred"], out / "dev_si_pred.txt", "si")

    best = fit(model, dataset, SICollator(special_ids(tokenizer)[2]), hp, device, amp, evaluate,
               "si_f1", save, out / "train_log.jsonl", cfg.seed, cfg.runtime.log_every)
    save_json({"best": best, "config": cfg.to_dict()}, out / "metrics.json")
    log.info("best dev SI F1 %.4f -> %s", best.get("si_f1", float("nan")), out)
    return best
