"""Inference for both stages and the end-to-end pipeline (article in -> spans + labels out)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data.bio import O, centrality, decode_bio
from .data.si_dataset import SICollator, article_windows
from .data.tc_dataset import SpanEncoder, TCCollator, TCDataset
from .data.tokenization import encode_article, special_ids
from .models.checkpoint import Bundle, load_bundle
from .techniques import NONE_LABEL
from .utils import autocast, get_device, resolve_precision


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ------------------------------------------------------------------------ stage 1
@torch.no_grad()
def predict_spans(model: nn.Module, tokenizer, articles: dict[str, str], max_length: int, stride: int,
                  batch_size: int, device: torch.device, amp_dtype: torch.dtype | None = None,
                  min_span_chars: int = 0, confidence_threshold: float = 0.0) -> pd.DataFrame:
    """Returns article_id, start, end, confidence (mean token P(inside) over the span)."""
    model.eval()
    cls, sep, pad = special_ids(tokenizer)
    collate = SICollator(pad)
    encodings, windows = {}, []
    for aid, text in articles.items():
        enc = encode_article(tokenizer, aid, text)
        encodings[aid] = enc
        windows.extend(article_windows(enc, max_length, stride, cls, sep))

    # per token: (label, prob, centrality of the window that produced it)
    best: dict[str, list] = {aid: [None] * len(e.ids) for aid, e in encodings.items()}
    for batch in _batches(windows, batch_size):
        inputs = {k: v.to(device) for k, v in collate(batch).items()}
        with autocast(device, amp_dtype):
            paths, probs = model.predict(**inputs)
        probs = probs.float().cpu().numpy()
        for item, path, prob in zip(batch, paths, probs):
            ws, we = item["window"]
            slots = best[item["article_id"]]
            for j, g in enumerate(range(ws, we)):
                c = centrality(g, (ws, we))
                if slots[g] is None or c > slots[g][2]:
                    slots[g] = (path[j + 1], float(prob[j + 1]), c)  # +1 skips CLS

    rows = []
    for aid, enc in encodings.items():
        labels = [s[0] if s else O for s in best[aid]]
        token_p = np.array([s[1] if s else 0.0 for s in best[aid]])
        starts = np.array([s for s, _ in enc.offsets]) if enc.offsets else np.zeros(0)
        ends = np.array([e for _, e in enc.offsets]) if enc.offsets else np.zeros(0)
        for s, e in decode_bio(enc.offsets, labels):
            inside = (starts < e) & (ends > s) & (ends > starts)
            conf = float(token_p[inside].mean()) if inside.any() else 0.0
            if e - s >= min_span_chars and conf >= confidence_threshold:
                rows.append({"article_id": aid, "start": s, "end": e, "confidence": conf})
    return pd.DataFrame(rows, columns=["article_id", "start", "end", "confidence"])


# ------------------------------------------------------------------------ stage 2
@torch.no_grad()
def classify_spans(model: nn.Module, tokenizer, articles: dict[str, str], spans: pd.DataFrame,
                   max_length: int, context_chars: int, batch_size: int, device: torch.device,
                   amp_dtype: torch.dtype | None = None) -> np.ndarray:
    """Softmax probabilities, one row per span row."""
    model.eval()
    num_labels = model.head[-1].out_features
    if spans.empty:
        return np.zeros((0, num_labels), dtype=np.float32)
    encoder = SpanEncoder(tokenizer, max_length, context_chars)
    ds = TCDataset.build(articles, spans, encoder)
    collate = TCCollator(encoder.pad)
    out = []
    for batch in _batches(ds.items, batch_size):
        inputs = {k: v.to(device) for k, v in collate(batch).items()}
        with autocast(device, amp_dtype):
            logits = model(**inputs)["logits"]
        out.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def assign_gold_span_labels(spans: pd.DataFrame, probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Gold-span TC: a span listed k times (k gold techniques) gets its top-k distinct labels.
    NONE is never predicted here, since every gold span is propaganda by definition."""
    probs = probs.copy()
    if NONE_LABEL in labels:
        probs[:, labels.index(NONE_LABEL)] = -1.0
    out = spans.copy().reset_index(drop=True)
    preds = [None] * len(out)
    keys = list(zip(out["article_id"], out["start"], out["end"]))
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    for idxs in groups.values():
        ranked = np.argsort(-probs[idxs[0]])
        for rank, i in enumerate(idxs):
            preds[i] = labels[ranked[rank % len(ranked)]]
    out["technique"] = preds
    out["technique_confidence"] = [float(probs[i, labels.index(t)]) for i, t in enumerate(preds)]
    return out


def assign_predicted_span_labels(spans: pd.DataFrame, probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """End-to-end: argmax per span; spans classified NONE are dropped (D4)."""
    out = spans.copy().reset_index(drop=True)
    if out.empty:
        out["technique"], out["technique_confidence"] = [], []
        return out
    idx = probs.argmax(1)
    out["technique"] = [labels[i] for i in idx]
    out["technique_confidence"] = probs[np.arange(len(idx)), idx].astype(float)
    return out[out["technique"] != NONE_LABEL].reset_index(drop=True)


# ------------------------------------------------------------------------ pipeline
class Pipeline:
    def __init__(self, si_run: str | Path, tc_run: str | Path, device: str = "auto", precision: str = "auto"):
        self.device = get_device(device)
        self.amp = resolve_precision(precision, self.device)
        self.si: Bundle = load_bundle(si_run, self.device)
        self.tc: Bundle = load_bundle(tc_run, self.device)

    def spans(self, articles: dict[str, str], batch_size: int = 16) -> pd.DataFrame:
        m = self.si.meta
        return predict_spans(self.si.model, self.si.tokenizer, articles, m["max_length"], m["stride"],
                             batch_size, self.device, self.amp, m.get("min_span_chars", 0),
                             m.get("confidence_threshold", 0.0))

    def classify(self, articles: dict[str, str], spans: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        m = self.tc.meta
        return classify_spans(self.tc.model, self.tc.tokenizer, articles, spans, m["max_length"],
                              m["context_chars"], batch_size, self.device, self.amp)

    def __call__(self, articles: dict[str, str]) -> pd.DataFrame:
        spans = self.spans(articles)
        probs = self.classify(articles, spans)
        return assign_predicted_span_labels(spans, probs, self.tc.meta["labels"])


# ------------------------------------------------------------------------ sentence view
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Lightweight sentence splitter (presentation only; scoring stays span-level)."""
    out, start = [], 0
    for m in _SENT_END.finditer(text):
        if text[start:m.start()].strip():
            out.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text)))
    return out


def sentence_view(text: str, preds: pd.DataFrame) -> list[dict]:
    """Roll predicted spans up to the 'critical sentences' of the original brief."""
    rows = []
    for s, e in sentence_spans(text):
        hits = preds[(preds["start"] < e) & (preds["end"] > s)]
        if hits.empty:
            continue
        rows.append({
            "start": s, "end": e, "sentence": text[s:e].strip(),
            "techniques": sorted(set(hits["technique"])),
            "confidence": float((hits["confidence"] * hits["technique_confidence"]).max()),
        })
    return rows
