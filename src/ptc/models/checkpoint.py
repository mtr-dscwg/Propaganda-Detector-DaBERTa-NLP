"""Build, save and reload models. Saved runs are self-contained (no re-download needed)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..data.tokenization import require_fast
from ..utils import load_json, save_json
from .span_classifier import SpanClassifier
from .span_tagger import SpanTagger


def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    require_fast(tok)
    return tok


def load_encoder(name_or_path: str, tokenizer, gradient_checkpointing: bool = False) -> nn.Module:
    encoder = AutoModel.from_pretrained(name_or_path)
    # DeBERTa-v3's embedding table (128,100) is larger than its tokenizer (~128,001), so the
    # marker tokens already have rows. Only grow, never shrink.
    if len(tokenizer) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tokenizer))
    if gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    return encoder


def save_bundle(model: nn.Module, tokenizer, out_dir: str | Path, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    model.encoder.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    save_json({**meta, "kind": model.kind}, out / "bundle.json")


@dataclass
class Bundle:
    model: nn.Module
    tokenizer: object
    meta: dict


def load_bundle(run_dir: str | Path, device: torch.device) -> Bundle:
    run = Path(run_dir)
    if not (run / "bundle.json").exists():
        raise FileNotFoundError(f"{run} is not a saved run (no bundle.json)")
    meta = load_json(run / "bundle.json")
    tokenizer = load_tokenizer(str(run))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(run))
    if len(tokenizer) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tokenizer))
    if meta["kind"] == "si":
        model = SpanTagger(encoder, use_crf=meta["use_crf"])
    else:
        model = SpanClassifier(encoder, num_labels=len(meta["labels"]))
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return Bundle(model.to(device).eval(), tokenizer, meta)
