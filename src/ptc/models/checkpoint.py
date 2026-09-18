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


def _as_fp32(encoder: nn.Module) -> nn.Module:
    """Master weights must be float32. transformers 5 keeps a checkpoint's stored dtype, and some
    DeBERTa-v3 safetensors are float16: AdamW's eps (1e-8) underflows to 0 in fp16, so the very
    first update divides by zero and turns every weight into NaN. Mixed precision (bf16 autocast)
    still applies on top of fp32 weights."""
    return encoder.float()


def load_encoder(name_or_path: str, tokenizer, gradient_checkpointing: bool = False) -> nn.Module:
    encoder = _as_fp32(AutoModel.from_pretrained(name_or_path))
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
    encoder = _as_fp32(AutoModel.from_config(AutoConfig.from_pretrained(run)))
    if len(tokenizer) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tokenizer))
    if meta["kind"] == "si":
        model = SpanTagger(encoder, use_crf=meta["use_crf"])
    else:
        model = SpanClassifier(encoder, num_labels=len(meta["labels"]))
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return Bundle(model.to(device).eval(), tokenizer, meta)