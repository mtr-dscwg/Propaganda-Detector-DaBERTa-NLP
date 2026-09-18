"""Configuration: dataclasses loaded from YAML, with dotted CLI overrides.

The backbone is a single flag (plan D5): ``xsmall``/``base``/``large`` map to
DeBERTa-v3 checkpoints; any other value is treated as a HF id or local path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

BACKBONES = {
    "xsmall": "microsoft/deberta-v3-xsmall",
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
    "roberta-large": "roberta-large",
    "modernbert-large": "answerdotai/ModernBERT-large",
}


@dataclass
class DataConfig:
    root: str = "data/raw/datasets"
    processed: str = "data/processed"
    dev_holdout_articles: int = 0
    holdout_seed: int = 0


@dataclass
class SIConfig:
    max_skipped_steps: int = 20
    max_length: int = 256
    stride: int = 64
    use_crf: bool = True
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 8
    eval_batch_size: int = 16
    lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum: int = 2
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    patience: int = 3
    min_span_chars: int = 0
    confidence_threshold: float = 0.0


@dataclass
class TCConfig:
    max_skipped_steps: int = 20
    max_length: int = 256
    context_chars: int = 400
    use_none: bool = False
    none_path: str = "data/processed/none_train.parquet"
    class_weight_power: float = 0.5
    dropout: float = 0.1
    epochs: int = 6
    batch_size: int = 8
    eval_batch_size: int = 32
    lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum: int = 2
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    patience: int = 2


@dataclass
class RuntimeConfig:
    device: str = "auto"
    precision: str = "auto"
    out_dir: str = "runs"
    log_every: int = 50


@dataclass
class Config:
    backbone: str = "base"
    seed: int = 13
    data: DataConfig = field(default_factory=DataConfig)
    si: SIConfig = field(default_factory=SIConfig)
    tc: TCConfig = field(default_factory=TCConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def backbone_id(self) -> str:
        return BACKBONES.get(self.backbone, self.backbone)

    @property
    def backbone_tag(self) -> str:
        return self.backbone if self.backbone in BACKBONES else Path(self.backbone).name

    def run_name(self, stage: str) -> str:
        if stage == "tc" and self.tc.use_none:
            stage = "tc-none"
        return f"{stage}-{self.backbone_tag}-s{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(current: Any, value: Any) -> Any:
    # YAML 1.1 parses "3e-5" (no dot) as a string; coerce by the default's type.
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply(obj: Any, values: dict, prefix: str = "") -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, _coerce(current, value))


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    cfg = Config()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        _apply(cfg, raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        dotted, raw_value = item.split("=", 1)
        nested: dict = {}
        cursor = nested
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
        _apply(cfg, nested)
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", help="xsmall | base | large | HF id | local path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config value, e.g. --set si.lr=2e-5 (repeatable)")


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = list(args.set)
    if args.backbone:
        overrides.append(f"backbone={args.backbone}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    path = args.config if args.config and Path(args.config).exists() else None
    return load_config(path, overrides)
