"""Seeding, device/precision selection, JSON I/O, console setup (Windows-friendly)."""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_console() -> None:
    """UTF-8 stdout and ANSI colours on Windows terminals (incl. VS Code)."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if os.name == "nt":
        os.system("")  # enables VT100 escape processing in cmd/PowerShell


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(preference: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype, or None for plain fp32."""
    if device.type != "cuda":
        return None  # bf16 autocast on MPS/CPU is not worth the trouble here
    if preference == "fp32":
        return None
    if preference == "fp16":
        return torch.float16
    if preference in ("bf16", "auto"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if preference == "auto" else None
    raise ValueError(f"Unknown precision {preference!r}")


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda:{props.name} ({props.total_memory / 2**30:.1f} GB, sm_{props.major}{props.minor})"
    return device.type


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def setup_logging(level: int = 20) -> None:
    import logging
    setup_console()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("transformers", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
