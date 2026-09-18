"""Generic fine-tuning loop: AdamW with split encoder/head LRs, linear warmup,
bf16/fp16 autocast on CUDA, grad accumulation, early stopping on a dev metric."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..utils import autocast

log = logging.getLogger(__name__)

NO_DECAY = ("bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight", "norm.weight")


def param_groups(model: nn.Module, lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    groups = {(e, d): [] for e in (True, False) for d in (True, False)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        decay = not any(k in name for k in NO_DECAY)
        groups[(is_encoder, decay)].append(p)
    out = []
    for (is_encoder, decay), params in groups.items():
        if params:
            out.append({"params": params, "lr": lr if is_encoder else head_lr,
                        "weight_decay": weight_decay if decay else 0.0})
    return out


def fit(model: nn.Module, dataset, collate: Callable, hp, device: torch.device,
        amp_dtype: torch.dtype | None, evaluate: Callable[[], dict], metric: str,
        save: Callable[[dict], None], log_path: Path, seed: int, log_every: int = 50) -> dict:
    """Train; after each epoch call `evaluate()`, keep the best by `metric` via `save()`."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate,
                        generator=generator, num_workers=0)  # 0 workers: safest on Windows
    optimizer = torch.optim.AdamW(param_groups(model, hp.lr, hp.head_lr, hp.weight_decay))
    steps_per_epoch = math.ceil(len(loader) / hp.grad_accum)
    total = steps_per_epoch * hp.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(hp.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    best, best_metrics, bad_epochs = -math.inf, {}, 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="\n") as logf:
        for epoch in range(1, hp.epochs + 1):
            model.train()
            t0, running, n = time.time(), 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader, 1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with autocast(device, amp_dtype):
                    loss = model(**batch)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch} step {step}")
                scaler.scale(loss / hp.grad_accum).backward()
                running += loss.item()
                n += 1
                if step % hp.grad_accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % log_every == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(loader), running / n)

            metrics = evaluate()
            metrics.update(epoch=epoch, train_loss=running / max(n, 1), epoch_seconds=time.time() - t0)
            logf.write(json.dumps(metrics) + "\n")
            logf.flush()
            improved = metrics[metric] > best
            log.info("epoch %d  %s=%.4f%s  (%.0fs)", epoch, metric, metrics[metric],
                     "  *best*" if improved else "", metrics["epoch_seconds"])
            if improved:
                best, best_metrics, bad_epochs = metrics[metric], dict(metrics), 0
                save(best_metrics)
            else:
                bad_epochs += 1
                if bad_epochs >= hp.patience:
                    log.info("early stop: no %s improvement for %d epochs", metric, hp.patience)
                    break
    return best_metrics
