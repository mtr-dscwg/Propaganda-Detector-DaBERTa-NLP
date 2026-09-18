"""Find where NaNs enter stage-1 training: weight loading, forward, backward, or the optimizer step.

    python scripts/diagnose_nan.py

Runs one real batch through each stage, fp32 and bf16, and tries the optimizer step with
PyTorch's default (foreach) AdamW and with the plain per-parameter implementation.
Paste the whole output back.
"""
import argparse
import copy

import torch

from ptc.config import add_config_args, config_from_args
from ptc.data.corpus import load_split
from ptc.data.si_dataset import SICollator, SIDataset
from ptc.data.tokenization import special_ids
from ptc.models.checkpoint import load_encoder, load_tokenizer
from ptc.models.span_tagger import SpanTagger
from ptc.training.loop import param_groups
from ptc.utils import autocast, describe_device, get_device, set_seed, setup_logging


def nonfinite(tensors) -> list[str]:
    return [name for name, t in tensors if t is not None and not torch.isfinite(t).all()]


def check_loaded(model) -> None:
    bad = nonfinite(model.named_parameters())
    print(f"[load] non-finite weight tensors: {len(bad)} {bad[:5]}")
    emb = model.encoder.get_input_embeddings().weight
    bad_rows = (~torch.isfinite(emb).all(dim=1)).nonzero().flatten().tolist()
    print(f"[load] embedding rows: {emb.shape[0]}, non-finite rows: {len(bad_rows)} "
          f"{bad_rows[:10]}{' ...' if len(bad_rows) > 10 else ''}")
    print(f"[load] max |weight| = {max(p.detach().abs().max().item() for p in model.parameters()):.3g}")


def one_step(model, batch, device, amp, lr, head_lr, foreach, label) -> None:
    model = copy.deepcopy(model).train()
    groups = param_groups(model, lr, head_lr, 0.01)
    opt = torch.optim.AdamW(groups, foreach=foreach)
    with autocast(device, amp):
        loss = model(**batch)["loss"]
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters()]
    gmax = max(g.abs().max().item() for _, g in grads if g is not None)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    bad_w = nonfinite(model.named_parameters())
    print(f"[{label}] loss={loss.item():.3f}  grad-norm={norm.item():.3g}  max|grad|={gmax:.3g}  "
          f"non-finite grads={len(nonfinite(grads))}  non-finite weights after step={len(bad_w)} {bad_w[:3]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    cfg = config_from_args(p.parse_args())
    setup_logging()
    set_seed(cfg.seed)
    device = get_device(cfg.runtime.device)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {describe_device(device)}")

    data = load_split(cfg.data.processed, "train")
    few = dict(list(data.articles.items())[:4])
    data.articles = few
    data.si = data.si[data.si["article_id"].isin(few)]
    tok = load_tokenizer(cfg.backbone_id)
    ds = SIDataset.from_split(data, tok, cfg.si.max_length, cfg.si.stride)
    batch = SICollator(special_ids(tok)[2])(ds.items[: cfg.si.batch_size])
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"batch shape {tuple(batch['input_ids'].shape)}, max token id {batch['input_ids'].max().item()}, "
          f"tokenizer size {len(tok)}")

    model = SpanTagger(load_encoder(cfg.backbone_id, tok), use_crf=cfg.si.use_crf).to(device)
    check_loaded(model)

    for amp_name, amp in (("fp32", None), ("bf16", torch.bfloat16 if device.type == "cuda" else None)):
        for foreach in (None, False):
            one_step(model, batch, device, amp, cfg.si.lr, cfg.si.head_lr, foreach,
                     f"{amp_name} {'foreach-default' if foreach is None else 'foreach-off'}")


if __name__ == "__main__":
    main()