"""Phase 0 gate: gold chars -> BIO -> chars over every article.

Fails (exit 1) if any token-aligned gold span is not reproduced exactly - that's a real
offset bug. Spans whose boundaries fall mid-token can't round-trip by construction; they
are counted and their cost is measured with the official SI scorer instead.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ptc.config import add_config_args, config_from_args
from ptc.data.bio import roundtrip
from ptc.data.corpus import load_split
from ptc.data.spans import spans_by_article
from ptc.data.tokenization import encode_article
from ptc.models.checkpoint import load_tokenizer
from ptc.scoring import score_si
from ptc.utils import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="train")
    p.add_argument("--report", default="runs/roundtrip_failures.jsonl")
    args = p.parse_args()
    cfg = config_from_args(args)
    setup_logging()

    data = load_split(cfg.data.processed, args.split)
    tok = load_tokenizer(cfg.backbone_id)
    gold = spans_by_article(data.si, merge=True)
    n_gold = exact = 0
    misaligned, bugs, decoded_rows = [], [], []
    for aid, text in data.articles.items():
        enc = encode_article(tok, aid, text)
        rt = roundtrip(aid, enc.offsets, gold.get(aid, []))
        n_gold += rt.n_gold
        exact += rt.exact
        misaligned += [(aid, s, e, text[s:e]) for s, e in rt.misaligned]
        bugs += [(aid, s, e, text[s:e]) for s, e in rt.bugs]
        decoded_rows += [{"article_id": aid, "start": s, "end": e} for s, e in rt.decoded]

    f1 = score_si(pd.DataFrame(decoded_rows, columns=["article_id", "start", "end"]), data.si)["f1"]
    print(f"backbone      {cfg.backbone_id}")
    print(f"articles      {len(data.articles)}")
    print(f"gold spans    {n_gold}")
    print(f"exact         {exact} ({exact / max(n_gold, 1):.2%})")
    print(f"mid-token     {len(misaligned)}  (tokenizer granularity, not a bug)")
    print(f"BUGS          {len(bugs)}  (token-aligned spans that failed to round-trip)")
    print(f"SI F1 of decoded gold vs gold: {f1:.4f}   <- the ceiling BIO imposes")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
        for kind, items in (("bug", bugs), ("mid_token", misaligned)):
            for aid, s, e, t in items:
                fh.write(json.dumps({"kind": kind, "article_id": aid, "start": s, "end": e, "text": t},
                                    ensure_ascii=False) + "\n")
    print(f"details -> {args.report}")
    sys.exit(1 if bugs else 0)


if __name__ == "__main__":
    main()
