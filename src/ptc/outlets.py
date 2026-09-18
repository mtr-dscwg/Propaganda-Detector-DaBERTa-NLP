"""D9 / Q1: recover per-article outlets for PTC by joining against proppy 1.0 (Zenodo 3271522).

1. Join on article ID (PTC IDs look like GDELT IDs, which proppy ships as article_ID).
2. Fallback: join on a normalised-text fingerprint of the opening paragraphs.

Proppy's TSVs are read header-less and the id / URL / text columns are detected from
their contents, so the join doesn't depend on getting the column order right.
Outlet = the URL's domain, unless a source-name column is given explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

log = logging.getLogger(__name__)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
FP_CHARS = 120


def _fingerprint(text: str) -> str | None:
    norm = _NON_ALNUM.sub("", text.lower())
    return hashlib.sha1(norm[:FP_CHARS].encode()).hexdigest() if len(norm) >= 40 else None


def _paragraph_prints(text: str, k: int = 4) -> list[str]:
    paras = [p for p in text.split("\n") if p.strip()]
    prints = [_fingerprint(p) for p in paras[:k]] + [_fingerprint(text)]
    return [p for p in prints if p]


def load_proppy(paths: list[str | Path], source_col: int | None = None) -> pd.DataFrame:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    frames = [pd.read_csv(p, sep="\t", header=None, dtype=str, quoting=csv.QUOTE_NONE,
                          on_bad_lines="warn", engine="python") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    sample = df.head(2000)
    id_col = max(df.columns, key=lambda c: sample[c].str.fullmatch(r"\d{6,12}", na=False).mean())
    url_col = max(df.columns, key=lambda c: sample[c].str.startswith("http", na=False).mean())
    text_col = max(df.columns, key=lambda c: sample[c].str.len().fillna(0).mean())
    log.info("proppy columns detected: id=%s url=%s text=%s", id_col, url_col, text_col)
    out = pd.DataFrame({
        "proppy_id": df[id_col].str.strip(),
        "url": df[url_col],
        "text": df[text_col].fillna("").str.replace(r"\\n", "\n", regex=True),
    })
    if source_col is not None:
        out["outlet"] = df[source_col]
    else:
        out["outlet"] = out["url"].map(lambda u: urlparse(str(u)).netloc.lower().removeprefix("www.") or None)
    return out


def join_outlets(articles: dict[str, str], proppy: pd.DataFrame) -> pd.DataFrame:
    by_id = proppy.drop_duplicates("proppy_id").set_index("proppy_id")
    by_print: dict[str, int] = {}
    for i, text in enumerate(proppy["text"]):
        for fp in _paragraph_prints(text):
            by_print.setdefault(fp, i)
    rows = []
    for aid, text in articles.items():
        row = {"article_id": aid, "outlet": None, "url": None, "match": None}
        if aid in by_id.index:
            hit = by_id.loc[aid]
            row.update(outlet=hit["outlet"], url=hit["url"], match="id")
        else:
            for fp in _paragraph_prints(text):
                if fp in by_print:
                    hit = proppy.iloc[by_print[fp]]
                    row.update(outlet=hit["outlet"], url=hit["url"], match="text")
                    break
        rows.append(row)
    return pd.DataFrame(rows)
