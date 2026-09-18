"""Scorers matching the PTC-SemEval20 definitions (Da San Martino et al., 2019/2020).

SI and FLC use character-level partial matching:

    P = (1/|S|) * sum_{s in S, t in T} |s ∩ t| / |s| * δ(l(s), l(t))
    R = (1/|T|) * sum_{s in S, t in T} |s ∩ t| / |t| * δ(l(s), l(t))

with δ ≡ 1 for SI (labels ignored). For SI, overlapping spans are merged on both sides
first. TC is scored given gold spans, so micro-F1 is span accuracy with multiset matching
when one span carries several gold techniques.

Verify against the organisers' scorer by reproducing the published baseline (0.31 SI)
with scripts/score.py before trusting any number from this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .data.spans import merge_spans

Row = tuple[int, int, str | None]


def _group(df: pd.DataFrame, labeled: bool) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    techs = df["technique"] if labeled else [None] * len(df)
    for aid, s, e, t in zip(df["article_id"], df["start"], df["end"], techs):
        if int(e) > int(s):
            out[str(aid)].append((int(s), int(e), t))
    return out


def _prf(p_sum: float, r_sum: float, n_pred: int, n_gold: int) -> dict:
    p = p_sum / n_pred if n_pred else 0.0
    r = r_sum / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "n_pred": n_pred, "n_gold": n_gold}


def span_prf(pred: pd.DataFrame, gold: pd.DataFrame, labeled: bool, merge: bool) -> dict:
    P, G = _group(pred, labeled), _group(gold, labeled)
    if merge:
        P = {k: [(s, e, None) for s, e in merge_spans((s, e) for s, e, _ in v)] for k, v in P.items()}
        G = {k: [(s, e, None) for s, e in merge_spans((s, e) for s, e, _ in v)] for k, v in G.items()}
    p_sum = r_sum = 0.0
    for aid in set(P) & set(G):
        for ps, pe, pl in P[aid]:
            for gs, ge, gl in G[aid]:
                if labeled and pl != gl:
                    continue
                inter = min(pe, ge) - max(ps, gs)
                if inter > 0:
                    p_sum += inter / (pe - ps)
                    r_sum += inter / (ge - gs)
    return _prf(p_sum, r_sum, sum(map(len, P.values())), sum(map(len, G.values())))


def score_si(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    return span_prf(pred, gold, labeled=False, merge=True)


def score_flc(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """End-to-end (predicted spans + predicted techniques), with a per-technique breakdown."""
    overall = span_prf(pred, gold, labeled=True, merge=False)
    per = {}
    for tech in sorted(set(gold["technique"]) | set(pred["technique"])):
        per[tech] = span_prf(pred[pred["technique"] == tech], gold[gold["technique"] == tech],
                             labeled=True, merge=False)
    overall["per_technique"] = per
    return overall


def score_tc(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Gold-span TC. `pred` has one row per gold row (same offsets), technique predicted."""
    def bag(df):
        out: dict[tuple, Counter] = defaultdict(Counter)
        for aid, s, e, t in zip(df["article_id"], df["start"], df["end"], df["technique"]):
            out[(str(aid), int(s), int(e))][t] += 1
        return out

    G, P = bag(gold), bag(pred)
    tp, n_pred, n_gold = Counter(), Counter(), Counter()
    for key in set(G) | set(P):
        g, p = G.get(key, Counter()), P.get(key, Counter())
        tp.update(g & p)
        n_pred.update(p)
        n_gold.update(g)
    total_tp, total_pred, total_gold = sum(tp.values()), sum(n_pred.values()), sum(n_gold.values())
    prec = total_tp / total_pred if total_pred else 0.0
    rec = total_tp / total_gold if total_gold else 0.0
    per = {}
    for tech in sorted(n_gold):
        p = tp[tech] / n_pred[tech] if n_pred[tech] else 0.0
        r = tp[tech] / n_gold[tech]
        per[tech] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                     "support": n_gold[tech]}
    return {
        "micro_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "macro_f1": sum(v["f1"] for v in per.values()) / len(per) if per else 0.0,
        "per_technique": per,
        "n": total_gold,
    }


def false_positive_report(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Predicted spans overlapping no gold span at all, overall and by predicted technique."""
    G = _group(gold, labeled=False)
    fp = Counter()
    total = Counter()
    labeled = "technique" in pred.columns
    for row in pred.itertuples(index=False):
        tech = row.technique if labeled else "span"
        total[tech] += 1
        if not any(row.start < ge and gs < row.end for gs, ge, _ in G.get(str(row.article_id), [])):
            fp[tech] += 1
    return {
        "fp_spans": sum(fp.values()),
        "pred_spans": sum(total.values()),
        "fp_rate": sum(fp.values()) / sum(total.values()) if total else 0.0,
        "by_technique": {t: {"fp": fp[t], "pred": total[t], "fp_rate": fp[t] / total[t]}
                         for t in sorted(total)},
    }


# ---------------------------------------------------------------- submission files
def write_submission(df: pd.DataFrame, path: str | Path, task: str) -> None:
    """Official formats: SI `id\\tstart\\tend`; TC/FLC `id\\ttechnique\\tstart\\tend`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in df.itertuples(index=False):
            if task == "si":
                fh.write(f"{r.article_id}\t{int(r.start)}\t{int(r.end)}\n")
            else:
                fh.write(f"{r.article_id}\t{r.technique}\t{int(r.start)}\t{int(r.end)}\n")


def read_submission(path: str | Path) -> pd.DataFrame:
    from .data.corpus import parse_label_file
    return parse_label_file(path)
