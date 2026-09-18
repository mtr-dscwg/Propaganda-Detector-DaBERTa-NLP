"""D8: always report SI, gold-span TC, and FLC end-to-end, plus the false-positive report
and (D9) a per-outlet split when an outlet map is supplied."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data.corpus import load_split
from .inference import Pipeline, assign_gold_span_labels, assign_predicted_span_labels
from .scoring import false_positive_report, score_flc, score_si, score_tc, write_submission
from .utils import save_json

log = logging.getLogger(__name__)


def evaluate(processed: str, si_run: str, tc_run: str, split: str, out_dir: str | Path,
             outlets_csv: str | None = None, min_outlet_articles: int = 3,
             device: str = "auto", precision: str = "auto") -> dict:
    out = Path(out_dir)
    data = load_split(processed, split)
    pipe = Pipeline(si_run, tc_run, device, precision)
    labels = pipe.tc.meta["labels"]

    spans = pipe.spans(data.articles)
    flc = assign_predicted_span_labels(spans, pipe.classify(data.articles, spans), labels)
    write_submission(spans, out / f"{split}_si.txt", "si")
    write_submission(flc, out / f"{split}_flc.txt", "flc")
    flc.to_json(out / f"{split}_flc.jsonl", orient="records", lines=True, force_ascii=False)

    results: dict = {"split": split, "si_run": str(si_run), "tc_run": str(tc_run)}
    if not data.has_gold:
        log.info("%s has no gold labels: wrote submission files only", split)
        save_json(results, out / f"{split}_metrics.json")
        return results

    gold_rows = data.tc[["article_id", "start", "end", "technique"]].reset_index(drop=True)
    tc_pred = assign_gold_span_labels(gold_rows.drop(columns="technique"),
                                      pipe.classify(data.articles, gold_rows), labels)
    write_submission(tc_pred, out / f"{split}_tc.txt", "tc")

    results["si"] = score_si(spans, data.si)
    results["tc_gold_spans"] = score_tc(tc_pred, gold_rows)
    results["flc_end_to_end"] = score_flc(flc, gold_rows)
    results["false_positives"] = false_positive_report(flc, gold_rows)
    results["headline"] = {
        "SI F1": results["si"]["f1"],
        "TC micro-F1 (gold spans)": results["tc_gold_spans"]["micro_f1"],
        "TC macro-F1 (gold spans)": results["tc_gold_spans"]["macro_f1"],
        "FLC F1 (end-to-end)": results["flc_end_to_end"]["f1"],
        "FP rate (end-to-end)": results["false_positives"]["fp_rate"],
    }
    if outlets_csv:
        results["by_outlet"] = by_outlet(outlets_csv, spans, flc, data, min_outlet_articles)
    save_json(results, out / f"{split}_metrics.json")
    return results


def by_outlet(outlets_csv: str, spans: pd.DataFrame, flc: pd.DataFrame, data, min_articles: int) -> dict:
    outlets = pd.read_csv(outlets_csv, dtype={"article_id": str})
    outlets = outlets[outlets["article_id"].isin(data.articles) & outlets["outlet"].notna()]
    res = {"coverage": len(outlets) / max(len(data.articles), 1), "outlets": {}}
    for outlet, grp in outlets.groupby("outlet"):
        ids = set(grp["article_id"])
        if len(ids) < min_articles:
            continue
        sel = lambda df: df[df["article_id"].isin(ids)]
        res["outlets"][outlet] = {
            "articles": len(ids),
            "si_f1": score_si(sel(spans), sel(data.si))["f1"],
            "flc_f1": score_flc(sel(flc), sel(data.tc))["f1"],
            "gold_spans_per_article": len(sel(data.si)) / len(ids),
            "pred_spans_per_article": len(sel(spans)) / len(ids),
        }
    return res


def format_report(results: dict) -> str:
    if "headline" not in results:
        return f"{results['split']}: no gold labels; predictions written."
    lines = [f"== {results['split']} ==", *(f"  {k:<28} {v:.4f}" for k, v in results["headline"].items())]
    lines.append("  per technique (gold-span TC F1 | FLC F1 | FP rate):")
    tc = results["tc_gold_spans"]["per_technique"]
    flc = results["flc_end_to_end"]["per_technique"]
    fp = results["false_positives"]["by_technique"]
    for t in sorted(tc, key=lambda k: -tc[k]["support"]):
        lines.append(f"    {t:<36} n={tc[t]['support']:<5} {tc[t]['f1']:.3f} | "
                     f"{flc.get(t, {}).get('f1', 0):.3f} | {fp.get(t, {}).get('fp_rate', 0):.2f}")
    if "by_outlet" in results:
        bo = results["by_outlet"]
        lines.append(f"  by outlet (coverage {bo['coverage']:.0%}):")
        for o, m in sorted(bo["outlets"].items(), key=lambda kv: -kv[1]["articles"]):
            lines.append(f"    {o:<32} n={m['articles']:<4} SI {m['si_f1']:.3f}  FLC {m['flc_f1']:.3f}")
    return "\n".join(lines)
