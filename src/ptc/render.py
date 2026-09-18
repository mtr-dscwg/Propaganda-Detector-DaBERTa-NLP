"""Text-based UI: highlight propaganda spans in the terminal, with technique and confidence.

Highlights are suggestions, not verdicts: SI tops out near 0.5 F1 even for the best systems.
"""
from __future__ import annotations

import html

import pandas as pd

from .techniques import DISPLAY_NAMES, TECHNIQUES

_PALETTE = [196, 208, 214, 220, 190, 118, 49, 45, 39, 33, 99, 135, 171, 205]
COLOURS = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(TECHNIQUES)}
RESET = "\033[0m"


def _style(tech: str) -> str:
    return f"\033[1;38;5;{COLOURS.get(tech, 250)}m"


def render_terminal(text: str, preds: pd.DataFrame, min_confidence: float = 0.0) -> str:
    """Inline highlights with numbered tags, then a legend of every span."""
    preds = preds[preds["confidence"] * preds["technique_confidence"] >= min_confidence]
    preds = preds.sort_values(["start", "end"]).reset_index(drop=True)
    out, pos, legend = [], 0, []
    for n, r in enumerate(preds.itertuples(index=False), 1):
        if r.start < pos:  # overlapping span: list it in the legend only
            legend.append((n, r))
            continue
        out.append(text[pos:r.start])
        out.append(f"{_style(r.technique)}{text[r.start:r.end]}{RESET}\033[2m[{n}]{RESET}")
        pos = r.end
        legend.append((n, r))
    out.append(text[pos:])
    lines = ["".join(out), "", "-" * 72]
    for n, r in legend:
        score = r.confidence * r.technique_confidence
        lines.append(f"[{n:>2}] {_style(r.technique)}{DISPLAY_NAMES.get(r.technique, r.technique):<34}{RESET}"
                     f" span {r.confidence:.2f}  label {r.technique_confidence:.2f}  -> {score:.2f}")
    return "\n".join(lines)


def render_sentences(rows: list[dict]) -> str:
    lines = ["Critical sentences:"]
    for r in sorted(rows, key=lambda r: -r["confidence"]):
        techs = ", ".join(DISPLAY_NAMES.get(t, t) for t in r["techniques"])
        lines.append(f"  ({r['confidence']:.2f}) {r['sentence']}\n         -> {techs}")
    return "\n".join(lines)


def render_html(text: str, preds: pd.DataFrame, title: str = "") -> str:
    preds = preds.sort_values(["start", "end"])
    parts, pos = [], 0
    for r in preds.itertuples(index=False):
        if r.start < pos:
            continue
        parts.append(html.escape(text[pos:r.start]))
        tip = f"{DISPLAY_NAMES.get(r.technique, r.technique)} (span {r.confidence:.2f}, label {r.technique_confidence:.2f})"
        parts.append(f'<mark title="{html.escape(tip)}" style="background:hsl({COLOURS.get(r.technique, 0)},80%,85%)">'
                     f"{html.escape(text[r.start:r.end])}</mark>")
        pos = r.end
    parts.append(html.escape(text[pos:]))
    return (f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<body style='font:16px/1.6 system-ui;max-width:48rem;margin:2rem auto;white-space:pre-wrap'>"
            f"{''.join(parts)}</body>")
