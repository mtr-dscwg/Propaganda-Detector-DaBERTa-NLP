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


# HTML hues (degrees on the colour wheel), grouped by technique family.
# Separate from COLOURS above, which holds xterm-256 codes for the terminal.
HTML_HUES = {
    # reds/oranges: attacks on people
    "Name_Calling,Labeling": 0,
    "Loaded_Language": 20,
    "Appeal_to_fear-prejudice": 38,
    # yellows/greens: distortion of facts
    "Exaggeration,Minimisation": 55,
    "Causal_Oversimplification": 80,
    "Black-and-White_Fallacy": 105,
    "Doubt": 130,
    # cyans/blues: appeals to authority and the crowd
    "Appeal_to_Authority": 175,
    "Bandwagon,Reductio_ad_hitlerum": 200,
    "Flag-Waving": 225,
    # purples/pinks: slogans and thought-stoppers
    "Slogans": 260,
    "Thought-terminating_Cliches": 285,
    "Repetition": 310,
    "Whataboutism,Straw_Men,Red_Herring": 335,
    "NONE": 0,  # unused in HTML output; NONE spans are dropped before rendering
}


def html_colour(technique: str, lightness: int = 85) -> str:
    """Background colour for a technique in HTML output."""
    hue = HTML_HUES.get(technique, 0)
    saturation = 0 if technique not in HTML_HUES else 75
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def render_html(text: str, preds: pd.DataFrame, title: str = "") -> str:
    preds = preds.sort_values(["start", "end"])

    legend = "".join(
        f'<span style="background:{html_colour(t)};padding:2px 8px;margin:0 6px 6px 0;'
        f'border-radius:3px;display:inline-block;font-size:0.85em">'
        f"{html.escape(DISPLAY_NAMES.get(t, t))}</span>"
        for t in sorted(set(preds["technique"]))
    )
    
    parts, pos = [], 0
    for r in preds.itertuples(index=False):
        if r.start < pos:
            continue
        parts.append(html.escape(text[pos:r.start]))
        tip = f"{DISPLAY_NAMES.get(r.technique, r.technique)} (span {r.confidence:.2f}, label {r.technique_confidence:.2f})"
        parts.append(f'<mark title="{html.escape(tip)}" style="background:{html_colour(r.technique)}">'
                     f"{html.escape(text[r.start:r.end])}</mark>")
        pos = r.end
    parts.append(html.escape(text[pos:]))

    return (f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<body style='font:16px/1.6 system-ui;max-width:48rem;margin:2rem auto'>"
            f"<div style='white-space:normal;margin-bottom:1.5rem'>"
            f"{legend}"                       
            f"</div>"
            f"<div style='white-space:pre-wrap'>{''.join(parts)}</div>"
            f"</body>")
