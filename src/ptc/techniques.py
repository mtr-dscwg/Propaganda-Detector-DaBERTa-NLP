"""The 14 PTC-SemEval20 techniques, spelled exactly as in the official label files."""
from __future__ import annotations

TECHNIQUES: tuple[str, ...] = (
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism,Straw_Men,Red_Herring",
)

NONE_LABEL = "NONE"  # D4: stage-1 false positives

DISPLAY_NAMES = {t: t.replace("_", " ").replace(",", " / ") for t in TECHNIQUES}
DISPLAY_NAMES[NONE_LABEL] = "None"


def label_list(use_none: bool) -> list[str]:
    labels = list(TECHNIQUES)
    if use_none:
        labels.append(NONE_LABEL)
    return labels


def validate_technique(name: str) -> str:
    if name not in TECHNIQUES and name != NONE_LABEL:
        raise ValueError(
            f"Unknown technique {name!r}. Expected one of the 14 PTC names "
            f"(check the label files weren't edited or re-encoded)."
        )
    return name
