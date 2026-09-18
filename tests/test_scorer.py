import pandas as pd
import pytest

from ptc.scoring import false_positive_report, score_flc, score_si, score_tc


def df(rows, labeled=False):
    cols = ["article_id", "start", "end"] + (["technique"] if labeled else [])
    return pd.DataFrame(rows, columns=cols)


def test_si_perfect():
    g = df([("1", 0, 10), ("1", 20, 30)])
    assert score_si(g, g)["f1"] == pytest.approx(1.0)


def test_si_partial_overlap_formula():
    # pred covers half of gold; P = 5/5 = 1, R = 5/10 = 0.5
    m = score_si(df([("1", 0, 5)]), df([("1", 0, 10)]))
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)


def test_si_merges_overlaps():
    pred = df([("1", 0, 6), ("1", 4, 10)])
    assert score_si(pred, df([("1", 0, 10)]))["f1"] == pytest.approx(1.0)


def test_si_other_article_no_credit():
    assert score_si(df([("2", 0, 10)]), df([("1", 0, 10)]))["f1"] == 0.0


def test_flc_requires_label_match():
    g = df([("1", 0, 10, "Doubt")], labeled=True)
    assert score_flc(df([("1", 0, 10, "Doubt")], True), g)["f1"] == pytest.approx(1.0)
    assert score_flc(df([("1", 0, 10, "Slogans")], True), g)["f1"] == 0.0


def test_tc_multiset_for_duplicate_spans():
    gold = df([("1", 0, 10, "Doubt"), ("1", 0, 10, "Slogans"), ("1", 20, 30, "Repetition")], True)
    pred = df([("1", 0, 10, "Slogans"), ("1", 0, 10, "Doubt"), ("1", 20, 30, "Doubt")], True)
    m = score_tc(pred, gold)
    assert m["micro_f1"] == pytest.approx(2 / 3)


def test_fp_report():
    gold = df([("1", 0, 10, "Doubt")], True)
    pred = df([("1", 5, 8, "Doubt"), ("1", 50, 60, "Slogans")], True)
    r = false_positive_report(pred, gold)
    assert r["fp_spans"] == 1 and r["by_technique"]["Slogans"]["fp"] == 1
