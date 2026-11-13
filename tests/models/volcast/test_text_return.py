import numpy as np
import pandas as pd
import pytest

from ai_core.models.volcast.text_return import article_labels, assign_sessions, daily_feature, train_articles

DATES = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-09"])     # Mon, Tue, Fri (Wed-Thu holiday)


def test_session_respects_close_and_holidays():
    pub = pd.Series(pd.to_datetime(["2026-01-05 07:00", "2026-01-05 07:45", "2026-01-06 09:00", "2026-01-03 10:00"]))
    got = assign_sessions(pub, DATES)
    assert list(got) == list(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-09", "2026-01-05"]))
    # 07:00 + 30 min <= 08:00 close -> same day; 07:45 + 30 min > close -> next session; after Tue close -> Fri; Saturday -> Monday


def test_session_after_last_close_is_nat():
    assert assign_sessions(pd.Series(pd.to_datetime(["2026-01-09 08:00"])), DATES).isna().all()


def _prices():
    days = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    return pd.DataFrame({"date": list(days) * 2, "symbol": ["AAA"] * 4 + ["BBB"] * 4,
                         "close": [10, 11, 12, 13, 20, 19, 22, 21.0]})


def _arts():
    return pd.DataFrame({"symbol": ["AAA", "BBB", "AAA"],
                         "published_at": pd.to_datetime(["2026-01-05 07:00", "2026-01-05 07:00", "2026-01-08 07:00"]),
                         "passage_text": ["a", "b", "c"]})


def test_article_labels_excess_and_absmove():
    ra, rb = 12 / 10 - 1, 22 / 20 - 1          # h=2 forward returns from 2026-01-05
    ex = article_labels(_arts(), _prices(), 2, "excess")
    assert len(ex) == 2                          # the 2026-01-08 article has no t+2 price -> dropped
    assert (ex.session == pd.Timestamp("2026-01-05")).all()
    assert ex.y.tolist() == pytest.approx([ra - (ra + rb) / 2, rb - (ra + rb) / 2])
    ab = article_labels(_arts(), _prices(), 2, "absmove")
    med = np.median([abs(ra), abs(rb)])
    assert ab.y.tolist() == pytest.approx([abs(ra) - med, abs(rb) - med])


def test_train_articles_purges_labels_overlapping_validation():
    dates = pd.bdate_range("2026-01-05", periods=10)
    lab = pd.DataFrame({"session": dates, "y": 0.0})
    tr = train_articles(lab, dates, "2026-01-12", 2)   # validation starts at position 5
    assert tr.session.tolist() == list(dates[:3])      # positions 3, 4 have i + 2 >= 5
    assert train_articles(lab, dates, "2026-01-12", 2, train_start="2026-01-06").session.tolist() == list(dates[1:3])


def test_daily_feature_window_and_empty_cells():
    arts = pd.DataFrame({"symbol": ["AAA", "AAA"],
                         "published_at": pd.to_datetime(["2026-01-05 07:00", "2026-01-05 09:00"]), "pred": [0.2, 0.6]})
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-05"]),
                       "symbol": ["AAA", "AAA", "AAA", "BBB"]})
    out = daily_feature(arts, df)
    assert out.ridge_24h.tolist() == pytest.approx([0.2, 0.6, 0.0, 0.0])
    assert out.has_ridge.tolist() == [1, 1, 0, 0]


def test_verdict_rule():
    from ai_core.models.volcast.text_return import verdict
    out = pd.DataFrame({"h": [2, 2, 2, 5, 5, 5], "diff": [0.01, 0.01, -0.001, 0.01, -0.002, 0.004]})
    assert verdict(out) == ({2: 2, 5: 2}, pytest.approx(0.0051667, abs=1e-6), True)
    assert verdict(out.assign(diff=[0.01, 0.01, -0.001, 0.01, -0.002, 0.003]))[2] is False    # mean 0.005, not > 0.005
    assert verdict(out.assign(diff=[0.05, -0.01, -0.001, 0.01, 0.01, 0.01]))[2] is False       # h=2 only 1/3
