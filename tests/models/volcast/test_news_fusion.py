from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai_core.models.volcast import news_fusion as NF, trial_forecast as tf, volcast
from ai_core.news.contract import ArticleCompanyResult

from .test_volcast import NO_BCTC

FOLDS = [("A", "2022-01-03", "2022-07-01", "2022-09-01", "2022-11-01"),
         ("B", "2022-01-03", "2022-11-01", "2023-01-02", "2023-03-01"),
         ("C", "2022-01-03", "2023-03-01", "2023-05-01", "2023-07-01")]
SYMS = list("ABCDEFGH")


def _frame(signal, seed=0):
    """Random features on 8 symbols; y2 follows one news column when `signal`, else nothing."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", "2023-06-30")
    d = pd.DataFrame({"date": np.repeat(days, len(SYMS)), "symbol": SYMS * len(days)})
    cols = list(dict.fromkeys(volcast.FEATS + NF.NEWS + ["ret1", "ret5", "vol10"]))
    d[cols] = rng.normal(size=(len(d), len(cols)))
    score = d["sentiment_abs_max"] if signal else rng.normal(size=len(d))
    d["y2"] = (score + 0.5 * rng.normal(size=len(d)) > 0).astype(float)
    return d


def test_fold_test_scores_only_the_test_window_with_the_kind_picked_on_validation():
    d = _frame(True)
    kind, te = NF.fold_test(d, 2, NF.ARMS["fusion"], FOLDS[0][1:])
    assert kind in volcast.MODELS and list(te.columns) == ["date", "symbol", "y", "p"]
    assert te["date"].min() >= pd.Timestamp("2022-09-01") and te["date"].max() < pd.Timestamp("2022-11-01")
    assert te["p"].between(0, 1).all() and len(te) == te[["date", "symbol"]].drop_duplicates().shape[0]


def test_r6_finds_news_signal_and_the_shuffled_control_loses_it():
    d = _frame(True)
    real = NF.r6({f[0]: d for f in FOLDS}, 2, FOLDS)
    assert real["mean"] < -0.05 and real["hi"] < 0 and real["fold_wins"] == 3 and real["verdict"] == "Có bằng chứng"
    assert set(real["kinds"]) == {"price", "fusion"} and real["n_days"] == real["per_fold"]["n_days"].sum()
    perm = NF.r6({f[0]: NF.shuffle_news(d, 0) for f in FOLDS}, 2, FOLDS)
    assert perm["mean"] > real["mean"] + 0.05


def test_r6_without_signal_is_not_evidence():
    d = _frame(False)
    assert NF.r6({f[0]: d for f in FOLDS}, 2, FOLDS)["verdict"] != "Có bằng chứng"


def test_shuffle_news_moves_news_between_symbols_of_the_same_date_only():
    d = _frame(True)
    s = NF.shuffle_news(d, 1)
    other = [c for c in d.columns if c not in NF.NEWS]
    pd.testing.assert_frame_equal(s[other], d[other])
    assert not s[NF.NEWS].equals(d[NF.NEWS])
    key = lambda x: x.sort_values(["date", *NF.NEWS])[["date", *NF.NEWS]].reset_index(drop=True)
    pd.testing.assert_frame_equal(key(s), key(d))                       # same vectors per date, kept whole
    assert not NF.shuffle_news(d, 2)[NF.NEWS].equals(s[NF.NEWS])


def _bars(n=80):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2026-01-01", periods=n)
    parts = []
    for s in ("AAA", "BBB", "CCC"):
        c = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        parts.append(pd.DataFrame({"date": days, "symbol": s, "close": c, "high": c * 1.01, "low": c * 0.99,
                                   "volume": rng.integers(1e5, 1e6, n).astype(float)}))
    return volcast.add_features(pd.concat(parts, ignore_index=True), NO_BCTC)


def _article(i, day, symbol="AAA"):
    pub = datetime(day.year, day.month, day.day, 2, 0, tzinfo=timezone.utc)          # 09:00 ICT, before the close
    return ArticleCompanyResult(f"r{i}", symbol, "s", pub, pub, "direct", 0.8, 0.1, 0.1, nlp_model_version="v")


def test_add_news_builds_ranked_counts_and_does_not_look_ahead():
    df = _bars()
    days = sorted(df["date"].unique())
    res = [_article(i, days[k]) for i, k in enumerate((30, 30, 31, 50))]
    d = NF.add_news(df, res, "v")
    assert len(d) == len(df) and set(NF.NEWS) <= set(d.columns) and d[NF.NEWS[:-1]].notna().all().all()
    assert d[tf.COUNTS].stack().between(0, 1).all()                                   # cross-sectional ranks
    row = d[(d.symbol == "AAA") & (d.date == days[30])].iloc[0]
    assert row["has_news"] == 1 and row["news_count_24h"] == 1.0                      # top of the day's rank
    assert d.loc[(d.symbol == "BBB") & (d.date == days[30]), "has_news"].iloc[0] == 0
    more = NF.add_news(df, res + [_article(9, days[60], "BBB")], "v")
    past = d["date"] < days[60]
    pd.testing.assert_frame_equal(more.loc[past, NF.NEWS], d.loc[past, NF.NEWS])


def test_walk_takes_a_feature_list():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2021-09-01", "2026-12-31")
    df = pd.concat(pd.DataFrame({"date": days, "symbol": s, "close": 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))),
                                 "volume": rng.integers(1e5, 1e6, len(days)).astype(float)}) for s in "ABCD")
    df = volcast.add_features(df.assign(high=df.close * 1.01, low=df.close * 0.99).reset_index(drop=True), NO_BCTC)
    df["extra"] = rng.normal(size=len(df))
    df.loc[df.index[:500], "extra"] = np.nan                                          # rows without the extra feature are dropped
    out = volcast.walk(df, 2, feats=volcast.FEATS + ["extra"])
    assert [r["fold"] for r in out] == ["A", "B", "C"]


def test_cli_writes_every_table(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2021-09-01", "2026-12-31")
    df = pd.concat(pd.DataFrame({"date": days, "symbol": s, "close": 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))),
                                 "volume": rng.integers(1e5, 1e6, len(days)).astype(float)}) for s in "ABCD")
    df = volcast.add_features(df.assign(high=df.close * 1.01, low=df.close * 0.99).reset_index(drop=True), NO_BCTC)
    for h in volcast.HORIZONS:
        df[f"y{h}"] = tf.add_targets(df.assign(close=df.adj), h, "absmove")[f"y{h}"]
    d = NF.add_news(df, [], "v")
    seen = {}
    monkeypatch.setattr(NF.storage, "connect", lambda: None)
    monkeypatch.setattr(NF, "datasets", lambda c, symbols, folds: (seen.update(folds=folds), {f[0]: d for f in folds})[1])
    NF.main(["--test-end", "2026-10-30", "--perms", "1", "--out", str(tmp_path / "r.txt")])
    assert seen["folds"][2][4] == "2026-10-30" and seen["folds"][0] == NF.FOLDS[0]
    text = (tmp_path / "r.txt").read_text()
    assert all(k in text for k in ("== R6", "== per fold", "== permutation control", "== Wilson gate")) and text.count("fusion") >= 4
