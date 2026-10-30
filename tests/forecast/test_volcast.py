import numpy as np
import pandas as pd
import pytest

from nlp_module import storage
from nlp_module.forecast import fundamentals, volcast

NO_BCTC = pd.DataFrame(columns=["symbol", "available_at"] + fundamentals.BCTC[1:]).astype(
    {"symbol": str, "available_at": "datetime64[ns]"} | {k: float for k in fundamentals.BCTC[1:]})


def _bars(n=60):
    rng = np.random.default_rng(0)
    d = pd.bdate_range("2026-01-01", periods=n)
    rows = []
    for s in ("AAA", "BBB"):
        c = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        rows.append(pd.DataFrame({"date": d, "symbol": s, "close": c, "high": c * 1.01, "low": c * 0.99,
                                  "volume": rng.integers(1e5, 1e6, n).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def test_features_do_not_see_the_future():
    df = _bars()
    before = volcast.add_features(df, NO_BCTC)
    cut = df.groupby("symbol").cumcount() > 40
    df2 = df.copy()
    df2.loc[cut, "close"] *= 1.7
    after = volcast.add_features(df2, NO_BCTC)
    keep = (df.groupby("symbol").cumcount() <= 40).to_numpy()
    pd.testing.assert_frame_equal(before.loc[keep, volcast.FEATS], after.loc[keep, volcast.FEATS])


def test_live_bars_use_only_post_close_reading():
    c = storage.connect(":memory:")
    q = "INSERT INTO market_quote (symbol, fetched_at, trading_date, matched_price, highest, lowest, matched, source) VALUES (?,?,?,?,?,?,1,'t')"
    for sym, t, p in [("FPT", "03:00", 55400), ("FPT", "08:05", 55700), ("FPT", "09:40", 55800), ("VCB", "03:00", 90000)]:
        c.execute(q, (sym, f"2026-10-28T{t}:00Z", "20261028", p, p + 100, p - 100))
    out = volcast.live_bars(c)
    assert out.symbol.tolist() == ["FPT"]
    assert out.close.iloc[0] == 55.8 and out.date.iloc[0] == pd.Timestamp("2026-10-28")


def test_live_bars_ref_is_converted_and_null_or_zero_is_nan():
    c = storage.connect(":memory:")
    q = "INSERT INTO market_quote (symbol, fetched_at, trading_date, ref_price, matched_price, highest, lowest, matched, source) VALUES (?,?,?,?,?,?,?,1,'t')"
    for sym, ref in [("AAA", 55000), ("BBB", None), ("CCC", 0)]:
        c.execute(q, (sym, "2026-10-28T08:05:00Z", "20261028", ref, 55800, 55900, 55700))
    out = volcast.live_bars(c).set_index("symbol").ref
    assert out["AAA"] == 55.0 and np.isnan(out["BBB"]) and np.isnan(out["CCC"])


def test_bars_history_wins_and_empty_live_keeps_numeric(tmp_path):
    (tmp_path / "FPT_stock.csv").write_text("time,open,high,low,close,volume\n2026-10-27 07:00:00,1,2,1,1.5,100\n")
    c = storage.connect(":memory:")
    b = volcast.bars(c, str(tmp_path))
    assert b.close.dtype == float and b.date.iloc[0] == pd.Timestamp("2026-10-27")
    c.execute("INSERT INTO market_quote (symbol, fetched_at, trading_date, matched_price, highest, lowest, volume, source)"
              " VALUES ('FPT','2026-10-27T08:30:00Z','20261027',9000,9000,9000,5,'t')")
    assert volcast.bars(c, str(tmp_path)).close.tolist() == [1.5]


def test_live_bars_take_last_row_not_last_non_null():
    c = storage.connect(":memory:")
    q = "INSERT INTO market_quote (symbol, fetched_at, trading_date, matched_price, highest, lowest, volume, source) VALUES ('FPT',?,'20261028',?,?,?,?,'t')"
    c.execute(q, ("2026-10-28T08:05:00Z", 55700, 56000, 55000, 10))
    c.execute(q, ("2026-10-28T09:00:00Z", 55800, None, 55100, None))
    out = volcast.live_bars(c)
    assert out.close.iloc[0] == 55.8 and out.low.iloc[0] == 55.1
    assert np.isnan(out.high.iloc[0]) and np.isnan(out.volume.iloc[0])


def test_pick_threshold():
    conf = np.repeat([0.4, 0.3, 0.2, 0.1], 25)
    ok = np.r_[np.ones(50, bool), np.arange(25) < 10, np.zeros(25, bool)]
    assert volcast.pick_threshold(conf, ok) == 0.2
    assert volcast.pick_threshold(conf, ok, min_n=80) is None
    assert volcast.pick_threshold(conf, np.zeros(100, bool)) is None
    assert volcast.pick_threshold(np.full(50, 0.3), np.arange(50) < 33) is None   # 0.66 point, bound ~0.52


def test_wilson():
    lo, hi = volcast.wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
    assert all(np.isnan(volcast.wilson(0, 0)))


def test_walk_reports_none_when_unreachable():
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2021-09-01", "2026-12-31")
    df = pd.concat(pd.DataFrame({"date": d, "symbol": s, "close": 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d)))),
                                 "volume": rng.integers(1e5, 1e6, len(d)).astype(float)}) for s in "ABCD")
    df = df.assign(high=df.close * 1.01, low=df.close * 0.99).reset_index(drop=True)
    out = volcast.walk(volcast.add_features(df, NO_BCTC), 2)
    assert [r["fold"] for r in out] == ["A", "B", "C"]
    for r in out:
        if r["model"] is None:
            assert r["threshold"] is None and not r["test_n"]
        else:
            assert r["model"] in volcast.MODELS and r["test_n"] > 0 and r["ci_lo"] <= r["test_acc"] <= r["ci_hi"]


class _Const:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.c_[1 - np.full(len(X), self.p), np.full(len(X), self.p)]


def _live_df():
    df = _bars(60)
    df["date"] = df.groupby("symbol").date.transform(lambda s: pd.bdate_range(end="2026-10-28", periods=len(s)))
    df = df[~((df.symbol == "BBB") & (df.date == "2026-10-28"))]   # BBB stops a day early
    return volcast.add_features(df, NO_BCTC)


def test_predict_latest_uses_each_symbols_own_last_day():
    b = {"version": volcast.VERSION, **{h: {"model": _Const(0.9), "threshold": 0.3, "kind": "logit"} for h in (2, 5)}}
    out = volcast.predict_latest(_live_df(), b)
    assert len(out) == 4
    assert out[out.symbol == "AAA"].base_date.unique().tolist() == ["2026-10-28"]
    assert out[out.symbol == "BBB"].base_date.unique().tolist() == ["2026-10-27"]
    assert set(out.call) == {"strong"}


def test_predict_latest_threshold_none_gives_no_call():
    b = {"version": volcast.VERSION, **{h: {"model": _Const(0.9), "threshold": None, "kind": "logit"} for h in (2, 5)}}
    assert volcast.predict_latest(_live_df(), b).call.isna().all()


def test_predict_latest_weak_and_below_threshold():
    mk = lambda p: {2: {"model": _Const(p), "threshold": 0.3, "kind": "logit"}, 5: {"model": _Const(p), "threshold": 0.3, "kind": "logit"}}
    assert set(volcast.predict_latest(_live_df(), mk(0.1)).call) == {"weak"}
    assert volcast.predict_latest(_live_df(), mk(0.6)).call.isna().all()


def test_fit_live_none_fold_falls_back_without_threshold():
    b = volcast.fit_live(_live_df(), 2, {"model": None, "threshold": None})
    assert b["kind"] == "logit" and b["threshold"] is None and hasattr(b["model"], "predict_proba")


def test_report_json_has_null_not_nan(tmp_path):
    p = tmp_path / "r.json"
    volcast.write_report([{"fold": "A", "ci_lo": float("nan"), "threshold": None, "test_acc": np.float64(0.7)}], str(p))
    txt = p.read_text()
    assert "NaN" not in txt and '"ci_lo": null' in txt


def test_bars_drop_live_only_tickers(tmp_path):
    (tmp_path / "FPT_stock.csv").write_text("time,open,high,low,close,volume\n2026-10-27 07:00:00,1,2,1,1.5,100\n")
    c = storage.connect(":memory:")
    c.executemany("INSERT INTO market_quote (symbol, fetched_at, trading_date, matched_price, highest, lowest, volume, matched, source)"
                  " VALUES (?, '2026-10-28T08:30:00Z', '20261028', 9000, 9000, 9000, 5, 1, 't')", [("FPT",), ("ZZZ",)])
    assert set(volcast.bars(c, str(tmp_path)).symbol) == {"FPT"}


def test_rule_score_is_symmetric_around_half():
    te = pd.DataFrame({"date": pd.Timestamp("2026-01-05"), "vol10": [0.01, 0.02, 0.03, 0.04, 0.05]})
    r = volcast._rule_score(te)
    assert np.isclose(r.max() - 0.5, 0.5 - r.min()) and r.min() < 0.5 < r.max()
    te = pd.DataFrame({"date": pd.Timestamp("2026-01-05"), "vol10": [0.01, 0.01, 0.03]})   # ties must not break symmetry
    r = volcast._rule_score(te)
    assert np.isclose(r.max() - 0.5, 0.5 - r.min())


def test_walk_rows_carry_rule_ci():
    rng = np.random.default_rng(2)
    d = pd.bdate_range("2021-09-01", "2026-12-31")
    df = pd.concat(pd.DataFrame({"date": d, "symbol": s, "close": 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d)))),
                                 "volume": rng.integers(1e5, 1e6, len(d)).astype(float)}) for s in "ABCD")
    df = df.assign(high=df.close * 1.01, low=df.close * 0.99).reset_index(drop=True)
    for r in volcast.walk(volcast.add_features(df, NO_BCTC), 2):
        assert "rule_ci_lo" in r and "rule_ci_hi" in r
        if r["model"]:
            assert r["rule_ci_lo"] <= r["rule_acc"] <= r["rule_ci_hi"]


def test_dividend_seam_fixes_live_ret1_only():
    df = _bars(30)
    df["ref"] = np.nan
    a = df.index[(df.symbol == "AAA")][-1]           # last AAA row = a live ex-dividend day
    df.loc[a, "close"] = df.loc[a - 1, "close"] * 0.7 # raw close vs unadjusted prior: -30%
    df.loc[a, "ref"] = df.loc[a, "close"] / 1.01      # exchange ref (adjusted prior close): +1% on the day
    b = df.index[(df.symbol == "BBB")][-1]            # history row (ref NaN) with a real -30% move
    df.loc[b, "close"] = df.loc[b - 1, "close"] * 0.7
    out = volcast.add_features(df, NO_BCTC).set_index(["symbol", "date"])
    assert out.loc[("AAA", df.loc[a, "date"]), "ret1"] == pytest.approx(0.01)
    assert out.loc[("BBB", df.loc[b, "date"]), "ret1"] == pytest.approx(-0.3)
    aaa = out.loc["AAA"]                              # the fix must land before vol10 is computed
    assert aaa.vol10.iloc[-1] == pytest.approx(aaa.ret1.iloc[-10:].std())
    assert aaa.ret5.iloc[-1] == pytest.approx((1 + aaa.ret1.iloc[-5:]).prod() - 1)   # level features skip the jump too
    assert abs(aaa.ma20.iloc[-1]) < 0.15


def test_dividend_seam_ignores_missing_or_zero_ref():
    df = _bars(30)
    df["ref"] = np.nan
    a = df.index[(df.symbol == "AAA")][-1]
    df.loc[a, "close"] = df.loc[a - 1, "close"] * 0.7
    df.loc[a, "ref"] = 0.0
    out = volcast.add_features(df, NO_BCTC)
    assert out[out.symbol == "AAA"].ret1.iloc[-1] == pytest.approx(-0.3)


def test_walk_honours_folds():
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2021-09-01", "2026-12-31")
    df = pd.concat(pd.DataFrame({"date": d, "symbol": s, "close": 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d)))),
                                 "volume": rng.integers(1e5, 1e6, len(d)).astype(float)}) for s in "ABCD")
    df = df.assign(high=df.close * 1.01, low=df.close * 0.99).reset_index(drop=True)
    out = volcast.walk(volcast.add_features(df, NO_BCTC), 2, folds=[volcast.FOLDS[0]])
    assert [r["fold"] for r in out] == ["A"]


def test_pick_threshold_keeps_wilson_bound():   # regression test, passes before and after Task 4
    for seed in range(20):
        rng = np.random.default_rng(seed)
        conf = rng.uniform(0, 0.5, 2000)
        ok = rng.random(2000) < 0.5 + conf
        t = volcast.pick_threshold(conf, ok)
        if t is not None:
            sel = conf >= t
            assert volcast.wilson(ok[sel].sum(), sel.sum())[0] >= 0.65 and sel.sum() >= 50


def _long_bars():
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2025-10-01", "2026-06-30")
    df = pd.concat(pd.DataFrame({"date": d, "symbol": s, "close": 20 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d)))),
                                 "volume": rng.integers(1e5, 1e6, len(d)).astype(float)}) for s in ("AAA", "BBB", "CCC"))
    return df.assign(high=df.close * 1.01, low=df.close * 0.99).reset_index(drop=True)


FEATS_AB = pd.DataFrame({"symbol": ["AAA", "BBB"], "available_at": pd.Timestamp("2025-04-01"),
                         "bc_np_yoy": [0.1, 0.2], "bc_roe": [0.3, 0.4], "bc_lev": [0.5, 0.6], "bc_size": [7.0, 8.0]})


def test_perm_feats_moves_whole_series_between_symbols_and_keeps_dates():
    f = pd.concat([FEATS_AB, FEATS_AB.assign(available_at=pd.Timestamp("2026-04-01"), bc_roe=[0.9, 0.8])], ignore_index=True)
    out = volcast._perm_feats(f, ["AAA", "BBB", "CCC"], np.random.default_rng(3))
    assert sorted(out.available_at) == sorted(f.available_at)   # dates are not shuffled
    for s, g in out.groupby("symbol"):   # each series is intact: one source symbol's rows with their own dates
        src = [t for t in ("AAA", "BBB") if (f[f.symbol == t].drop(columns="symbol").reset_index(drop=True)
               .equals(g.drop(columns="symbol").sort_values("available_at").reset_index(drop=True)))]
        assert len(src) == 1


def test_walk_hold_trains_without_the_symbol_and_tests_only_on_it():
    df = volcast.add_features(_long_bars(), FEATS_AB)
    out = volcast.walk(df, 2, folds=[("A", "2025-10-01", "2026-02-01", "2026-03-01", "2026-06-01")], hold="CCC")
    assert [r["fold"] for r in out] == ["A"] and (out[0]["test_n"] <= (df.symbol == "CCC").sum())


def test_bctc_check_keys_and_shapes():
    df = volcast.add_features(_long_bars(), FEATS_AB)
    folds = [("A", "2025-10-01", "2026-02-01", "2026-03-01", "2026-06-01")]
    out = volcast.bctc_check(df, 2, folds, n_perm=2, bctc=FEATS_AB)
    assert {"real", "perm", "loto"} <= set(out)
    assert 0 <= out["real"] <= 1 and len(out["perm"]) == 2 and all(0 <= p <= 1 for p in out["perm"])
    assert set(out["loto"]) == {"AAA", "BBB", "CCC"}
    assert out["perm"] == volcast.bctc_check(df, 2, folds, n_perm=2, bctc=FEATS_AB)["perm"]   # seeded


def test_print_bctc_applies_the_rule(capsys, monkeypatch):
    df = volcast.add_features(_long_bars(), FEATS_AB)
    monkeypatch.setattr(volcast.fundamentals, "bctc_features", lambda: FEATS_AB)
    volcast._print_bctc(df, [("A", "2025-10-01", "2026-02-01", "2026-03-01", "2026-06-01")], n_perm=2)
    out = capsys.readouterr().out
    assert "6-cell: real=" in out and "conclusion:" in out and "CCC" in out
