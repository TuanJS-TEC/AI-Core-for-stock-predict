import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from nlp_module.forecast import trial_forecast as tf
from nlp_module.contract import ArticleCompanyResult


class TrialForecastTest(unittest.TestCase):
    def test_text_features_help_when_signal_exists_and_purge_holds(self):
        rng = np.random.default_rng(0)
        days = pd.bdate_range("2025-01-01", periods=400)
        sent = rng.choice([-1, 0, 1], size=len(days))
        # price at t+1 moves with the sentiment of articles published on day t (signal), plus noise
        ret = np.r_[0, 0.012 * sent[:-1]] + rng.normal(0, 0.006, len(days))
        close = 100 * np.cumprod(1 + ret)
        path = tempfile.mktemp(suffix=".csv")
        pd.DataFrame({"date": days, "symbol": "AAA", "close": close}).to_csv(path, index=False)
        results = []
        for i, (d, s) in enumerate(zip(days, sent)):
            if s:
                pub = datetime(d.year, d.month, d.day, 2, 0, tzinfo=timezone.utc)   # 09:00 ICT, before 15:00 close
                p = (0.8, 0.1, 0.1) if s > 0 else (0.1, 0.1, 0.8)
                results.append(ArticleCompanyResult(f"r{i}", "AAA", "s", pub, pub, "direct", *p))
        df = tf.add_targets(tf.load_prices(path), 2)
        df = tf.add_text(df, results, "backtest", "baseline-v0")
        with_text, price_only = tf.run(df, 2, use_text=True), tf.run(df, 2, use_text=False)
        self.assertGreater(with_text["auc"], price_only["auc"] + 0.05)
        self.assertLess(with_text["n_train"] + with_text["n_test"], len(days))   # embargo/purge removed rows


    def test_news_surprise_ranks_raw_values_within_each_date(self):
        d = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-03"] * 3), "symbol": list("ABC") * 2,
                          "news_count_24h": [5, 1, 3, 0, 9, 9], "vol10": [0.1, 0.3, 0.2, 0.3, 0.2, 0.1]})
        out = tf.add_news_surprise(d)
        day1 = out[out.date == "2025-01-02"].set_index("symbol").news_surprise
        self.assertEqual(day1.idxmax(), "A")   # most news, calmest price
        self.assertEqual(day1.idxmin(), "B")   # least news, most volatile
        day2 = out[out.date == "2025-01-03"].set_index("symbol").news_surprise
        self.assertEqual(day2.idxmin(), "A")   # unaffected by day 1: A has the fewest news that day
        self.assertEqual(day2.idxmax(), "C")   # tied news, calmer than B
        self.assertTrue(out.news_surprise.notna().all())
        d.loc[0, "vol10"] = np.nan
        self.assertTrue(np.isnan(tf.add_news_surprise(d).news_surprise[0]))

    def test_add_text_imputes_never_seen_symbol_as_720h_and_zero_decay(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2025-03-03")], "symbol": ["AAA"]})
        pub = datetime(2025, 3, 3, 2, 0, tzinfo=timezone.utc)
        other = ArticleCompanyResult("r0", "BBB", "s", pub, pub, "direct", 0.8, 0.1, 0.1)
        out = tf.add_text(df, [other], "backtest", "baseline-v0")
        self.assertEqual(out.hours_since_last_news[0], 720.0)
        self.assertEqual(out.sentiment_decay[0], 0.0)
        self.assertEqual(out.has_news[0], 0.0)
        old = datetime(2024, 1, 2, 2, 0, tzinfo=timezone.utc)   # >720h before the cutoff
        mine = ArticleCompanyResult("r1", "AAA", "s", old, old, "direct", 0.8, 0.1, 0.1)
        self.assertEqual(tf.add_text(df, [mine], "backtest", "baseline-v0").hours_since_last_news[0], 720.0)

    def test_cross_sectional_rank_removes_a_level_shift(self):
        # same cross-sectional ORDER every day, but the level doubles in the second half (corpus growth)
        days = pd.bdate_range("2025-01-01", periods=20)
        d = pd.DataFrame([{"date": day, "symbol": s, "news_count_72h": (i + 1) * (1 + (k >= 10))}
                          for k, day in enumerate(days) for i, s in enumerate("ABC")])
        half = days[10]
        before, after = tf.drift(d, ["news_count_72h"], half)["news_count_72h"]
        self.assertAlmostEqual(after, before * 2)                      # the drift the forecaster trips on
        r = tf.rank_cs(d, ["news_count_72h"])
        lo, hi = tf.drift(r, ["news_count_72h"], half)["news_count_72h"]
        self.assertEqual(lo, hi)                                       # rank is invariant to the level shift
        self.assertEqual(sorted(r[r.date == days[0]]["news_count_72h"].round(3)), [0.333, 0.667, 1.0])


    def test_excess_target_strips_the_market_move(self):
        # every ticker rises, so raw direction is all 1s and unlearnable; AAA rises fastest, so only it beats the mean
        days = pd.bdate_range("2025-01-01", periods=10)
        rows = [{"date": d, "symbol": s, "close": 100 * (1 + g) ** i}
                for i, d in enumerate(days) for s, g in (("AAA", 0.03), ("BBB", 0.01), ("CCC", 0.01))]
        df = pd.DataFrame(rows).sort_values(["symbol", "date"])
        raw = tf.add_targets(df.copy(), 2)["y2"].dropna()
        exc = tf.add_targets(df.copy(), 2, "excess")
        self.assertEqual(set(raw), {1.0})                                      # no information left in direction
        got = exc.dropna(subset=["y2"]).groupby("symbol")["y2"].mean().to_dict()
        self.assertEqual(got, {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0})


    def test_absmove_target_ignores_sign_and_ranks_magnitude(self):
        # AAA swings hardest, BBB falls hard, CCC barely moves: magnitude must flag AAA and BBB, not CCC
        days = pd.bdate_range("2025-01-01", periods=10)
        rows = [{"date": d, "symbol": s, "close": 100 * (1 + g) ** i}
                for i, d in enumerate(days) for s, g in (("AAA", 0.05), ("BBB", -0.04), ("CCC", 0.001))]
        df = pd.DataFrame(rows).sort_values(["symbol", "date"])
        got = tf.add_targets(df.copy(), 2, "absmove").dropna(subset=["y2"]).groupby("symbol")["y2"].mean().to_dict()
        self.assertEqual(got, {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0})   # median of 3 => only the largest clears it
        self.assertEqual(tf.add_targets(df.copy(), 2, "excess").dropna(subset=["y2"])
                         .groupby("symbol")["y2"].mean()["BBB"], 0.0)   # BBB loses on direction, so the two differ

    def test_prices_from_stock_keeps_only_requested_symbols(self):
        with tempfile.TemporaryDirectory() as d:
            for sym in ("AAA", "BBB", "CCC"):
                pd.DataFrame({"time": [f"2021-01-0{i} 07:00:00" for i in (4, 5, 6)], "open": 1.0, "high": 2.0,
                              "low": 1.0, "close": [10.0, 11.0, 12.0], "volume": 100}).to_csv(f"{d}/{sym}_stock.csv", index=False)
            got = tf.prices_from_stock(d, {"AAA", "BBB"})
        self.assertEqual(list(got.columns), ["date", "symbol", "close"])
        self.assertEqual(len(got), 6)
        self.assertEqual(set(got.symbol), {"AAA", "BBB"})
        self.assertEqual(got.date.iloc[0], "2021-01-04")

    def test_check_prices_flags_calendar_gap_and_big_move(self):
        days = pd.bdate_range("2025-01-01", periods=5).strftime("%Y-%m-%d")
        df = pd.DataFrame([{"date": d, "symbol": s, "close": 10.0} for s in ("AAA", "BBB") for d in days])
        got = tf.check_prices(df)
        self.assertEqual((got["n_symbols"], got["same_calendar"], got["n_nan"], got["n_big_moves"]), (2, True, 0, 0))
        self.assertFalse(tf.check_prices(df.drop(df.index[-2]))["same_calendar"])   # BBB loses one day
        two = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"] * 2, "symbol": ["AAA"] * 2 + ["BBB"] * 2,
                            "close": [10.0, 10.1, 20.0, 20.1]}).iloc[[2, 3, 0, 1]]   # unsorted; AAA->BBB boundary is a 2x jump
        self.assertEqual(tf.check_prices(two)["n_big_moves"], 0)
        df.loc[4, "close"] = 10.702                                               # +7.02%: a limit day rounded past 0.07
        self.assertEqual(tf.check_prices(df)["n_big_moves"], 0)
        df.loc[4, "close"] = 12.0                                                 # AAA's last close jumps +20%
        self.assertEqual(tf.check_prices(df)["n_big_moves"], 1)
        self.assertEqual(tf.check_prices(df)["n_dup"], 0)
        dup = pd.concat([df, df.iloc[[0]]])                                       # AAA's first day appears twice
        self.assertEqual(tf.check_prices(dup)["n_dup"], 2)


if __name__ == "__main__":
    unittest.main()
