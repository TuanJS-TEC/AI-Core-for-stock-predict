import numpy as np
import pandas as pd
import pytest

from ai_core.models.rangecast import rangecast as R


def _frames(n=200, syms=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, name="session_date")
    out = {}
    for k, s in enumerate(syms):
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.01 * (k + 1), n)))
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
        hi = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n)); lo = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
        out[s] = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1e6}, index=idx)
    return out


def test_features_equal_mmf_har_raw_terms():
    pytest.importorskip("torch")
    from ai_core.models.mmf import har
    fr = _frames()
    x = R.features(fr)
    assert list(x.columns) == R.FEATS
    for s, f in fr.items():
        pd.testing.assert_frame_equal(x.loc[s][R.HAR], har._raw_har(f)[R.HAR], check_names=False)
    assert np.allclose(x["har_rv_w_mkt"].loc["AAA"].dropna(), x["har_rv_w"].groupby(level=1).mean().dropna())


def test_future_prices_do_not_change_past_features():
    fr = _frames()
    base = R.features(fr)
    for f in fr.values():
        f.iloc[150:, :4] *= 1.3
    after = R.features(fr)
    keep = base.index.get_level_values(1) < fr["AAA"].index[150]
    pd.testing.assert_frame_equal(base[keep], after[keep])


def test_targets_look_only_forward_and_mark_unmatured_rows():
    fr = _frames()
    t = R.targets(fr, 5)
    a = t.loc["AAA"]
    c = fr["AAA"]["close"]
    assert a["status"].iloc[-5:].eq("not_matured").all() and a["status"].iloc[:-5].eq("ok").all()
    assert a["r"].iloc[10] == pytest.approx(np.log(c.iloc[15] / c.iloc[10]))
    r1 = np.log(c / c.shift(1))
    assert a["logrv"].iloc[10] == pytest.approx(np.log((r1.iloc[11:16] ** 2).sum() + 1e-8))


def test_band_is_ordered_and_inside_the_exchange_limit():
    lo, hi = R.band(np.array([1e-4, 0.02, 5.0]), (-1.3, 1.2), 5)
    assert (lo < 0).all() and (hi > 0).all()
    assert (lo >= 5 * np.log(0.93) - 1e-12).all() and (hi <= 5 * np.log(1.07) + 1e-12).all()
    assert lo[1] == pytest.approx(-1.3 * 0.02) and hi[1] == pytest.approx(1.2 * 0.02)


def test_coverage_on_synthetic_gaussian_returns():
    rng = np.random.default_rng(1)
    sig = rng.uniform(0.005, 0.03, 20000)
    r = sig * rng.standard_normal(20000)
    z = R.z_quantiles(r[:10000], sig[:10000])
    lo, hi = R.band(sig[10000:], z, 5)
    cov = ((r[10000:] >= lo) & (r[10000:] <= hi)).mean()
    assert abs(cov - 0.80) < 0.03


def test_pinball_is_zero_only_at_the_quantile_and_penalises_the_right_side():
    assert R.pinball(np.array([0.0]), np.array([0.0]), 0.1)[0] == 0
    assert R.pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert R.pinball(np.array([-1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.1)


def test_verdict_thresholds():
    assert R.verdict(0.80, 0.79, -0.001, -0.002) == "Dùng được"
    assert R.verdict(0.80, 0.79, 0.001, -0.002) == "Dùng được, có lưu ý"
    assert R.verdict(0.76, 0.80, -0.001, -0.002) == "Dùng được, có lưu ý"
    assert R.verdict(0.74, 0.80, -0.001, -0.002) == "Không đạt"
    assert R.verdict(0.80, 0.86, -0.001, -0.002) == "Không đạt"


def test_risk_level_percentiles():
    hist = np.arange(1.0, 101.0)
    assert R.risk_level(np.r_[hist, 69.5]) == "normal" and R.risk_level(np.r_[hist, 70.5]) == "high"      # 69% / 70% of the past below
    assert R.risk_level(np.r_[hist, 30.5]) == "low" and R.risk_level(np.r_[hist, 31.5]) == "normal"
    assert R.risk_level(np.r_[np.arange(1.0, 40.0), 1000.0]) == "normal"      # fewer than 60 values


# --- live bundle and daily prediction -------------------------------------------------------------------------------
from ai_core.news import storage


def _cut(fr, n):
    return {s: f.iloc[:n] for s, f in fr.items()}


def test_train_bundle_holds_out_the_last_126_sessions():
    fr = _frames(n=420)
    b = R.train_bundle(fr)
    assert b["version"] == R.VERSION
    for h in R.HORIZONS:
        d = R.frame(fr, h); d = d[d["usable"]]
        cut = np.sort(d["as_of_session"].unique())[-R.HOLD]
        old, new = d[d["as_of_session"] < cut], d[d["as_of_session"] >= cut]
        m = R.fit(old[R.FEATS], old["logrv"])
        assert np.allclose(b[h]["model"].coef_, m.coef_)
        assert b[h]["z"] == pytest.approx(R.z_quantiles(new["r"], R.sigma(m, new[R.FEATS])))


def test_predict_uses_the_newest_stored_session_and_never_rewrites_an_older_one():
    fr = _frames(n=421)
    b = R.train_bundle(_cut(fr, 400))
    c = storage.connect(":memory:")
    first = R.predict_rows(_cut(fr, 420), b)
    assert len(first) == 3 * len(R.HORIZONS) and {r[1] for r in first} == {str(fr["AAA"].index[419].date())}
    assert R.write_rows(c, first) == 6 and R.write_rows(c, first) == 0
    b2 = R.train_bundle(_cut(fr, 410))                       # a retrained bundle must not change a stored day
    assert R.write_rows(c, R.predict_rows(_cut(fr, 420), b2)) == 0
    assert R.write_rows(c, R.predict_rows(fr, b)) == 6
    stored = [tuple(r) for r in c.execute("SELECT * FROM range_forecast WHERE base_date = ? ORDER BY symbol, h", (first[0][1],))]
    assert stored == sorted(first, key=lambda r: (r[0], r[2]))
    for sym, base, h, close, lo, hi, sig, level, version, _ in first:
        assert lo < 0 < hi and level in ("low", "normal", "high") and version == R.VERSION
        assert close == pytest.approx(fr[sym]["close"].iloc[419])


def test_ticker_with_a_gap_or_a_jump_in_the_har_window_gets_no_row_others_do():
    fr = _frames(n=420)
    b = R.train_bundle(_cut(fr, 400))
    gap = {s: f.copy() for s, f in fr.items()}
    gap["BBB"].iloc[410, :4] = np.nan
    assert {r[0] for r in R.predict_rows(gap, b)} == {"AAA", "CCC"}
    jump = {s: f.copy() for s, f in fr.items()}
    jump["CCC"].iloc[412:, :4] *= 1.10                       # a re-based series: one session moves 10%
    assert {r[0] for r in R.predict_rows(jump, b)} == {"AAA", "BBB"}


def test_predict_without_a_usable_bundle_exits_with_the_train_hint(tmp_path):
    import joblib
    with pytest.raises(SystemExit, match="rangecast train"):
        R.load_bundle(tmp_path / "missing.joblib")
    joblib.dump({"version": "rangecast-v0"}, tmp_path / "old.joblib")
    with pytest.raises(SystemExit, match="rangecast train"):
        R.load_bundle(tmp_path / "old.joblib")
