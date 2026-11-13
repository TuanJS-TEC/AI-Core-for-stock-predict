import numpy as np
import pandas as pd

from ai_core.prices.data import Panel
from ai_core.prices.dataset import LABEL_COLS, build_dataset, inference_features
from ai_core.prices.price_features import FEATURES

SYMS = ["AAA", "BBB", "CCC"]
MARKET_DEPENDENT = {"mret1", "mret5", "mvol20", "ex_ret1", "ex_ret5"}


def _panel(n=300, drop_last_of=None):
    dates = pd.bdate_range("2024-01-01", periods=n)
    ohlcv = {}
    for i, s in enumerate(SYMS):
        rng = np.random.default_rng(i)
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = close * np.exp(rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * np.exp(rng.uniform(0, 0.005, n))
        low = np.minimum(open_, close) * np.exp(-rng.uniform(0, 0.005, n))
        vol = rng.uniform(1e5, 1e6, n)
        ohlcv[s] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=dates
        )
    if drop_last_of:
        ohlcv[drop_last_of] = ohlcv[drop_last_of].iloc[:-1]
    log_close = {s: np.log(f["close"].to_numpy()) for s, f in ohlcv.items()}
    return Panel(dates, dates.to_numpy(), log_close, ohlcv=ohlcv, calendar=dates)


def test_offline_online_parity():
    p = _panel()
    ds, inf = build_dataset(p, 2), inference_features(p)
    for s in SYMS:
        off = ds[(ds["symbol"] == s) & (ds["as_of_session"] == p.calendar[-1])].iloc[0]
        on = inf[inf["symbol"] == s].iloc[0]
        pd.testing.assert_series_equal(off[FEATURES].astype(float), on[FEATURES].astype(float), check_names=False)


def test_inference_has_no_target_columns():
    assert set(inference_features(_panel()).columns).isdisjoint(LABEL_COLS)


def test_missing_last_session_marks_only_that_symbol():
    full = inference_features(_panel()).set_index("symbol")
    miss = inference_features(_panel(drop_last_of="BBB")).set_index("symbol")
    assert not miss.loc["BBB", "features_ok"]
    assert miss.loc[["AAA", "CCC"], "features_ok"].all()
    own = [f for f in FEATURES if f not in MARKET_DEPENDENT]
    pd.testing.assert_frame_equal(miss.loc[["AAA", "CCC"], own], full.loc[["AAA", "CCC"], own])


def test_last_rows_not_matured():
    ds = build_dataset(_panel(), 5)
    for s in SYMS:
        st = ds.loc[ds["symbol"] == s, "status"].to_numpy()
        assert (st == "not_matured").sum() == 5
        assert (st[-5:] == "not_matured").all()


def test_training_rows_require_features_and_ok_label():
    ds = build_dataset(_panel(), 2)
    finite = np.isfinite(ds[FEATURES].to_numpy()).all(axis=1)
    assert (ds["features_ok"].to_numpy() == finite).all()
    assert (ds["features_ok"] & (ds["status"] == "ok")).sum() == (finite & ds["label"].notna().to_numpy()).sum()
    assert (ds["features_ok"] & (ds["status"] == "ok")).sum() > 0
    assert list(ds.columns) == ["symbol", "as_of_session", "h", *FEATURES, "flag_hl_equal", "features_ok", *LABEL_COLS]
