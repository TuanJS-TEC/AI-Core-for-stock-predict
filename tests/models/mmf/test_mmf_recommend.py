import numpy as np, pandas as pd, pytest

from ai_core.models.mmf import recommend as R


def _toy_prices():
    return pd.DataFrame({"symbol": "AAA", "as_of_session": pd.bdate_range("2024-01-01", periods=8),
                         "open": 10.0 + np.arange(8), "close": 20.0 + np.arange(8)})


def _toy_rows():
    """60 sessions x 5 symbols: rows with a wide margin earn 0.02, the others -0.01; one row has no future bars."""
    r = np.random.default_rng(0)
    n, gap = 300, r.uniform(0, 0.4, 300)
    net = np.where(gap >= 0.1, 0.02, -0.01).astype(float)
    net[-1] = np.nan
    return pd.DataFrame({"symbol": np.tile(list("ABCDE"), 60), "as_of_session": np.repeat(pd.bdate_range("2024-01-01", periods=60), 5),
                         "p_up": 0.3 + gap, "p_down": np.full(n, 0.3), "net_ret": net})


def test_trade_return_uses_next_open_and_exit_close():
    f = _toy_prices()
    r = R.trade_return(f, 2, cost=0.0045)
    assert r.iloc[0] == pytest.approx(np.log(f.close.iloc[3] / f.open.iloc[1]) - 0.0045)
    assert r.iloc[-1:].isna().all()
    assert r.isna().sum() == 3     # t + h + 1 past the last bar


def test_trade_return_is_per_symbol_and_keeps_the_frame_order():
    a, b = _toy_prices(), _toy_prices().assign(symbol="BBB", open=lambda d: d.open * 2, close=lambda d: d.close * 3)
    f = pd.concat([a, b], ignore_index=True).iloc[::-1]
    r = R.trade_return(f, 2, cost=0.0)
    assert r.index.equals(f.index)
    assert r.loc[0] == pytest.approx(np.log(23 / 11))
    assert r.loc[8] == pytest.approx(np.log(23 * 3 / 22))
    assert np.isnan(r.loc[5])      # never reads the next symbol's bars


def test_margin_needs_min_trades():
    assert R.pick_margin(np.array([.9] * 10), np.array([.1] * 10), np.ones(10), min_n=50) is None


def test_margin_maximises_mean_net_and_ignores_rows_without_return():
    d = _toy_rows()
    m = R.pick_margin(d.p_up.to_numpy(), d.p_down.to_numpy(), d.net_ret.to_numpy(), min_n=50)
    assert m == pytest.approx(0.1)
    net = np.full(300, np.nan); net[:40] = 1.0
    assert R.pick_margin(d.p_up.to_numpy(), d.p_down.to_numpy(), net, min_n=50) is None


def test_labels_three_way():
    assert list(R.label(np.array([.6, .1, .4]), np.array([.2, .6, .35]), 0.2)) == ["Ưu tiên mua", "Tránh", "Không đủ tin cậy"]


def test_simulate_reports_coverage_with_result():
    out = R.simulate(_toy_rows(), 0.1, 2)
    assert {"coverage", "n", "n_days", "mean_net", "rand_median", "ci_lo", "ci_hi"} <= set(out)
    d = _toy_rows().dropna(subset=["net_ret"])
    k = d.p_up - d.p_down >= 0.1
    assert out["n"] == k.sum() and out["coverage"] == pytest.approx(k.mean()) and out["n_days"] == d[k].as_of_session.nunique()
    assert out["mean_net"] == pytest.approx(0.02)
    assert out["rand_median"] == pytest.approx(d.net_ret.mean(), abs=2e-3)   # random draw of the same size: no edge
    assert out["ci_lo"] == pytest.approx(0.02) and out["ci_hi"] == pytest.approx(0.02)


def test_simulate_without_trades_reports_zero_coverage():
    out = R.simulate(_toy_rows(), 0.9, 2)
    assert out["coverage"] == 0 and out["n"] == 0 and np.isnan(out["mean_net"]) and np.isnan(out["ci_lo"])


def test_block_rows_read_no_price_after_the_block():
    f, end = _toy_prices(), pd.bdate_range("2024-01-01", periods=8)[5]
    rows = f.loc[f.as_of_session <= end, ["symbol", "as_of_session"]]
    a = R.with_net_ret(rows, f, 2, end, cost=0.0)
    assert a.net_ret.iloc[2] == pytest.approx(np.log(f.close.iloc[5] / f.open.iloc[3]))     # exit on the block's last session stays
    assert a.net_ret.iloc[3:].isna().all()                                                  # exit past it does not
    later = f.assign(close=f.close.where(f.as_of_session <= end, 999.0), open=f.open.where(f.as_of_session <= end, 999.0))
    pd.testing.assert_series_equal(a.net_ret, R.with_net_ret(rows, later, 2, end, cost=0.0).net_ret)


def test_simulate_passes_block_to_bootstrap(monkeypatch):
    seen, real = [], R.block_bootstrap_dates
    monkeypatch.setattr(R, "block_bootstrap_dates", lambda d, block, *a: (seen.append(block), real(d, block, *a))[1])
    R.simulate(_toy_rows(), 0.1, 2)
    R.simulate(_toy_rows(), 0.1, 2, block=5)
    assert seen == [R.PC.BOOTSTRAP_BLOCK, 5]
