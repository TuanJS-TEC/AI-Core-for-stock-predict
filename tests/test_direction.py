import numpy as np
import pandas as pd
import pytest

from ai_core import config as C
from ai_core.direction import FEATS, add_labels, build_frame, folds, go, long_mask, run_fold, select_on_val

N, WARM = 40, 20


def _frame():
    rng = np.random.default_rng(1)
    parts = []
    for s, sd in zip("ABC", (0.02, 0.02, 0.002)):   # C is calm enough for the cost floor to bind
        close = 10 * np.exp(np.cumsum(rng.normal(0, sd, N)))
        d = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=N), "symbol": s, "close": close,
                          "open": close * np.exp(rng.normal(0, 0.005, N))})
        d["vol20"] = np.log(d.close).diff().rolling(WARM).std()   # as volcast.add_features makes it: NaN for the first 20 rows
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_tradable_return_uses_next_open_to_close_t1h():
    d = _frame()
    out = add_labels(d, 2)
    a, o = d[d.symbol == "B"].reset_index(drop=True), out[out.symbol == "B"].reset_index(drop=True)
    t = 10
    assert o.r_trade2[t] == pytest.approx(np.log(a.close[t + 3] / a.open[t + 1]))


def test_excess_and_tau():
    d = _frame()
    h = 5
    out = add_labels(d, h)
    full = out.dropna(subset=[f"r_trade{h}"])
    assert (full.groupby("date")[f"r_ex{h}"].sum().abs() < 1e-12).all()
    ok = out.dropna(subset=["vol20", f"r_ex{h}"])
    assert np.allclose(ok[f"tau{h}"], np.maximum(C.COST, C.KAPPA * ok.vol20 * np.sqrt(h)))
    assert (ok[f"tau{h}"] > C.COST).any() and (ok[f"tau{h}"] == C.COST).any()   # both branches of the max are exercised
    r, tau = ok[f"r_ex{h}"], ok[f"tau{h}"]
    assert (ok[f"y{h}"] == np.where(r > tau, 1.0, np.where(r < -tau, -1.0, 0.0))).all()
    assert set(ok[f"y{h}"]) == {-1.0, 0.0, 1.0}


def test_label_nan_without_vol_or_future():
    h = 2
    out = add_labels(_frame(), h)
    for _, g in out.groupby("symbol"):
        g = g.sort_values("date")
        assert g[f"y{h}"].iloc[-(h + 1):].isna().all()
        assert g[f"y{h}"].iloc[:WARM].isna().all()
        assert g[f"y{h}"].iloc[WARM:-(h + 1)].notna().all()


@pytest.mark.parametrize("h", [2, 5])
def test_build_frame_real_data(h):
    df = build_frame(h)
    assert set(df.symbol) == set(C.UNIVERSE)
    assert {f"r_trade{h}", f"r_ex{h}", f"tau{h}", f"y{h}", *FEATS} <= set(df.columns)
    y = df[f"y{h}"].dropna()
    frac = y.value_counts(normalize=True)
    print(f"h={h} n={len(y)} balance", frac.round(3).to_dict())
    assert all(0.05 < frac[k] < 0.9 for k in (-1.0, 0.0, 1.0))


def _prices(n_sym=6):
    """Synthetic OHLCV through fold C's test window, with planted mean reversion so the models have something to find."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2025-03-03", "2026-10-29")
    parts = []
    for i in range(n_sym):
        x = np.zeros(len(dates))
        for t in range(5, len(dates)):
            x[t] = -0.15 * x[t - 5:t].sum() + rng.normal(0, 0.02)
        close = 20 * np.exp(np.cumsum(x))
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, len(dates)))
        parts.append(pd.DataFrame({"date": dates, "symbol": f"S{i}", "open": open_, "close": close,
                                   "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * 0.99,
                                   "volume": rng.lognormal(12, 0.3, len(dates))}))
    return pd.concat(parts, ignore_index=True)


def _fold_c(prices, h=2):
    return {name: rest for name, *rest in folds(build_frame(h, prices), h)}["C"]


def test_long_mask_threshold():
    assert long_mask(np.array([0.75, 0.5, 0.25]), np.array([0.25, 0.25, 0.25]), 0.25).tolist() == [True, True, False]
    assert long_mask(np.array([0.3]), np.array([0.5]), 0.0).tolist() == [False]   # AVOID is never LONG


def test_go_rule():
    row = lambda net, rand=0.0, kind="logit": {"kind": kind, "net": net, "rand_median": rand}
    assert go([row(0.01), row(0.02), row(-0.01)], 0.001)
    assert not go([row(0.01), row(-0.02), row(-0.01)], 0.001)
    assert not go([row(0.01), row(0.02), row(0.01)], -0.001)
    assert not go([row(0.01), row(0.02, kind=None), row(-0.01)], 0.001)          # kind=None is a loss
    assert not go([row(0.01, 0.02), row(0.02, 0.03), row(0.01)], 0.001)         # beats rand in only 1/3


def test_purged_split_no_leakage():
    p = _prices()
    tr, va, _, te = _fold_c(p)
    moved = p.copy()
    moved.loc[moved.date >= "2026-07-01", "close"] *= 1.5                         # every fold C test row
    tr2, va2, _, te2 = _fold_c(moved)
    assert not np.allclose(te.r_trade2, te2.r_trade2)                            # the change does reach test labels
    pd.testing.assert_frame_equal(tr, tr2)                                       # features and labels of train/val
    pd.testing.assert_frame_equal(va, va2)
    a, b = select_on_val(tr, va, 2), select_on_val(tr2, va2, 2)
    assert a["kind"] is not None and a["val_n"] >= 50
    assert (a["kind"], a["C"], a["m"], a["val_n"]) == (b["kind"], b["C"], b["m"], b["val_n"])
    assert abs(a["val_net"] - b["val_net"]) < 1e-12


def test_select_on_val_none_below_50_trades():
    tr, va, _, _ = _fold_c(_prices())
    assert select_on_val(tr, va.head(49), 2)["kind"] is None


def test_run_fold_keys():
    tr, va, trva, te = _fold_c(_prices())
    r = run_fold(tr, va, trva, te, 2, np.random.default_rng(0))
    assert {"kind", "C", "m", "val_net", "val_n", "n", "hit", "net", "net_ex", "ci_lo", "ci_hi", "rand_median",
            "turnover", "max_dd", "e0_prior"} <= set(r)
    assert r["kind"] is not None and r["n"] > 0 and r["ci_lo"] <= r["net_ex"] <= r["ci_hi"]
    assert 0 <= r["hit"] <= 1 and r["max_dd"] >= 0 and 0 <= r["turnover"] <= 1
    assert r["e0_prior"] == pytest.approx((te.r_trade2 - C.COST).mean())
