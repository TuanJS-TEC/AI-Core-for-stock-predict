import numpy as np
import pandas as pd
import pytest

from ai_core import config as C
from ai_core.direction import FEATS, add_labels, build_frame

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
