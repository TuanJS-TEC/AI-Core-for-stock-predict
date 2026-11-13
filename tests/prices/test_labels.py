import math

import numpy as np
import pandas as pd

from ai_core.prices import config as C
from ai_core.prices.data import assign_labels
from ai_core.prices.labels import make_labels

COLS = ["target_session", "label_end_date", "realized_return", "theta_used", "label", "status", "label_version"]


def _frame(n, seed=0, gap=None):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    if gap is not None:
        close[gap] = np.nan
    return pd.DataFrame({"close": close}, index=pd.bdate_range("2025-01-01", periods=n))


def test_target_follows_calendar_not_rows():
    g, h = 70, 5
    f = _frame(140, gap=g)
    out = make_labels(f, h)
    row = out.iloc[g - 2]
    assert row["target_session"] == f.index[g + 3]
    assert row["status"] == "missing_session"
    assert pd.isna(row["label"])
    assert out.iloc[g + 61]["status"] == "ok"


def test_theta_formula_and_prefix_invariance():
    h = 5
    rets = np.where(np.arange(150) % 2 == 0, 0.01, -0.012)
    f = pd.DataFrame({"close": 20 * np.exp(np.cumsum(rets))}, index=pd.bdate_range("2025-01-01", periods=150))
    ret_1 = np.log(f["close"] / f["close"].shift(1)).to_numpy()
    out = make_labels(f, h)
    expected = C.K_ADAPT * np.std(ret_1[41:101], ddof=1) * math.sqrt(h)
    assert math.isclose(out["theta_used"].iloc[100], expected, rel_tol=1e-12)
    pre = make_labels(f.iloc[:110], h)
    pd.testing.assert_series_equal(pre["theta_used"], out["theta_used"].iloc[:110])


def test_boundary_and_sign():
    t = 0.02
    assert assign_labels(np.array([-t, t, t + 1e-9, -t - 1e-9]), t).tolist() == [1, 1, 2, 0]


def test_not_matured_is_na_never_flat():
    h = 5
    out = make_labels(_frame(120), h)
    tail = out.iloc[-h:]
    assert (tail["status"] == "not_matured").all()
    assert tail["label"].isna().all()
    assert tail["realized_return"].isna().all()
    assert tail["target_session"].isna().all()


def test_constant_price_gives_bad_sigma():
    f = pd.DataFrame({"close": np.full(120, 20.0)}, index=pd.bdate_range("2025-01-01", periods=120))
    out = make_labels(f, 2)
    assert (out["status"].iloc[60:-2] == "bad_sigma").all()
    assert out["label"].isna().all()


def test_label_version_and_columns():
    f = _frame(120)
    out = make_labels(f, 2)
    assert list(out.columns) == COLS
    assert out.index.name == "as_of_session"
    assert out.index.equals(f.index)
    assert (out["label_version"] == C.LABEL_VERSION).all()
    assert out["label_end_date"].equals(out["target_session"])
    assert str(out["label"].dtype) == "Int64"
    assert set(out["status"]) <= {"ok", "not_matured", "missing_session", "bad_sigma"}


def test_ok_labels_match_independent_computation():
    f = _frame(300, seed=1, gap=150)
    out = make_labels(f, 5)
    ok = out["status"] == "ok"
    r, th = out.loc[ok, "realized_return"].to_numpy(), out.loc[ok, "theta_used"].to_numpy()
    expected = [2 if x > t else 0 if x < -t else 1 for x, t in zip(r, th)]
    assert out.loc[ok, "label"].astype(int).tolist() == expected
    assert min(expected.count(c) for c in (0, 1, 2)) >= 10  # all three classes, non-trivially
    assert str(out["label"].dtype) == "Int64"
    assert out.loc[~ok, "label"].isna().all()
    assert {"not_matured", "missing_session", "bad_sigma"} <= set(out["status"])
