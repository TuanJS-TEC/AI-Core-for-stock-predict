import numpy as np
import pandas as pd

from ai_core.models.volcast import trial_forecast as tf
from ai_core.models.volcast.walk_forward import PRICE, RANK_COLS, _split, forecast_fold


def test_fold_windows_are_purged_and_disjoint():
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    rng = np.random.default_rng(0)
    d = pd.concat(pd.DataFrame({"date": dates, "symbol": s}) for s in ("AAA", "BBB")).reset_index(drop=True)
    for col in PRICE + tf.TEXT:
        d[col] = rng.normal(size=len(d))
    h = 5
    d[f"y{h}"] = rng.integers(0, 2, len(d)).astype(float)
    ds = {rk: (tf.rank_cs(d, cols) if cols else d) for rk, cols in RANK_COLS.items()}
    res, pick, n_tr, n_va, n_te = forecast_fold(ds, h, "2022-01-01", "2024-01-01", "2024-07-01", "2025-01-01")
    i_val = list(dates).index(pd.Timestamp("2024-01-01"))
    i_test = list(dates).index(pd.Timestamp("2024-07-01"))
    assert n_tr == 2 * (i_val - h)              # last h sessions before validation dropped
    assert n_va == 2 * (i_test - h - i_val)     # last h sessions before test dropped from validation too
    assert n_te == 2 * (len(dates) - i_test)
    assert pick in res and pick in ("price", "text/none", "text/counts", "text/all")
    assert set(res) == {"price", "text/none", "text/counts", "text/all"}   # rank mode is part of the val search


def test_split_order_ignores_rows_outside_the_fold():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"date": np.repeat(pd.bdate_range("2022-01-03", periods=300), 3), "symbol": ["A", "B", "C"] * 300})
    for col in PRICE + ["y2"]:
        d[col] = rng.normal(size=len(d))
    more = pd.concat([d, d.tail(30).assign(date=lambda x: x.date + pd.Timedelta(days=400))])
    folds = ("2022-01-01", "2022-06-01", "2022-09-01", "2023-01-01")
    for a, b in zip(_split(d.sample(frac=1, random_state=0), 2, *folds), _split(more, 2, *folds)):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_split_purge_overrides_h():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=400)
    d = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * 400})
    for col in PRICE + ["y2"]:
        d[col] = rng.normal(size=len(d))
    tr, va, *_ = _split(d, 2, "2022-01-01", "2022-09-01", "2023-01-01", "2023-06-01", purge=3)
    gap = list(dates).index(va.date.min()) - list(dates).index(tr.date.max())
    assert gap - 1 == 3   # sessions dropped between last train day and first val day == purge
    tr0, *_ = _split(d, 2, "2022-01-01", "2022-09-01", "2023-01-01", "2023-06-01")
    assert list(dates).index(va.date.min()) - list(dates).index(tr0.date.max()) - 1 == 2   # default still h
