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


# --- nhãn trung hòa thị trường "xs" (phụ lục 2026-11-11) ---
from ai_core.prices.labels import make_xs_labels  # noqa: E402

XS_SYMS = ["A", "B", "C", "D"]


def _xs_rets(n=80, seed=0):
    return np.random.default_rng(seed).normal(0, 0.01, (n, len(XS_SYMS)))


def _xs_frames(rets):
    idx = pd.bdate_range("2025-01-01", periods=len(rets))
    return {s: pd.DataFrame({"close": 20 * np.exp(np.cumsum(rets[:, i]))}, index=idx) for i, s in enumerate(XS_SYMS)}


def test_xs_columns_and_formula():
    h, t = 5, 65
    rets = _xs_rets()
    out = make_xs_labels(_xs_frames(rets), h)
    assert sorted(out) == XS_SYMS
    a = out["A"]
    assert list(a.columns) == COLS and a.index.name == "as_of_session"
    assert (a["label_end_date"].dropna() == a["target_session"].dropna()).all()
    assert (a["label_version"] == "xs-k0.43-w60-v1").all()
    e = rets - np.median(rets, axis=1, keepdims=True)
    assert math.isclose(a["theta_used"].iloc[t], 0.43 * np.std(e[t - 59:t + 1, 0], ddof=1) * math.sqrt(h), rel_tol=1e-9)
    assert math.isclose(a["realized_return"].iloc[t], e[t + 1:t + h + 1, 0].sum(), rel_tol=1e-9, abs_tol=1e-12)
    assert a["status"].iloc[t] == "ok" and a["status"].iloc[59] == "bad_sigma"
    assert (a["status"].iloc[-h:] == "not_matured").all() and a["label"].iloc[-h:].isna().all()


def test_xs_market_shift_changes_no_label():
    rets = _xs_rets()
    shift = np.random.default_rng(1).normal(0, 0.03, (len(rets), 1))     # cùng một cú sốc cho mọi mã mỗi phiên
    a, b = make_xs_labels(_xs_frames(rets), 2), make_xs_labels(_xs_frames(rets + shift), 2)
    for s in XS_SYMS:
        pd.testing.assert_series_equal(a[s]["label"], b[s]["label"])
        pd.testing.assert_series_equal(a[s]["status"], b[s]["status"])
    assert a["A"]["label"].notna().sum() > 10


def test_xs_no_lookahead():
    h, t = 5, 68
    rets = _xs_rets()
    base = make_xs_labels(_xs_frames(rets), h)
    late = rets.copy(); late[t + h + 1:] += 0.05 * np.arange(len(XS_SYMS))   # đổi giá sau t+h, khác nhau theo mã
    other = make_xs_labels(_xs_frames(late), h)
    nxt = rets.copy(); nxt[t + 1:] += 0.05 * np.arange(len(XS_SYMS))         # đổi từ hàng t+1
    theta = make_xs_labels(_xs_frames(nxt), h)
    for s in XS_SYMS:
        pd.testing.assert_series_equal(base[s]["label"].iloc[:t + 1], other[s]["label"].iloc[:t + 1])
        pd.testing.assert_series_equal(base[s]["theta_used"].iloc[:t + 1], theta[s]["theta_used"].iloc[:t + 1])


def test_xs_missing_close_of_one_symbol_blocks_every_symbol():
    h, g = 2, 70
    frames = _xs_frames(_xs_rets())
    frames["B"].iloc[g, 0] = np.nan
    out = make_xs_labels(frames, h)
    for s in XS_SYMS:
        assert (out[s]["status"].iloc[g - h:g + 1] == "missing_session").all()
        assert out[s]["label"].iloc[g - h:g + 1].isna().all()
        assert out[s]["status"].iloc[g - h - 1] == "ok"
        assert (out[s]["status"].iloc[g + 1:-h] != "ok").all()               # cửa sổ 60 phiên chứa phiên thiếu: không ok


def test_xs_up_and_down_roughly_balanced():
    out = make_xs_labels(_xs_frames(_xs_rets(n=600, seed=3)), 5)
    lab = pd.concat([o["label"] for o in out.values()]).dropna().astype(int)
    up, down = (lab == 2).mean(), (lab == 0).mean()
    assert up > 0.15 and down > 0.15 and abs(up - down) < 0.1


# --- nhãn độ lớn biến động "mag" (phụ lục 2026-11-12) ---
from ai_core.prices.labels import make_mag_labels  # noqa: E402

MAG_SYMS = ["A", "B", "C", "D", "E"]


def _mag_frames(rets):
    idx = pd.bdate_range("2025-01-01", periods=len(rets))
    return {s: pd.DataFrame({"close": 20 * np.exp(np.cumsum(rets[:, i]))}, index=idx) for i, s in enumerate(MAG_SYMS)}


def _mag_rets(n=60, seed=0):
    return np.random.default_rng(seed).normal(0, 0.01, (n, len(MAG_SYMS)))


def test_mag_formula_and_base_rate():
    h, t = 2, 30
    rets = _mag_rets()
    out = make_mag_labels(_mag_frames(rets), h)
    a = out["A"]
    assert list(a.columns) == COLS and (a["label_version"] == "mag-median-v1").all()
    r = np.abs(rets[t + 1:t + h + 1].sum(axis=0))
    assert math.isclose(a["theta_used"].iloc[t], np.median(r), rel_tol=1e-9)
    assert [int(out[s]["label"].iloc[t]) for s in MAG_SYMS] == [int(x > np.median(r)) for x in r]
    assert sum(int(out[s]["label"].iloc[t]) for s in MAG_SYMS) == 2            # 5 mã: 2 Mạnh, mã ở trung vị là Yếu
    assert (a["status"].iloc[-h:] == "not_matured").all() and a["label"].iloc[-h:].isna().all()
    assert set(a["label"].dropna().unique()) <= {0, 1}


def test_mag_no_lookahead_and_missing_and_seam():
    h = 2
    rets = _mag_rets()
    base = make_mag_labels(_mag_frames(rets), h)
    late = rets.copy(); late[40:] += 0.02                                       # đổi tương lai sau phiên 39
    changed = make_mag_labels(_mag_frames(late), h)
    for s in MAG_SYMS:
        pd.testing.assert_series_equal(base[s]["label"].iloc[:38], changed[s]["label"].iloc[:38])
    fr = _mag_frames(rets); fr["B"].iloc[20, 0] = np.nan                        # B thiếu phiên 20
    miss = make_mag_labels(fr, h)
    assert (miss["A"]["status"].iloc[18:21] == "missing_session").all() and miss["A"]["status"].iloc[17] == "ok"
    jump = rets.copy(); jump[20, 2] = 0.2                                       # C nhảy 22% ở phiên 20
    sm = make_mag_labels(_mag_frames(jump), h)
    assert (sm["C"]["status"].iloc[18:20] == "seam").all() and sm["C"]["label"].iloc[18:20].isna().all()
    assert sm["A"]["status"].iloc[18] == "ok"
    r = np.abs(jump[19:21].sum(axis=0))[[0, 1, 3, 4]]                           # trung vị không tính C
    assert math.isclose(sm["A"]["theta_used"].iloc[18], np.median(r), rel_tol=1e-9)
