"""Kiểm thử src/price_features.py: đặc trưng nhân quả trên bảng lịch phiên (dữ liệu tổng hợp)."""
import numpy as np
import pandas as pd

from src import price_features as F

N = 250
T = 150  # hàng "hiện tại" cho các test nhân quả


def make_frames(n=N, syms=("AAA", "BBB", "CCC"), seed=7):
    idx = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(seed)
    out = {}
    for s in syms:
        c = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.005, n))
        h = np.maximum(o, c) * np.exp(rng.uniform(0.001, 0.01, n))
        lo = np.minimum(o, c) * np.exp(-rng.uniform(0.001, 0.01, n))
        v = rng.integers(1_000, 100_000, n).astype(float)
        out[s] = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)
    return out


def run(frames, sym="AAA"):
    m = F.market_features(frames)
    return F.compute(frames[sym], m), m


def test_prefix_invariance():
    frames = make_frames()
    full, _ = run(frames)
    prefix, _ = run({s: f.iloc[: T + 1] for s, f in frames.items()})
    t = full.index[T]
    pd.testing.assert_frame_equal(full.loc[:t], prefix.loc[:t])


def test_future_perturbation():
    frames = make_frames()
    base, mb = run(frames)
    bumped = {s: f.copy() for s, f in frames.items()}
    for f in bumped.values():
        f.iloc[T + 1 :] = f.iloc[T + 1 :] * 1.7
    pert, mp = run(bumped)
    pd.testing.assert_frame_equal(base.iloc[: T + 1], pert.iloc[: T + 1])
    pd.testing.assert_frame_equal(mb.iloc[: T + 1], mp.iloc[: T + 1])


def test_scale_invariance():
    frames = make_frames()
    base, _ = run(frames)
    scaled = {s: f.assign(**{c: f[c] * 7.3 for c in ("open", "high", "low", "close")}) for s, f in frames.items()}
    sc, _ = run(scaled)
    np.testing.assert_allclose(
        sc[F.FEATURES].to_numpy(), base[F.FEATURES].to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_equal_high_low_flagged_not_inf():
    frames = make_frames()
    g = 100
    f = frames["AAA"]
    f.iloc[g, f.columns.get_loc("high")] = f.iloc[g]["close"]
    f.iloc[g, f.columns.get_loc("low")] = f.iloc[g]["close"]
    out, _ = run(frames)
    assert np.isnan(out["close_loc"].iloc[g])
    assert out["flag_hl_equal"].iloc[g]
    assert out["flag_hl_equal"].sum() == 1
    assert not np.isinf(out.to_numpy(dtype=float)).any()


def test_missing_session_row_is_nan_not_filled():
    frames = make_frames()
    g = 100
    frames["AAA"].iloc[g] = np.nan
    out, _ = run(frames)
    assert out["ret_1"].iloc[[g, g + 1]].isna().all()
    assert out["ret_1"].iloc[g + 2 :].notna().all()
    assert out["std_20"].iloc[g : g + 21].isna().all()
    assert out["std_20"].iloc[g + 21 :].notna().all()
    # ret_60 chỉ dùng hai đầu mút (t và t-60): NaN đúng tại g và g+60; cửa sổ trượt 60 thì NaN tới g+59
    assert out["ret_60"].iloc[[g, g + 60]].isna().all()
    assert out["ret_60"].iloc[g + 1 : g + 60].notna().all()
    assert out["ret_60"].iloc[g + 61 :].notna().all()
    assert out["d_ma60"].iloc[g : g + 60].isna().all()
    assert out["d_ma60"].iloc[g + 60 :].notna().all()
    assert not out["flag_hl_equal"].iloc[g]


def test_volume_nan_stays_nan_zero_volume_is_valid():
    frames = make_frames()
    f = frames["AAA"]
    f.iloc[100, f.columns.get_loc("volume")] = np.nan
    f.iloc[200, f.columns.get_loc("volume")] = 0.0
    out, _ = run(frames)
    assert np.isnan(out["v_ma20"].iloc[100])
    assert np.isfinite(out["logv_rel60"].iloc[200])
    assert out["v_ma20"].iloc[200] == 0.0


def test_feature_count_and_names():
    assert len(F.FEATURES) == 25
    assert len(set(F.FEATURES)) == 25
    out, m = run(make_frames())
    assert list(out.columns) == F.FEATURES + ["flag_hl_equal"]
    assert list(m.columns) == ["mret1", "mret5", "mvol20"]
