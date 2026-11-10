import numpy as np, pandas as pd, pytest
from ai_core.prices import config as PC
from ai_core.prices.data import build_panel, load_raw
from ai_core.models.mmf import config as C, frame as FR, windows as W
from ai_core.models.mmf.text import TEXT_FEATS

def no_text(version, symbols, sessions):
    idx = pd.MultiIndex.from_product([symbols, sessions], names=["symbol", "as_of_session"])
    out = pd.DataFrame(0.0, index=idx, columns=TEXT_FEATS).reset_index()
    return out

@pytest.fixture(scope="module")
def panel():
    return build_panel(PC.TICKERS, load_raw())

@pytest.fixture(scope="module")
def frame(panel):
    return FR.build_frame(panel, 5, "x", text_fn=no_text)

def test_usable_rows_start_2022_and_hl_equal_is_kept(frame):
    u = frame[frame.usable]
    assert u.as_of_session.min() >= pd.Timestamp(C.START)
    assert (u.status == "ok").all()
    assert frame.close_loc[frame.flag_hl_equal & (frame.as_of_session >= C.START)].eq(0.5).all()

def test_future_change_does_not_touch_past_features(panel):
    base = FR.build_frame(panel, 5, "x", text_fn=no_text)
    cut = pd.Timestamp("2025-06-02")
    panel2 = build_panel(PC.TICKERS, {k: v.assign(close=v.close.where(v.index <= cut, v.close * 2),
                                                  high=v.high.where(v.index <= cut, v.high * 2)) for k, v in load_raw().items()})
    other = FR.build_frame(panel2, 5, "x", text_fn=no_text)
    cols = [c for g in FR.GROUPS.values() for c in g]
    a, b = (f[f.as_of_session <= cut].sort_values(["symbol", "as_of_session"])[cols].reset_index(drop=True) for f in (base, other))
    pd.testing.assert_frame_equal(a, b)

def test_scaler_fits_on_given_rows_only(frame):
    tr = frame[frame.usable & (frame.as_of_session < "2024-01-01")]
    s = W.Scalers.fit(tr).apply(frame)
    got = s[s.index.isin(tr.index)]["ret_5"]
    assert abs(got.mean()) < 0.02 and abs(got.std() - 1) < 0.02

def test_window_with_missing_session_is_dropped(frame):
    f = frame.copy()
    hole = pd.Timestamp("2024-03-14")
    f.loc[(f.symbol == "FPT") & (f.as_of_session == hole), FR.GROUPS["price"]] = np.nan
    rows = f[f.usable & f.as_of_session.between("2024-03-01", "2024-04-30")]
    s = W.make_samples(W.Scalers.fit(f[f.usable]).apply(f), rows, PC.TICKERS)
    fpt = pd.to_datetime(s.as_of[s.symbol == "FPT"])
    cal = sorted(f.as_of_session.unique())
    blocked = set(cal[cal.index(hole):cal.index(hole) + C.LP])     # every window holding the missing session
    assert not blocked & set(fpt)
    assert (fpt < hole).any()                                      # windows before the hole survive
    assert np.isfinite(s.x_price).all()
    vcb = pd.to_datetime(s.as_of[s.symbol == "VCB"])
    assert hole in set(vcb)

def test_window_ends_at_sample_session(frame):
    f = frame[frame.symbol == "FPT"].sort_values("as_of_session")
    row = f[f.as_of_session == "2024-03-14"]
    s = W.make_samples(frame, row, PC.TICKERS)
    assert s.x_price.shape == (1, C.LP, len(FR.GROUPS["price"]))
    j = FR.GROUPS["price"].index("ret_5")
    assert s.x_price[0, -1, j] == pytest.approx(float(row.ret_5.iloc[0]))
    assert s.row_index[0] == row.index[0]

def test_real_retention_at_least_99_percent(frame):
    u = frame[frame.usable]
    s = W.make_samples(frame, u, PC.TICKERS)
    assert len(s) >= 0.99 * len(u)
    assert (s.row_index == u.index[np.isin(u.index, s.row_index)]).all()   # block order kept

def test_block_rows_only_selects_x_fund_comes_from_scaled_frame(frame):
    u = frame[frame.usable & frame.as_of_session.between("2024-03-01", "2024-04-30")]
    sc = W.Scalers.fit(frame[frame.usable]).apply(frame)
    a, b = (W.make_samples(sc, r, PC.TICKERS) for r in (u, sc.loc[u.index]))
    np.testing.assert_array_equal(a.x_fund, b.x_fund)
    assert np.abs(a.x_fund).max() <= W.CLIP
