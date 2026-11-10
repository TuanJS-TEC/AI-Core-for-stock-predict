import numpy as np, pandas as pd
from pandas.testing import assert_frame_equal
from mmf import context as X
from src.price_features import FEATURES

CAL = pd.bdate_range("2024-01-01", periods=300)
SERIES = ("spx", "vix", "brent", "usdvnd")


def walk(seed, n=len(CAL)):
    return 20 * np.exp(np.random.default_rng(seed).normal(0, 0.02, n).cumsum())


def world(n_sym=40, cal=CAL):
    frames = {}
    for i in range(n_sym):
        c = walk(i)[:len(cal)]
        frames[f"S{i:02d}"] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e5}, index=cal)
    ic = np.mean([f["close"] for f in frames.values()], axis=0)
    index = pd.DataFrame({"open": ic, "high": ic, "low": ic, "close": ic, "volume": 1e9}, index=cal)
    glob = pd.concat([pd.DataFrame({"series": s, "date": cal, "close": walk(100 + j)[:len(cal)]}) for j, s in enumerate(SERIES)], ignore_index=True)
    return frames, index, glob


def feats(frames, index, glob):
    return X.context_features(frames, index, glob).set_index(["symbol", "as_of_session"]).sort_index()


def test_names_and_count():
    assert len(X.CTX_FEATS) == 19 and len(set(X.CTX_FEATS)) == 19 and not set(X.CTX_FEATS) & set(FEATURES)
    out = X.context_features(*world())
    assert list(out.columns) == ["symbol", "as_of_session", *X.CTX_FEATS] and len(out) == 40 * len(CAL)
    assert np.isfinite(out[X.CTX_FEATS].to_numpy()[-40:]).all()          # the last session is fully defined


def test_prefix_invariance():
    frames, index, glob = world()
    cut = CAL[:200]
    short = feats({s: f.loc[cut] for s, f in frames.items()}, index.loc[cut], glob[glob.date <= cut[-1]])
    full = feats(frames, index, glob)
    assert_frame_equal(short, full[full.index.get_level_values(1) <= cut[-1]])


def test_future_perturbation():
    frames, index, glob = world()
    t = CAL[200]
    bump = lambda f: f.where(pd.Series(f.index <= t, index=f.index), f * 1.7, axis=0)
    g2 = glob.assign(close=glob.close.where(glob.date <= t, glob.close * 1.7))
    a, b = feats(frames, index, glob), feats({s: bump(f) for s, f in frames.items()}, bump(index), g2)
    keep = a.index.get_level_values(1) <= t
    assert_frame_equal(a[keep], b[keep])
    assert not a[~keep].equals(b[~keep])


def test_same_day_global_bar_is_not_used():
    frames, index, glob = world()
    t, nxt = CAL[250], CAL[251]
    g2 = glob.assign(close=glob.close.where(~((glob.series == "spx") & (glob.date == t)), glob.close * 1.5))
    a, b = feats(frames, index, glob), feats(frames, index, g2)
    assert a.loc[("S00", t), "spx_ret1"] == b.loc[("S00", t), "spx_ret1"]
    assert a.loc[("S00", nxt), "spx_ret1"] != b.loc[("S00", nxt), "spx_ret1"]


def test_thin_cross_section_is_nan():
    xs = ["breadth_ma20", "adv_share", "xs_rank5", "xs_rank20"]
    assert X.context_features(*world(29))[xs].isna().all().all()
    r = X.context_features(*world(30))["xs_rank5"].dropna()
    assert len(r) and r.gt(0).all() and r.le(1).all()


def test_missing_session_stays_nan():
    frames, index, glob = world()
    g = CAL[150]
    frames["S03"] = frames["S03"].copy()
    frames["S03"].loc[g] = np.nan
    out = feats(frames, index, glob)
    assert np.isnan(out.loc[("S03", g), "rel_ix5"]) and np.isnan(out.loc[("S03", g), "xs_rank5"])
    assert np.isnan(out.loc[("S03", CAL[151]), "beta60"])                 # the window still holds the gap
    assert np.isfinite(out.loc[("S04", g), X.CTX_FEATS].to_numpy(dtype=float)).all()


def test_no_inf():
    frames, index, glob = world()
    frames["S05"] = frames["S05"].assign(close=frames["S05"].close.where(frames["S05"].index != CAL[100], 0.0))
    v = X.context_features(frames, index, glob)[X.CTX_FEATS].to_numpy(dtype=float)
    assert not np.isinf(v).any()
