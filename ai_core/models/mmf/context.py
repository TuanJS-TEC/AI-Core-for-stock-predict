"""Causal context features of the v2 variants: VNINDEX, market breadth, global series, cross-sectional rank.

Every value at session t uses rows <= t of the session calendar, and global bars dated strictly before t.
Rolling is trailing with min_periods = window. A cross-section with fewer than MIN_XS symbols gives NaN.
"""
import numpy as np
import pandas as pd

from .globaldata import asof_before

CTX_FEATS = ["ix_ret1", "ix_ret5", "ix_ret20", "ix_std20", "ix_dd60", "breadth_ma20", "adv_share",
             "spx_ret1", "spx_ret5", "vix_log", "vix_chg5", "brent_ret5", "usdvnd_ret5", "usdvnd_ret20",
             "rel_ix5", "rel_ix20", "beta60", "xs_rank5", "xs_rank20"]
MIN_XS = 30


def _ret(c, k):
    return np.log(c / c.shift(k))


def _share(hit, valid):
    """Row share of `hit` among `valid` cells; NaN below MIN_XS valid cells."""
    n = valid.sum(axis=1)
    return ((hit & valid).sum(axis=1) / n).where(n >= MIN_XS)


def _global(glob, cal):
    """Returns computed on each series' own bars, then joined as-of strictly before the session."""
    g = {name: d.set_index("date")["close"].sort_index() for name, d in glob.groupby("series")}
    spec = {"spx_ret1": ("spx", 1), "spx_ret5": ("spx", 5), "vix_chg5": ("vix", 5), "brent_ret5": ("brent", 5),
            "usdvnd_ret5": ("usdvnd", 5), "usdvnd_ret20": ("usdvnd", 20)}
    out = {k: asof_before(_ret(g[s], n), cal) for k, (s, n) in spec.items()}
    out["vix_log"] = asof_before(np.log(g["vix"]), cal)
    return pd.DataFrame(out, index=cal)


def context_features(frames, index, glob):
    """Long table symbol, as_of_session, *CTX_FEATS: one row per (symbol in frames, session of index.index)."""
    cal = index.index
    close = pd.DataFrame({s: f["close"] for s, f in frames.items()}).reindex(cal)
    close = close.where(close > 0)
    r = {k: _ret(close, k) for k in (1, 5, 20)}
    ic = index["close"]
    ix = {k: _ret(ic, k) for k in (1, 5, 20)}
    ma20 = close.rolling(20, min_periods=20).mean()
    market = pd.DataFrame({"ix_ret1": ix[1], "ix_ret5": ix[5], "ix_ret20": ix[20],
                           "ix_std20": ix[1].rolling(20, min_periods=20).std(),
                           "ix_dd60": ic / ic.rolling(60, min_periods=60).max() - 1,
                           "breadth_ma20": _share(close > ma20, ma20.notna() & close.notna()),
                           "adv_share": _share(r[1] > 0, r[1].notna())}, index=cal).join(_global(glob, cal))
    var = ix[1].rolling(60, min_periods=60).var()
    beta = r[1].rolling(60, min_periods=60).cov(ix[1]).div(var.where(var > 0), axis=0)
    rank = {k: r[k].rank(axis=1, pct=True).where(r[k].notna().sum(axis=1) >= MIN_XS, axis=0) for k in (5, 20)}
    parts = {}
    for s in frames:
        parts[s] = market.assign(rel_ix5=r[5][s] - ix[5], rel_ix20=r[20][s] - ix[20], beta60=beta[s],
                                 xs_rank5=rank[5][s], xs_rank20=rank[20][s])[CTX_FEATS]
    out = pd.concat(parts, names=["symbol", "as_of_session"]).replace([np.inf, -np.inf], np.nan)
    return out.reset_index()
