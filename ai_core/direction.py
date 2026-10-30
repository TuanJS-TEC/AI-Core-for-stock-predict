"""Tradable direction labels and the modelling frame. Entry is the next session's open (the signal is known at the
15:00 close), exit is the close h sessions after that: log(close[t+h+1] / open[t+1]). The label is the cross-sectional
excess of that return against a cost-aware band. Models (E3 logit, E4 gbm) are picked on val, LONG-only, scored once
on test; go() is the pre-registered GO/NO-GO rule (spec Bước 2)."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C   # first: puts NLP_Module and Pipeline_V1.2 on sys.path
from . import splits
from .prices import load_prices
from nlp_module.forecast import fundamentals, volcast  # noqa: E402
from src import config as PC  # noqa: E402  (Pipeline_V1.2: bootstrap block/n/level)
from src.metrics import block_bootstrap_dates  # noqa: E402

EXRET = ["exret1", "exret5", "exret20"]
FEATS = [f for f in volcast.FEATS if f not in fundamentals.BCTC] + EXRET


def add_labels(df: pd.DataFrame, h: int, cost: float = C.COST, kappa: float = C.KAPPA) -> pd.DataFrame:
    """Needs date, symbol, open, close, vol20. y is NaN (never 0) where vol20 or the h+1 future sessions are missing."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    df[f"r_trade{h}"] = np.log(g.close.shift(-(h + 1)) / g.open.shift(-1))
    df[f"r_ex{h}"] = df[f"r_trade{h}"] - df.groupby("date")[f"r_trade{h}"].transform("mean")
    df[f"tau{h}"] = np.maximum(cost, kappa * df.vol20 * np.sqrt(h))   # np.maximum propagates NaN from the warm-up
    r, tau = df[f"r_ex{h}"], df[f"tau{h}"]
    df[f"y{h}"] = np.where(r > tau, 1.0, np.where(r < -tau, -1.0, 0.0))
    df.loc[r.isna() | tau.isna(), f"y{h}"] = np.nan
    return df


def build_frame(h: int, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """prices: load_prices() output (the 20-ticker universe when None). Warm-up rows with NaN features are dropped,
    as volcast.walk does, since logit cannot take NaN."""
    prices = load_prices() if prices is None else prices
    df = volcast.add_features(prices, fundamentals.bctc_features(C.NLP / "data/bctc.parquet"))
    for k in (1, 5, 20):
        df[f"exret{k}"] = df[f"ret{k}"] - df.groupby("date")[f"ret{k}"].transform("mean")
    return add_labels(df, h).dropna(subset=FEATS).reset_index(drop=True)


CS = (0.01, 0.1, 1.0)
MODELS = {"logit": lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)),   # E3, multinomial
          "gbm": lambda c: volcast.MODELS["gbm"]()}                                                     # E4
GRID = [("logit", c) for c in CS] + [("gbm", None)]
MS = np.round(np.arange(0.0, 0.51, 0.02), 2)
MIN_N, N_RAND = 50, 1000


def folds(df, h):
    """splits.folds with a 1 + h session purge: the label reads close[t+1+h]."""
    return splits.folds(df, h, purge=h + 1)


def _fit(d, h, kind, c):
    return CalibratedClassifierCV(MODELS[kind](c), method="sigmoid", cv=3).fit(d[FEATS], d[f"y{h}"])


def _probs(model, d):
    """(P(UP), P(DOWN))."""
    p, k = model.predict_proba(d[FEATS]), list(model.classes_)
    return p[:, k.index(1.0)], p[:, k.index(-1.0)]


def long_mask(p_up, p_down, m) -> np.ndarray:
    return np.asarray(p_up) - np.asarray(p_down) >= m


def select_on_val(tr, va, h) -> dict:
    """Fit + calibrate each (kind, C) on train; pick (kind, C, m) with the best mean net per LONG trade on val over
    >= MIN_N trades. Ties keep the first in GRID x MS order. kind=None when nothing reaches MIN_N."""
    net = va[f"r_trade{h}"].to_numpy() - C.COST
    best = {"kind": None, "C": None, "m": None, "val_net": float("nan"), "val_n": 0}
    for kind, c in GRID:
        up, down = _probs(_fit(tr, h, kind, c), va)
        for m in MS:
            k = long_mask(up, down, m)
            if k.sum() >= MIN_N and (best["kind"] is None or net[k].mean() > best["val_net"]):
                best = {"kind": kind, "C": c, "m": float(m), "val_net": float(net[k].mean()), "val_n": int(k.sum())}
    return best


def boot_ci(daily, rng) -> tuple[float, float]:
    """CI of the mean net excess per LONG trade: whole test dates resampled in blocks (Pipeline_V1.2 settings). daily
    has one row per test date (s = summed net_ex of that day's trades, k = their count); concat folds' to pool."""
    s, k = daily.s.to_numpy(), daily.k.to_numpy()
    stats = [s[i].sum() / k[i].sum() for i in block_bootstrap_dates(daily.index.to_numpy(), PC.BOOTSTRAP_BLOCK,
                                                                   PC.BOOTSTRAP_N, rng) if k[i].sum()]
    a = (1 - PC.CI_LEVEL) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def run_fold(tr, va, trva, te, h, rng) -> dict:
    """Select on val, refit the pick on train+val, score test once. net = mean(r_trade - COST) per LONG trade, net_ex
    the same on r_ex. e0_prior (E0) = net of going LONG on every test row. turnover and max_dd are of the daily
    equal-weight portfolio of that day's LONG trades; a trade spans h+1 sessions, so h+1 staggered sleeves each hold
    1/(h+1) of capital. A fold without a pick or without test trades has NaN metrics (go() counts it as a loss)."""
    sel = select_on_val(tr, va, h)
    te = te.assign(net=te[f"r_trade{h}"] - C.COST, net_ex=te[f"r_ex{h}"] - C.COST)
    nan = float("nan")
    out = {**sel, "n": 0, "hit": nan, "net": nan, "net_ex": nan, "ci_lo": nan, "ci_hi": nan, "rand_median": nan,
           "turnover": nan, "max_dd": nan, "e0_prior": float(te.net.mean()), "daily": None}
    if sel["kind"] is None:
        return out
    k = long_mask(*_probs(_fit(trva, h, sel["kind"], sel["C"]), te), sel["m"])
    n = int(k.sum())
    if not n:
        return {**out, "n": 0}
    t, all_net = te[k], te.net.to_numpy()
    daily = te.assign(k=k.astype(int), s=np.where(k, te.net_ex, 0.0), p=np.where(k, te.net, 0.0)) \
        .groupby("date")[["k", "s", "p"]].sum()
    eq = np.r_[0.0, np.cumsum((daily.p / daily.k.where(daily.k > 0, 1)) / (h + 1))]
    w = te.assign(w=k.astype(float)).pivot_table(index="date", columns="symbol", values="w", fill_value=0.0)
    w = w.div(w.sum(axis=1).where(lambda x: x > 0, 1), axis=0)
    lo, hi = boot_ci(daily, rng)
    return {**out, "n": n, "hit": float((t.net > 0).mean()), "net": float(t.net.mean()), "net_ex": float(t.net_ex.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "rand_median": float(np.median([all_net[rng.choice(len(te), n, replace=False)].mean() for _ in range(N_RAND)])),
            "turnover": float(0.5 * w.diff().abs().sum(axis=1).iloc[1:].mean()),
            "max_dd": float((np.maximum.accumulate(eq) - eq).max()), "daily": daily}


def go(rows: list[dict], pooled_ci_lo: float) -> bool:
    """Pre-registered (spec Bước 2), per h: net > 0 in >= 2/3 folds AND net > rand_median in >= 2/3 folds AND the
    pooled 3-fold CI lower bound of net_ex > 0. A fold with kind=None is a loss."""
    ok = [r for r in rows if r["kind"] is not None]
    wins = lambda f: 3 * sum(bool(f(r)) for r in ok) >= 2 * len(rows)
    return wins(lambda r: r["net"] > 0) and wins(lambda r: r["net"] > r["rand_median"]) and pooled_ci_lo > 0
