"""Tradable direction labels and the modelling frame. Entry is the next session's open (the signal is known at the
15:00 close), exit is the close h sessions after that: log(close[t+h+1] / open[t+1]). The label is the cross-sectional
excess of that return against a cost-aware band."""
import numpy as np
import pandas as pd

from . import config as C   # first: puts NLP_Module on sys.path
from .prices import load_prices
from nlp_module.forecast import fundamentals, volcast  # noqa: E402

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


def build_frame(h: int) -> pd.DataFrame:
    df = volcast.add_features(load_prices(), fundamentals.bctc_features(C.NLP / "data/bctc.parquet"))
    for k in (1, 5, 20):
        df[f"exret{k}"] = df[f"ret{k}"] - df.groupby("date")[f"ret{k}"].transform("mean")
    return add_labels(df, h)
