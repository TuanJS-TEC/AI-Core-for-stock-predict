"""Train-fitted standardisation and leak-safe sample windows (price window of LP sessions, text window of LT)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .frame import GROUPS

BINARY = ["has_news", "has_bctc", "is_bank"]
CLIP = 5.0
COLS = [c for g in GROUPS.values() for c in g if c not in BINARY]


@dataclass
class Scalers:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, train_rows, cols=None):
        """Mean and std per column (default COLS) over the rows given (non-finite values skipped)."""
        x = train_rows[COLS if cols is None else list(cols)].replace([np.inf, -np.inf], np.nan)
        return cls(x.mean(), x.std().replace(0.0, 1.0).fillna(1.0))

    def apply(self, frame):
        """Standardise, clip to [-5, 5]; NaN stays NaN so make_samples can drop the window."""
        out, cols = frame.copy(), list(self.mean.index)
        out[cols] = ((frame[cols] - self.mean) / self.std).clip(-CLIP, CLIP)
        return out


@dataclass
class Samples:
    x_price: np.ndarray      # (N, LP, Fp) float32
    x_text: np.ndarray       # (N, LT, Ft)
    x_fund: np.ndarray       # (N, Ff), the sample session only
    ticker: np.ndarray       # (N,) int64 index into tickers
    y: np.ndarray            # (N,) int64, -1 when unlabelled
    symbol: np.ndarray
    as_of: np.ndarray
    label_end: np.ndarray
    row_index: np.ndarray = None    # frame index of each kept row

    def __len__(self):
        return len(self.y)


def make_samples(scaled_frame, block_rows, tickers, lp=C.LP, lt=C.LT, price_cols=None):
    """Windows end at the sample session. A window with a missing session or any non-finite value is dropped, never filled.

    price_cols: columns of the price window (default GROUPS["price"]; the ctx variant adds the context features)."""
    f, price_cols = scaled_frame, GROUPS["price"] if price_cols is None else list(price_cols)
    cal = pd.DatetimeIndex(np.sort(f["as_of_session"].unique()))
    parts = {k: [] for k in ("xp", "xt", "xf", "ticker", "idx")}
    for t, sym in enumerate(tickers):
        rows = block_rows[block_rows["symbol"].astype(str) == sym]
        if rows.empty:
            continue
        mine = f[f["symbol"].astype(str) == sym].set_index("as_of_session")
        mine = mine[~mine.index.duplicated()].reindex(cal)             # sessions on the calendar, NaN where absent
        pos = cal.get_indexer(rows["as_of_session"])
        keep = pos >= max(lp, lt) - 1                                  # windows must fit inside the calendar
        rows, pos = rows[keep], pos[keep]
        xp = mine[price_cols].to_numpy(np.float32)[pos[:, None] + np.arange(1 - lp, 1)]
        xt = mine[GROUPS["text"]].to_numpy(np.float32)[pos[:, None] + np.arange(1 - lt, 1)]
        xf = mine[GROUPS["fund"]].to_numpy(np.float32)[pos]            # scaled frame only; block_rows just selects
        ok = (np.isfinite(xp).all((1, 2)) & np.isfinite(xt).all((1, 2)) & np.isfinite(xf).all(1))
        for k, v in zip(parts, (xp, xt, xf, np.full(len(rows), t), rows.index.to_numpy())):
            parts[k].append(v[ok])
    cat = lambda k, shape: np.concatenate(parts[k]) if parts[k] else np.empty(shape)
    xp, xt = cat("xp", (0, lp, len(price_cols))), cat("xt", (0, lt, len(GROUPS["text"])))
    xf, tk, idx = cat("xf", (0, len(GROUPS["fund"]))), cat("ticker", (0,)).astype(np.int64), cat("idx", (0,))
    order = np.argsort(pd.Index(block_rows.index).get_indexer(idx), kind="stable")   # keep block_rows order
    xp, xt, xf, tk, idx = xp[order], xt[order], xf[order], tk[order], idx[order]
    kept = f.loc[idx]
    return Samples(xp.astype(np.float32), xt.astype(np.float32), xf.astype(np.float32), tk,
                   kept["label"].fillna(-1).to_numpy(np.int64), kept["symbol"].astype(str).to_numpy(),
                   kept["as_of_session"].to_numpy(), kept["label_end_date"].to_numpy(), idx)
