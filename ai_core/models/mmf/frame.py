"""Joined (symbol, session) frame: price features and labels, point-in-time financials, per-session text features."""
import numpy as np
import pandas as pd

from . import config as C
from . import fundamentals, text
from src.dataset import build_dataset
from src.price_features import FEATURES

GROUPS = {"price": FEATURES, "fund": fundamentals.FUND_FEATS, "text": text.TEXT_FEATS}
KEY = ["symbol", "as_of_session"]
LABEL_COLS = ["label", "status", "label_end_date", "realized_return"]


def build_frame(panel, h, nlp_version, text_fn=text.text_features):
    """One row per (symbol, calendar session), usable or not. usable = status ok, session >= START, price group finite."""
    df = build_dataset(panel, h)
    df["symbol"] = df["symbol"].astype(str)
    px = pd.concat({s: panel.on_calendar(s)[["open", "close"]] for s in sorted(panel.ohlcv)}, names=KEY).reset_index()
    df = df.merge(px, on=KEY, how="left", validate="one_to_one")
    df.loc[df["flag_hl_equal"].astype(bool), "close_loc"] = 0.5      # limit up/down day, known at t
    df[GROUPS["fund"]] = fundamentals.add_fundamentals(df[[*KEY, "close"]])[GROUPS["fund"]]
    tx = text_fn(nlp_version, sorted(df["symbol"].unique()), panel.calendar)
    tx["symbol"] = tx["symbol"].astype(str)
    df = df.merge(tx[[*KEY, *GROUPS["text"]]], on=KEY, how="left", validate="one_to_one")
    df["usable"] = ((df["status"].astype(str) == "ok") & (df["as_of_session"] >= pd.Timestamp(C.START))
                    & np.isfinite(df[GROUPS["price"]].to_numpy(dtype=float)).all(axis=1))
    return df[[*KEY, "open", "close", "flag_hl_equal", *GROUPS["price"], *GROUPS["fund"], *GROUPS["text"],
               *LABEL_COLS, "usable"]]
