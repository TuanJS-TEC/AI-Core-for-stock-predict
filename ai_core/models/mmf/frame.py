"""Joined (symbol, session) frame: price features and labels, point-in-time financials, per-session text features."""
import numpy as np
import pandas as pd

from . import config as C
from . import fundamentals, text
from ai_core.prices.dataset import build_dataset
from ai_core.prices.labels import make_mag_labels, make_xs_labels
from ai_core.prices.price_features import FEATURES

GROUPS = {"price": FEATURES, "fund": fundamentals.FUND_FEATS, "text": text.TEXT_FEATS}
KEY = ["symbol", "as_of_session"]
LABEL_COLS = ["label", "status", "label_end_date", "realized_return"]


def build_frame(panel, h, nlp_version, text_fn=text.text_features, label="adaptive"):
    """One row per (symbol, calendar session), usable or not. usable = status ok, session >= START, price group finite.

    label "xs" replaces LABEL_COLS with the market-neutral labels (prereg addendum 2026-11-11), "mag" with the magnitude
    labels (addendum 2026-11-12: 0 = Yếu, 1 = Mạnh, class 2 never occurs); features are untouched."""
    if label not in ("adaptive", "xs", "mag"):
        raise ValueError(f"unknown label kind: {label}")
    df = build_dataset(panel, h)
    df["symbol"] = df["symbol"].astype(str)
    if label != "adaptive":
        xs = pd.concat({"xs": make_xs_labels, "mag": make_mag_labels}[label]({s: panel.on_calendar(s) for s in sorted(panel.ohlcv)}, h), names=KEY).reset_index()
        df = df.drop(columns=LABEL_COLS).merge(xs[[*KEY, *LABEL_COLS]], on=KEY, how="left", validate="one_to_one")
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
