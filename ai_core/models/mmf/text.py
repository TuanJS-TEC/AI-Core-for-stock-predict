"""Per-session news-sentiment features from fold-specific PhoBERT scores (nlp.db), cut off at each session close."""
import argparse
import dataclasses
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from nlp_module import pipeline
from nlp_module.contract import build_snapshot
from nlp_module.forecast.trial_forecast import CLOSE_UTC

TEXT_FEATS = ["has_news", "news_count_24h", "news_count_72h", "unique_source_count", "hours_since_last_news",
              "difficult_ratio", "direct_news_ratio", "risk_flag_72h", "sentiment_mean", "sentiment_std",
              "sentiment_mean_24h", "sentiment_abs_max", "sentiment_shock_24h", "sentiment_decay"]
COUNTS = ["news_count_24h", "news_count_72h", "unique_source_count", "risk_flag_72h"]   # log1p
MAX_HOURS = 240.0


def features_from_results(results, symbols, sessions, version):
    """-> symbol, as_of_session, *TEXT_FEATS. A missing value is 0 (hours since last news: 1.0), never NaN."""
    # uncertain is ignored: abstention is off, and the flag must not drop a score the model did produce
    results = [dataclasses.replace(r, uncertain=False) for r in results]
    rows = []
    for s in symbols:
        mine = [r for r in results if r.symbol == s]   # build_snapshot scans its input per cutoff
        for d in sessions:
            cut = datetime.combine(d.date(), CLOSE_UTC, tzinfo=timezone.utc)
            f = build_snapshot(mine, s, cut, "backtest", version).features
            rows.append({"symbol": s, "as_of_session": d, **{k: f[k] for k in TEXT_FEATS}})
    df = pd.DataFrame(rows)
    h = df["hours_since_last_news"].astype(float)
    df["hours_since_last_news"] = (h.clip(upper=MAX_HOURS) / MAX_HOURS).fillna(1.0)
    df[COUNTS] = np.log1p(df[COUNTS].astype(float))
    rest = [c for c in TEXT_FEATS if c != "hours_since_last_news"]
    df[rest] = df[rest].astype(float).fillna(0.0)
    return df[["symbol", "as_of_session", *TEXT_FEATS]]


def text_features(version, symbols, sessions):
    """Cached features for symbols x sessions; the cache key holds the result count, so new scores invalidate it."""
    sessions = pd.DatetimeIndex(sessions)
    c = sqlite3.connect(f"file:{C.NLP_DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    results = pipeline.load_results(c, version)
    c.close()
    path = C.DATA / f"text_{version}_{len(results)}_{sessions.max():%Y%m%d}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["symbol"] = df["symbol"].astype(str)
        if len(df) == len(symbols) * len(sessions) and set(df["symbol"]) == set(symbols):
            return df
    df = features_from_results(results, symbols, sessions, version)
    C.DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def score_missing(version, model_path):
    """Score the articles this model has not seen yet (the only write to nlp.db). Run from NLP_Module/."""
    from nlp_module import storage
    from nlp_module.entities import Linker, load_aliases
    from nlp_module.sentiment.phobert import PhoBert
    model, linker = PhoBert(str(model_path), version), Linker(load_aliases())
    return pipeline.process(storage.connect(), model, linker, abstain_entropy=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"]); ap.add_argument("--version", required=True); ap.add_argument("--path", required=True)
    a = ap.parse_args()
    print(f"{a.version}: {score_missing(a.version, a.path)} results written")
