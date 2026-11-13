"""Direct Text->Return regression: TF-IDF + Ridge maps each direct article to its ticker's forward return relative to
the day's cross-section, the mean prediction over 24h becomes a daily feature, and a forecaster trained on the
validation window (where Ridge predictions are out-of-sample) is scored once on test, per walk-forward fold.
python -m ai_core.models.volcast.text_return --target absmove|excess"""
import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_core import paths
from . import trial_forecast as tf
from ai_core.news.contract import BACKTEST_DELAY
from .walk_forward import FOLDS, PRICE, _split

CLOSE = pd.Timedelta(hours=tf.CLOSE_UTC.hour, minutes=tf.CLOSE_UTC.minute)
RIDGE = ["ridge_24h", "has_ridge"]
HS = (2, 5)


def assign_sessions(published, dates):
    """First session whose close (date + CLOSE_UTC) is at/after published + BACKTEST_DELAY (naive UTC); NaT if none."""
    dates = pd.DatetimeIndex(sorted(dates))
    i = (dates + CLOSE).searchsorted(published + BACKTEST_DELAY, side="left")
    return pd.Series(dates.append(pd.DatetimeIndex([pd.NaT]))[np.minimum(i, len(dates))], index=published.index)


def article_labels(articles, prices, h, target):
    """Adds session and y: the ticker's r = close[t+h]/close[t] - 1 minus the day's cross-sectional mean r (excess)
    or |r| minus the day's median |r| (absmove). Articles without a session or a t+h price are dropped."""
    p = prices.sort_values(["symbol", "date"])
    r = p.groupby("symbol")["close"].shift(-h) / p["close"] - 1
    if target == "excess":
        y = r - r.groupby(p["date"]).transform("mean")
    elif target == "absmove":
        y = r.abs() - r.abs().groupby(p["date"]).transform("median")
    else:
        raise ValueError(f"unknown target {target!r}")
    lab = pd.DataFrame({"session": p["date"], "symbol": p["symbol"], "y": y}).dropna()
    a = articles.assign(session=assign_sessions(articles["published_at"], p["date"].unique()))
    return a.merge(lab, on=["session", "symbol"], how="inner")


def train_articles(labeled, dates, val_start, h, train_start=None):
    """Session in [train_start, val_start) AND position(session) + h < position(first session >= val_start), so no
    training label window (t -> t+h) reaches into validation."""
    dates = pd.DatetimeIndex(sorted(dates))
    iv = dates.searchsorted(pd.Timestamp(val_start))
    keep = dates.get_indexer(labeled["session"]) + h < iv
    if train_start is not None:
        keep &= (labeled["session"] >= pd.Timestamp(train_start)).values
    return labeled[keep]


def texts(a):
    return ("[" + a["symbol"] + "] " + a["passage_text"]).tolist()


def fit_ridge(texts, y):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), Ridge(alpha=10.0)).fit(texts, y)


def daily_feature(articles_with_pred, df):
    """ridge_24h = mean pred of the symbol's articles with published_at + BACKTEST_DELAY in (cutoff - 24h, cutoff],
    cutoff = date + CLOSE_UTC; has_ridge 0/1. Imputation (explicit): no article => ridge_24h 0, has_ridge 0."""
    out = df.copy()
    s, n = np.zeros(len(df)), np.zeros(len(df))
    cut = (df["date"] + CLOSE).values
    for sym, g in articles_with_pred.groupby("symbol"):
        g = g.assign(_t=g["published_at"] + BACKTEST_DELAY).sort_values("_t")
        t, cs = g["_t"].values, np.concatenate([[0.0], np.cumsum(g["pred"].values)])
        m = (df["symbol"] == sym).values
        hi = np.searchsorted(t, cut[m], side="right")
        lo = np.searchsorted(t, cut[m] - np.timedelta64(24, "h"), side="right")
        s[m], n[m] = cs[hi] - cs[lo], hi - lo
    out["has_ridge"] = (n > 0).astype(int)
    out["ridge_24h"] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return out


def load_articles(db=paths.NEWS / "nlp.db", symbols=()):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)   # read-only
    a = pd.read_sql("""SELECT r.symbol, v.published_at, r.passage_text FROM article_company_result r
                       JOIN document_version v USING(document_version_id)
                       WHERE r.nlp_model_version='baseline-tfidf-lr-v1' AND r.relation='direct'
                         AND v.is_representative=1 AND v.published_at IS NOT NULL""", c)
    c.close()
    a = a[a.symbol.isin(set(symbols))].copy()
    a["published_at"] = pd.to_datetime([datetime.fromisoformat(p.replace("Z", "")) for p in a.published_at])
    return a.reset_index(drop=True)


def _lr_auc(va, te, cols, h):
    sc = StandardScaler().fit(va[cols])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(va[cols]), va[f"y{h}"])
    return roc_auc_score(te[f"y{h}"], m.predict_proba(sc.transform(te[cols]))[:, 1])


def run(target, prices_path=paths.NEWS / "prices.csv", db=paths.NEWS / "nlp.db"):
    prices = tf.load_prices(prices_path)
    dates = pd.DatetimeIndex(sorted(prices.date.unique()))
    arts = load_articles(db, prices.symbol.unique())
    rows = []
    for h in HS:
        lab = article_labels(arts, prices, h, target)
        d0 = tf.add_targets(prices.copy(), h, target)
        d0 = d0[d0.date >= "2021-10-21"]   # first collected news day, as walk_forward
        for name, t0, v0, s0, e0 in FOLDS:
            tr = train_articles(lab, dates, v0, h, train_start=t0)
            model = fit_ridge(texts(tr), tr["y"])
            d = daily_feature(arts.assign(pred=model.predict(texts(arts))), d0)
            _, va, _, te = _split(d, h, t0, v0, s0, e0)
            tn = te[te.has_ridge == 1]
            p, pr = _lr_auc(va, te, PRICE, h), _lr_auc(va, te, PRICE + RIDGE, h)
            rows.append({"fold": name, "h": h, "n_train_articles": len(tr), "n_val": len(va), "n_test": len(te),
                         "auc_price": p, "auc_price_ridge": pr, "diff": pr - p,
                         "auc_ridge_uni": roc_auc_score(tn[f"y{h}"], tn["ridge_24h"]), "n_test_has_ridge": len(tn)})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows), len(arts)


def verdict(out):
    """Pre-registered: diff > 0 in >= 2 of 3 folds at BOTH h=2 and h=5, AND mean diff over the 6 cells > 0.005."""
    wins = {h: int((out[out.h == h]["diff"] > 0).sum()) for h in HS}
    mean = float(out["diff"].mean())
    ok = all(w >= 2 for w in wins.values()) and mean > 0.005
    return wins, mean, ok


def report(target, out, n_articles):
    wins, mean, ok = verdict(out)
    return "\n".join([
        f"== target: {target} ==",
        f"direct articles (baseline-tfidf-lr-v1, representative, priced tickers): {n_articles}",
        out.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "rule: diff > 0 in >= 2/3 folds at both h=2 and h=5, and mean diff over 6 cells > 0.005",
        f"folds with diff > 0: h=2 {wins[2]}/3, h=5 {wins[5]}/3; mean diff {mean:+.4f}",
        "conclusion: " + ("Text->Return ADDS VALUE" if ok else "Text->Return does NOT add value"), ""])


def write_block(path, target, text):
    """Replace this target's block in the report file, keep the other target's."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = p.read_text().split("== target: ")[1:] if p.exists() else []
    keep = ["== target: " + b for b in old if not b.startswith(f"{target} ==")]
    p.write_text("\n".join(b.rstrip("\n") + "\n" for b in sorted(keep + [text])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["absmove", "excess"])
    ap.add_argument("--prices", default=str(paths.NEWS / "prices.csv"))
    ap.add_argument("--db", default=str(paths.NEWS / "nlp.db"))
    ap.add_argument("--out", default=str(paths.NEWS / "reports/text_return_report.txt"))
    a = ap.parse_args(argv)
    t = time.time()
    out, n = run(a.target, a.prices, a.db)
    text = report(a.target, out, n) + f"runtime {time.time() - t:.0f}s\n"
    print(text)
    write_block(a.out, a.target, text)
    return out


if __name__ == "__main__":
    main()
