"""Price + BCTC against the same features plus news, on the magnitude target (absmove).

Rules: pre-registered entry of 2026-11-11 in docs/news/model_report.md (R6). Nothing here changes the live volcast.
Use: python -m ai_core.models.volcast.news_fusion --test-end 2026-10-30
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ai_core import paths
from ai_core.models.mmf import evaluate as E   # torch-free: the date block bootstrap and the registered verdict scale
from ai_core.news import storage
from ai_core.news.pipeline import load_results
from . import trial_forecast as tf
from . import volcast as V
from .walk_forward import FOLDS, _split

NEWS = tf.TEXT + tf.NEWS3
ARMS = {"price": V.FEATS, "fusion": V.FEATS + NEWS, "news": NEWS}
REPORT = paths.NEWS / "reports/news_fusion_report.txt"


def add_news(df, results, version):
    """df (add_features output) + NEWS at each session close: add_text, news_surprise on raw counts, then counts ranked per date."""
    d = pd.concat(tf.add_text(g.copy(), [r for r in results if r.symbol == s], "backtest", version) for s, g in df.groupby("symbol"))
    return tf.rank_cs(tf.add_news_surprise(d.loc[df.index]), tf.COUNTS)


def shuffle_news(d, seed):
    """Control: each date's NEWS vectors handed to another symbol of that date, kept whole; everything else untouched."""
    rng, src = np.random.default_rng(seed), np.arange(len(d))
    for i in d.groupby("date").indices.values():
        src[i] = rng.permutation(i)
    out = d.copy()
    out[NEWS] = d[NEWS].to_numpy()[src]
    return out


def fold_test(d, h, feats, dates):
    """(model kind, test frame date/symbol/y/p). Kind = lower validation log-loss when fit on train; refit on train + val."""
    y = f"y{h}"
    tr, va, trva, te = _split(d, h, *dates)
    val = {m: log_loss(va[y], mk().fit(tr[feats], tr[y]).predict_proba(va[feats])[:, 1], labels=[0, 1]) for m, mk in V.MODELS.items()}
    kind = min(val, key=val.get)
    p = V.MODELS[kind]().fit(trva[feats], trva[y]).predict_proba(te[feats])[:, 1]
    return kind, te[["date", "symbol"]].assign(y=te[y].to_numpy(), p=p)


def _row_ll(t):
    p = np.clip(t["p"].to_numpy(), 1e-12, 1 - 1e-12)
    return -(t["y"].to_numpy() * np.log(p) + (1 - t["y"].to_numpy()) * np.log(1 - p))


def r6(ds, h, folds=FOLDS, a="fusion", b="price"):
    """Paired test log-loss difference a - b pooled over the folds (negative favours a). ds: fold name -> frame with
    ARMS columns and y{h}; both arms are scored on the rows complete for every feature."""
    kinds, per_fold, parts = {a: {}, b: {}}, [], []
    for name, *dates in folds:
        d = ds[name].dropna(subset=[*V.FEATS, *NEWS, f"y{h}"])
        t = {}
        for arm in (a, b):
            kinds[arm][name], t[arm] = fold_test(d, h, ARMS[arm], dates)
        m = t[a][["date"]].assign(a=_row_ll(t[a]), b=_row_ll(t[b]))
        mean, lo, hi = E.paired_diff(m["a"], m["b"], m["date"])
        per_fold.append({"h": h, "fold": name, "mean": mean, "lo": lo, "hi": hi, "n": len(m), "n_days": m["date"].nunique(),
                         **{f"{k}_{arm}": f(t[arm]["y"], t[arm]["p"]) for arm in (a, b)
                            for k, f in (("logloss", lambda y, p: log_loss(y, p, labels=[0, 1])), ("auc", roc_auc_score), ("brier", brier_score_loss))},
                         f"kind_{a}": kinds[a][name], f"kind_{b}": kinds[b][name]})
        parts.append(m)
    m, per_fold = pd.concat(parts, ignore_index=True), pd.DataFrame(per_fold)
    mean, lo, hi = E.paired_diff(m["a"], m["b"], m["date"])
    wins = int((per_fold["mean"] < 0).sum())
    return {"h": h, "a": a, "b": b, "mean": mean, "lo": lo, "hi": hi, "fold_wins": wins, "n_folds": len(per_fold), "n": len(m),
            "n_days": int(m["date"].nunique()), "verdict": E.verdict(mean, hi, wins, len(per_fold)), "kinds": kinds, "per_fold": per_fold}


def datasets(c, symbols, folds=FOLDS):
    """fold name -> add_features frame with that fold's sentiment (phobert-fold<name>) and both targets."""
    df = V.add_features(V.bars(c, symbols=symbols))
    for h in V.HORIZONS:
        df[f"y{h}"] = tf.add_targets(df.assign(close=df.adj), h, "absmove")[f"y{h}"]
    return {name: add_news(df, load_results(c, f"phobert-fold{name}"), f"phobert-fold{name}") for name, *_ in folds}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-end", required=True, help="YYYY-MM-DD, end of fold C's test window (2026-10-30 in the registered run)")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    folds = [(*f[:4], a.test_end if f[0] == "C" else f[4]) for f in FOLDS]
    ds = datasets(storage.connect(), V.SYMBOLS, folds)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    lines = [f"news fusion on absmove, symbols={V.SYMBOLS}, test_end={a.test_end}, perms={a.perms}"]
    head = ["h", "a", "b", "mean", "lo", "hi", "fold_wins", "n_folds", "n", "n_days", "verdict"]
    main_rows, folds_rows, perm_rows, gate = [], [], [], []
    for h in V.HORIZONS:
        real = r6(ds, h, folds)
        main_rows.append({k: real[k] for k in head})
        folds_rows.append(real["per_fold"])
        news = r6(ds, h, folds, a="news")
        main_rows.append({k: news[k] for k in head} | {"verdict": "(báo cáo kèm)"})
        perm = [r6({k: shuffle_news(d, s) for k, d in ds.items()}, h, folds)["mean"] for s in range(a.perms)]
        perm_rows.append({"h": h, "real": real["mean"], "perm_mean": np.mean(perm), "perm_sd": np.std(perm, ddof=1) if len(perm) > 1 else np.nan,
                          "perm_min": min(perm, default=np.nan), "perm_max": max(perm, default=np.nan), "perm_at_or_below_real": sum(p <= real["mean"] for p in perm)})
        for arm in ("price", "fusion"):
            for f in folds:      # walk labels the frame itself, so it gets every row and drops the incomplete ones after
                gate += [{"arm": arm, **{k: r[k] for k in ("fold", "h", "model", "test_acc", "test_cov", "test_n", "rule_acc")}}
                         for r in V.walk(ds[f[0]], h, [f], feats=ARMS[arm])]
    for title, t in (("R6 (fusion - price) and the news-only arm", pd.DataFrame(main_rows)), ("per fold", pd.concat(folds_rows)),
                     ("permutation control (news shuffled between symbols within a date)", pd.DataFrame(perm_rows)),
                     ("Wilson gate, report only", pd.DataFrame(gate))):
        lines += ["", f"== {title} ==", t.round(4).to_string(index=False)]
    text = "\n".join(lines) + "\n"
    with open(a.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
