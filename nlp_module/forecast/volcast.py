"""Volatility forecaster over OHLCV: bar loading (history CSVs + live board quotes), price features and the annual
audited-statement features (fundamentals.BCTC; needs data/bctc.parquet, see README).
Prices are in thousand VND throughout (live quotes are VND, divided by 1000 on the way in)."""
import argparse
import glob
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import storage
from . import fundamentals
from . import trial_forecast as tf
from .trial_forecast import CLOSE_UTC
from .walk_forward import FOLDS, _split

FEATS = ["ret1", "ret5", "ret20", "vol10", "vol20", "ma20", "range", "vz", "mret1", "mret5"] + fundamentals.BCTC
VERSION = "volcast-v2.1"   # v1 = price only; v2 adds BCTC; v2.1 = Wilson-bound gate + purge/seam/sort fixes
BUNDLE = "data/models/volcast.joblib"
REPORT = "data/reports/volcast_report.json"
HORIZONS = (2, 5)
COLS = ["date", "symbol", "close", "high", "low", "volume"]
STOCK_DIR = Path(__file__).resolve().parents[3] / "Pipeline_V1.2" / "stock"   # the one canonical price source (the sync keeps it current)
SYMBOLS = "ACB FPT GAS HPG MBB MSN MWG SSI STB TCB VCB VIC VNM VPB VRE".split()   # live set; new files in STOCK_DIR do not change it
RET_BAND = 0.072   # exchange daily limit is 7% (+ price rounding); a bigger daily move means a corporate action or a calendar gap


def load_history(stock_dir=STOCK_DIR, symbols=None):
    """symbols=None reads every *_stock.csv in stock_dir."""
    parts = []
    for p in sorted(glob.glob(os.path.join(stock_dir, "*_stock.csv"))):
        if symbols is not None and os.path.basename(p).split("_")[0] not in symbols:
            continue
        d = pd.read_csv(p, parse_dates=["time"])
        d["date"] = d["time"].dt.normalize()
        parts.append(d.assign(symbol=os.path.basename(p).split("_")[0]))
    return pd.concat(parts, ignore_index=True)[COLS]


def live_bars(c):
    """Last reading per (symbol, trading_date) taken after the 15:00 close: earlier readings are intraday, and a
    mid-session high/low/volume would understate the day's range."""
    q = pd.read_sql("SELECT symbol, fetched_at, trading_date, ref_price, matched_price, highest, lowest, volume "
                    "FROM market_quote WHERE matched = 1", c)
    q["date"] = pd.to_datetime(q["trading_date"], format="%Y%m%d")
    t = pd.to_datetime(q["fetched_at"], utc=True).dt.tz_localize(None)
    q = q[t >= q["date"] + pd.Timedelta(hours=CLOSE_UTC.hour)].assign(t=t)
    q = q.sort_values("t").drop_duplicates(["symbol", "date"], keep="last")  # last ROW: groupby.last() would splice earlier non-nulls over NULLs
    ref = pd.to_numeric(q.ref_price, errors="coerce") / 1000
    q = q.assign(close=q.matched_price / 1000, high=q.highest / 1000, low=q.lowest / 1000, ref=ref.where(ref > 0))
    return q[COLS + ["ref"]].astype({"close": float, "high": float, "low": float, "volume": float, "ref": float})  # empty table reads as object


def bars(c, stock_dir=STOCK_DIR, symbols=None):
    """History wins on a shared (symbol, date): it is the vendor's final bar, live is our own last read."""
    # ponytail: sessions missed by both sources (22-23/10 at the seam, 28/10 machine off) are skipped, not reindexed: live
    # ret1 comes from the exchange ref so it stays a one-session return, but rolling windows hold fewer calendar days
    # and the skipped sessions' returns are lost from ret5/ret20/ma20. Upgrade: backfill the bars if a source appears.
    h, lv = load_history(stock_dir, symbols), live_bars(c)
    lv = lv[lv.symbol.isin(h.symbol.unique())]   # tickers outside the history universe would skew mret*, the label median, predictions
    df = pd.concat([h, lv], ignore_index=True)
    df = df.drop_duplicates(["symbol", "date"], keep="first")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_features(df, bctc=None):
    """Backward-looking only. Warm-up rows (first ~20 sessions per symbol) keep NaN features; callers drop them.
    bctc: fundamentals.bctc_features() output, loaded from data/bctc.parquet when None."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["ret1"] = df.groupby("symbol").close.pct_change(1)
    if "ref" in df:   # live rows: the exchange ref is the previous session's close adjusted for corporate actions
        live = df.ref > 0
        df.loc[live, "ret1"] = df.close[live] / df.ref[live] - 1
    df["adj"] = (1 + df.ret1.fillna(0)).groupby(df.symbol).cumprod()   # price index, continuous across live ex-dates
    g = df.groupby("symbol")
    df["ret5"], df["ret20"] = g.adj.pct_change(5), g.adj.pct_change(20)
    df["vol10"] = g.ret1.transform(lambda x: x.rolling(10).std())
    df["vol20"] = g.ret1.transform(lambda x: x.rolling(20).std())
    df["ma20"] = df.adj / g.adj.transform(lambda x: x.rolling(20).mean()) - 1
    df["range"] = (df.high - df.low) / df.close
    df["vz"] = np.log1p(df.volume) - g.volume.transform(lambda x: np.log1p(x).rolling(20).mean())
    df["mret1"] = df.groupby("date").ret1.transform("mean")
    df["mret5"] = df.groupby("date").ret5.transform("mean")
    return fundamentals.add_bctc(df, fundamentals.bctc_features() if bctc is None else bctc)


MODELS = {"logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
          "gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                        min_samples_leaf=100, random_state=0)}


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    cen, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return cen - h, cen + h


def pick_threshold(conf, ok, min_acc=0.65, min_n=50):
    """Smallest confidence c (= widest coverage) whose rows conf >= c have a Wilson 95% lower bound >= min_acc over at
    least min_n calls, else None. The point accuracy just reaching min_acc (v2) regressed to ~0.57 on test in fold C."""
    o = np.argsort(-conf, kind="stable")
    cs, k = conf[o], np.arange(1, len(conf) + 1)
    edge = np.r_[cs[1:] != cs[:-1], True]   # a threshold keeps every tied row, so only cut between distinct values
    lb = np.array([wilson(a, n)[0] for a, n in zip(np.cumsum(ok[o]), k)])
    good = np.flatnonzero(edge & (k >= min_n) & (lb >= min_acc))
    return float(cs[good[-1]]) if len(good) else None


def _rule_score(te):
    """Per-date ordinal rank centred as (rank - 0.5) / n, so the top and bottom ticker sit symmetrically around 0.5."""
    g = te.groupby("date").vol10
    return ((g.rank(method="first") - 0.5) / g.transform("count")).to_numpy()


def _rule_acc(te, n, y):
    """Baseline: no model, just the day's cross-sectional vol10 rank (high -> strong, low -> weak), most extreme n rows.
    Returns (accuracy, hits)."""
    if not n:
        return float("nan"), 0
    r = _rule_score(te)
    top = np.argsort(-np.abs(r - 0.5), kind="stable")[:int(n)]
    k = int(((r[top] > 0.5) == (te[y].to_numpy()[top] > 0.5)).sum())
    return k / int(n), k


def walk(df, h):
    """df = add_features output. Per fold: fit on train, pick the threshold on val, refit on train+val, score test once.
    Strong (y=1) means the ticker's |forward return| beats the day's cross-sectional median."""
    y = f"y{h}"
    d = tf.add_targets(df.assign(close=df.adj), h, "absmove").dropna(subset=FEATS)   # adj: labels across ex-dates too
    out = []
    for name, *dates in FOLDS:
        tr, va, trva, te = _split(d, h, *dates)
        best = None
        for m, mk in MODELS.items():
            pv = mk().fit(tr[FEATS], tr[y]).predict_proba(va[FEATS])[:, 1]
            cv, okv = np.abs(pv - 0.5), (pv > 0.5) == (va[y].to_numpy() > 0.5)
            t = pick_threshold(cv, okv)
            if t is not None and (best is None or okv[cv >= t].mean() > best[1]):
                best = (m, okv[cv >= t].mean(), t, (cv >= t).mean())
        row = dict(fold=name, h=h, model=None, threshold=None, val_acc=np.nan, val_cov=np.nan, test_acc=np.nan,
                   test_cov=np.nan, test_n=0, ci_lo=np.nan, ci_hi=np.nan, rule_acc=np.nan,
                   rule_ci_lo=np.nan, rule_ci_hi=np.nan)
        if best:
            m, va_acc, t, va_cov = best
            pt = MODELS[m]().fit(trva[FEATS], trva[y]).predict_proba(te[FEATS])[:, 1]
            sel = np.abs(pt - 0.5) >= t
            n, k = int(sel.sum()), int((((pt > 0.5) == (te[y].to_numpy() > 0.5)) & sel).sum())
            ra, rk = _rule_acc(te, n, y)
            row.update(model=m, threshold=t, val_acc=float(va_acc), val_cov=float(va_cov), test_n=n,
                       test_cov=n / len(te), test_acc=k / n if n else np.nan, rule_acc=ra,
                       **dict(zip(("ci_lo", "ci_hi"), wilson(k, n))), **dict(zip(("rule_ci_lo", "rule_ci_hi"), wilson(rk, n))))
        out.append(row)
    return out


def fit_live(df, h, fold_row):
    """Refit fold C's winning model kind on every labelled row. No winner -> logit with threshold None (no calls)."""
    kind, y = fold_row["model"] or "logit", f"y{h}"
    d = tf.add_targets(df.assign(close=df.adj), h, "absmove").dropna(subset=FEATS + [y])
    return dict(model=MODELS[kind]().fit(d[FEATS], d[y]), threshold=fold_row["threshold"] if fold_row["model"] else None,
                kind=kind)


def predict_latest(df, bundle):
    """Per symbol, its own last row with complete features (no label needed), scored at each horizon."""
    last = df.dropna(subset=FEATS).groupby("symbol").tail(1).sort_values("symbol")
    out = []
    for h in HORIZONS:
        b = bundle[h]
        p = b["model"].predict_proba(last[FEATS])[:, 1]
        t = b["threshold"]
        call = [None if t is None or abs(x - 0.5) < t else "strong" if x > 0.5 else "weak" if x < 0.5 else None for x in p]
        out.append(pd.DataFrame({"symbol": last.symbol.to_numpy(), "base_date": last.date.dt.strftime("%Y-%m-%d").to_numpy(),
                                 "h": h, "p_strong": p, "call": call}))
    return pd.concat(out, ignore_index=True)


def _num(v):
    v = v.item() if isinstance(v, np.generic) else v
    return None if isinstance(v, float) and math.isnan(v) else v


def write_report(rows, path=REPORT):
    doc = {"version": VERSION, "created_at": storage.now(), "rows": [{k: _num(v) for k, v in r.items()} for r in rows]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def _walks(df):
    return {h: walk(df, h) for h in HORIZONS}


def _print(w):
    print("h fold model thr   val_acc val_cov test_acc test_cov test_n  CI95            rule  ruleCI95")
    for r in (r for h in HORIZONS for r in w[h]):
        f = lambda v, n=3: "  -  " if _num(v) is None else f"{v:.{n}f}"
        print(f"{r['h']} {r['fold']}    {str(r['model']):5} {f(r['threshold'])} {f(r['val_acc'])}   {f(r['val_cov'])}   "
              f"{f(r['test_acc'])}    {f(r['test_cov'])}    {r['test_n']:<6} [{f(r['ci_lo'])},{f(r['ci_hi'])}] {f(r['rule_acc'])} [{f(r['rule_ci_lo'])},{f(r['rule_ci_hi'])}]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="volcast")
    ap.add_argument("cmd", choices=["eval", "train", "predict"])
    ap.add_argument("--symbols", default="fixed", help="fixed = SYMBOLS, all = every file in STOCK_DIR, or A,B,...")
    a = ap.parse_args(argv)
    cmd = a.cmd
    syms = SYMBOLS if a.symbols == "fixed" else None if a.symbols == "all" else a.symbols.split(",")
    c = storage.connect()
    df = add_features(bars(c, symbols=syms))
    if cmd in ("eval", "train"):
        w = _walks(df)
        _print(w)
        if cmd == "train":   # eval only prints: writing the report here would desync it from the live bundle
            write_report([r for h in HORIZONS for r in w[h]])
            os.makedirs(os.path.dirname(BUNDLE), exist_ok=True)
            joblib.dump({"version": VERSION, **{h: fit_live(df, h, w[h][-1]) for h in HORIZONS}}, BUNDLE)
            print(f"wrote {BUNDLE}")
        return
    if not os.path.exists(BUNDLE):
        raise SystemExit(f"{BUNDLE} missing: run `python -m nlp_module.forecast.volcast train` first")
    out = predict_latest(df, joblib.load(BUNDLE))
    t = storage.now()
    c.executemany("INSERT OR REPLACE INTO vol_forecast VALUES (?,?,?,?,?,?,?)",
                  [(r.symbol, r.base_date, int(r.h), float(r.p_strong), r.call, VERSION, t) for r in out.itertuples()])
    c.commit()
    print(f"vol_forecast: wrote {len(out)} rows, {out.call.notna().sum()} calls")


if __name__ == "__main__":
    main()
