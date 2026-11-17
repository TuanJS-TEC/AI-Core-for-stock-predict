"""rangecast: an 80% price band and a volatility level for T+2 and T+5, from a pooled HAR regression.

sigma_hat = sqrt(exp(linear(HAR terms of the ticker and their mean over the tickers of the session))) forecasts the size
of the h-session move; the band is close * exp(z10 * sigma_hat) .. close * exp(z90 * sigma_hat), z10/z90 being empirical
quantiles of r / sigma_hat on rows the regression was not fit on, clipped to the exchange limit compounded over h.
Everything of session t uses OHLC up to the close of t. Prereg: addendum R11 of docs/reports/mmf_preregistration.md.
Use: python -m ai_core.models.rangecast.rangecast eval --stock-dir data/prices_2018/snapshots/<id>
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ai_core import paths
from ai_core.models.mmf import config as MC
from ai_core.models.mmf.evaluate import paired_diff
from ai_core.prices import config as PC
from ai_core.prices.data import build_panel
from ai_core.prices.labels import make_mag_labels
from ai_core.prices.splits import Fold, folds_from_config, split_samples

VERSION = "rangecast-v1"
HAR = ["har_rv_d", "har_rv_w", "har_rv_m", "har_pk_d", "har_pk_w", "har_pk_m", "har_gap_w"]
FEATS = HAR + [c + "_mkt" for c in HAR]
Q = (0.10, 0.90)
LIMIT = 0.07                         # HOSE daily price limit
EPS = 1e-8
RISK_WINDOW = 250; RISK_MIN = 60; RISK_HIGH = 70; RISK_LOW = 30
KEY = ["symbol", "as_of_session"]
HORIZONS = PC.HORIZONS


def _har_raw(f):
    """The HAR terms of one symbol; the formulas of ai_core.models.mmf.har._raw_har, kept here so this module needs no torch."""
    r2 = np.log(f["close"] / f["close"].shift(1)) ** 2
    pk = np.log(f["high"] / f["low"]) ** 2 / (4 * np.log(2))
    gap2 = np.log(f["open"] / f["close"].shift(1)) ** 2
    lg = lambda s, w: np.log(s.rolling(w, min_periods=w).mean() + EPS)
    out = pd.DataFrame({"har_rv_d": lg(r2, 1), "har_rv_w": lg(r2, 5), "har_rv_m": lg(r2, 22),
                        "har_pk_d": lg(pk, 1), "har_pk_w": lg(pk, 5), "har_pk_m": lg(pk, 22), "har_gap_w": lg(gap2, 5)})
    return out.replace([np.inf, -np.inf], np.nan)


def features(frames):
    """{symbol: OHLC frame on the calendar} -> FEATS indexed (symbol, as_of_session): raw HAR terms and, per session,
    their mean over the symbols with a value (the common level of the market)."""
    raw = pd.concat({s: _har_raw(f) for s, f in frames.items()}, names=KEY)
    mkt = raw.groupby(level="as_of_session").transform("mean")
    return raw.join(mkt.add_suffix("_mkt"))[FEATS]


def targets(frames, h):
    """r (h-session log return), status and label_end_date of the mag label rows, logrv = ln(sum of the h daily r^2 after t)."""
    lab = pd.concat(make_mag_labels(frames, h), names=KEY)
    rv = pd.concat({s: (np.log(f["close"] / f["close"].shift(1)) ** 2).rolling(h, min_periods=h).sum().shift(-h) for s, f in frames.items()}, names=KEY)
    return pd.DataFrame({"r": lab["realized_return"].astype(float), "status": lab["status"].astype(str),
                         "label_end_date": lab["label_end_date"], "logrv": np.log(rv + EPS)})


def naive_sigma(frames, h, window=20):
    return pd.concat({s: np.log(f["close"] / f["close"].shift(1)).rolling(window, min_periods=window).std() * np.sqrt(h)
                      for s, f in frames.items()}, names=KEY)


def fit(x, logrv):
    return LinearRegression().fit(np.asarray(x, dtype=float), np.asarray(logrv, dtype=float))


def sigma(model, x):
    return np.sqrt(np.exp(model.predict(np.asarray(x, dtype=float))))


def z_quantiles(r, sig):
    z = np.asarray(r, dtype=float) / np.asarray(sig, dtype=float)
    return float(np.quantile(z, Q[0])), float(np.quantile(z, Q[1]))


def band(sig, z, h):
    """(lo, hi) log-return bounds, kept on their own side of 0 and inside the limit compounded over h sessions."""
    sig = np.asarray(sig, dtype=float)
    lo = np.clip(min(z[0], -EPS) * sig, h * np.log(1 - LIMIT), -EPS)
    hi = np.clip(max(z[1], EPS) * sig, EPS, h * np.log(1 + LIMIT))
    return lo, hi


def pinball(y, q, tau):
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.maximum(tau * d, (tau - 1) * d)


def risk_level(sig_hist):
    """Level of the last value against the (up to RISK_WINDOW - 1) values before it; too little history reads as normal."""
    past = np.asarray(sig_hist, dtype=float)[-RISK_WINDOW:-1]
    past = past[np.isfinite(past)]
    if len(past) < RISK_MIN - 1:
        return "normal"
    pct = 100 * (past < sig_hist[-1]).mean()
    return "high" if pct >= RISK_HIGH else "low" if pct <= RISK_LOW else "normal"


def verdict(cov2, cov5, d2_hi, d5_hi):
    """Pre-registered reading: coverage of both h and the upper CI bound of pinball(har) - pinball(naive20)."""
    cov = (cov2, cov5)
    if not all(0.75 <= c <= 0.85 for c in cov):
        return "Không đạt"
    return "Dùng được" if all(0.77 <= c <= 0.83 for c in cov) and d2_hi < 0 and d5_hi < 0 else "Dùng được, có lưu ý"


# --- evaluation on the frozen folds (R11) ---------------------------------------------------------------------------
ARMS = ("har", "naive20", "hist")


def frames_of(panel):
    return {s: panel.on_calendar(s) for s in sorted(panel.ohlcv)}


def frame(fr, h):
    """{symbol: OHLC frame} -> one row per (symbol, session) with FEATS, the targets and the naive sigma;
    usable = ok label and finite inputs."""
    d = features(fr).join(targets(fr, h)).join(naive_sigma(fr, h).rename("naive")).reset_index()
    d["usable"] = (d["status"] == "ok") & np.isfinite(d[[*FEATS, "naive", "logrv", "r"]].to_numpy(dtype=float)).all(axis=1)
    return d


def _levels(d, sig):
    """Risk level per row from the symbol's own running sigma_hat (rows in session order)."""
    s = pd.Series(sig, index=d.index)
    return s.groupby(d["symbol"].to_numpy()).transform(
        lambda x: x.rolling(RISK_WINDOW, min_periods=1).apply(lambda w: {"low": 0, "normal": 1, "high": 2}[risk_level(w)], raw=True))


def eval_fold(d, h, fold):
    """Eval rows of one fold with each arm's band: columns arm, lo, hi (log-return bounds) beside r and the keys."""
    b = split_samples(d[d["usable"]], fold)
    tr = pd.concat([b["train"], b["tune"]])
    model = fit(tr[FEATS], tr["logrv"])
    out = []
    z = z_quantiles(b["calib"]["r"], sigma(model, b["calib"][FEATS]))
    lo, hi = band(sigma(model, b["eval"][FEATS]), z, h)
    ok = d[np.isfinite(d[FEATS].to_numpy(dtype=float)).all(axis=1)].sort_values(KEY)
    lev = _levels(ok, sigma(model, ok[FEATS])).reindex(b["eval"].index)
    out.append(b["eval"][[*KEY, "r"]].assign(arm="har", lo=lo, hi=hi, level=lev.map({0: "low", 1: "normal", 2: "high"})))
    z = z_quantiles(b["calib"]["r"], b["calib"]["naive"])
    lo, hi = band(b["eval"]["naive"], z, h)
    out.append(b["eval"][[*KEY, "r"]].assign(arm="naive20", lo=lo, hi=hi, level="all"))
    q = tr.groupby("symbol")["r"].quantile(list(Q)).unstack()
    e = b["eval"]
    out.append(e[[*KEY, "r"]].assign(arm="hist", lo=np.clip(e["symbol"].map(q[Q[0]]).to_numpy(), h * np.log(1 - LIMIT), -EPS),
                                     hi=np.clip(e["symbol"].map(q[Q[1]]).to_numpy(), EPS, h * np.log(1 + LIMIT)), level="all"))
    return pd.concat(out, ignore_index=True).assign(h=h, fold=fold.name)


def _score(g):
    r = g["r"].to_numpy()
    return {"n": len(g), "n_days": g["as_of_session"].nunique(), "coverage": ((r >= g["lo"]) & (r <= g["hi"])).mean(), "below": (r < g["lo"]).mean(),
            "above": (r > g["hi"]).mean(), "width_pct": 100 * (np.exp(g["hi"]) - np.exp(g["lo"])).mean(), "pinball": g["pb"].mean()}


def evaluate(panel):
    folds = [*folds_from_config(), *(Fold(n, *map(pd.Timestamp, e)) for n, *e in MC.BACK_FOLDS)]
    fr = frames_of(panel)
    p = pd.concat([eval_fold(frame(fr, h), h, f) for h in HORIZONS for f in folds], ignore_index=True)
    p["pb"] = pinball(p["r"], p["lo"], Q[0]) + pinball(p["r"], p["hi"], Q[1])
    pooled = [{"h": h, "arm": a, **_score(g)} for (h, a), g in p.groupby(["h", "arm"])]
    by_fold = [{"h": h, "fold": f, "arm": a, **_score(g)} for (h, f, a), g in p.groupby(["h", "fold", "arm"])]
    har = p[p["arm"] == "har"]
    by_symbol = [{"h": h, "symbol": s, **_score(g)} for (h, s), g in har.groupby(["h", "symbol"])]
    by_level = [{"h": h, "level": lv, **_score(g)} for (h, lv), g in har.groupby(["h", "level"])]
    diffs = []
    for h in HORIZONS:
        a = har[har["h"] == h].sort_values([*KEY]).reset_index(drop=True)
        for other in ("naive20", "hist"):
            o = p[(p["arm"] == other) & (p["h"] == h)].sort_values([*KEY]).reset_index(drop=True)
            if not a[KEY].equals(o[KEY]):
                raise ValueError(f"h={h}: har and {other} eval rows differ")
            mean, lo, hi = paired_diff(a["pb"], o["pb"], a["as_of_session"], n=2000)
            wins = sum(a["pb"][a["fold"] == f].mean() < o["pb"][o["fold"] == f].mean() for f in a["fold"].unique())
            diffs.append({"h": h, "so_sanh": f"har - {other}", "mean": mean, "lo": lo, "hi": hi, "fold_wins": f"{wins}/{a['fold'].nunique()}"})
    cov = {r["h"]: r["coverage"] for r in pooled if r["arm"] == "har"}
    hi = {r["h"]: r["hi"] for r in diffs if r["so_sanh"] == "har - naive20"}
    return {"pooled": pooled, "by_fold": by_fold, "by_symbol": by_symbol, "by_level": by_level, "diffs": diffs,
            "verdict": verdict(cov[2], cov[5], hi[2], hi[5]), "dataset_id": panel.dataset_id}


def report(res):
    md = lambda rows: pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")
    return (f"# R11: dải giá 80% của rangecast, 6 fold (vùng phát triển)\n\nDataset: {res['dataset_id']}\n\n**Kết luận theo luật: {res['verdict']}**\n\n"
            f"## Gộp 6 fold\n\n{md(res['pooled'])}\n\n## Pinball ghép cặp (âm là har tốt hơn)\n\n{md(res['diffs'])}\n\n"
            f"## har theo mức rủi ro\n\n{md(res['by_level'])}\n\n## har theo mã\n\n{md(res['by_symbol'])}\n\n## Theo fold\n\n{md(res['by_fold'])}\n")


# --- live: bundle and daily rows --------------------------------------------------------------------------------------
BUNDLE = paths.RANGECAST / "rangecast.joblib"
HOLD = 126                           # newest sessions kept out of the fit; the two quantiles come from them
SEAM_SESSIONS = 23                   # the monthly HAR window plus the return into it
TRAIN_HINT = "run `.venv/bin/python -m ai_core.models.rangecast.rangecast train` first"


def train_bundle(fr):
    """{"version", h: {"model", "z"}}: regression on every labelled row before the last HOLD sessions, z from those."""
    out = {"version": VERSION}
    for h in HORIZONS:
        d = frame(fr, h)
        d = d[d["usable"]]
        cut = np.sort(d["as_of_session"].unique())[-HOLD]
        old, new = d[d["as_of_session"] < cut], d[d["as_of_session"] >= cut]
        model = fit(old[FEATS], old["logrv"])
        out[h] = {"model": model, "z": z_quantiles(new["r"], sigma(model, new[FEATS]))}
    return out


def load_bundle(path=BUNDLE):
    import joblib
    try:
        b = joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(f"{path} missing: {TRAIN_HINT} (rangecast train)") from None
    if b.get("version") != VERSION:
        raise SystemExit(f"{path} is {b.get('version')}, this code is {VERSION}: {TRAIN_HINT} (rangecast train)")
    return b


def predict_rows(fr, bundle, created_at=None):
    """range_forecast tuples for the newest session of the frames. A symbol whose HAR window has a missing bar or a
    daily move above the sync band (a re-based series) gets no row: its band would be built on a broken history."""
    from datetime import datetime, timezone
    created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    x = features(fr)
    out = []
    for s, f in fr.items():
        base = f.index[-1]
        mine = x.loc[s]
        seam = (f["close"].pct_change(fill_method=None).tail(SEAM_SESSIONS).abs() > PC.SYNC_RET_BAND).any()
        if seam or not np.isfinite(mine.loc[base].to_numpy(dtype=float)).all():
            continue
        hist = mine[np.isfinite(mine.to_numpy(dtype=float)).all(axis=1)].tail(RISK_WINDOW)
        for h in HORIZONS:
            sig = sigma(bundle[h]["model"], hist[FEATS])
            lo, hi = band(sig[-1:], bundle[h]["z"], h)
            out.append((s, str(pd.Timestamp(base).date()), h, float(f["close"].iloc[-1]), float(np.expm1(lo[0])), float(np.expm1(hi[0])),
                        float(sig[-1]), risk_level(sig), VERSION, created_at))
    return out


def write_rows(c, rows):
    """Insert the rows not stored yet (a stored forecast is never rewritten); returns how many were added."""
    n = c.total_changes
    c.executemany("INSERT OR IGNORE INTO range_forecast VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    return c.total_changes - n


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rangecast", description=__doc__)
    ap.add_argument("cmd", choices=["eval", "train", "predict"])
    ap.add_argument("--stock-dir")
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        if not a.stock_dir:
            ap.error("eval needs --stock-dir (a frozen snapshot)")
        out = report(evaluate(build_panel(stock_dir=a.stock_dir)))
        paths.RANGECAST.mkdir(parents=True, exist_ok=True)
        (paths.RANGECAST / "eval.md").write_text(out)
        print(out)
        return
    fr = frames_of(build_panel(stock_dir=a.stock_dir))       # default: the live price store
    if a.cmd == "train":
        import joblib
        paths.RANGECAST.mkdir(parents=True, exist_ok=True)
        joblib.dump(train_bundle(fr), BUNDLE)
        print(f"wrote {BUNDLE}")
        return
    from ai_core.news import storage
    rows = predict_rows(fr, load_bundle())
    print(f"range_forecast: {write_rows(storage.connect(), rows)} rows added of {len(rows)} ({VERSION}, base {rows[0][1] if rows else '-'})")


if __name__ == "__main__":
    main()
