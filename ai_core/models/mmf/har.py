"""R9 (prereg addendum, spec docs/superpowers/specs/2026-11-12-mag-har-events-design.md): a relative HAR baseline and a
point-in-time corporate-event calendar for the magnitude label "mag", as two nested logistic regressions.

Features of session t use OHLCV up to the close of t and events announced strictly before t (public_date < t).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as B, compare as CP, config as C, evaluate as E, experiment as X, mag as M
from ai_core.prices.data import build_panel
from ai_core.prices.events import load as load_events
from ai_core.prices.splits import Fold, folds_from_config

HAR_FEATS = ["har_rv_d", "har_rv_w", "har_rv_m", "har_pk_d", "har_pk_w", "har_pk_m", "har_gap_w"]
EV_FEATS = ["ev_exr_in_h", "ev_mtg_in_h", "ev_next", "ev_new5"]
EPS = 1e-8; NEXT_CAP = 20; NEW_WINDOW = 5
EXR_CODES = ("DIV", "ISS")           # event day = exright_date
MTG_CODES = ("AGME", "EGME")         # event day = display_date1, the meeting day (audit 2026-11-12)
KEY = ["symbol", "as_of_session"]


def _raw_har(f):
    """HAR features of one symbol before the cross-sectional demeaning (index = f.index)."""
    r2 = np.log(f["close"] / f["close"].shift(1)) ** 2
    pk = np.log(f["high"] / f["low"]) ** 2 / (4 * np.log(2))
    gap2 = np.log(f["open"] / f["close"].shift(1)) ** 2
    lg = lambda s, w: np.log(s.rolling(w, min_periods=w).mean() + EPS)
    out = pd.DataFrame({"har_rv_d": lg(r2, 1), "har_rv_w": lg(r2, 5), "har_rv_m": lg(r2, 22),
                        "har_pk_d": lg(pk, 1), "har_pk_w": lg(pk, 5), "har_pk_m": lg(pk, 22), "har_gap_w": lg(gap2, 5)})
    return out.replace([np.inf, -np.inf], np.nan)


def har_features(frames):
    """{symbol: OHLC frame on the calendar} -> HAR_FEATS indexed (symbol, as_of_session), minus each session's mean
    over the symbols with a finite value (the label ranks symbols within a session)."""
    raw = pd.concat({s: _raw_har(f) for s, f in frames.items()}, names=KEY)
    return raw - raw.groupby(level="as_of_session").transform("mean")


def event_features(events, calendar, symbols, h):
    """EV_FEATS for every (symbol, session of calendar); never NaN. Positions are counted on `calendar`;
    an event day that is not a session maps to the next session."""
    n = len(calendar)
    parts = {}
    for s in symbols:
        exr, mtg, new = np.zeros(n), np.zeros(n), np.zeros(n)
        nxt = np.full(n, float(NEXT_CAP))
        for e in events[events["symbol"] == s].itertuples():
            if pd.isna(e.public_date):
                continue
            kp = calendar.searchsorted(e.public_date, side="right")       # first session strictly after the announcement
            new[kp:kp + NEW_WINDOW] += 1
            day = e.exright_date if e.event_code in EXR_CODES else e.display_date1 if e.event_code in MTG_CODES else pd.NaT
            if pd.isna(day):
                continue
            ep = calendar.searchsorted(day, side="left")
            if ep >= n or ep <= kp:
                continue
            (exr if e.event_code in EXR_CODES else mtg)[max(kp, ep - h):ep] = 1
            t = np.arange(kp, ep)
            nxt[t] = np.minimum(nxt[t], ep - t)
        parts[s] = pd.DataFrame({"ev_exr_in_h": exr, "ev_mtg_in_h": mtg, "ev_next": np.minimum(nxt, NEXT_CAP) / NEXT_CAP,
                                 "ev_new5": new}, index=pd.Index(calendar, name="as_of_session"))
    return pd.concat(parts, names=KEY)


# --- the two arms ---------------------------------------------------------------------------------------------------
ARMS = {"logit_har": HAR_FEATS, "logit_har_ev": HAR_FEATS + EV_FEATS}
CLIP = 5.0
MIN_FINITE = 0.95                    # share of train rows with every HAR feature finite
OUTS = ("calib", "eval")


def _rows(b, k):
    r = b.blocks[k]
    return pd.MultiIndex.from_arrays([r["symbol"].astype(str), pd.to_datetime(r["as_of_session"])], names=KEY)


def _fit(b, extra, cols):
    """Logistic regression (baselines' logit settings) on train rows; scaler from train only, NaN -> 0.
    Only the HAR features are clipped to ±5: a rare event flag standardised sits above 5 and must keep its size."""
    x = {k: extra.reindex(_rows(b, k))[cols].to_numpy(float) for k in ("train", *OUTS)}
    har = [i for i, c in enumerate(cols) if c in HAR_FEATS]
    if np.isfinite(x["train"][:, har]).all(axis=1).mean() < MIN_FINITE:
        raise ValueError("HAR features missing on most train rows: features and bundle rows do not align")
    mu, sd = np.nanmean(x["train"], axis=0), np.nanstd(x["train"], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    z = {}
    for k, v in x.items():
        v = (v - mu) / sd
        v[:, har] = np.clip(v[:, har], -CLIP, CLIP)
        z[k] = np.nan_to_num(v)
    y = b.blocks["train"]["label"].to_numpy(float)
    ok = np.isfinite(y)
    return B._model("logit").fit(z["train"][ok], y[ok].astype(int)), z


def predict(b, extra, cols):
    """Probabilities (n, 3) for blocks calib and eval, in block row order. extra: indexed (symbol, as_of_session)."""
    clf, z = _fit(b, extra, cols)
    return {k: B._proba(clf, z[k]) for k in OUTS}


def derangement(symbols, seed):
    """{symbol: another symbol}, a random permutation without fixed points."""
    rng, syms = np.random.default_rng(seed), list(symbols)
    while True:
        perm = list(rng.permutation(syms))
        if all(a != b for a, b in zip(syms, perm)):
            return dict(zip(syms, map(str, perm)))


def _check_reference(b, ref_dir):
    """The eval rows of the bundle equal those of the reference run's prior_fixed (the mag run), before any fit."""
    path = Path(ref_dir) / f"h{b.h}" / b.fold.name / "prior_fixed" / "seed0.parquet"
    d = pd.read_parquet(path, columns=["symbol", "as_of_session", "block"])
    d = d[d["block"].astype(str) == "eval"]
    ref = set(zip(d["symbol"].astype(str), pd.to_datetime(d["as_of_session"])))
    got = set(zip(map(str, b.samples["eval"].symbol), pd.to_datetime(b.samples["eval"].as_of)))
    if ref != got:
        raise ValueError(f"h{b.h} {b.fold.name}: eval rows differ from the reference run {path} ({len(ref ^ got)} not shared)")


def _coef(clf, cols):
    """Coefficients of the strong class (binary: sklearn keeps one row, for class 1)."""
    cl = list(clf.classes_)
    if 1 not in cl:
        return {}
    w = clf.coef_[0] if len(cl) == 2 else clf.coef_[cl.index(1)]
    return dict(zip(cols, w.round(5).tolist()))


def run(panel, events, horizons, folds, run_dir, placebo_seeds=(), build=None, ref_dir=None):
    """Per (h, fold): logit_har and logit_har_ev, written like a baseline (alpha fit on calib), plus coef.json.
    placebo_seeds: instead of the two arms, for each seed the event table is handed to other symbols (derangement) and
    only logit_har_ev runs, as logit_har_ev_p{seed}; one bundle per (h, fold) serves every seed.
    ref_dir: a run whose prior_fixed eval rows must equal the bundle's, checked before any fit."""
    build = build or (lambda p, h, f: X.fold_bundle(p, h, f, label="mag"))
    syms = sorted(panel.ohlcv)
    if placebo_seeds:
        jobs = [(f"logit_har_ev_p{s}", ARMS["logit_har_ev"], events.assign(symbol=events["symbol"].map(derangement(syms, s))))
                for s in placebo_seeds]
    else:
        jobs = [(arm, cols, events) for arm, cols in ARMS.items()]
    har = har_features({s: panel.on_calendar(s) for s in syms})
    for h in horizons:
        extras = {}
        for fold in folds:
            b = build(panel, h, fold)
            if ref_dir is not None:
                _check_reference(b, ref_dir)
            for arm, cols, ev in jobs:
                path = Path(run_dir) / f"h{h}" / fold.name / arm / "seed0.parquet"
                if X._reusable(path, b, "none"):
                    continue
                key = id(ev)
                if key not in extras:
                    extras[key] = har.join(event_features(ev, panel.calendar, syms, h), how="outer")
                clf, z = _fit(b, extras[key], cols)
                p = {k: B._proba(clf, z[k]) for k in OUTS}
                alpha = E.fit_alpha(b.samples["calib"].y, p["calib"], X._prior_rows(b, "calib"))
                X._write_preds(path, b, arm, 0, "none", p, alpha)
                (path.parent / "coef.json").write_text(json.dumps({"classes": clf.classes_.tolist(), "coef_strong": _coef(clf, cols)}, indent=1))
                print(f"h{h} {fold.name} {arm} alpha={alpha}", flush=True)


# --- R9 scoring -----------------------------------------------------------------------------------------------------
EXTRA = (("kèm: logit_har - prior_fixed", "logit_har", "prior_fixed"), ("kèm: logit_har_ev - prior_fixed", "logit_har_ev", "prior_fixed"),
         ("kèm: logit_har - dl_price", "logit_har", "dl_price"), ("kèm: logit_har_ev - dl_price", "logit_har_ev", "dl_price"),
         ("kèm: logit_har_ev - fusion_concat", "logit_har_ev", "fusion_concat"))


def check_rows(ph):
    """The eval (symbol, session) set of logit_har equals prior_fixed's in every fold; else the two runs do not pair."""
    ev = ph[ph["block"] == "eval"]
    for f in sorted(ev["fold"].unique()):
        g = ev[ev["fold"] == f]
        keys = {a: set(zip(d["symbol"], d["as_of_session"])) for a, d in g.groupby("arm") if a in ("logit_har", "prior_fixed")}
        if len(keys) != 2 or keys["logit_har"] != keys["prior_fixed"]:
            raise ValueError(f"fold {f}: eval rows of logit_har and prior_fixed differ or one is missing")


def r9(ph, h):
    """Registered R9a, R9b, then the reported-only pairs."""
    folds = sorted(ph["fold"].unique())
    same = lambda arm: M._same(arm, folds)
    return [M._row(ph, h, n, same(a), same(b)) for n, a, b in
            (("R9a", "logit_har_ev", "logit_har"), ("R9b", "logit_har", "prior_adaptive"), *EXTRA)]


def load_joint(har_dir, mag_dir, h, folds):
    ph = pd.concat([CP.load_h(mag_dir, h), CP.load_h(har_dir, h)], ignore_index=True)
    ph = M.two_class(ph[ph["fold"].isin(folds)])
    check_rows(ph)
    return ph


def placebo(ph, h):
    """Pooled R9a of the real events vs the same difference for each placebo arm."""
    folds = sorted(ph["fold"].unique())
    real = CP.compare(ph, M._same("logit_har_ev", folds), M._same("logit_har", folds))["mean"]
    fake = [CP.compare(ph, M._same(a, folds), M._same("logit_har", folds))["mean"]
            for a in sorted(ph["arm"].unique()) if a.startswith("logit_har_ev_p")]
    if not fake:
        return {"h": h, "real": real, "n_placebo": 0}
    q = np.quantile(fake, [0.05, 0.5, 0.95])
    return {"h": h, "real": real, "n_placebo": len(fake), "p05": q[0], "p50": q[1], "p95": q[2],
            "share_placebo_le_real": float(np.mean(np.array(fake) <= real))}


def coefs(har_dir, h, folds):
    out = []
    for f in folds:
        p = Path(har_dir) / f"h{h}" / f / "logit_har_ev" / "coef.json"
        if p.exists():
            c = json.loads(p.read_text())["coef_strong"]
            out.append({"h": h, "fold": f, **{k: c.get(k) for k in EV_FEATS}})
    return out


def main(argv=None):
    folds = {f.name: f for f in [*folds_from_config(), *(Fold(n, *map(pd.Timestamp, e)) for n, *e in C.BACK_FOLDS)]}
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-dir", required=True)
    ap.add_argument("--events", help="frozen event file, data/prices/events/<day>.parquet (needed to run)")
    ap.add_argument("--h", type=int, nargs="+", default=list(C.HORIZONS))
    ap.add_argument("--folds", nargs="+", choices=list(folds), default=["A", "B", "C", "D", "E", "Fb"])
    ap.add_argument("--placebo", type=int, default=0, help="run N placebo event tables (seeds 0..N-1) instead of the arms")
    ap.add_argument("--score", action="store_true", help="score R9 from the written predictions")
    ap.add_argument("--mag-dir", default=str(C.DATA / "exp/mag/24251acc1a5695bd"))
    ap.add_argument("--out", default=str(C.DATA / "exp/har/r9.md"))
    ap.add_argument("--start", help="first usable session instead of C.START (longer price history; not R9)")
    a = ap.parse_args(argv)
    C.START = a.start or C.START
    panel = build_panel(stock_dir=a.stock_dir)
    run_dir = C.DATA / "exp/har" / X._dataset_id(panel)[:16]
    if not a.score:
        if not a.events:
            ap.error("--events is needed to run the arms")
        run(panel, load_events(a.events), a.h, [folds[n] for n in a.folds], run_dir, placebo_seeds=tuple(range(a.placebo)),
            ref_dir=a.mag_dir)
        (run_dir / "events.json").write_text(json.dumps({"events": a.events, "sha256": hashlib.sha256(Path(a.events).read_bytes()).hexdigest()}))
        return
    main_rows, fold_rows, scores, plac, co = [], [], [], [], []
    for h in a.h:
        ph = load_joint(run_dir, a.mag_dir, h, a.folds)
        real = ph[~ph["arm"].str.startswith("logit_har_ev_p")]
        rows = r9(real, h)
        main_rows += rows; fold_rows += M.per_fold(real, h, rows); scores += M.arm_scores(real, h)
        plac.append(placebo(ph, h)); co += coefs(run_dir, h, a.folds)
    md = lambda rows, cols=None: pd.DataFrame(rows)[cols or list(rows[0])].to_markdown(index=False, floatfmt=".4f")
    auc = max(s["auc"] for s in scores if np.isfinite(s["auc"]))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# R9 (nhãn mag): baseline HAR và lịch sự kiện, folds {' '.join(a.folds)}\n\nRun: {run_dir}\nMốc: {a.mag_dir}\nSự kiện: {(run_dir / 'events.json').read_text()}\n\n"
                   f"AUC Mạnh/Yếu lớn nhất trên eval: {auc:.3f} (ngưỡng kiểm tra 0.75)\n\n## So sánh (a - b, âm là a tốt hơn)\n\n{md(main_rows, M.COLS)}\n\n"
                   f"## Theo từng fold\n\n{md(fold_rows)}\n\n## Giả dược (R9a gộp: thật so với bảng sự kiện hoán vị giữa các mã)\n\n{md(plac)}\n\n"
                   f"## Hệ số của đặc trưng sự kiện (logit_har_ev, lớp Mạnh, đặc trưng đã chuẩn hóa)\n\n{md(co) if co else '(không có)'}\n\n"
                   f"## Mọi arm trên eval\n\n{md(scores)}\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
