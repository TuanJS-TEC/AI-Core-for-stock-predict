"""Tradable direction labels and the modelling frame. Entry is the next session's open (the signal is known at the
15:00 close), exit is the close h sessions after that: log(close[t+h+1] / open[t+1]). The label is the cross-sectional
excess of that return against a cost-aware band. Models (E3 logit, E4 gbm) are picked on val, LONG-only, scored once
on test; go() is the pre-registered GO/NO-GO rule (spec Bước 2)."""
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C   # first: puts NLP_Module and Pipeline_V1.2 on sys.path
from . import splits
from .prices import load_prices
from nlp_module.forecast import fundamentals, volcast  # noqa: E402
from src import config as PC  # noqa: E402  (Pipeline_V1.2: bootstrap block/n/level)
from src.stats import block_bootstrap_dates  # noqa: E402

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


def build_frame(h: int, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """prices: load_prices() output (the 20-ticker universe when None). Warm-up rows with NaN features are dropped,
    as volcast.walk does, since logit cannot take NaN."""
    prices = load_prices() if prices is None else prices
    df = volcast.add_features(prices, fundamentals.bctc_features(C.NLP / "data/bctc.parquet"))
    for k in (1, 5, 20):
        df[f"exret{k}"] = df[f"ret{k}"] - df.groupby("date")[f"ret{k}"].transform("mean")
    return add_labels(df, h).dropna(subset=FEATS).reset_index(drop=True)


CS = (0.01, 0.1, 1.0)
MODELS = {"logit": lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)),   # E3, multinomial
          "gbm": lambda c: volcast.MODELS["gbm"]()}                                                     # E4
GRID = [("logit", c) for c in CS] + [("gbm", None)]
MS = np.round(np.arange(0.0, 0.51, 0.02), 2)
MIN_N, N_RAND = 50, 1000


def folds(df, h):
    """splits.folds with a 1 + h session purge: the label reads close[t+1+h]."""
    return splits.folds(df, h, purge=h + 1)


def _fit(d, h, kind, c):
    return CalibratedClassifierCV(MODELS[kind](c), method="sigmoid", cv=3).fit(d[FEATS], d[f"y{h}"])


def _probs(model, d):
    """(P(UP), P(DOWN))."""
    p, k = model.predict_proba(d[FEATS]), list(model.classes_)
    return p[:, k.index(1.0)], p[:, k.index(-1.0)]


def long_mask(p_up, p_down, m) -> np.ndarray:
    return np.asarray(p_up) - np.asarray(p_down) >= m


def select_on_val(tr, va, h, grid=GRID) -> dict:
    """Fit + calibrate each (kind, C) on train; pick (kind, C, m) with the best mean net per LONG trade on val over
    >= MIN_N trades. Ties keep the first in GRID x MS order. kind=None when nothing reaches MIN_N."""
    net = va[f"r_trade{h}"].to_numpy() - C.COST
    best = {"kind": None, "C": None, "m": None, "val_net": float("nan"), "val_n": 0}
    for kind, c in grid:
        up, down = _probs(_fit(tr, h, kind, c), va)
        for m in MS:
            k = long_mask(up, down, m)
            if k.sum() >= MIN_N and (best["kind"] is None or net[k].mean() > best["val_net"]):
                best = {"kind": kind, "C": c, "m": float(m), "val_net": float(net[k].mean()), "val_n": int(k.sum())}
    return best


def boot_ci(daily, rng) -> tuple[float, float]:
    """CI of the mean net excess per LONG trade: whole test dates resampled in blocks (Pipeline_V1.2 settings). daily
    has one row per test date (s = summed net_ex of that day's trades, k = their count); concat folds' to pool."""
    s, k = daily.s.to_numpy(), daily.k.to_numpy()
    stats = [s[i].sum() / k[i].sum() for i in block_bootstrap_dates(daily.index.to_numpy(), PC.BOOTSTRAP_BLOCK,
                                                                   PC.BOOTSTRAP_N, rng) if k[i].sum()]
    if not stats:   # no resample holds a trade
        return float("nan"), float("nan")
    a = (1 - PC.CI_LEVEL) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def sleeve_turnover(w, h) -> float:
    """w: date x symbol equal-weight target of each day's LONG trades. The book holds h+1 sleeves of 1/(h+1) capital;
    day t buys w_t's sleeve and sells w_{t-h-1}'s (an empty book before the first test day). Mean one-way turnover."""
    return float((0.5 * (w - w.shift(h + 1, fill_value=0.0)).abs().sum(axis=1) / (h + 1)).mean())


def run_fold(tr, va, trva, te, h, rng, grid=GRID) -> dict:
    """Select on val, refit the pick on train+val, score test once. net = mean(r_trade - COST) per LONG trade, net_ex
    the same on r_ex. e0_prior (E0) = net of going LONG on every test row. turnover and max_dd are of one portfolio:
    each day's LONG trades equal-weighted into a sleeve; a trade spans h+1 sessions, so h+1 staggered sleeves each
    hold 1/(h+1) of capital. A fold without a pick or without test trades has NaN metrics (go() counts it as a loss)."""
    sel = select_on_val(tr, va, h, grid)
    te = te.assign(net=te[f"r_trade{h}"] - C.COST, net_ex=te[f"r_ex{h}"] - C.COST)
    nan = float("nan")
    out = {**sel, "n": 0, "hit": nan, "net": nan, "net_ex": nan, "ci_lo": nan, "ci_hi": nan, "rand_median": nan,
           "turnover": nan, "max_dd": nan, "e0_prior": float(te.net.mean()), "daily": None}
    if sel["kind"] is None:
        return out
    k = long_mask(*_probs(_fit(trva, h, sel["kind"], sel["C"]), te), sel["m"])
    n = int(k.sum())
    if not n:
        return {**out, "n": 0}
    t, all_net = te[k], te.net.to_numpy()
    daily = te.assign(k=k.astype(int), s=np.where(k, te.net_ex, 0.0), p=np.where(k, te.net, 0.0)) \
        .groupby("date")[["k", "s", "p"]].sum()
    eq = np.r_[0.0, np.cumsum((daily.p / daily.k.where(daily.k > 0, 1)) / (h + 1))]
    w = te.assign(w=k.astype(float)).pivot_table(index="date", columns="symbol", values="w", fill_value=0.0)
    w = w.div(w.sum(axis=1).where(lambda x: x > 0, 1), axis=0)
    lo, hi = boot_ci(daily, rng)
    return {**out, "n": n, "hit": float((t.net > 0).mean()), "net": float(t.net.mean()), "net_ex": float(t.net_ex.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "rand_median": float(np.median([all_net[rng.choice(len(te), n, replace=False)].mean() for _ in range(N_RAND)])),
            "turnover": sleeve_turnover(w, h),
            "max_dd": float((np.maximum.accumulate(eq) - eq).max()), "daily": daily}


def go(rows: list[dict], pooled_ci_lo: float) -> bool:
    """Pre-registered (spec Bước 2), per h: net > 0 in >= 2/3 folds AND net > rand_median in >= 2/3 folds AND the
    pooled 3-fold CI lower bound of net_ex > 0. A fold with kind=None is a loss."""
    ok = [r for r in rows if r["kind"] is not None]
    wins = lambda f: 3 * sum(bool(f(r)) for r in ok) >= 2 * len(rows)
    return wins(lambda r: r["net"] > 0) and wins(lambda r: r["net"] > r["rand_median"]) and pooled_ci_lo > 0


REPORT = C.ROOT / "docs/reports/direction_report.md"
E1_TABLE = C.PIPELINE / "outputs/results/pooled_embed_k0.43/final_table.csv"
COLS = ["h", "fold", "model", "kind", "C", "m", "val_n", "n", "hit", "net", "net_ex", "ci_lo", "ci_hi", "rand_median",
        "turnover", "max_dd"]
NAN = float("nan")


def run_h(df, h) -> tuple[list[dict], list[str], bool]:
    """Table rows and verdict lines for one h. GO inputs come only from the SEL runs (full GRID, one rng(0) over
    A, B, C, then the pooled CI). The E3/E4 rows re-run run_fold on one kind's grid with their own rng(0): table only."""
    fl = list(folds(df, h))
    rng = np.random.default_rng(0)
    sel = [{"fold": f[0], **run_fold(*f[1:], h, rng)} for f in fl]
    daily = [r["daily"] for r in sel if r["daily"] is not None]
    lo, hi = boot_ci(pd.concat(daily), rng) if daily else (NAN, NAN)
    ok = go(sel, lo)
    rows = [{"h": h, "fold": r["fold"], "model": "E0 prior (LONG all)", "n": len(t[4]), "net": r["e0_prior"]}
            for r, t in zip(sel, fl)]
    rows += [{"h": h, "model": f"SEL ({r['kind']})", **r} for r in sel]
    for name, kind in (("E3", "logit"), ("E4", "gbm")):
        rng_k = np.random.default_rng(0)
        rows += [{"h": h, "fold": f[0], "model": name, **run_fold(*f[1:], h, rng_k, [g for g in GRID if g[0] == kind])}
                 for f in fl]
    pos = sum(r["kind"] is not None and r["net"] > 0 for r in sel)
    beat = sum(r["kind"] is not None and r["net"] > r["rand_median"] for r in sel)
    lines = [f"- h={h}: (1) net > 0 in {pos}/3 folds; (2) net > rand_median in {beat}/3 folds; "
             f"(3) pooled net_ex 95% CI [{lo:.5f}, {hi:.5f}], lower bound > 0: {lo > 0} -> **{'GO' if ok else 'NO-GO'}**"]
    return rows, lines, ok


def e1_rows() -> pd.DataFrame:
    t = pd.read_csv(E1_TABLE)
    return t[(t["method"] == "model") & t["headline"].astype(bool)][
        ["T", "L", "config", "n_tickers", "date_from", "date_to", "hit_rate_mean", "f1_macro_mean", "f1_ci_lo",
         "f1_ci_hi", "kappa_mean", "directional_share_mean"]]


def main(frame=build_frame, report=REPORT, hs=C.HORIZONS):
    t0 = time.time()
    rows, lines, gos = [], [], []
    for h in hs:
        r, l, ok = run_h(frame(h), h)
        rows, lines = rows + r, lines + l
        gos += [h] if ok else []
    table = pd.DataFrame(rows).reindex(columns=COLS).sort_values(["h", "fold", "model"], kind="stable")
    decision = f"GO h={','.join(map(str, gos))}" if gos else "NO-GO"
    out = "\n".join([
        f"\n### Run {datetime.now().astimezone().isoformat(timespec='seconds')} ({time.time() - t0:.0f} s)\n",
        "E0/E3/E4 × fold × h on test. SEL = the (kind, C, m) chosen on val (the GO candidate). E3/E4 = the same "
        "machinery restricted to one kind (table only). net/net_ex = mean per LONG trade after COST; ci = 95% block "
        "bootstrap of net_ex; rand_median = random-same-count median of net (SEL rows' value is the GO input).\n",
        table.to_markdown(index=False, floatfmt=".4f"),
        "\nE1 (Pipeline_V1.2 pooled_embed_k0.43, headline rows; split 70/15/15, dữ liệu trước khi sửa, không cùng "
        "fold; 15 tickers; classification metrics only, that file has no net-return columns):\n",
        e1_rows().to_markdown(index=False, floatfmt=".4f"),
        "\nGO/NO-GO (pre-registered rule above):\n", *lines, f"\n**Decision: {decision}**\n"])
    print(out)
    with open(report, "a") as f:
        f.write(out)
    return decision


if __name__ == "__main__":
    main()
