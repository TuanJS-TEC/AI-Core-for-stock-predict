"""Shadow report for the Step-5 decision: is the live board healthy, and does live volcast hold up out of sample?
Run: `NLP_Module/.venv/bin/python -m ai_core.fusion.shadow >> docs/reports/shadow_report.md` (only after >= 40 sessions of
live data; with fewer the text says so).
Definitions (same as volcast.walk's test_acc):
- call -> label: "strong" predicts y = 1 (the ticker's |forward return| over h sessions beats the day's cross-sectional
  median), "weak" predicts y = 0. Correct = (call == "strong") == (y == 1). NULL call = abstain: not counted at all.
- matured: y{h} exists, i.e. the symbol has at least h sessions in `hist` after base_date (exactly h counts). A call whose
  base_date is not in `hist`, or with fewer future sessions, is pending.
- the label is trial_forecast.add_targets(df.assign(close=df.adj), h, "absmove") on `hist`, exactly as volcast.walk builds it.
- label basket: `hist` is filtered to volcast.SYMBOLS (read at call time) inside shadow_report, because the "strong"
  threshold is the day's cross-sectional median of |forward return| and so depends on which tickers are in the frame.
  The model is trained/scored on that basket; load_prices() returns all 20 UNIVERSE tickers, so callers need not filter.
- DATA_INVALID share covers every row on the board (all 20 symbols), not only the volcast ones.
- shadow_report calls volcast.add_features itself (needs date, symbol, close, high, low, volume); only `adj` matters here,
  so the BCTC join gets an empty table: the report then needs neither data/bctc.parquet nor a working directory."""
import json

import pandas as pd

from . import config as C
from . import live
from .prices import load_prices
from ai_core.news import storage                               # noqa: E402
from ai_core.models.volcast import fundamentals, volcast        # noqa: E402
from ai_core.models.volcast import trial_forecast as tf         # noqa: E402

MIN_SESSIONS = 40
REPORT = C.NLP / volcast.REPORT                              # volcast.REPORT is relative to NLP_Module


def _no_bctc() -> pd.DataFrame:
    return pd.DataFrame({"symbol": pd.Series(dtype=object), "available_at": pd.Series(dtype="datetime64[ns]"),
                         **{k: pd.Series(dtype=float) for k in fundamentals.BCTC[1:]}})


def _labels(hist: pd.DataFrame) -> dict:
    """{(symbol, 'YYYY-MM-DD', h): y} for every row whose label exists (h future sessions in `hist`)."""
    df = volcast.add_features(hist, _no_bctc())
    out = {}
    for h in volcast.HORIZONS:
        d = tf.add_targets(df.assign(close=df.adj), h, "absmove").dropna(subset=[f"y{h}"])
        out.update({(s, t.strftime("%Y-%m-%d"), h): y for s, t, y in zip(d.symbol, d.date, d[f"y{h}"])})
    return out


def _acc(correct: list[bool], n_pending: int) -> dict:
    n = len(correct)
    return dict(n_matured=n, n_correct=sum(correct), accuracy=sum(correct) / n if n else None, n_pending=n_pending)


def shadow_report(c, hist: pd.DataFrame, report_path=REPORT) -> dict:
    """c: NLP DB connection; hist: load_prices() frame. accuracy / data_invalid_share are None when there is nothing to divide."""
    n_sessions, n_rows, n_bad = c.execute(
        "SELECT COUNT(DISTINCT base_date), COUNT(*), COALESCE(SUM(decision='DATA_INVALID'), 0) "
        "FROM fusion_decision WHERE version=?", (live.VERSION,)).fetchone()
    y = _labels(hist[hist["symbol"].isin(volcast.SYMBOLS)])
    ok, pending = {h: [] for h in volcast.HORIZONS}, dict.fromkeys(volcast.HORIZONS, 0)
    for r in c.execute("SELECT symbol, base_date, h, call FROM vol_forecast WHERE model_version=? AND call IS NOT NULL",
                       (volcast.VERSION,)):
        label = y.get((r["symbol"], r["base_date"], r["h"]))
        if label is None:
            pending[r["h"]] += 1
        else:
            ok[r["h"]].append((r["call"] == "strong") == (label == 1.0))
    try:
        with open(report_path) as f:
            rows = json.load(f)["rows"]
    except FileNotFoundError:
        rows = []
    fold_c = {r["h"]: dict(test_acc=r["test_acc"], test_n=r["test_n"]) for r in rows if r["fold"] == "C"}
    return dict(n_sessions=n_sessions, n_rows=n_rows, n_data_invalid=n_bad,
                data_invalid_share=n_bad / n_rows if n_rows else None,
                **_acc([x for v in ok.values() for x in v], sum(pending.values())),
                by_h={h: _acc(ok[h], pending[h]) for h in volcast.HORIZONS}, fold_c=fold_c)


def render(r: dict) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    out = [f"## Shadow report ({live.VERSION}, volcast {volcast.VERSION}) generated {storage.now()}", "",
           f"- sessions: {r['n_sessions']}; rows {r['n_rows']}; DATA_INVALID {r['n_data_invalid']} ({f(r['data_invalid_share'])}), all board symbols"]
    out.append(f"- volcast labels use the {len(volcast.SYMBOLS)}-symbol volcast basket (like fold C)")
    if r["n_sessions"] < MIN_SESSIONS:
        out.append(f"- CAVEAT: less than {MIN_SESSIONS} sessions: not enough for the Step-5 decision")
    out.append(f"- live volcast calls: matured {r['n_matured']}, correct {r['n_correct']}, accuracy {f(r['accuracy'])}, pending {r['n_pending']}")
    for h, a in r["by_h"].items():
        t = r["fold_c"].get(h, dict(test_acc=None, test_n=0))
        out.append(f"  - h={h}: live {a['n_correct']}/{a['n_matured']} = {f(a['accuracy'])} (pending {a['n_pending']})"
                   f" | fold C test_acc {f(t['test_acc'])} (n={t['test_n']})")
    return "\n".join(out)


def main(c=None, hist: pd.DataFrame | None = None) -> str:
    c = storage.connect() if c is None else c
    txt = render(shadow_report(c, load_prices() if hist is None else hist))
    print(txt)
    return txt


if __name__ == "__main__":
    main()
