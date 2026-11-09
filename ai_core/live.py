"""Live board: one fusion_decision row per (symbol, h) for the symbol's newest session.

Direction was NO-GO (Task 10), so the board only ever says NO_TRADE or DATA_INVALID; p_up/p_down/margin stay NULL.
Run: `NLP_Module/.venv/bin/python -m ai_core.live` (after the 15:20 price sync).
Write rules (primary key = symbol, base_date, h, version):
- NO_TRADE and SEAM rows use INSERT OR REPLACE: they describe the data of that base_date itself, so a rerun
  may legitimately refresh them.
- MISSING_SESSION rows use INSERT OR IGNORE: the status says "the data is older than due", not that base_date's own
  decision was wrong. From 15:00 until the 15:20 sync, hist still ends on yesterday, so a run would otherwise
  overwrite yesterday's valid row (and its created_at) with DATA_INVALID, permanently, since after the sync the new
  session is written under a new key. With IGNORE a brand-new key is still recorded as DATA_INVALID (fail-closed),
  an existing row is kept, and a later valid run for that key replaces the DATA_INVALID row.
DATA_INVALID rows carry NULL p_strong/call: such a row never displays a signal."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import config as C                                    # first: puts Pipeline_V1.2 and NLP_Module on sys.path
from .prices import load_prices
from nlp_module import storage                               # noqa: E402
from nlp_module.contract import build_snapshot               # noqa: E402
from nlp_module.forecast import volcast                      # noqa: E402
from nlp_module.forecast.trial_forecast import CLOSE_UTC     # noqa: E402
from nlp_module.pipeline import load_results                 # noqa: E402
from src import sessions as P                                # noqa: E402

VERSION = "decision-v0-nogo"
NLP_MODEL = "baseline-tfidf-lr-v1"
SEAM_SESSIONS = 21
VN = "Asia/Ho_Chi_Minh"                                      # SESSION_DATA_READY (15:00) is Vietnam wall-clock time


def holidays(hist: pd.DataFrame) -> np.ndarray:
    """datetime64[D]: weekdays with no session in `hist` (any symbol), plus the declared C.MARKET_HOLIDAYS.
    Same rule as src.sessions.market_holidays, which only needs `.dates` from its panel."""
    return P.market_holidays(SimpleNamespace(dates=pd.DatetimeIndex(sorted(hist["date"].unique()))))


def data_status(hist: pd.DataFrame, symbol: str, now: pd.Timestamp, holidays: np.ndarray) -> str | None:
    """SEAM | MISSING_SESSION | None. `now` must be tz-aware.
    Precedence: SEAM first. A seam (|ret1| > RET_BAND: split/dividend gap) makes the price history itself untrustworthy
    for 21 sessions, while a missing session clears itself at the next sync; both end in DATA_INVALID."""
    d = hist[hist["symbol"] == symbol].sort_values("date")
    if (d["close"].pct_change().tail(SEAM_SESSIONS).abs() > volcast.RET_BAND).any():
        return "SEAM"
    if P.missing_sessions(d["date"].iloc[-1], now.tz_convert(VN), holidays):
        return "MISSING_SESSION"
    return None


def decide(status: str | None) -> tuple[str, str]:
    return ("DATA_INVALID", status) if status else ("NO_TRADE", "no direction model (NO-GO)")


def rows(hist, results, vol, now, symbols=C.UNIVERSE) -> list[tuple]:
    """fusion_decision tuples. `vol` = {(symbol, base_date, h): (p_strong, call)} from vol_forecast (current volcast
    version); a symbol/date without a row (the 5 tickers outside volcast.SYMBOLS) keeps NULLs and says so in `reason`."""
    hol, t, out = holidays(hist), storage.now(), []
    for s in symbols:
        base = hist.loc[hist["symbol"] == s, "date"].max()
        cutoff = datetime.combine(base.date(), CLOSE_UTC, tzinfo=timezone.utc)
        mine = [r for r in results if r.symbol == s]
        snap = build_snapshot(mine, s, cutoff, "live", NLP_MODEL)
        f, win = snap.features, set(snap.lineage)
        flags = json.dumps(sorted({k for r in mine if r.result_id in win for k in r.risk_flags}))
        status = data_status(hist, s, now, hol)
        decision, reason = decide(status)
        base = str(base.date())
        for h in C.HORIZONS:
            ps, call = (None, None) if status else vol.get((s, base, h), (None, None))   # DATA_INVALID shows no signal
            note = "" if ps is not None or status else "; no volcast row" + ("" if s in volcast.SYMBOLS else " (symbol outside volcast.SYMBOLS)")
            out.append((s, base, h, decision, None, None, None, ps, call, int(f["has_news"]), f["sentiment_mean_24h"],
                        flags, reason + note, VERSION, t))
    return out


def main(now: pd.Timestamp | None = None, c=None, hist: pd.DataFrame | None = None) -> list[tuple]:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    c = storage.connect() if c is None else c
    hist = load_prices() if hist is None else hist
    vol = {(r["symbol"], r["base_date"], r["h"]): (r["p_strong"], r["call"]) for r in
           c.execute("SELECT * FROM vol_forecast WHERE model_version=?", (volcast.VERSION,))}
    out = rows(hist, load_results(c, NLP_MODEL), vol, now)
    for verb, keep in (("REPLACE", lambda r: r[12] != "MISSING_SESSION"), ("IGNORE", lambda r: r[12] == "MISSING_SESSION")):
        c.executemany(f"INSERT OR {verb} INTO fusion_decision VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [r for r in out if keep(r)])
    c.commit()
    return out


if __name__ == "__main__":
    out = main()
    print(f"fusion_decision: evaluated {len(out)} rows ({VERSION}); MISSING_SESSION rows never replace an existing row")
