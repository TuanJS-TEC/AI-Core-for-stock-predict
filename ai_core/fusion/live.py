"""Live board: one fusion_decision row per (symbol, h) for the symbol's newest session.

The row is a trend label, not a trade signal: LONG = tăng, AVOID = giảm, NO_TRADE = đi ngang (the table's CHECK only
allows these names; the app shows the Vietnamese words). Flat comes from volcast (p_strong < 0.5: the move is expected
to stay under the day's cross-sectional median), up/down from the direction model's P(UP) vs P(DOWN). That model was
NO-GO as a trading rule (docs/reports/direction_report.md), so up/down carries no proven edge. The news snapshot does
not change the label; `reason` only says whether it agrees with it.
Run: `.venv/bin/python -m ai_core.fusion.live` (after the 15:20 price sync).
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

from . import config as C
from . import direction as D
from .prices import load_prices
from ai_core.news import storage                               # noqa: E402
from ai_core.news.contract import build_snapshot               # noqa: E402
from ai_core.models.volcast import volcast                      # noqa: E402
from ai_core.models.volcast.trial_forecast import CLOSE_UTC     # noqa: E402
from ai_core.news.pipeline import load_results                 # noqa: E402
from ai_core.prices import sessions as P                                # noqa: E402

VERSION = "decision-v1-trend"
NLP_MODEL = "baseline-tfidf-lr-v1"
SEAM_SESSIONS = 21
VN = "Asia/Ho_Chi_Minh"                                      # SESSION_DATA_READY (15:00) is Vietnam wall-clock time


def holidays(hist: pd.DataFrame) -> np.ndarray:
    """datetime64[D]: weekdays with no session in `hist` (any symbol), plus the declared C.MARKET_HOLIDAYS.
    Same rule as ai_core.prices.sessions.market_holidays, which only needs `.dates` from its panel."""
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


def direction_probs(hist: pd.DataFrame) -> dict:
    """{(symbol, base_date, h): (P(UP), P(DOWN))} for each symbol's newest session, refit on every labelled row.
    ponytail: one fixed model (logit, C=0.01, the most regularised of direction.GRID), refit per run (a few seconds);
    save a bundle like volcast's if the fit gets slow or the kind should be re-picked on val."""
    out = {}
    for h in C.HORIZONS:
        d = D.build_frame(h, hist)
        last = d.groupby("symbol").tail(1)
        up, down = D._probs(D._fit(d.dropna(subset=[f"y{h}"]), h, "logit", 0.01), last)
        out.update({(s, str(t.date()), h): (float(u), float(dn)) for s, t, u, dn in zip(last.symbol, last.date, up, down)})
    return out


def trend(ps, up, down) -> str:
    """LONG = tăng, AVOID = giảm, NO_TRADE = đi ngang. Without volcast (ps None) flat is the direction model's own
    argmax; without direction probabilities the label is never up/down."""
    if up is None or (ps < 0.5 if ps is not None else 1 - up - down >= max(up, down)):
        return "NO_TRADE"
    return "LONG" if up >= down else "AVOID"


SENT_BAND = 0.1   # |sentiment_mean_24h| below this reads as neutral
WORDS = {"LONG": "tăng", "AVOID": "giảm", "NO_TRADE": "đi ngang"}


def explain(label, ps, call, up, down, f, flags) -> str:
    """Vietnamese reason for the board: what the two models said, then whether the news agrees with the label."""
    mag = "không có volcast" if ps is None else \
        f"biên độ {'mạnh' if ps >= 0.5 else 'yếu'} (p_strong {ps:.2f}, {'đạt' if call else 'chưa đạt'} ngưỡng tin cậy)"
    dr = "không có xác suất hướng" if up is None else f"P(tăng) {up:.2f}, P(giảm) {down:.2f}"
    s = f["sentiment_mean_24h"]
    if not f["has_news"]:
        news = "Không có tin trong 72 giờ"
    elif s is None:
        news = "Có tin trong 72 giờ, không có tin trong 24 giờ"
    else:
        tone = "LONG" if s > SENT_BAND else "AVOID" if s < -SENT_BAND else "NO_TRADE"
        news = (f"Tin 24 giờ {('tích cực', 'tiêu cực', 'trung tính')[('LONG', 'AVOID', 'NO_TRADE').index(tone)]} ({s:+.2f}): "
                f"{'ủng hộ' if tone == label else 'trái chiều với'} nhãn {WORDS[label]}")
    return f"{mag}; {dr}. {news}" + (f". Cờ rủi ro: {', '.join(flags)}" if flags else "")


def rows(hist, results, vol, now, symbols=C.UNIVERSE, dirp=None) -> list[tuple]:
    """fusion_decision tuples. `vol` = {(symbol, base_date, h): (p_strong, call)} from vol_forecast (current volcast
    version); a symbol/date without a row (the 5 tickers outside volcast.SYMBOLS) keeps NULLs and says so in `reason`.
    `dirp` = direction_probs() output, same key."""
    hol, t, out, dirp = holidays(hist), storage.now(), [], dirp or {}
    for s in symbols:
        base = hist.loc[hist["symbol"] == s, "date"].max()
        cutoff = datetime.combine(base.date(), CLOSE_UTC, tzinfo=timezone.utc)
        mine = [r for r in results if r.symbol == s]
        snap = build_snapshot(mine, s, cutoff, "live", NLP_MODEL)
        f, win = snap.features, set(snap.lineage)
        flags = sorted({k for r in mine if r.result_id in win for k in r.risk_flags})
        status = data_status(hist, s, now, hol)
        base = str(base.date())
        for h in C.HORIZONS:
            (ps, call), (up, down) = ((None, None),) * 2 if status else \
                (vol.get((s, base, h), (None, None)), dirp.get((s, base, h), (None, None)))   # DATA_INVALID shows no signal
            decision = "DATA_INVALID" if status else trend(ps, up, down)
            out.append((s, base, h, decision, up, down, None if up is None else up - down, ps, call, int(f["has_news"]),
                        f["sentiment_mean_24h"], json.dumps(flags), status or explain(decision, ps, call, up, down, f, flags),
                        VERSION, t))
    return out


def main(now: pd.Timestamp | None = None, c=None, hist: pd.DataFrame | None = None, dirp: dict | None = None) -> list[tuple]:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    c = storage.connect() if c is None else c
    hist = load_prices() if hist is None else hist
    vol = {(r["symbol"], r["base_date"], r["h"]): (r["p_strong"], r["call"]) for r in
           c.execute("SELECT * FROM vol_forecast WHERE model_version=?", (volcast.VERSION,))}
    out = rows(hist, load_results(c, NLP_MODEL), vol, now, dirp=direction_probs(hist) if dirp is None else dirp)
    for verb, keep in (("REPLACE", lambda r: r[12] != "MISSING_SESSION"), ("IGNORE", lambda r: r[12] == "MISSING_SESSION")):
        c.executemany(f"INSERT OR {verb} INTO fusion_decision VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [r for r in out if keep(r)])
    c.commit()
    return out


if __name__ == "__main__":
    out = main()
    print(f"fusion_decision: evaluated {len(out)} rows ({VERSION}); MISSING_SESSION rows never replace an existing row")
    print(pd.Series([r[3] for r in out]).value_counts().to_string())
