"""Live board (SQLite fusion_decision) -> StockSenseVN PostgreSQL (market_batches, forecasts, market_summary).

StockSenseVN (sibling repo ../StockSenseVN) expects a Python "model worker" that only talks to it through its database
(its SDD, decisions D2/D3); its API shows the batch with the newest data_date. This module is that worker's write step.
- label: LONG -> up, AVOID -> down, NO_TRADE -> hold (the trend labels of ai_core.fusion.live).
- probabilities: hold = 1 - p_strong (volcast); p_strong is split up/down in the ratio P(up):P(down) of the direction
  model. The label is live.trend's, not necessarily the argmax of the three numbers.
- confidence is never "high": the direction model is NO-GO (docs/reports/direction_report.md). "mid" only for a hold
  whose volcast call passed its own reliability threshold ("weak"); everything else is "low". Batch state is
  "experimental": StockSenseVN's web shows a "mô hình thử nghiệm" note for it (EXPERIMENTAL_FORECAST_NOTE).
- Forecasts are immutable (StockSenseVN SDD 6.3): ON CONFLICT DO NOTHING, a rerun never changes stored numbers; a
  stored row that no longer matches fusion_decision (a later fusion.live REPLACE) is kept and counted in a warning.
  Rows that are DATA_INVALID or miss p_strong / P(up) are not pushed, so a half-built row is never frozen. A batch is
  only created when at least one forecast lands in it: an empty newest batch would blank every card in the app.
  batch_completed_at moves only when a run adds a forecast to the batch, so the app shows when forecasts were made.
- A symbol missing from the newest batch (its sync failed, or a SEAM) is not copied forward here: StockSenseVN's query
  shows that symbol's newest forecast of the last 7 days with a stale tag (SDD AC-06/28).
- The cron pushes only the last SINCE_DAYS dates, so batches an admin deleted are not recreated and a symbol enabled
  later is not back-filled into old batches.
Run: STOCKSENSE_DATABASE_URL=postgresql://... .venv/bin/python -m ai_core.fusion.stocksense (after ai_core.fusion.live).
"""
import json
import os
from collections import defaultdict
from datetime import date, timedelta

from ai_core.news import storage
from .live import VERSION
from ai_core.models.rangecast.rangecast import VERSION as RANGE_VERSION

LABEL = {"LONG": "up", "AVOID": "down", "NO_TRADE": "hold"}
SINCE_DAYS = 10   # the cron only pushes recent dates: older batches stay as StockSenseVN keeps them


def to_forecast(row) -> dict | None:
    """One fusion_decision row (mapping) -> a StockSenseVN forecast dict, or None if it must not be pushed."""
    if row["decision"] not in LABEL or row["p_strong"] is None or row["p_up"] is None or row["p_down"] is None:
        return None
    ps, pu, pd = float(row["p_strong"]), float(row["p_up"]), float(row["p_down"])
    hold = round(1 - ps, 7)
    up = round(ps * (pu / (pu + pd) if pu + pd > 0 else 0.5), 7)
    label = LABEL[row["decision"]]
    return {"symbol": row["symbol"], "data_date": row["base_date"], "horizon": f"t{row['h']}", "label": label,
            "up": up, "hold": hold, "down": round(max(0.0, 1 - up - hold), 7),
            "confidence": "mid" if label == "hold" and row["call"] == "weak" else "low"}


def push(sq, pg, version: str = VERSION, since: str | None = None) -> int:
    """Insert the forecasts of `version` (base_date >= `since`, all dates if None) that StockSenseVN does not have yet;
    returns how many rows were inserted. Stored rows whose numbers no longer match fusion_decision are kept and counted."""
    by_date = defaultdict(list)
    for r in sq.execute("SELECT * FROM fusion_decision WHERE version = ? AND (? IS NULL OR base_date >= ?)",
                        (version, since, since)):
        if (f := to_forecast(r)) is not None:
            by_date[f["data_date"]].append(f)
    inserted = differ = 0
    for day, fs in sorted(by_date.items()):
        enabled = {s for (s,) in pg.execute("SELECT symbol FROM symbols WHERE analysis_enabled AND symbol = ANY(%s)",
                                            ([f["symbol"] for f in fs],))}
        fs = [f for f in fs if f["symbol"] in enabled]
        if not fs:
            continue
        with pg.transaction():
            # The no-op DO UPDATE only makes RETURNING give the id of an existing batch; its timestamp moves below,
            # and only when this run adds a forecast to it.
            (batch,) = pg.execute(
                """INSERT INTO market_batches (data_date, batch_completed_at, model_version, state, source_status)
                   VALUES (%s, now(), %s, 'experimental', 'ai_core')
                   ON CONFLICT (data_date, model_version) DO UPDATE SET batch_completed_at = market_batches.batch_completed_at
                   RETURNING id""", (day, version)).fetchone()
            added = 0
            for f in fs:
                key = (batch, f["symbol"], f["horizon"])
                if pg.execute(
                        """INSERT INTO forecasts (batch_id, symbol, horizon, label, probability_up, probability_hold,
                                                  probability_down, confidence)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (batch_id, symbol, horizon) DO NOTHING""",
                        (*key, f["label"], f["up"], f["hold"], f["down"], f["confidence"])).rowcount:
                    added += 1
                    continue
                old = pg.execute("""SELECT label, probability_up, probability_hold, probability_down FROM forecasts
                                    WHERE batch_id = %s AND symbol = %s AND horizon = %s""", key).fetchone()
                differ += old[0] != f["label"] or any(abs(float(o) - f[k]) > 1e-6 for o, k in zip(old[1:], ("up", "hold", "down")))
            if added:
                pg.execute("UPDATE market_batches SET batch_completed_at = now() WHERE id = %s", (batch,))
            inserted += added
            # Label shares on the overview count T+5 labels, as StockSenseVN's own mocks/market.js does. Breadth columns
            # are only its fallback when the price board is empty, so they stay 0.
            pg.execute("""INSERT INTO market_summary (batch_id, advances, nochanges, declines, labels_up, labels_hold, labels_down)
                          SELECT %(b)s, 0, 0, 0, count(*) FILTER (WHERE label = 'up'), count(*) FILTER (WHERE label = 'hold'),
                                 count(*) FILTER (WHERE label = 'down')
                          FROM forecasts WHERE batch_id = %(b)s AND horizon = 't5'
                          ON CONFLICT (batch_id) DO UPDATE SET labels_up = EXCLUDED.labels_up,
                            labels_hold = EXCLUDED.labels_hold, labels_down = EXCLUDED.labels_down""", {"b": batch})
    if differ:
        print(f"stocksense: WARNING {differ} stored forecasts differ from fusion_decision ({version}); "
              "kept as stored, forecasts are immutable")
    return inserted


COVERAGE = 0.8      # the band is the central 80% (rangecast.Q)


def push_ranges(sq, pg, version: str = RANGE_VERSION, since: str | None = None) -> int:
    """Insert the range_forecast rows of `version` (base_date >= `since`) that StockSenseVN does not have yet into
    forecast_ranges (its migration 025); returns how many were inserted. Same rules as `push`: only analysis_enabled
    symbols, a batch only when a row lands in it, stored rows are never changed. The news snapshot (has_news,
    sentiment_mean_24h, risk_flags) is the one fusion.live stored for the same symbol, session and horizon; without it
    the row says "no news". Nothing is written to `forecasts`: this model has no up/hold/down label."""
    by_date = defaultdict(list)
    for r in sq.execute(
            """SELECT r.symbol, r.base_date, r.h, r.lo_pct, r.hi_pct, r.risk_level, d.has_news, d.sentiment_mean_24h, d.risk_flags
               FROM range_forecast r LEFT JOIN fusion_decision d
                 ON d.symbol = r.symbol AND d.base_date = r.base_date AND d.h = r.h AND d.version = ?
               WHERE r.model_version = ? AND (? IS NULL OR r.base_date >= ?)""", (VERSION, version, since, since)):
        by_date[r["base_date"]].append(r)
    inserted = 0
    for day, rows in sorted(by_date.items()):
        enabled = {s for (s,) in pg.execute("SELECT symbol FROM symbols WHERE analysis_enabled AND symbol = ANY(%s)",
                                            ([r["symbol"] for r in rows],))}
        rows = [r for r in rows if r["symbol"] in enabled]
        if not rows:
            continue
        with pg.transaction():
            (batch,) = pg.execute(
                """INSERT INTO market_batches (data_date, batch_completed_at, model_version, state, source_status)
                   VALUES (%s, now(), %s, 'experimental', 'ai_core')
                   ON CONFLICT (data_date, model_version) DO UPDATE SET batch_completed_at = market_batches.batch_completed_at
                   RETURNING id""", (day, version)).fetchone()
            added = sum(pg.execute(
                """INSERT INTO forecast_ranges (batch_id, symbol, horizon, low_pct, high_pct, coverage, risk_level,
                                                has_news, sentiment_24h, risk_flags)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (batch_id, symbol, horizon) DO NOTHING""",
                (batch, r["symbol"], f"t{r['h']}", round(r["lo_pct"], 6), round(r["hi_pct"], 6), COVERAGE, r["risk_level"],
                 bool(r["has_news"]), r["sentiment_mean_24h"], json.loads(r["risk_flags"] or "[]"))).rowcount for r in rows)
            if added:
                pg.execute("UPDATE market_batches SET batch_completed_at = now() WHERE id = %s", (batch,))
            inserted += added
            # Level shares on the overview count the T+5 rows, like the label shares of `push`.
            pg.execute("""INSERT INTO market_summary (batch_id, advances, nochanges, declines, labels_up, labels_hold, labels_down,
                                                      vol_low, vol_normal, vol_high)
                          SELECT %(b)s, 0, 0, 0, 0, 0, 0, count(*) FILTER (WHERE risk_level = 'low'),
                                 count(*) FILTER (WHERE risk_level = 'normal'), count(*) FILTER (WHERE risk_level = 'high')
                          FROM forecast_ranges WHERE batch_id = %(b)s AND horizon = 't5'
                          ON CONFLICT (batch_id) DO UPDATE SET vol_low = EXCLUDED.vol_low, vol_normal = EXCLUDED.vol_normal,
                            vol_high = EXCLUDED.vol_high""", {"b": batch})
    return inserted


def connect(url: str):
    """Autocommit connection that fails fast: the pushes run inside live_update.sh's live.lock, so a hang would make
    every later live run skip."""
    import psycopg
    return psycopg.connect(url, autocommit=True, connect_timeout=10,
                           options="-c statement_timeout=60000 -c lock_timeout=30000")


def main() -> None:
    url = os.environ.get("STOCKSENSE_DATABASE_URL")
    if not url:
        print("stocksense: STOCKSENSE_DATABASE_URL not set, skip")
        return
    with connect(url) as pg:
        # Since 2026-11-13 the web shows the price range and volatility level (rangecast), not the trend labels: `push`
        # (fusion_decision -> forecasts) is kept for a way back but no longer called.
        n = push_ranges(storage.connect(), pg, since=str(date.today() - timedelta(days=SINCE_DAYS)))
        (day,) = pg.execute("SELECT max(data_date)::text FROM market_batches WHERE model_version = %s", (RANGE_VERSION,)).fetchone()
    print(f"stocksense: {n} ranges inserted ({RANGE_VERSION}, newest batch {day})")


if __name__ == "__main__":
    main()
