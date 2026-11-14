"""TCBS active buy/sell share per session -> data/tcbs/supply_demand.parquet (date, symbol, active_buy_percent,
active_sell_percent, buy_sell_ratio).

The source serves only about one month of sessions, so the history is built by running this every session (cron after
the close, like ai_core.fusion.finhay). A session's value is known after its close. Runs on ~/.venv_6x (vnstock 6) and
needs a live TCBS session (jobs/tcbs_login.sh); with an expired session the run fails and the next good run backfills
whatever the one-month window still holds.

Use: VNSTOCK_OAPI_SESSION_BACKEND=file ~/.venv_6x/bin/python -m ai_core.fusion.tcbs_flow
"""
import logging
import time

import pandas as pd

from ai_core import paths
from ai_core.prices import config as PC
from .finhay import upsert

OUT = paths.DATA / "tcbs" / "supply_demand.parquet"
log = logging.getLogger(__name__)


def rows(symbol, df):
    """Source frame (tz-aware `time`, one row per session) -> rows keyed by naive session date and symbol."""
    d = df.rename(columns={"time": "date"})
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    return d.assign(symbol=symbol)[["date", "symbol", "active_buy_percent", "active_sell_percent", "buy_sell_ratio"]]


def sync(symbols, fetch=None, delay=1.0):
    if fetch is None:
        from vnstock import Market   # vnstock 6, only in ~/.venv_6x
        fetch = lambda s: Market("tcbs").equity(s).supply_demand_daily()
    parts, failed = [], []
    for s in symbols:
        try:
            parts.append(rows(s, fetch(s)))
        except Exception as e:       # one bad symbol must not lose the day for the others
            failed.append(s)
            log.warning("%s: %s: %s", s, type(e).__name__, str(e)[:200])
        time.sleep(delay)
    if not parts:
        raise RuntimeError(f"no symbol fetched ({len(failed)} failed): is the TCBS session live?")
    out = upsert(OUT, pd.concat(parts, ignore_index=True), ["symbol", "date"])
    log.info("supply_demand: %d rows, %s .. %s, failed: %s", len(out), out["date"].min().date(), out["date"].max().date(),
             failed or "none")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync(sorted(PC.TICKERS))
