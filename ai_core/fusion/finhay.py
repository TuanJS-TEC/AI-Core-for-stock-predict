"""Finhay / HayInsights Open API -> training data under data/finhay/ (docs: https://developer.hayinsights.com/api-reference).

netflows.parquet     date, symbol, investor (foreign|proprietary), buy/sell/net value in VND, one row per session.
                     The API serves only ~3 months of daily points, so the history is built by running this every session
                     (cron, after the 15:20 price sync). A session's flow is known after its close: same availability as close.
interbank_vn.parquet date + SBV interbank rates (%) and volumes (bn VND) by tenor, full history on first run.

Key: env FINHAY_API_KEY, else the file AI_Core/.finhay_key (chmod 600, never commit it).
Use: python -m ai_core.fusion.finhay
"""
import json
import logging
import os
import time
import urllib.request
from datetime import date, timedelta
from urllib.error import HTTPError
from urllib.parse import urlencode

import pandas as pd

from ai_core import paths
from . import config as C

BASE = "https://api-v2.hayinsights.com/openapi/v1"
OUT = paths.DATA / "finhay"
log = logging.getLogger(__name__)


def _key():
    return os.environ.get("FINHAY_API_KEY") or (C.ROOT / ".finhay_key").read_text().strip()


def get(path, **params):
    """GET one endpoint -> `data`. Waits out 429 (budget is weight per minute) and retries 5xx."""
    url = f"{BASE}/{path}" + (f"?{urlencode(params)}" if params else "")
    req = urllib.request.Request(url, headers={"X-API-Key": _key(), "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["data"]
        except HTTPError as e:
            if attempt == 3 or (e.code != 429 and e.code < 500):
                raise
            wait = int(e.headers.get("Retry-After") or 5 * (attempt + 1))
            log.warning("%s -> HTTP %d, retry in %ds", path, e.code, wait)
            time.sleep(wait)


def upsert(path, new, keys):
    """Merge new rows into a parquet file; on a key clash the new row wins (the source restates recent sessions)."""
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    out = new.drop_duplicates(keys, keep="last").sort_values(keys, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def netflow_rows(symbol, investor, data):
    d = pd.DataFrame(data["points"])
    if d.empty:
        return d
    assert data["resolution"] == "day", data["resolution"]   # 6m/1y are monthly sums, not sessions
    d = d.rename(columns={"period": "date"}).drop(columns="sessions")
    d["date"] = pd.to_datetime(d["date"])                     # tz-naive midnight, the ai_core.fusion.prices key
    return d.assign(symbol=symbol, investor=investor)


def sync_netflows(symbols):
    parts = [netflow_rows(s, inv, get(f"market/vn/equities/{s}/netflows", investor=inv, range="3m"))
             for s in symbols for inv in ("foreign", "proprietary")]
    out = upsert(OUT / "netflows.parquet", pd.concat(parts, ignore_index=True), ["symbol", "investor", "date"])
    log.info("netflows: %d rows, %s .. %s", len(out), out["date"].min().date(), out["date"].max().date())


def sync_interbank():
    path = OUT / "interbank_vn.parquet"
    stop = pd.read_parquet(path)["date"].max().date() - timedelta(days=30) if path.exists() else date(1990, 1, 1)
    to, parts = date.today(), []
    while to > stop:                                          # one call spans at most 1,098 days
        frm = max(stop, to - timedelta(days=1097))
        d = pd.DataFrame(get("macro/interbank-rates/vn", **{"from": frm.isoformat(), "to": to.isoformat()}))
        if d.empty:
            break
        parts.append(d.dropna(axis=1, how="all"))           # old chunks lack some tenors entirely
        to = frm - timedelta(days=1)
    d = pd.concat(parts, ignore_index=True).rename(columns={"ngay": "date"})   # live fields are Vietnamese, not the documented names
    d["date"] = pd.to_datetime(d["date"])
    out = upsert(path, d, ["date"])
    log.info("interbank_vn: %d rows, %s .. %s", len(out), out["date"].min().date(), out["date"].max().date())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sync_netflows(sorted(p.name.split("_")[0] for p in C.STOCK_DIR.glob("*_stock.parquet")))
    sync_interbank()
