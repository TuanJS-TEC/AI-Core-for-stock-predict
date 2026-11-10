"""Global daily series (S&P 500, VIX, Brent, USD/VND) from the Yahoo chart API, joined to sessions strictly as-of.

A bar is usable for session t only when its exchange-local date is before t: the US close of date D lands at about
03:00-04:00 Vietnam time on D+1, after the Vietnamese close of D. The API is unofficial and may change.
Use: python -m ai_core.models.mmf.globaldata
"""
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import config as C

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d"


def parse_chart(payload, now=None):
    """date (exchange-local day of the bar), close; null closes dropped, the last bar of a day kept.

    The bar of a session still open at `now` (epoch seconds) is dropped: its close is not final."""
    r = payload["chart"]["result"][0]
    ts = np.asarray(r["timestamp"], dtype="int64")
    close = pd.to_numeric(pd.Series(r["indicators"]["quote"][0]["close"]), errors="coerce")
    # ponytail: without currentTradingPeriod in meta the last bar is kept as is
    period = r["meta"].get("currentTradingPeriod", {}).get("regular")
    if now is not None and period and now < period["end"]:
        keep = ts < period["start"]
        ts, close = ts[keep], close[keep]
    day = pd.to_datetime(ts + int(r["meta"]["gmtoffset"]), unit="s").normalize()
    d = pd.DataFrame({"date": day, "close": close.to_numpy()})
    return d.dropna().drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch(symbol, start="2021-01-01"):
    url = URL.format(sym=quote(symbol), p1=int(pd.Timestamp(start).timestamp()), p2=int(time.time()) + 86400)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
        return parse_chart(json.load(r), now=time.time())


def sync(out=C.GLOBAL_FILE):
    """Long table series, date, close of every C.GLOBAL_SERIES, written whole or not at all."""
    table = pd.concat([fetch(sym).assign(series=name) for name, sym in C.GLOBAL_SERIES.items()], ignore_index=True)[["series", "date", "close"]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        table.to_parquet(tmp, index=False)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return table


def asof_before(s, calendar, stale_days=C.GLOBAL_STALE_DAYS):
    """Per session t, the value of the last bar dated strictly before t; NaN when none or older than stale_days."""
    s = s.sort_index()
    calendar = pd.DatetimeIndex(calendar)
    pos = s.index.searchsorted(calendar, side="left") - 1
    out = np.full(len(calendar), np.nan)
    ok = pos >= 0
    ok[ok] = (calendar[ok] - s.index[pos[ok]]).days <= stale_days
    out[ok] = s.to_numpy(dtype=float)[pos[ok]]
    return pd.Series(out, index=calendar)


if __name__ == "__main__":
    t = sync()
    print(t.groupby("series")["date"].agg(["min", "max", "count"]).to_string())
