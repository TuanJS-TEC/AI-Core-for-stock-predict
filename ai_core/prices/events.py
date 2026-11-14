"""Corporate-event calendar from VCI (vnstock source "vci"): dividends, share issues, shareholder meetings.

Every event keeps public_date (the day it was announced), so a feature can use only what was known at a session.
One parquet per download day under paths.EVENTS: the day's view of the vendor, kept for point-in-time checks.
Use: python -m ai_core.prices.events
"""
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ai_core import paths

CODES = ("DIV", "ISS", "AGME", "AGMR", "EGME")
COLUMNS = ["symbol", "event_id", "event_code", "public_date", "record_date", "exright_date", "display_date1"]
_RAW = {"id": "event_id", "eventCode": "event_code", "publicDate": "public_date", "recordDate": "record_date",
        "exrightDate": "exright_date", "displayDate1": "display_date1"}
PAGE = 100
TRIES = 3


def normalize(raw, symbol):
    """VCI records (camelCase) -> COLUMNS; dates as day-level datetime64, codes outside CODES and duplicate ids dropped."""
    df = pd.DataFrame(raw, columns=list(_RAW)).rename(columns=_RAW).assign(symbol=symbol)
    for c in ("public_date", "record_date", "exright_date", "display_date1"):
        df[c] = pd.to_datetime(df[c].astype("string").str[:10], errors="coerce")
    df = df[df["event_code"].isin(CODES)].drop_duplicates("event_id")
    return df[COLUMNS].reset_index(drop=True)


def fetch(symbols, since="2020-01-01", pause=3.5):
    """All CODES events of each symbol since `since`, paged; `pause` seconds between calls (guest limit 20/min)."""
    from vnstock import Company
    out = []
    for s in symbols:
        c, raw, page = Company(symbol=s, source="vci"), [], 0
        while True:
            for attempt in range(TRIES):     # the vendor answers 502 now and then
                try:
                    got = c._fetch_events(event_codes=",".join(CODES), from_date=since.replace("-", ""), page=page, size=PAGE)
                    break
                except ConnectionError:
                    if attempt == TRIES - 1:
                        raise
                    time.sleep(pause * 4)
            raw += got
            time.sleep(pause)
            if len(got) < PAGE:
                break
            page += 1
        out.append(normalize(raw, s))
    return pd.concat(out, ignore_index=True) if out else normalize([], "")


def save(df, day):
    paths.EVENTS.mkdir(parents=True, exist_ok=True)
    path = paths.EVENTS / f"{day}.parquet"
    df.to_parquet(path, index=False)
    return path


def load(path):
    return pd.read_parquet(path)


def main():
    from ai_core.prices.config import TICKERS
    df = fetch(TICKERS)
    path = save(df, datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat())
    print(path, len(df), "events")
    print(df.groupby("symbol").size().to_string())


if __name__ == "__main__":
    main()
