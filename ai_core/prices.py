"""One price loader for the fusion layer: Pipeline_V1.2's parquet files -> long frame (date, symbol, OHLCV).
`date` is the trading day (tz-naive midnight), the same key NLP's trial_forecast/walk_forward use."""
import pandas as pd

from . import config as C


def load_prices(symbols=C.UNIVERSE, stock_dir=C.STOCK_DIR) -> pd.DataFrame:
    parts = []
    for s in symbols:
        d = pd.read_parquet(stock_dir / f"{s}_stock.parquet")
        d["date"] = d["time"].dt.tz_localize(None).dt.normalize()   # sessions are stamped 07:00 UTC on the trading day
        parts.append(d.drop(columns="time").assign(symbol=s))
    out = pd.concat(parts, ignore_index=True)
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]].sort_values(["symbol", "date"], ignore_index=True)
