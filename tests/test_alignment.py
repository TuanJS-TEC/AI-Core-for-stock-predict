import pandas as pd

from ai_core import config as C
from ai_core.prices import load_prices


def test_prices_aligned_and_unique():
    df = load_prices()
    assert set(df.symbol) == set(C.UNIVERSE)
    assert not df.duplicated(["symbol", "date"]).any()
    assert df[["close", "high", "low", "volume"]].notna().all().all()
    days = df.groupby("symbol")["date"].agg(frozenset)
    assert days.nunique() == 1                       # identical calendars: cross-sectional ranks need it
    assert df.date.dt.tz is None and (df.date == df.date.dt.normalize()).all()


def test_pipeline_prices_cover_nlp_copy():
    """Pipeline's copy is the canonical one: same values as NLP's stale copy on every date NLP has."""
    old = pd.read_csv(C.NLP / "data/stock/FPT_stock.csv", parse_dates=["time"])
    old["date"] = old["time"].dt.normalize()
    new = load_prices(["FPT"]).set_index("date")["close"]
    cmp = old.set_index("date")["close"].reindex(new.index).dropna()
    assert len(cmp) > 1400 and (cmp == new.loc[cmp.index]).all()
