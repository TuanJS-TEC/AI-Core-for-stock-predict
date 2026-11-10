import pandas as pd

from ai_core.fusion import config as C
from ai_core.fusion.prices import load_prices


def test_prices_aligned_and_unique():
    df = load_prices()
    assert set(df.symbol) == set(C.UNIVERSE)
    assert not df.duplicated(["symbol", "date"]).any()
    assert df[["close", "high", "low", "volume"]].notna().all().all()
    days = df.groupby("symbol")["date"].agg(frozenset)
    assert days.nunique() == 1                       # identical calendars: cross-sectional ranks need it
    assert df.date.dt.tz is None and (df.date == df.date.dt.normalize()).all()


def test_volcast_reads_canonical_prices():
    from ai_core.models.volcast import volcast
    h = volcast.load_history(symbols=volcast.SYMBOLS).set_index(["symbol", "date"]).close.sort_index()
    p = load_prices().set_index(["symbol", "date"]).close.sort_index()
    assert set(h.index.get_level_values(0)) == set(volcast.SYMBOLS) and set(volcast.SYMBOLS) <= set(C.UNIVERSE)   # volcast widens to all 20 in Task 7
    assert (p.loc[h.index] == h).all()


def test_load_history_symbols_filter(tmp_path):
    from ai_core.models.volcast import volcast
    for s in ("AAA", "BBB"):
        pd.DataFrame({"time": ["2026-01-02"], "close": [1.0], "high": [1.0], "low": [1.0], "volume": [1]}
                     ).to_csv(tmp_path / f"{s}_stock.csv", index=False)
    assert set(volcast.load_history(str(tmp_path), symbols=["AAA"]).symbol) == {"AAA"}
    assert set(volcast.load_history(str(tmp_path)).symbol) == {"AAA", "BBB"}


def test_no_corporate_action_seam():
    from ai_core.models.volcast.trial_forecast import check_prices
    assert check_prices(load_prices())["n_big_moves"] == 0
