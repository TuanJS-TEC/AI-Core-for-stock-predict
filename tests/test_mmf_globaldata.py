import numpy as np, pandas as pd, pytest
from mmf import globaldata as G


def test_parse_chart_drops_null_close_and_uses_exchange_date():
    payload = {"chart": {"result": [{"meta": {"gmtoffset": -18000}, "timestamp": [1762785000, 1762871400],
                                     "indicators": {"quote": [{"close": [10.0, None]}]}}]}}
    d = G.parse_chart(payload)
    assert d.date.tolist() == [pd.Timestamp("2025-11-10")] and d.close.tolist() == [10.0]


def test_parse_chart_keeps_one_row_per_day():
    payload = {"chart": {"result": [{"meta": {"gmtoffset": 0}, "timestamp": [1762785000, 1762788600],
                                     "indicators": {"quote": [{"close": [10.0, 11.0]}]}}]}}
    assert G.parse_chart(payload).close.tolist() == [11.0]


def _live(start, end):
    return {"chart": {"result": [{"meta": {"gmtoffset": 0, "currentTradingPeriod": {"regular": {"start": start, "end": end}}},
                                  "timestamp": [1762698600, 1762785000], "indicators": {"quote": [{"close": [10.0, 11.0]}]}}]}}


def test_parse_chart_drops_the_bar_of_a_session_still_open():
    # second bar opened at 1762785000 and its session ends at +23400 s
    assert G.parse_chart(_live(1762785000, 1762808400), now=1762790000).close.tolist() == [10.0]
    assert G.parse_chart(_live(1762785000, 1762808400), now=1762808400).close.tolist() == [10.0, 11.0]
    assert G.parse_chart(_live(1762871400, 1762894800), now=1762880000).close.tolist() == [10.0, 11.0]   # open session has no bar yet
    # futures: the open session shows up as two bars (its start and the live quote), both dropped
    assert G.parse_chart(_live(1762698600, 1762808400), now=1762790000).close.tolist() == []


def test_asof_before_is_strict():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-19", "2026-10-20"]))
    assert G.asof_before(s, pd.to_datetime(["2026-10-20", "2026-10-21"])).tolist() == [1.0, 2.0]


def test_asof_before_stale_is_nan():
    s = pd.Series([1.0], index=pd.to_datetime(["2026-10-09"]))
    out = G.asof_before(s, pd.to_datetime(["2026-10-09", "2026-10-16", "2026-10-19"]))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1.0 and np.isnan(out.iloc[2])


def test_sync_writes_nothing_when_one_series_fails(tmp_path, monkeypatch):
    calls = []

    def fetch(symbol, start="2021-01-01"):
        calls.append(symbol)
        if len(calls) == 2:
            raise OSError("down")
        return pd.DataFrame({"date": pd.to_datetime(["2026-10-19"]), "close": [1.0]})

    monkeypatch.setattr(G, "fetch", fetch)
    with pytest.raises(OSError):
        G.sync(tmp_path / "g.parquet")
    assert list(tmp_path.iterdir()) == []


def test_sync_writes_long_table(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "fetch", lambda symbol, start="2021-01-01": pd.DataFrame({"date": pd.to_datetime(["2026-10-19"]), "close": [1.0]}))
    out = G.sync(tmp_path / "sub" / "g.parquet")
    assert list(out.columns) == ["series", "date", "close"] and sorted(out.series) == sorted(G.C.GLOBAL_SERIES)
    assert pd.read_parquet(tmp_path / "sub" / "g.parquet").equals(out)
