import pandas as pd

from ai_core.finhay import netflow_rows, upsert


def test_netflow_rows_and_upsert(tmp_path):
    data = {"resolution": "day", "points": [
        {"period": "2026-10-28", "buyValue": 5, "sellValue": 3, "netValue": 2, "sessions": 1},
        {"period": "2026-10-29", "buyValue": 1, "sellValue": 4, "netValue": -3, "sessions": 1}]}
    d = netflow_rows("FPT", "foreign", data)
    assert list(d.columns) == ["date", "buyValue", "sellValue", "netValue", "symbol", "investor"]
    assert d["date"].iloc[0] == pd.Timestamp("2026-10-28")

    p, keys = tmp_path / "n.parquet", ["symbol", "investor", "date"]
    upsert(p, d, keys)
    restated = d.iloc[[1]].assign(netValue=-9)
    out = upsert(p, restated, keys)                 # rerun: same session restated -> one row, new value wins
    assert len(out) == 2 and out["netValue"].tolist() == [2, -9]
