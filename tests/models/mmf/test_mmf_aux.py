import numpy as np, pandas as pd, pytest
from ai_core.models.mmf import aux as A, config as C

CAL = pd.bdate_range("2023-01-02", periods=600)


def write(d, sym, dates, seed=0):
    """A {sym}_stock.parquet random walk on `dates`, time = date + 07:00 UTC."""
    c = 20 * np.exp(np.random.default_rng(seed).normal(0, 0.02, len(dates)).cumsum())
    pd.DataFrame({"time": (pd.DatetimeIndex(dates) + pd.Timedelta(hours=7)).tz_localize("UTC"), "open": c * 0.99, "high": c * 1.02,
                  "low": c * 0.98, "close": c, "volume": 1000}).to_parquet(d / f"{sym}_stock.parquet", index=False)


def test_load_frames_reindexes_without_filling(tmp_path):
    write(tmp_path, "AAA", CAL.delete(300))
    f = A.load_frames(tmp_path, CAL)["AAA"]
    assert f.index.equals(CAL) and list(f.columns) == ["open", "high", "low", "close", "volume"]
    assert f.loc[CAL[300]].isna().all()
    assert np.isfinite(f.loc[CAL[301]]).all() and f.close[CAL[301]] != f.close[CAL[299]]


def test_short_history_symbol_is_dropped(tmp_path):
    write(tmp_path, "BBB", CAL[-499:])
    write(tmp_path, "CCC", CAL[-500:])
    frames = A.load_frames(tmp_path, CAL)
    assert list(frames) == ["CCC"]
    assert frames["CCC"].close.iloc[:100].isna().all() and frames["CCC"].close.iloc[100:].notna().all()


def test_thesis_tickers_and_index_are_excluded(tmp_path):
    for s in ("AAA", "FPT", C.INDEX):
        write(tmp_path, s, CAL)
    assert list(A.load_frames(tmp_path, CAL)) == ["AAA"]
    ix = A.load_index(tmp_path, CAL)
    assert ix.index.equals(CAL) and ix.close.notna().all()


def test_load_index_missing_raises(tmp_path):
    with pytest.raises(ValueError):
        A.load_index(tmp_path, CAL)


def test_sync_index_keeps_bars_with_broken_ohlc_order(tmp_path):
    raw = pd.DataFrame({"time": pd.to_datetime(["2026-10-19 07:00", "2026-10-20 07:00", "2026-10-21 07:00"]), "open": [10.0, 10, 10],
                        "high": [11.0, 9, 11], "low": [9.0, 9, 9], "close": [10.0, 10.5, np.nan], "volume": [1, 1, 1]})
    A.sync_index(tmp_path, fetch=lambda: raw)
    ix = A.load_index(tmp_path, pd.bdate_range("2026-10-19", periods=3))
    assert ix.close.tolist()[:2] == [10.0, 10.5] and np.isnan(ix.close.iloc[2])


def glob_file(path, days=CAL):
    pd.DataFrame({"series": "spx", "date": days, "close": 1.0}).to_parquet(path, index=False)
    return path


def ext_dir(tmp_path):
    d = tmp_path / "ext"
    d.mkdir()
    for i, s in enumerate(("AAA", "BBB", C.INDEX)):
        write(d, s, CAL, seed=i)
    return d


def test_freeze_is_idempotent_and_load_detects_tampering(tmp_path):
    d, g = ext_dir(tmp_path), glob_file(tmp_path / "global.parquet")
    a = A.freeze(d, g, tmp_path / "aux")
    assert A.freeze(d, g, tmp_path / "aux") == a and a.name == A.aux_id_of(a)[:16]
    aux = A.load(a, CAL)
    assert list(aux.frames) == ["AAA", "BBB"] and aux.aux_id == A.aux_id_of(a) and len(aux.glob) == len(CAL)
    assert aux.index.close.notna().all()
    glob_file(a / "global.parquet", CAL[:-1])
    with pytest.raises(ValueError, match="sửa"):
        A.load(a, CAL)


def test_load_detects_tampered_stock_file(tmp_path):
    a = A.freeze(ext_dir(tmp_path), glob_file(tmp_path / "global.parquet"), tmp_path / "aux")
    write(a, "AAA", CAL, seed=9)
    with pytest.raises(ValueError):
        A.load(a, CAL)


def test_load_refuses_aux_older_than_calendar(tmp_path):
    a = A.freeze(ext_dir(tmp_path), glob_file(tmp_path / "global.parquet"), tmp_path / "aux")
    with pytest.raises(ValueError, match="cũ hơn"):
        A.load(a, CAL.append(pd.bdate_range(CAL[-1], periods=2)[1:]))
