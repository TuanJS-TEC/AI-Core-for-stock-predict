"""Kiểm thử src/snapshot.py: content hash, manifest, dataset_id và snapshot bất biến (dữ liệu tổng hợp trong tmp_path)."""
import json

import numpy as np
import pandas as pd
import pytest

import sync_stock_parquet as sync
from src import config as C
from src import snapshot as S
from src.data import build_panel, load_raw

NO_HOLIDAYS = np.array([], "datetime64[D]")
SYMS = ["AAA", "BBB"]


def make_df(base: float, drop: int | None = None) -> pd.DataFrame:
    days = pd.bdate_range("2026-10-19", periods=12)
    close = base + np.arange(12, dtype=float)
    df = pd.DataFrame({"time": (days + pd.Timedelta(hours=C.SESSION_HOUR_UTC)).tz_localize("UTC"),
                       "open": close, "high": close + 1, "low": close - 1, "close": close,
                       "volume": np.arange(12, dtype="int64") * 100})
    return df.drop(index=drop).reset_index(drop=True) if drop is not None else df


def write_stock(d, frames: dict[str, pd.DataFrame]) -> None:
    for sym, df in frames.items():
        df.to_parquet(d / C.STOCK_FILE.format(ticker=sym), index=False)


@pytest.fixture
def stock(tmp_path):
    write_stock(tmp_path, {"AAA": make_df(10.0), "BBB": make_df(20.0)})
    return tmp_path


def manifest_of(d, fetched_at="2026-11-09T08:00:00+00:00"):
    return S.build_manifest(d, "kbs", fetched_at, True, holidays=NO_HOLIDAYS)


def test_content_hash_changes_when_one_ohlc_value_changes_with_same_rows_and_last_date():
    df = make_df(10.0)
    other = df.copy()
    other.loc[5, "high"] += 0.5
    assert len(df) == len(other) and df.time.iloc[-1] == other.time.iloc[-1]
    assert S.content_hash(df) != S.content_hash(other)
    assert S.content_hash(df) == S.content_hash(df.sample(frac=1, random_state=0))


def test_dataset_id_changes_with_content_not_with_fetched_at(stock):
    a = manifest_of(stock, "2026-11-09T08:00:00+00:00")
    b = manifest_of(stock, "2026-11-10T08:00:00+00:00")
    assert a["fetched_at"] != b["fetched_at"] and a["dataset_id"] == b["dataset_id"]
    df = make_df(20.0)
    df.loc[3, "high"] += 0.5
    write_stock(stock, {"BBB": df})
    assert manifest_of(stock)["dataset_id"] != a["dataset_id"]


def test_manifest_records_policy_unknown_and_gaps(stock):
    m = manifest_of(stock)
    assert m["adjustment_policy"] == "unknown" and m["source"] == "kbs" and m["fetched"] is True
    assert set(m["tickers"]) == set(SYMS) and m["quality"]["AAA"] == {"unexplained_gaps": []}
    assert m["tickers"]["AAA"]["rows"] == 12 and set(m["libs"]) == {"vnstock", "pandas", "pyarrow"}
    assert isinstance(m["holidays_unverified"], int)
    write_stock(stock, {"AAA": make_df(10.0, drop=4)})       # bỏ 2026-10-23, ngày thường không nghỉ
    assert S.build_manifest(stock, "kbs", "x", False, holidays=NO_HOLIDAYS)["quality"]["AAA"] == \
        {"unexplained_gaps": ["2026-10-23"]}


def test_write_manifest_atomic_leaves_no_tmp(stock):
    m = manifest_of(stock)
    S.write_manifest(stock, m)
    assert json.loads((stock / "manifest.json").read_text()) == m
    assert not list(stock.glob("*.tmp"))


def test_freeze_is_idempotent_and_refuses_overwrite(stock):
    S.write_manifest(stock, manifest_of(stock))
    did = S.freeze(stock)
    assert S.freeze(stock) == did and did == manifest_of(stock)["dataset_id"]
    snap = S.snapshot_dir(stock, did)
    assert snap == stock / "snapshots" / did[:16] and (snap / "manifest.json").exists()
    (snap / C.STOCK_FILE.format(ticker="AAA")).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        S.freeze(stock)


def test_revision_isolation(stock):
    S.write_manifest(stock, manifest_of(stock))
    did = S.freeze(stock)
    old = build_panel(SYMS, stock_dir=stock)
    write_stock(stock, {"AAA": make_df(50.0), "BBB": make_df(60.0)})
    S.write_manifest(stock, manifest_of(stock))
    new = build_panel(SYMS, stock_dir=stock)
    assert new.dataset_id != did
    frozen = build_panel(SYMS, stock_dir=S.snapshot_dir(stock, did))
    assert frozen.dataset_id == did == old.dataset_id
    assert all(np.array_equal(frozen.log_close[s], old.log_close[s]) for s in SYMS)
    assert not np.array_equal(frozen.log_close["AAA"], new.log_close["AAA"])


def test_panel_without_manifest_has_none_dataset_id(stock):
    assert build_panel(SYMS, stock_dir=stock).dataset_id is None


def test_injected_raw_gets_no_dataset_id(stock):
    S.write_manifest(stock, manifest_of(stock))
    assert build_panel(SYMS, load_raw(SYMS, stock), stock_dir=stock).dataset_id is None
    assert build_panel(SYMS, stock_dir=stock).dataset_id == manifest_of(stock)["dataset_id"]


def test_manifest_not_matching_loaded_data_gives_none(stock):
    S.write_manifest(stock, manifest_of(stock))
    write_stock(stock, {"AAA": make_df(10.0, drop=11), "BBB": make_df(20.0, drop=11)})   # parquet đổi, manifest cũ
    with pytest.warns(UserWarning):
        assert build_panel(SYMS, stock_dir=stock).dataset_id is None


def test_manifest_missing_ticker_gives_none(stock):
    m = manifest_of(stock)
    del m["tickers"]["BBB"]
    S.write_manifest(stock, m)
    with pytest.warns(UserWarning):
        assert build_panel(SYMS, stock_dir=stock).dataset_id is None


@pytest.mark.parametrize("text", ["{not json", "{}", '{"dataset_id": "x"}'])
def test_malformed_manifest_gives_none_without_raising(stock, text):
    (stock / "manifest.json").write_text(text)
    with pytest.warns(UserWarning):
        assert build_panel(SYMS, stock_dir=stock).dataset_id is None


def test_sync_main_survives_manifest_failure_and_removes_stale_manifest(stock, monkeypatch):
    for sym, base in (("AAA", 10.0), ("BBB", 20.0)):
        df = make_df(base)
        df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(stock / f"{sym}_stock.csv", index=False)
    (stock / "manifest.json").write_text('{"dataset_id": "stale"}')

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(sync, "write_manifest", boom)
    assert sync.main(["--stock-dir", str(stock), "--no-fetch"]) is True      # giá trị của check_alignment
    assert not (stock / "manifest.json").exists()
