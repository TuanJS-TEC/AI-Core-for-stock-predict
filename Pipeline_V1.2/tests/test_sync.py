"""Kiểm thử sync_stock_parquet.py: mỗi phiên đúng một dòng, không gọi vnstock khi --no-fetch."""
import pandas as pd

import sync_stock_parquet as S
from src import config as C


def frame(times, close):
    n = len(times)
    return pd.DataFrame({"time": times, "open": close, "high": close, "low": close, "close": close,
                         "volume": [100] * n})


def test_normalize_sessions_dedups_same_day_with_different_hours_and_keeps_last():
    df = frame(["2026-10-15 07:00:00", "2026-10-16 07:00:00", "2026-10-16 00:00:00", "2026-10-19 00:00:00"],
               [1.0, 2.0, 2.5, 3.0])
    out = S.normalize_sessions(df)
    assert len(out) == 3
    assert out["time"].is_unique and out["time"].is_monotonic_increasing
    assert (out["time"].dt.hour == C.SESSION_HOUR_UTC).all() and str(out["time"].dt.tz) == "UTC"
    assert out.loc[out["time"].dt.day == 16, "close"].item() == 2.5      # bản ghi sau (vnstock) được giữ


def test_normalize_sessions_keeps_local_date_of_tz_aware_times():
    t = pd.to_datetime(["2026-10-16 00:00:00", "2026-10-19 00:00:00"]).tz_localize("Asia/Ho_Chi_Minh")
    out = S.normalize_sessions(frame(t, [1.0, 2.0]))
    assert [str(d.date()) for d in out["time"]] == ["2026-10-16", "2026-10-19"]


def test_process_without_fetch_cleans_existing_duplicates(tmp_path):
    csv = tmp_path / "AAA_stock.csv"
    frame(["2026-10-15 07:00:00", "2026-10-16 00:00:00", "2026-10-16 07:00:00", "2026-10-19 00:00:00"],
          [1.0, 2.0, 2.0, 3.0]).to_csv(csv, index=False)
    S.process_stock_file(str(csv), str(tmp_path), update_csv=True, delay_between_calls=0, fetch=False)
    pq = pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA"))
    assert len(pq) == 3 and pq["time"].is_unique
    back = pd.read_csv(csv)
    assert list(back["time"]) == ["2026-10-15 07:00:00", "2026-10-16 07:00:00", "2026-10-19 07:00:00"]
    # chạy lại không đổi gì (idempotent)
    S.process_stock_file(str(csv), str(tmp_path), update_csv=True, delay_between_calls=0, fetch=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA")), pq)


def test_check_alignment_flags_ticker_missing_last_day(tmp_path):
    full = ["2026-10-15 07:00:00", "2026-10-16 07:00:00"]
    for t, times in (("AAA", full), ("BBB", full[:1])):
        S.normalize_sessions(frame(times, [1.0] * len(times))).to_parquet(tmp_path / C.STOCK_FILE.format(ticker=t))
    assert not S.check_alignment(str(tmp_path), ["AAA", "BBB"])
    assert S.check_alignment(str(tmp_path), ["AAA"])


def test_merge_csv_utc_with_vnstock_ho_chi_minh_times(tmp_path, monkeypatch):
    """Lỗi thật: CSV đã chuẩn hóa (UTC) ghép với vnstock (Asia/Ho_Chi_Minh) thành cột object lẫn múi giờ."""
    csv = tmp_path / "AAA_stock.csv"
    frame(["2026-10-21 07:00:00", "2026-10-22 07:00:00", "2026-10-23 07:00:00"], [1.0, 2.0, 3.0]).to_csv(csv, index=False)
    t = pd.to_datetime(["2026-10-23 07:00:00", "2026-10-26 07:00:00"]).tz_localize("Asia/Ho_Chi_Minh")
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda *a, **k: frame(t, [3.004, 3.1]))   # chồng lấn lệch < SYNC_BASIS_TOL: cùng chuẩn giá
    S.process_stock_file(str(csv), str(tmp_path), update_csv=True, delay_between_calls=0, fetch=True)
    pq = pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA"))
    assert [str(d.date()) for d in pq["time"]] == ["2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26"]
    assert (pq["time"].dt.hour == C.SESSION_HOUR_UTC).all()
    assert pq["close"].tolist() == [1.0, 2.0, 3.004, 3.1]             # phiên trùng lấy bản vnstock


def test_wall_clock_accepts_mixed_object_column():
    s = pd.Series([pd.Timestamp("2026-10-23 07:00", tz="UTC"), pd.Timestamp("2026-10-26 07:00", tz="Asia/Ho_Chi_Minh"),
                   pd.Timestamp("2026-10-27 00:00")], dtype=object)
    assert [str(v) for v in S.wall_clock(s)] == ["2026-10-23 07:00:00", "2026-10-26 07:00:00", "2026-10-27 00:00:00"]


def _run(tmp_path, monkeypatch, stored, responder):
    """Chạy process_stock_file với fetch giả; trả về (danh sách start_date đã gọi, parquet)."""
    csv = tmp_path / "AAA_stock.csv"
    stored.to_csv(csv, index=False)
    calls = []

    def fake(symbol, start_date, end_date, max_retries=3):
        calls.append(start_date)
        return responder(start_date)
    monkeypatch.setattr(S, "fetch_latest_ohlcv", fake)
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0)
    return calls, pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA"))


def test_overlap_mismatch_triggers_full_refetch(tmp_path, monkeypatch):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    full = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: full[full.time >= s].reset_index(drop=True))
    assert calls == ["2026-10-27", "2026-10-26"]                  # incremental, then full from the first stored day
    assert list(pq.close) == [22.55, 22.38, 22.35]


def test_big_move_after_merge_triggers_full_refetch(tmp_path, monkeypatch):
    # overlap 2026-10-27 matches (33.55) but 10-28 = 22.35 (-33%): the new rows sit on another basis
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.55, 22.35])
    full = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: inc if s == "2026-10-27" else full)
    assert len(calls) == 2
    assert list(pq.close) == [22.55, 22.38, 22.35]


def test_consistent_update_fetches_once(tmp_path, monkeypatch):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.55, 33.9])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: inc)
    assert len(calls) == 1
    assert list(pq.close) == [33.8, 33.55, 33.9]


def test_old_gap_does_not_refetch(tmp_path, monkeypatch):
    # stored history already holds a >RET_BAND jump (missing sessions): not a basis change
    stored = frame(["2026-10-20 07:00:00", "2026-10-21 07:00:00", "2026-10-22 07:00:00"], [30.0, 33.0, 33.1])
    inc = frame(["2026-10-22 07:00:00", "2026-10-23 07:00:00"], [33.1, 33.2])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: inc)
    assert len(calls) == 1
    assert list(pq.close) == [30.0, 33.0, 33.1, 33.2]


def test_empty_fetch_keeps_data(tmp_path, monkeypatch):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: pd.DataFrame())
    assert len(calls) == 1
    assert list(pq.close) == [33.8, 33.55]
