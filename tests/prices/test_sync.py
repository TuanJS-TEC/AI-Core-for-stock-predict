"""Kiểm thử ai_core/prices/sync.py: mỗi phiên đúng một dòng, không gọi vnstock khi --no-fetch."""
import pandas as pd

import ai_core.prices.sync as S
from ai_core.prices import config as C


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
    assert calls == ["2026-10-26", "2026-10-26"]                  # overlap start (< 20 sessions stored = first day), then full
    assert list(pq.close) == [22.55, 22.38, 22.35]


def test_big_move_after_merge_triggers_full_refetch(tmp_path, monkeypatch):
    # overlap 2026-10-27 matches (33.55) but 10-28 = 22.35 (-33%): the new rows sit on another basis
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.55, 22.35])
    full = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    seq = iter([inc, full])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: next(seq))
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


def test_empty_refetch_keeps_stored_data(tmp_path, monkeypatch, caplog):
    # basis changed (overlap 10-27 mismatches) but the full refetch comes back empty (rate limit):
    # the mixed series [33.8, 22.38, 22.35] must not be written; parquet and CSV keep the stored rows
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.38, 22.35])
    seq = iter([inc, pd.DataFrame()])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: next(seq))
    assert calls == ["2026-10-26", "2026-10-26"]
    assert list(pq.close) == [33.8, 33.55]
    assert list(pd.read_csv(tmp_path / "AAA_stock.csv").close) == [33.8, 33.55]
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_truncated_refetch_keeps_stored_data(tmp_path, monkeypatch, caplog):
    days = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27"]
    stored = frame([d + " 07:00:00" for d in days], [33.0, 33.1, 33.2, 33.3, 33.4, 33.8, 33.55])
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.38, 22.35])
    short = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    seq = iter([inc, short])
    calls, pq = _run(tmp_path, monkeypatch, stored, lambda s: next(seq))
    assert len(calls) == 2
    assert list(pq.close) == list(stored.close)
    assert list(pd.read_csv(tmp_path / "AAA_stock.csv").close) == list(stored.close)
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_refetch_sleeps_before_next_call(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(S.time, "sleep", slept.append)
    csv = tmp_path / "AAA_stock.csv"
    frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55]).to_csv(csv, index=False)
    full = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda sym, start_date, end_date, **k: full[full.time >= start_date].reset_index(drop=True))
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0.5)
    assert slept == [0.5, 0.5]                                     # one pause per vnstock call


def _run_full(tmp_path, monkeypatch, responder):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.8, 33.55, 22.35])
    csv = tmp_path / "AAA_stock.csv"
    stored.to_csv(csv, index=False)
    calls = []

    def fake(symbol, start_date, end_date, max_retries=3):
        calls.append(start_date)
        return responder()
    monkeypatch.setattr(S, "fetch_latest_ohlcv", fake)
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0, full_refresh=True)
    return calls, pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA"))


def test_full_refresh_replaces_history(tmp_path, monkeypatch):
    full = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    calls, pq = _run_full(tmp_path, monkeypatch, lambda: full)
    assert calls == ["2026-10-26"]                                # one fetch, from the first stored day
    assert list(pq.close) == [22.55, 22.38, 22.35]


def test_full_refresh_empty_keeps_data(tmp_path, monkeypatch):
    calls, pq = _run_full(tmp_path, monkeypatch, lambda: pd.DataFrame())
    assert len(calls) == 1
    assert list(pq.close) == [33.8, 33.55, 22.35]


def test_full_refresh_truncated_keeps_data(tmp_path, monkeypatch, caplog):
    # the stored history starts 10-26; a refetch that starts weeks later must not replace it
    stored = frame(["2026-09-28 07:00:00", "2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.0, 33.8, 33.55])
    csv = tmp_path / "AAA_stock.csv"
    stored.to_csv(csv, index=False)
    late = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [22.55, 22.38])
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda *a, **k: late)
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0, full_refresh=True)
    assert list(pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA")).close) == [33.0, 33.8, 33.55]
    assert list(pd.read_csv(csv).close) == [33.0, 33.8, 33.55]
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_drop_open_session_before_close():
    df = frame(["2026-10-29 07:00:00", "2026-10-30 07:00:00"], [1.0, 2.0])
    out = S.drop_open_session(df, pd.Timestamp("2026-10-30 09:45", tz="Asia/Ho_Chi_Minh"))
    assert list(out.close) == [1.0]


def test_drop_open_session_after_close():
    df = frame(["2026-10-29 07:00:00", "2026-10-30 07:00:00"], [1.0, 2.0])
    out = S.drop_open_session(df, pd.Timestamp("2026-10-30 15:00", tz="Asia/Ho_Chi_Minh"))
    assert list(out.close) == [1.0, 2.0]


def test_init_writes_csv_from_start(tmp_path, monkeypatch):
    calls = []

    def fake(sym, start_date, end_date, **k):
        calls.append((sym, start_date))
        return frame(["2021-01-04 00:00:00", "2021-01-05 00:00:00"], [1.0, 2.0])

    monkeypatch.setattr(S, "fetch_latest_ohlcv", fake)
    path = S.init_stock_file("bid", str(tmp_path))
    assert calls == [("BID", "2021-01-04")] and path == str(tmp_path / "BID_stock.csv")
    back = pd.read_csv(path)
    assert list(back.columns) == list(C.RAW_COLS)
    assert list(back["time"]) == ["2021-01-04 07:00:00", "2021-01-05 07:00:00"]


def test_init_does_not_overwrite(tmp_path, monkeypatch):
    csv = tmp_path / "BID_stock.csv"
    csv.write_text("keep")
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    assert S.init_stock_file("BID", str(tmp_path)) == str(csv)
    assert csv.read_text() == "keep"


def test_init_returns_none_on_empty_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda *a, **k: pd.DataFrame())
    assert S.init_stock_file("BID", str(tmp_path)) is None
    assert not (tmp_path / "BID_stock.csv").exists()


def _sessions(n):
    days = pd.bdate_range("2026-09-01", periods=n)
    return S.normalize_sessions(frame([d.strftime("%Y-%m-%d 07:00:00") for d in days], [10.0 + i * 0.01 for i in range(n)]))


def test_clean_types_drops_nan_volume_instead_of_zeroing():
    df = frame(["2026-10-15 07:00:00", "2026-10-16 07:00:00"], [1.0, 2.0])
    df["volume"] = [100, None]
    out = S.clean_types(df)
    assert len(out) == 1 and out.close.tolist() == [1.0]
    assert out.volume.dtype == "int64"


def test_clean_types_keeps_real_zero_volume_and_logs_dropped_rows(caplog):
    df = frame(["2026-10-15 07:00:00", "2026-10-16 07:00:00"], [1.0, -2.0])
    df["volume"] = [0, 100]
    out = S.clean_types(df)
    assert out.volume.tolist() == [0]
    assert any(r.levelname == "ERROR" and "nonpositive_price" in r.getMessage() for r in caplog.records)


def test_basis_changed_sees_open_high_low():
    stored = S.normalize_sessions(frame(["2026-10-15 07:00:00", "2026-10-16 07:00:00"], [1.0, 2.0]))
    merged = stored.copy()
    merged.loc[0, "open"] += 0.5
    assert S.basis_changed(stored, merged)
    assert not S.basis_changed(stored, stored)


def test_quote_gets_explicit_source(monkeypatch):
    kw = {}

    class FakeQuote:
        def __init__(self, **k):
            kw.update(k)

        def history(self, **k):
            return pd.DataFrame()
    monkeypatch.setattr(S, "Quote", FakeQuote)
    S.fetch_latest_ohlcv("AAA", "2026-10-01", "2026-10-02")
    assert kw == {"source": C.SYNC_SOURCE, "symbol": "AAA"}


def test_overlap_start_is_20th_last_session():
    stored = _sessions(30)
    assert S.overlap_start(stored) == stored.time.iloc[-20].strftime("%Y-%m-%d")
    short = _sessions(5)
    assert S.overlap_start(short) == short.time.iloc[0].strftime("%Y-%m-%d")


def test_process_fetches_from_overlap_start(tmp_path, monkeypatch):
    stored = _sessions(30)
    calls, _ = _run(tmp_path, monkeypatch, stored, lambda s: pd.DataFrame())
    assert calls == [S.overlap_start(stored)]


def test_raw_response_saved_before_merge(tmp_path, monkeypatch):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.55, 33.9])
    _run(tmp_path, monkeypatch, stored, lambda s: inc)
    assert len(list((tmp_path / "raw").glob("*/AAA.parquet"))) == 1


def test_failed_csv_write_keeps_both_old_files(tmp_path, monkeypatch):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55])
    _run(tmp_path, monkeypatch, stored, lambda s: pd.DataFrame())
    pq, csv = tmp_path / C.STOCK_FILE.format(ticker="AAA"), tmp_path / "AAA_stock.csv"
    before = (pq.read_bytes(), csv.read_bytes())

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    inc = frame(["2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.55, 33.9])
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda *a, **k: inc)
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0)
    assert (pq.read_bytes(), csv.read_bytes()) == before
    assert not list(tmp_path.glob("*.tmp"))


def _full_refresh_with_bad_rows(tmp_path, monkeypatch, volumes):
    stored = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [33.8, 33.55, 22.35])
    csv = tmp_path / "AAA_stock.csv"
    stored.to_csv(csv, index=False)
    bad = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    bad["volume"] = volumes
    monkeypatch.setattr(S, "fetch_latest_ohlcv", lambda *a, **k: bad)
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0, full_refresh=True)
    return pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA"))


def test_full_refetch_all_rows_invalid_keeps_stored_data(tmp_path, monkeypatch, caplog):
    pq = _full_refresh_with_bad_rows(tmp_path, monkeypatch, [None, None, None])     # clean_types drops every row
    assert list(pq.close) == [33.8, 33.55, 22.35]
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_full_refetch_losing_a_mid_row_keeps_stored_data(tmp_path, monkeypatch):
    pq = _full_refresh_with_bad_rows(tmp_path, monkeypatch, [100, None, 100])        # would drop 10-27 for good
    assert list(pq.close) == [33.8, 33.55, 22.35]
