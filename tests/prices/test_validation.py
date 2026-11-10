"""Kiểm thử src/validation.py: cách ly dòng OHLCV lỗi và lịch ngày nghỉ."""
import numpy as np
import pandas as pd

from src import validation as V

NO_HOLIDAYS = np.array([], "datetime64[D]")


def test_check_ohlcv_quarantines_with_reason():
    nan = float("nan")
    df = pd.DataFrame({"time": pd.date_range("2026-10-19", periods=5),
                       "open": [10, 10, 10, 10, -1], "high": [11, 9, 11, 11, 1], "low": [9, 8, 9, 9, 1],
                       "close": [10, 10, 10, 10, 1], "volume": [100, 100, nan, -5, 0]})
    good, bad = V.check_ohlcv(df)
    assert good.index.tolist() == [0]
    assert dict(zip(bad.index, bad.reason)) == {1: "ohlc_order", 2: "volume_invalid", 3: "volume_invalid",
                                                4: "nonpositive_price"}


def test_zero_volume_is_valid():
    df = pd.DataFrame({"time": [pd.Timestamp("2026-10-19")], "open": [10.0], "high": [11.0], "low": [9.0],
                       "close": [10.0], "volume": [0]})
    good, bad = V.check_ohlcv(df)
    assert good.index.tolist() == [0] and bad.empty


def test_unexplained_gaps():
    d = pd.DatetimeIndex(["2026-10-19", "2026-10-20", "2026-10-22", "2026-10-23"])
    assert V.unexplained_gaps(d, NO_HOLIDAYS).tolist() == [pd.Timestamp("2026-10-21")]
    assert len(V.unexplained_gaps(d, np.array(["2026-10-21"], "datetime64[D]"))) == 0
    weekend = pd.DatetimeIndex(["2026-10-23", "2026-10-26"])          # thứ Sáu rồi thứ Hai: T7/CN không là gap
    assert len(V.unexplained_gaps(weekend, NO_HOLIDAYS)) == 0


def test_trading_calendar_observed_day_beats_holiday_list():
    hol = np.array(["2026-10-21"], "datetime64[D]")
    cal = V.trading_calendar("2026-10-19", "2026-10-23", hol)
    assert pd.Timestamp("2026-10-21") not in cal and len(cal) == 4
    cal = V.trading_calendar("2026-10-19", "2026-10-23", hol, observed=[pd.Timestamp("2026-10-21")])
    assert pd.Timestamp("2026-10-21") in cal and len(cal) == 5 and cal.is_monotonic_increasing


def test_load_holidays_unions_csv_and_config(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text("date,verified\n2026-04-30,false\n")
    h = V.load_holidays(f)
    assert np.datetime64("2026-04-30") in h and np.datetime64("2027-01-01") in h
