"""Panel giữ OHLCV chuẩn, lịch phiên và các ngày thiếu không giải thích được."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import Panel, build_panel

TICKERS = ["AAA", "BBB", "CCC"]
DAYS = pd.bdate_range("2024-01-02", periods=80)
GAP = DAYS[40]
NEXT = DAYS[41]


@pytest.fixture
def stock(tmp_path):
    """3 mã, 80 phiên, phiên thứ 40 bị bỏ ở cả 3 mã."""
    for k, t in enumerate(TICKERS):
        close = 10.0 + k + np.arange(80) * 0.1
        df = pd.DataFrame({"time": (DAYS + pd.Timedelta(hours=C.SESSION_HOUR_UTC)).tz_localize("UTC"),
                           "open": close, "high": close + 1, "low": close - 1, "close": close,
                           "volume": np.arange(80, dtype="int64") * 100 + 1})
        df.drop(index=40).reset_index(drop=True).to_parquet(tmp_path / C.STOCK_FILE.format(ticker=t), index=False)
    return tmp_path


def holidays(*days):
    return np.array(days, dtype="datetime64[D]")


def test_gap_not_in_holidays_is_reported_and_left_nan(stock):
    with pytest.warns(UserWarning, match="1 "):
        panel = build_panel(TICKERS, stock_dir=stock, holidays=holidays())
    assert list(panel.gaps) == [GAP]
    bars = panel.on_calendar("AAA")
    assert bars.loc[GAP].isna().all()
    assert bars.loc[NEXT].notna().all()          # không điền: hàng liền sau vẫn là giá thật
    assert bars.loc[NEXT, "close"] == pytest.approx(10.0 + 41 * 0.1)


def test_gap_listed_as_holiday_is_not_a_gap(stock):
    panel = build_panel(TICKERS, stock_dir=stock, holidays=holidays(GAP))
    assert len(panel.gaps) == 0
    assert GAP not in panel.calendar


def test_data_day_beats_holiday_list(stock):
    panel = build_panel(TICKERS, stock_dir=stock, holidays=holidays(GAP, NEXT))
    assert NEXT in panel.calendar
    assert panel.on_calendar("AAA").loc[NEXT].notna().all()


def test_log_close_unchanged(stock):
    panel = build_panel(TICKERS, stock_dir=stock, holidays=holidays(GAP))
    for t in TICKERS:
        assert np.array_equal(panel.log_close[t], np.log(panel.ohlcv[t]["close"].to_numpy()))
    assert list(panel.ohlcv["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    assert (panel.ohlcv["AAA"].dtypes == np.float64).all()
    assert panel.ohlcv["AAA"].index.equals(panel.dates)


def test_legacy_panel_constructor_still_works():
    dates = pd.bdate_range("2024-01-02", periods=5)
    panel = Panel(dates, dates.to_numpy(), {"AAA": np.zeros(5)})
    assert panel.ohlcv is None
    with pytest.raises(ValueError, match="ohlcv"):
        panel.on_calendar("AAA")
