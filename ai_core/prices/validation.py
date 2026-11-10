"""Kiểm tra dữ liệu giá ngày: cách ly dòng OHLCV lỗi và lịch ngày nghỉ của HOSE.

Dòng lỗi không bị sửa (không điền, không clip); chỉ tách ra kèm lý do. Volume 0 là hợp lệ.
"""
import numpy as np
import pandas as pd

from . import config as C

OHLC = ["open", "high", "low", "close"]


def check_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (good, bad); bad giữ chỉ số gốc, thêm cột `reason`, lý do đầu tiên theo thứ tự kiểm."""
    p = df[OHLC]
    o, h, l, c = (df[k] for k in OHLC)
    order = (h < pd.concat([o, c, l], axis=1).max(axis=1)) | (l > pd.concat([o, c, h], axis=1).min(axis=1))
    reason = pd.Series(np.select([p.isna().any(axis=1) | (p <= 0).any(axis=1), order,
                                  df["volume"].isna() | (df["volume"] < 0)],
                                 ["nonpositive_price", "ohlc_order", "volume_invalid"], default=""),
                       index=df.index)
    return df[reason == ""], df[reason != ""].assign(reason=reason[reason != ""])


def load_holidays(path=C.HOLIDAYS_FILE) -> np.ndarray:
    """Ngày nghỉ (datetime64[D]) trong CSV `date,verified` hợp với C.MARKET_HOLIDAYS."""
    csv = pd.read_csv(path)["date"].to_numpy().astype("datetime64[D]")
    return np.union1d(csv, np.array(C.MARKET_HOLIDAYS, dtype="datetime64[D]"))


def trading_calendar(first, last, holidays: np.ndarray, observed=()) -> pd.DatetimeIndex:
    """Ngày thường trong [first, last] trừ ngày nghỉ, hợp với ngày đã quan sát có dữ liệu thật."""
    days = pd.bdate_range(first, last)
    days = days[~np.isin(days.values.astype("datetime64[D]"), holidays)]
    return days.union(pd.DatetimeIndex(observed))


def unexplained_gaps(dates: pd.DatetimeIndex, holidays: np.ndarray) -> pd.DatetimeIndex:
    """Ngày thường trong [min, max] không có trong dates và không là ngày nghỉ."""
    days = pd.bdate_range(dates.min(), dates.max())
    return days[~days.isin(dates) & ~np.isin(days.values.astype("datetime64[D]"), holidays)]
