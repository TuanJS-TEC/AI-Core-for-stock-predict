"""Lịch phiên HOSE: ngày nghỉ, phiên thứ n sau một ngày, phiên lẽ ra đã có dữ liệu lúc now."""
import numpy as np
import pandas as pd

from . import config as C

WEEKMASK = "1111100"                        # thứ Hai .. thứ Sáu
ONE_DAY = np.timedelta64(1, "D")


def market_holidays(panel) -> np.ndarray:
    """Ngày nghỉ (datetime64[D]): ngày thường không có phiên trong dữ liệu và ngày khai báo trong C.MARKET_HOLIDAYS.
    panel chỉ cần có `.dates`."""
    past = pd.bdate_range(panel.dates[0], panel.dates[-1]).difference(panel.dates)
    return np.union1d(past.values.astype("datetime64[D]"), np.array(C.MARKET_HOLIDAYS, dtype="datetime64[D]"))


def day(d) -> np.datetime64:
    return np.datetime64(pd.Timestamp(d).date(), "D")


def session_after(d, n: int, holidays: np.ndarray) -> pd.Timestamp:
    """Phiên giao dịch thứ n sau ngày d (d là ngày nghỉ thì tính từ phiên kế tiếp)."""
    return pd.Timestamp(np.busday_offset(day(d), n, roll="forward", weekmask=WEEKMASK, holidays=holidays))


def latest_ready_session(now: pd.Timestamp, holidays: np.ndarray) -> pd.Timestamp:
    """Phiên mới nhất lẽ ra đã có dữ liệu lúc now: hôm nay nếu là ngày giao dịch và đã qua C.SESSION_DATA_READY,
    nếu không thì phiên giao dịch trước đó."""
    d = day(now) if now.strftime("%H:%M") >= C.SESSION_DATA_READY else day(now) - ONE_DAY
    return pd.Timestamp(np.busday_offset(d, 0, roll="backward", weekmask=WEEKMASK, holidays=holidays))


def missing_sessions(last_date, now: pd.Timestamp, holidays: np.ndarray) -> list[pd.Timestamp]:
    """Các phiên sau last_date lẽ ra đã có dữ liệu lúc now (rỗng nếu dữ liệu đã mới nhất)."""
    days = np.arange(day(last_date) + ONE_DAY, day(latest_ready_session(now, holidays)) + ONE_DAY)
    return [pd.Timestamp(d) for d in days[np.is_busday(days, weekmask=WEEKMASK, holidays=holidays)]]
