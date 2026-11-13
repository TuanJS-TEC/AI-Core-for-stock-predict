"""Đọc dữ liệu giá đã đồng bộ (stock/*.parquet) thành Panel: trục ngày chung, OHLCV, lịch phiên, khoảng trống.

Mọi mã phải có cùng dãy ngày, mỗi phiên một dòng (ai_core/prices/sync.py bảo đảm). Panel không chứa gì về mô hình."""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .validation import load_holidays, trading_calendar, unexplained_gaps


def assign_labels(r: np.ndarray, theta: float) -> np.ndarray:
    """0 = Giảm, 1 = Đi ngang, 2 = Tăng."""
    lab = np.full(len(r), C.CLASS_FLAT, dtype=np.int64)
    lab[r > theta] = C.CLASS_UP
    lab[r < -theta] = C.CLASS_DOWN
    return lab


# ---------------------------------------------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------------------------------------------
def stock_path(ticker: str, stock_dir=None):
    return Path(C.STOCK_DIR if stock_dir is None else stock_dir) / C.STOCK_FILE.format(ticker=ticker)


def load_ohlcv(ticker: str, stock_dir=None) -> pd.DataFrame:
    """OHLCV của một mã, chỉ số là ngày (đã bỏ giờ) để căn các mã, cột stamp giữ thời điểm gốc (UTC, không múi giờ)."""
    df = pd.read_parquet(stock_path(ticker, stock_dir), columns=list(C.RAW_COLS)).sort_values(C.TIME_COL)
    t = pd.to_datetime(df[C.TIME_COL])
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)
    df["stamp"] = t                              # thời điểm gốc, duy nhất theo dòng, dùng làm cột date đầu ra
    df.index = pd.DatetimeIndex(t.dt.normalize(), name="date")
    return df.drop(columns=C.TIME_COL)


def load_raw(tickers=C.TICKERS, stock_dir=None) -> dict[str, pd.DataFrame]:
    return {t: load_ohlcv(t, stock_dir) for t in tickers}


def check_aligned(raw: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Mọi mã phải có đúng cùng dãy ngày theo từng dòng, mỗi phiên một dòng."""
    first = next(iter(raw))
    ref = raw[first].index
    for t, df in raw.items():
        if not df.index.equals(ref) or not df["stamp"].equals(raw[first]["stamp"]):
            raise ValueError(f"{t}: dãy ngày khác mã {first}, không gộp theo ngày được")
    dup = ref[ref.duplicated()]
    if len(dup):
        raise ValueError(f"Ngày lặp {[str(d.date()) for d in dup]}; chạy `python jobs/sync_prices.py --no-fetch` "
                         "để làm sạch trước")
    return ref


@dataclass
class Panel:
    dates: pd.DatetimeIndex              # trục ngày chung của mọi mã (đã bỏ giờ)
    stamps: np.ndarray                   # thời điểm gốc từng dòng (datetime64, duy nhất), dùng làm ngày đầu ra
    log_close: dict[str, np.ndarray]     # mã -> (n,)
    dataset_id: str | None = None        # từ stock_dir/manifest.json; None nếu thư mục chưa có manifest
    ohlcv: dict[str, pd.DataFrame] | None = None   # mã -> OHLCV float64, chỉ số session_date (= dates); None nếu dựng tay
    calendar: pd.DatetimeIndex | None = None       # ngày thường trừ ngày nghỉ, hợp với ngày có dữ liệu
    gaps: pd.DatetimeIndex | None = None           # ngày thường không có dữ liệu và không là ngày nghỉ

    def on_calendar(self, ticker: str) -> pd.DataFrame:
        """OHLCV của mã trên lịch phiên; ngày thiếu là NaN, không điền."""
        if self.ohlcv is None or self.calendar is None:
            raise ValueError("Panel không có ohlcv/calendar; dựng bằng build_panel")
        return self.ohlcv[ticker].reindex(self.calendar)


def manifest_id(stock_dir, raw: dict[str, pd.DataFrame], tickers) -> str | None:
    """dataset_id của stock_dir/manifest.json nếu manifest khớp dữ liệu đã đọc (số dòng và phiên cuối từng mã), ngược
    lại None. Không bao giờ raise: manifest chỉ là metadata. Thiếu file thì im lặng, hỏng/lệch thì warnings.warn."""
    path = Path(C.STOCK_DIR if stock_dir is None else stock_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text())
        for t in tickers:
            info = m["tickers"][t]
            last = pd.Timestamp(info["last"])
            last = last.tz_convert(None) if last.tzinfo is not None else last
            if info["rows"] != len(raw[t]) or last != raw[t]["stamp"].iloc[-1]:
                raise ValueError(f"{t} lệch dữ liệu đã đọc")
        return m["dataset_id"]
    except Exception as e:
        warnings.warn(f"{path} không khớp dữ liệu đã đọc, bỏ dataset_id: {e!r}")
        return None


def build_panel(tickers=C.TICKERS, raw: dict[str, pd.DataFrame] | None = None, stock_dir=None,
                holidays: np.ndarray | None = None) -> Panel:
    injected = raw
    raw = load_raw(tickers, stock_dir) if raw is None else raw
    dates = check_aligned(raw)
    stamps = raw[next(iter(raw))]["stamp"].to_numpy()
    dataset_id = manifest_id(stock_dir, raw, tickers) if injected is None else None   # raw bơm vào: không rõ nguồn
    holidays = load_holidays() if holidays is None else holidays
    gaps = unexplained_gaps(dates, holidays)
    if len(gaps):
        warnings.warn(f"{len(gaps)} ngày thường không có dữ liệu và không là ngày nghỉ, đầu tiên {gaps[0].date()}")
    ohlcv = {t: raw[t][list(C.PRICE_COLS) + [C.VOLUME_COL]].astype(np.float64).rename_axis("session_date")
             for t in tickers}
    return Panel(dates, stamps, {t: np.log(raw[t][C.TARGET_COL].to_numpy(np.float64)) for t in tickers}, dataset_id,
                 ohlcv, trading_calendar(dates[0], dates[-1], holidays, observed=dates), gaps)
