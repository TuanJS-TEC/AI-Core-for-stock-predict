"""Đặc trưng nhân quả giá/khối lượng/thị trường trên bảng lịch phiên.

Đầu vào: bảng từ Panel.on_calendar (open, high, low, close, volume; phiên thiếu = cả hàng NaN, không điền).
Mọi giá trị tại hàng t chỉ dùng các hàng <= t. Rolling trailing, min_periods = độ dài cửa sổ; shift theo hàng lịch.
Module thuần hàm: không I/O, không trạng thái toàn cục.
"""
import math

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "ret_60",
    "d_ma20", "d_ma60", "dd_60",
    "std_10", "std_20", "std_60", "std_ratio_20_60", "downside_20",
    "c_o", "h_l", "o_pc", "close_loc",
    "v_ma20", "logv_rel60",
    "mret1", "mret5", "mvol20", "ex_ret1", "ex_ret5",
]
MARKET_COLS = ["mret1", "mret5", "mvol20"]


def _ret(close: pd.Series, k: int) -> pd.Series:
    return np.log(close / close.shift(k))


def market_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Proxy thị trường = trung bình đều log-return 1 phiên của các mã (cùng chỉ số lịch)."""
    r = pd.DataFrame({s: _ret(f["close"], 1) for s, f in frames.items()})
    mret1 = r.mean(axis=1, skipna=True).where(r.notna().sum(axis=1) >= math.ceil(len(frames) / 2))
    return pd.DataFrame(
        {
            "mret1": mret1,
            "mret5": mret1.rolling(5, min_periods=5).sum(),
            "mvol20": mret1.rolling(20, min_periods=20).std(),
        },
        index=r.index,
    )[MARKET_COLS]


def compute(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    o, h, lo, c, v = (frame[k] for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=frame.index)
    for k in (1, 2, 5, 10, 20, 60):
        out[f"ret_{k}"] = _ret(c, k)
    r1 = out["ret_1"]

    out["d_ma20"] = c / c.rolling(20, min_periods=20).mean() - 1
    out["d_ma60"] = c / c.rolling(60, min_periods=60).mean() - 1
    out["dd_60"] = c / c.rolling(60, min_periods=60).max() - 1

    for w in (10, 20, 60):
        out[f"std_{w}"] = r1.rolling(w, min_periods=w).std()
    out["std_ratio_20_60"] = out["std_20"] / out["std_60"].where(out["std_60"] > 0)
    out["downside_20"] = np.sqrt((r1.clip(upper=0) ** 2).rolling(20, min_periods=20).mean())

    out["c_o"] = np.log(c / o)
    out["h_l"] = np.log(h / lo)
    out["o_pc"] = np.log(o / c.shift(1))
    hl_equal = h == lo  # NaN so sánh -> False
    out["close_loc"] = (c - lo) / (h - lo).where(~hl_equal)

    out["v_ma20"] = v / v.shift(1).rolling(20, min_periods=20).mean()
    out["logv_rel60"] = np.log1p(v) - np.log1p(v).shift(1).rolling(60, min_periods=60).mean()

    out[MARKET_COLS] = market[MARKET_COLS].reindex(frame.index)
    out["ex_ret1"] = r1 - out["mret1"]
    out["ex_ret5"] = out["ret_5"] - out["mret5"]

    out = out[FEATURES].replace([np.inf, -np.inf], np.nan)  # mẫu số 0 (vd. v_ma20) -> NaN, không inf
    out["flag_hl_equal"] = hl_equal
    return out
