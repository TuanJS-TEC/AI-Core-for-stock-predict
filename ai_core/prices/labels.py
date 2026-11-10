"""Nhãn ba lớp thích ứng theo biến động, có phiên bản, trên bảng lịch phiên.

Đầu vào: bảng từ Panel.on_calendar (cột close; phiên thiếu = NaN, không điền), chỉ số = lịch phiên.
θ và σ tại hàng t chỉ dùng các hàng <= t; chỉ target_session / realized_return nhìn tới t+h.
Thứ tự status: not_matured (t+h vượt cuối lịch) -> missing_session (NaN close trong [t, t+h]) -> bad_sigma -> ok.
"""
import numpy as np
import pandas as pd

from . import config as C
from .data import assign_labels

COLUMNS = ["target_session", "label_end_date", "realized_return", "theta_used", "label", "status", "label_version"]


def make_labels(frame: pd.DataFrame, h: int, k: float = C.K_ADAPT, window: int = C.SIGMA_WINDOW) -> pd.DataFrame:
    close = frame["close"]
    ret_1 = np.log(close / close.shift(1))
    sigma = ret_1.rolling(window, min_periods=window).std()
    theta = k * sigma * np.sqrt(h)
    target_session = frame.index.to_series().shift(-h)
    realized = np.log(close.shift(-h) / close)

    status = pd.Series("ok", index=frame.index, dtype=object)
    status[~(theta > 0)] = "bad_sigma"                                   # NaN hoặc <= 0
    status[close.isna().rolling(h + 1, min_periods=h + 1).sum().shift(-h) > 0] = "missing_session"
    status[target_session.isna()] = "not_matured"                        # ghi cuối: ưu tiên cao nhất

    label = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    ok = (status == "ok").to_numpy()
    # assign_labels so sánh r với theta theo từng phần tử, nên truyền được mảng theta riêng cho từng hàng
    label[ok] = assign_labels(realized.to_numpy()[ok], theta.to_numpy()[ok])

    out = pd.DataFrame(
        {
            "target_session": target_session,
            "label_end_date": target_session,
            "realized_return": realized,
            "theta_used": theta,
            "label": label,
            "status": status,
            "label_version": C.LABEL_VERSION,
        },
        index=frame.index,
    )[COLUMNS]
    out.index.name = "as_of_session"
    return out
