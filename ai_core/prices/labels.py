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


XS_VERSION = "xs-k0.43-w60-v1"


def make_xs_labels(frames: dict[str, pd.DataFrame], h: int, k: float = C.K_ADAPT, window: int = C.SIGMA_WINDOW) -> dict[str, pd.DataFrame]:
    """Nhãn trung hòa thị trường (phụ lục 2026-11-11 của mmf_preregistration.md): mã -> bảng cùng COLUMNS với make_labels.

    e = ret 1 phiên trừ median qua các mã cùng phiên; θ = k·σ_e·√h với σ_e trên `window` phiên kết thúc tại t;
    realized_return = tổng e của t+1..t+h. Một mã thiếu close ở một phiên làm e của mọi mã ở phiên đó là NaN (median
    không bỏ NaN), nên hàng có phiên thiếu trong [t, t+h] là missing_session và trong cửa sổ σ là bad_sigma, cho mọi mã."""
    close = pd.DataFrame({s: f["close"] for s, f in frames.items()})
    e = np.log(close / close.shift(1))
    e = e.sub(e.median(axis=1, skipna=False), axis=0)
    theta = k * e.rolling(window, min_periods=window).std() * np.sqrt(h)
    realized = e.rolling(h, min_periods=h).sum().shift(-h)
    target_session = close.index.to_series().shift(-h)
    missing = close.isna().any(axis=1).rolling(h + 1, min_periods=h + 1).sum().shift(-h) > 0

    out = {}
    for s in close.columns:
        status = pd.Series("ok", index=close.index, dtype=object)
        status[~(theta[s] > 0)] = "bad_sigma"
        status[missing] = "missing_session"
        status[target_session.isna()] = "not_matured"
        label = pd.Series(pd.NA, index=close.index, dtype="Int64")
        ok = (status == "ok").to_numpy()
        label[ok] = assign_labels(realized[s].to_numpy()[ok], theta[s].to_numpy()[ok])
        d = pd.DataFrame({"target_session": target_session, "label_end_date": target_session, "realized_return": realized[s],
                          "theta_used": theta[s], "label": label, "status": status, "label_version": XS_VERSION},
                         index=close.index)[COLUMNS]
        d.index.name = "as_of_session"
        out[s] = d
    return out


MAG_VERSION = "mag-median-v1"


def make_mag_labels(frames: dict[str, pd.DataFrame], h: int, band: float = C.SYNC_RET_BAND) -> dict[str, pd.DataFrame]:
    """Nhãn độ lớn biến động (phụ lục 2026-11-12 của mmf_preregistration.md): mã -> bảng cùng COLUMNS với make_labels.

    r = ln(close[t+h]/close[t]); label 1 (Mạnh) nếu |r| > trung vị |r| các mã cùng phiên, ngược lại 0 (Yếu); theta_used = trung vị.
    Một mã thiếu close trong [t, t+h] làm hàng của mọi mã là missing_session. Mã có |lợi suất 1 phiên| > band trong t+1..t+h
    (nghi đổi chuẩn điều chỉnh giá) có status seam và không tham gia trung vị."""
    close = pd.DataFrame({s: f["close"] for s, f in frames.items()})
    realized = np.log(close.shift(-h) / close)
    jump = (close / close.shift(1) - 1).abs() > band
    seam = jump.astype(float).rolling(h, min_periods=h).max().shift(-h) > 0
    mag = realized.abs().mask(seam)
    med = mag.median(axis=1)
    target_session = close.index.to_series().shift(-h)
    missing = close.isna().any(axis=1).rolling(h + 1, min_periods=h + 1).sum().shift(-h) > 0

    out = {}
    for s in close.columns:
        status = pd.Series("ok", index=close.index, dtype=object)
        status[seam[s]] = "seam"
        status[missing] = "missing_session"
        status[target_session.isna()] = "not_matured"
        label = pd.Series(pd.NA, index=close.index, dtype="Int64")
        ok = (status == "ok").to_numpy()
        label[ok] = (mag[s].to_numpy()[ok] > med.to_numpy()[ok]).astype(np.int64)
        d = pd.DataFrame({"target_session": target_session, "label_end_date": target_session, "realized_return": realized[s],
                          "theta_used": med, "label": label, "status": status, "label_version": MAG_VERSION},
                         index=close.index)[COLUMNS]
        d.index.name = "as_of_session"
        out[s] = d
    return out
