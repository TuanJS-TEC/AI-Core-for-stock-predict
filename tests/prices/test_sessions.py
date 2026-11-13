"""Lịch phiên: ngày mục tiêu bỏ qua cuối tuần/ngày nghỉ, phiên thiếu lúc now."""
import numpy as np
import pandas as pd
import pytest

from ai_core.prices import config as C, sessions as S
from ai_core.prices.data import build_panel, load_raw


@pytest.fixture(scope="module")
def panel():
    return build_panel(C.TICKERS, load_raw())


def test_target_date_skips_weekends_and_holidays(panel):
    hol = S.market_holidays(panel)
    d = panel.dates
    for T in C.HORIZONS:                                            # trùng phiên thật của dữ liệu, kể cả qua Tết
        got = np.busday_offset(d[:-T].values.astype("datetime64[D]"), T, weekmask=S.WEEKMASK, holidays=hol)
        np.testing.assert_array_equal(got, d[T:].values.astype("datetime64[D]"))
    assert S.session_after("2026-04-29", 2, hol) == pd.Timestamp("2026-05-05")    # nghỉ 30/4, 1/5
    assert S.session_after("2026-10-23", 2, hol) == pd.Timestamp("2026-10-27")    # qua cuối tuần
    assert S.session_after("2026-12-31", 2, hol) == pd.Timestamp("2027-01-05")    # nghỉ 1/1 khai báo trong config
    fri = "2026-10-23"
    assert S.missing_sessions(fri, pd.Timestamp("2026-10-25 10:00"), hol) == []           # Chủ Nhật
    assert S.missing_sessions(fri, pd.Timestamp("2026-10-26 10:00"), hol) == []           # thứ Hai, phiên chưa xong
    assert S.missing_sessions(fri, pd.Timestamp("2026-10-26 16:00"), hol) == [pd.Timestamp("2026-10-26")]
    assert S.missing_sessions("2026-04-29", pd.Timestamp("2026-05-02 09:00"), hol) == []  # 30/4, 1/5 là ngày nghỉ
