import numpy as np
import pandas as pd
import pytest

import crosscheck_tcbs as X
from ai_core.prices import config as C

DAYS = pd.bdate_range("2026-10-01", periods=10)


def bars(days=DAYS, close=20.0, volume=1_000_000.0):
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": volume},
                        index=pd.DatetimeIndex(days, name="date"))


def test_identical_sources_give_no_rows():
    assert X.compare(bars(), bars()).empty


def test_price_difference_above_tolerance_only():
    fresh = bars()
    fresh.loc[DAYS[3], "close"] += 0.02
    fresh.loc[DAYS[5], "close"] += C.SYNC_BASIS_TOL / 2
    out = X.compare(bars(), fresh)
    assert out[["session", "field"]].values.tolist() == [[DAYS[3], "close"]]
    assert out.iloc[0]["tcbs"] == pytest.approx(20.02)


def test_volume_flagged_on_relative_difference():
    fresh = bars()
    fresh.loc[DAYS[2], "volume"] *= 1.05
    fresh.loc[DAYS[4], "volume"] *= 1.01
    out = X.compare(bars(), fresh)
    assert out[["session", "field"]].values.tolist() == [[DAYS[2], "volume"]]


def test_missing_sessions_inside_window_only():
    stored = bars()
    fresh = bars(DAYS.delete(6).append(pd.DatetimeIndex([DAYS[-1] + pd.offsets.BDay()])))
    out = X.compare(stored, fresh)
    assert out[["session", "field"]].values.tolist() == [[DAYS[6], "missing_in_tcbs"]]


def test_stored_sessions_before_the_fetch_window_are_ignored():
    assert X.compare(bars(), bars(DAYS[5:])).empty


class OtpRequiredError(Exception):
    pass


def write_stock(tmp_path, symbols):
    for s in symbols:
        b = bars().reset_index()
        b.insert(0, "time", pd.to_datetime(b.pop("date")) + pd.Timedelta(hours=C.SESSION_HOUR_UTC))
        b.to_parquet(tmp_path / C.STOCK_FILE.format(ticker=s))


def test_closed_session_skips_the_run_without_writing(tmp_path):
    write_stock(tmp_path, ["AAA"])
    def fetch(symbol, start, end):
        raise OtpRequiredError("Phiên tcbs chưa mở")
    out = tmp_path / "crosscheck.csv"
    res = X.run(tmp_path, out, fetch, delay=0)
    assert res["status"] == "skipped" and not out.exists()


def test_run_appends_differences_and_fetch_errors(tmp_path):
    write_stock(tmp_path, ["AAA", "BBB"])
    def fetch(symbol, start, end):
        if symbol == "BBB":
            raise RuntimeError("HTTP 500")
        f = bars()
        f.loc[DAYS[3], "close"] += 0.1
        return f
    out = tmp_path / "crosscheck.csv"
    for _ in range(2):
        res = X.run(tmp_path, out, fetch, delay=0)
    assert res == {"status": "ok", "symbols": 2, "differences": 1, "errors": 1}
    log = pd.read_csv(out)
    assert list(log.columns) == ["checked_at", "symbol", "session", "field", "stored", "tcbs", "note"]
    assert log.groupby("symbol").field.apply(list).to_dict() == {"AAA": ["close", "close"],
                                                                  "BBB": ["fetch_error", "fetch_error"]}
    assert np.isclose(log[log.symbol == "AAA"].tcbs, 20.1).all()
    assert log[log.symbol == "BBB"].note.str.contains("HTTP 500").all()
