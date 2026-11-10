import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ai_core.fusion import config as C
from ai_core.fusion import live
from ai_core.news import storage
from ai_core.news.contract import ArticleCompanyResult
from ai_core.models.volcast import volcast

VN = "Asia/Ho_Chi_Minh"
THU_1600 = pd.Timestamp("2026-10-29 16:00", tz=VN)          # a Thursday, after the 15:00 data-ready time


def _hist(last, n=60, symbol="ACB", rets=None):
    """n weekday sessions ending at `last`, tiny alternating returns unless `rets` {index_from_end: ret1} says otherwise."""
    d = pd.bdate_range(end=last, periods=n)
    r = np.where(np.arange(n) % 2, 0.004, -0.004)
    for k, v in (rets or {}).items():
        r[n - 1 - k] = v
    return pd.DataFrame({"date": d, "symbol": symbol, "close": 100 * np.cumprod(1 + r)})


def _status(hist, now=THU_1600, hol=None):
    return live.data_status(hist, "ACB", now, live.holidays(hist) if hol is None else hol)


def test_status_ok():
    assert _status(_hist("2026-10-29")) is None
    assert _status(_hist("2026-10-28"), now=pd.Timestamp("2026-10-29 10:00", tz=VN)) is None   # before 15:00 today is not due


def test_status_missing_session():
    assert _status(_hist("2026-10-27")) == "MISSING_SESSION"   # Wed + Thu are due, data stops Tuesday


def test_status_declared_holiday_ok():
    h = _hist("2026-10-27")
    assert _status(h, now=pd.Timestamp("2026-10-28 16:00", tz=VN), hol=np.array(["2026-10-28"], "datetime64[D]")) is None


def test_status_seam():
    assert _status(_hist("2026-10-29", rets={5: -0.33})) == "SEAM"
    assert _status(_hist("2026-10-29", rets={21: -0.33})) is None          # older than the 21-session window


def test_status_precedence_seam_first():
    assert _status(_hist("2026-10-27", rets={5: -0.33})) == "SEAM"          # both hold: SEAM (prices untrustworthy) wins


def test_decide_data_invalid_never_trades():
    assert live.decide("SEAM")[0] == "DATA_INVALID"
    assert live.decide("MISSING_SESSION") == ("DATA_INVALID", "MISSING_SESSION")
    assert live.decide(None) == ("NO_TRADE", "no direction model (NO-GO)")


def test_holidays_gap_and_declared():
    h = _hist("2026-10-29").pipe(lambda d: d[d.date != "2026-10-14"])     # 10-14 is a Wednesday without a session
    hol = live.holidays(h)
    assert np.datetime64("2026-10-14") in hol and np.datetime64("2027-01-01") in hol
    assert np.datetime64("2026-10-13") not in hol and hol.dtype == np.dtype("datetime64[D]")
    assert not np.isin(np.array(["2026-10-17", "2026-10-18"], "datetime64[D]"), hol).any()   # weekends are not holidays


def test_fusion_decision_table_exists():
    c = storage.connect(":memory:")
    cols = c.execute("PRAGMA table_info(fusion_decision)").fetchall()
    assert [r["name"] for r in cols] == ["symbol", "base_date", "h", "decision", "p_up", "p_down", "margin", "p_strong",
                                         "call", "has_news", "sentiment_mean_24h", "risk_flags", "reason", "version",
                                         "created_at"]
    assert [r["name"] for r in sorted(cols, key=lambda r: r["pk"]) if r["pk"]] == ["symbol", "base_date", "h", "version"]


def _res(symbol, flags=(), uncertain=False, hours_before_cutoff=5, cutoff=datetime(2026, 10, 29, 8, tzinfo=timezone.utc)):
    t = cutoff - pd.Timedelta(hours=hours_before_cutoff)
    return ArticleCompanyResult(f"r-{symbol}-{hours_before_cutoff}", symbol, "cafef", t, t, "direct", 0.6, 0.3, 0.1,
                                uncertain, False, True, live.NLP_MODEL, tuple(flags))


def _vol(c, symbol, base="2026-10-29"):
    c.executemany("INSERT INTO vol_forecast VALUES (?,?,?,?,?,?,?)",
                  [(symbol, base, h, 0.61, "strong", volcast.VERSION, "x") for h in C.HORIZONS])


def test_rows_news_flags_and_extra_ticker():
    hist = pd.concat([_hist("2026-10-29", symbol=s) for s in ("ACB", "BID")])
    results = [_res("ACB", ["lawsuit"]), _res("ACB", hours_before_cutoff=30)]
    vol = {("ACB", "2026-10-29", 2): (0.61, "strong")}
    rows = {(r[0], r[2]): r for r in live.rows(hist, results, vol, THU_1600, ["ACB", "BID"])}
    a, b = rows["ACB", 2], rows["BID", 2]
    assert a[1] == "2026-10-29" and a[3] == "NO_TRADE" and a[4:7] == (None, None, None) and a[7:9] == (0.61, "strong")
    assert a[9] == 1 and a[10] is not None and json.loads(a[11]) == ["lawsuit"]
    assert rows["ACB", 5][7:9] == (None, None)                       # no vol_forecast row for h=5 in `vol`
    assert b[3] == "NO_TRADE" and b[7:9] == (None, None) and b[9] == 0 and b[10] is None and json.loads(b[11]) == []
    assert "no volcast row" in b[12] and b[12].startswith("no direction model (NO-GO)")
    assert a[13] == "decision-v0-nogo"


def test_main_writes_40_rows_in_memory():
    c = storage.connect(":memory:")
    _vol(c, "ACB")
    hist = pd.concat([_hist("2026-10-29", symbol=s) for s in C.UNIVERSE])
    seam = _hist("2026-10-29", symbol="VIC", rets={5: -0.33})
    hist = pd.concat([hist[hist.symbol != "VIC"], seam])
    live.main(now=THU_1600, c=c, hist=hist)
    rows = c.execute("SELECT * FROM fusion_decision").fetchall()
    assert len(rows) == len(C.UNIVERSE) * 2 and {r["version"] for r in rows} == {"decision-v0-nogo"}
    by = {(r["symbol"], r["h"]): r for r in rows}
    assert by["ACB", 2]["p_strong"] == 0.61 and by["ACB", 2]["decision"] == "NO_TRADE"
    assert by["GAS", 5]["has_news"] == 0 and by["GAS", 5]["risk_flags"] == "[]"      # no news: no exception
    assert by["BID", 2]["p_strong"] is None and "no volcast row" in by["BID", 2]["reason"]
    assert by["VIC", 2]["decision"] == "DATA_INVALID" and by["VIC", 2]["reason"].startswith("SEAM")
    live.main(now=THU_1600, c=c, hist=hist)                              # idempotent: INSERT OR REPLACE
    assert c.execute("SELECT count(*) FROM fusion_decision").fetchone()[0] == 40


def _universe(last, **kw):
    return pd.concat([_hist(last, symbol=s, **kw) for s in C.UNIVERSE])


def test_missing_session_run_never_overwrites_valid_row():
    c, hist = storage.connect(":memory:"), _universe("2026-10-28")           # data ends Wednesday
    _vol(c, "ACB", "2026-10-28")
    live.main(now=pd.Timestamp("2026-10-28 16:00", tz=VN), c=c, hist=hist)   # session closed: valid rows
    c.execute("UPDATE fusion_decision SET created_at='2000-01-01T00:00:00Z'")   # sentinel: a rewrite would change it
    before = [tuple(r) for r in c.execute("SELECT * FROM fusion_decision ORDER BY symbol, h")]
    assert len(before) == 40 and {r[3] for r in before} == {"NO_TRADE"}
    live.main(now=pd.Timestamp("2026-10-29 15:10", tz=VN), c=c, hist=hist)   # Thursday, before the sync: Thu is due
    assert [tuple(r) for r in c.execute("SELECT * FROM fusion_decision ORDER BY symbol, h")] == before


def test_missing_session_new_key_recorded_then_valid_run_replaces_it():
    c, hist = storage.connect(":memory:"), _universe("2026-10-28")
    _vol(c, "ACB", "2026-10-28")
    live.main(now=pd.Timestamp("2026-10-29 15:10", tz=VN), c=c, hist=hist)   # nothing stored yet: fail-closed row is written
    r = c.execute("SELECT * FROM fusion_decision WHERE symbol='ACB' AND h=2").fetchall()
    assert len(r) == 1 and r[0]["decision"] == "DATA_INVALID" and r[0]["reason"] == "MISSING_SESSION"
    live.main(now=pd.Timestamp("2026-10-29 10:00", tz=VN), c=c, hist=hist)   # Thursday not due yet: valid, replaces the row
    r = c.execute("SELECT * FROM fusion_decision WHERE symbol='ACB' AND h=2").fetchall()
    assert len(r) == 1 and r[0]["decision"] == "NO_TRADE" and r[0]["p_strong"] == 0.61


def test_data_invalid_row_carries_no_signal():
    c = storage.connect(":memory:")
    _vol(c, "ACB", "2026-10-28")
    hist = _universe("2026-10-28").query("symbol != 'ACB'")
    hist = pd.concat([hist, _hist("2026-10-28", symbol="ACB", rets={5: -0.33})])
    live.main(now=pd.Timestamp("2026-10-28 16:00", tz=VN), c=c, hist=hist)
    r = c.execute("SELECT * FROM fusion_decision WHERE symbol='ACB'").fetchall()
    assert len(r) == 2 and all(x["decision"] == "DATA_INVALID" and x["reason"] == "SEAM" for x in r)
    assert all(x["p_strong"] is None and x["call"] is None for x in r)


def test_board_constants_match_live():
    """The board imports live's version; the app's hard-coded NLP model must still match live's."""
    import re
    from ai_core.fusion import board
    assert board.FUSION_VERSION == live.VERSION
    app = (C.ROOT / "ai_core" / "fusion" / "app.py").read_text(encoding="utf-8")
    used = re.findall(r'snapshot_passages\(con, symbol, pd\.Timestamp\(cutoff\), "([^"]+)"\)', app)
    assert used == [live.NLP_MODEL]
