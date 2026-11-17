import pandas as pd
import pytest

from ai_core.fusion import tcbs_flow


def _frame(day, buy):
    t = pd.Timestamp(day, tz="Asia/Ho_Chi_Minh") + pd.Timedelta(hours=7)
    return pd.DataFrame({"time": [t], "active_buy_percent": [buy], "active_sell_percent": [100 - buy], "buy_sell_ratio": [1.0]})


def test_sync_keys_by_session_and_survives_one_bad_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr(tcbs_flow, "OUT", tmp_path / "sd.parquet")

    def fetch(s):
        if s == "BAD":
            raise RuntimeError("boom")
        return _frame("2026-11-13", 50.4)

    out = tcbs_flow.sync(["FPT", "BAD"], fetch=fetch, delay=0)
    assert out["date"].tolist() == [pd.Timestamp("2026-11-13")] and out["symbol"].tolist() == ["FPT"]
    out = tcbs_flow.sync(["FPT"], fetch=lambda s: _frame("2026-11-13", 51.0), delay=0)   # restated session: one row
    assert len(out) == 1 and out["active_buy_percent"].iloc[0] == 51.0
    with pytest.raises(RuntimeError):
        tcbs_flow.sync(["BAD"], fetch=fetch, delay=0)                                    # dead session must not pass silently
