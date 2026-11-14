import sys
import types

import pandas as pd

from ai_core.prices import events as EV


def _rec(i, code, pub="2026-11-02T00:00:00", exr="2026-11-11T00:00:00"):
    return {"id": i, "eventCode": code, "publicDate": pub, "recordDate": "2026-11-12T00:00:00",
            "exrightDate": exr, "displayDate1": exr}


def test_normalize_keeps_known_codes_and_parses_dates():
    df = EV.normalize([_rec("a", "DIV"), _rec("b", "DDINS")], "VNM")
    assert list(df.columns) == EV.COLUMNS and len(df) == 1
    r = df.iloc[0]
    assert r.symbol == "VNM" and r.event_code == "DIV"
    assert r.public_date == pd.Timestamp("2026-11-02") and r.exright_date == pd.Timestamp("2026-11-11")


def test_normalize_drops_duplicate_ids():
    assert len(EV.normalize([_rec("a", "DIV"), _rec("a", "DIV")], "VNM")) == 1


def test_normalize_missing_dates_are_nat():
    assert pd.isna(EV.normalize([_rec("a", "AGMR", exr=None)], "VNM").iloc[0].exright_date)


def test_normalize_empty_has_columns():
    assert list(EV.normalize([], "VNM").columns) == EV.COLUMNS


def test_fetch_paginates(monkeypatch):
    pages = {0: [_rec(f"x{i}", "DIV") for i in range(100)], 1: [_rec(f"y{i}", "ISS") for i in range(3)]}
    calls = []

    class FakeCompany:
        def __init__(self, symbol, source):
            assert source == "vci"
            self.symbol = symbol

        def _fetch_events(self, event_codes, from_date, page, size):
            calls.append(page)
            assert event_codes == ",".join(EV.CODES) and from_date == "20200101" and size == 100
            return pages.get(page, [])

    monkeypatch.setitem(sys.modules, "vnstock", types.SimpleNamespace(Company=FakeCompany))
    df = EV.fetch(["VNM"], pause=0)
    assert len(df) == 103 and calls == [0, 1]


def test_fetch_retries_transient_errors(monkeypatch):
    fails = [ConnectionError("502"), ConnectionError("502")]

    class FlakyCompany:
        def __init__(self, symbol, source):
            pass

        def _fetch_events(self, **kw):
            if fails:
                raise fails.pop()
            return [_rec("a", "DIV")]

    monkeypatch.setitem(sys.modules, "vnstock", types.SimpleNamespace(Company=FlakyCompany))
    assert len(EV.fetch(["VNM"], pause=0)) == 1
