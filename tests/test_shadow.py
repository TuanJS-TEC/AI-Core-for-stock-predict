import json

import numpy as np
import pandas as pd

from ai_core import config as C
from ai_core import live, shadow
from nlp_module import storage
from nlp_module.forecast import volcast

DATES = pd.bdate_range("2026-10-01", periods=10)            # d0..d9


def _hist():
    """A drifts +0.1%/day, B +3%/day: B's |forward return| always beats the 2-ticker median, A's never does. So
    strong = B, weak = A, for every horizon and every base date that has h future sessions."""
    rows = [dict(date=d, symbol=s, open=p, high=p * 1.01, low=p * 0.99, close=p, volume=1000)
            for s, g in (("A", 1.001), ("B", 1.03)) for d, p in zip(DATES, 100 * g ** np.arange(10))]
    return pd.DataFrame(rows)


def _db(calls, decisions=()):
    """calls: (symbol, base_date index, h, call); decisions: (symbol, base_date, decision) for live.VERSION."""
    c = storage.connect(":memory:")
    c.executemany("INSERT INTO vol_forecast VALUES (?,?,?,?,?,?,?)",
                  [(s, str(DATES[i].date()), h, 0.6, call, volcast.VERSION, "t") for s, i, h, call in calls])
    c.executemany("INSERT INTO fusion_decision VALUES (?,?,2,?,NULL,NULL,NULL,NULL,NULL,0,NULL,'[]','r',?,'t')",
                  [(s, d, dec, live.VERSION) for s, d, dec in decisions])
    return c


def _rep(calls):
    return shadow.shadow_report(_db(calls), _hist(), report_path="/nonexistent")


def test_brief_three_matured_two_correct_one_pending():
    r = shadow.shadow_report(_db([("A", 0, 2, "weak"),      # matured, A is weak -> correct
                                  ("B", 0, 2, "strong"),    # matured, B is strong -> correct
                                  ("A", 3, 2, "strong"),    # matured, wrong
                                  ("B", 8, 2, "strong"),    # only d9 follows -> pending
                                  ("A", 1, 2, None)]),      # abstain: not a call at all
                             _hist(), report_path="/nonexistent")
    assert (r["n_matured"], r["n_correct"], r["n_pending"]) == (3, 2, 1)
    assert r["accuracy"] == 2 / 3


def test_other_model_version_ignored():
    c = _db([("A", 0, 2, "weak")])
    c.execute("INSERT INTO vol_forecast VALUES ('B','2026-10-01',2,0.6,'strong','volcast-old','t')")
    assert shadow.shadow_report(c, _hist(), report_path="/nonexistent")["n_matured"] == 1


def test_maturation_is_per_horizon_and_exact_at_the_edge():
    r = _rep([("B", 7, 2, "strong"),    # h=2 at d7: d8, d9 = exactly 2 future sessions -> matured (edge)
              ("B", 8, 2, "strong"),    # h=2 at d8: 1 future session -> pending
              ("B", 6, 5, "strong"),    # h=5 at d6: 3 future sessions -> pending
              ("B", 4, 5, "strong"),    # h=5 at d4: d5..d9 = exactly 5 -> matured (edge)
              ("A", 4, 5, "strong")])   # matured, wrong
    assert r["by_h"][2] == dict(n_matured=1, n_correct=1, accuracy=1.0, n_pending=1)
    assert r["by_h"][5] == dict(n_matured=2, n_correct=1, accuracy=0.5, n_pending=1)
    assert (r["n_matured"], r["n_correct"], r["n_pending"]) == (3, 2, 2)


def test_no_calls_no_division_by_zero():
    r = _rep([("B", 8, 2, "strong")])                    # only a pending call
    assert (r["n_matured"], r["accuracy"], r["n_pending"]) == (0, None, 1)
    r = shadow.shadow_report(_db([]), _hist(), report_path="/nonexistent")
    assert (r["n_sessions"], r["data_invalid_share"], r["n_matured"], r["accuracy"]) == (0, None, 0, None)


def test_sessions_and_data_invalid_share():
    d = lambda i: str(DATES[i].date())
    dec = [("A", d(8), "NO_TRADE"), ("B", d(8), "DATA_INVALID"), ("A", d(9), "NO_TRADE"), ("B", d(9), "NO_TRADE")]
    c = _db([], decisions=dec)
    c.execute("INSERT INTO fusion_decision VALUES ('A','2026-09-01',2,'DATA_INVALID',NULL,NULL,NULL,NULL,NULL,0,NULL,'[]','r','other','t')")
    r = shadow.shadow_report(c, _hist(), report_path="/nonexistent")
    assert (r["n_sessions"], r["n_rows"], r["n_data_invalid"], r["data_invalid_share"]) == (2, 4, 1, 0.25)


def test_fold_c_accuracy_from_report(tmp_path):
    rows = [dict(fold="B", h=5, test_acc=0.9, test_n=10), dict(fold="C", h=2, test_acc=None, test_n=0),
            dict(fold="C", h=5, test_acc=0.58, test_n=507)]
    p = tmp_path / "rep.json"
    p.write_text(json.dumps({"version": volcast.VERSION, "created_at": "x", "rows": rows}))
    r = shadow.shadow_report(_db([]), _hist(), report_path=p)
    assert r["fold_c"] == {2: dict(test_acc=None, test_n=0), 5: dict(test_acc=0.58, test_n=507)}
    assert shadow.shadow_report(_db([]), _hist(), report_path=tmp_path / "missing.json")["fold_c"] == {}


def test_text_caveat_under_40_sessions():
    txt = shadow.render(_rep([]))
    assert "less than 40 sessions: not enough for the Step-5 decision" in txt
