import os

import pytest

from ai_core.fusion.stocksense import push, to_forecast
from ai_core.news import storage

VERSION = "decision-v1-trend"


def row(**kw):
    r = dict(symbol="FPT", base_date="2026-11-12", h=2, decision="LONG", p_up=0.5, p_down=0.3, margin=0.2,
             p_strong=0.6, call=None, has_news=0, sentiment_mean_24h=None, risk_flags="[]", reason="", version=VERSION,
             created_at="2026-11-12T08:30:00Z")
    r.update(kw)
    return r


def test_to_forecast_maps_labels_and_probs():
    f = to_forecast(row(decision="LONG", h=2, p_up=0.6, p_down=0.2, p_strong=0.7, call=None))
    assert (f["label"], f["horizon"], f["symbol"], f["data_date"]) == ("up", "t2", "FPT", "2026-11-12")
    assert f["hold"] == pytest.approx(0.3) and f["up"] == pytest.approx(0.525) and f["down"] == pytest.approx(0.175)
    assert f["confidence"] == "low"
    assert to_forecast(row(decision="AVOID", h=5))["label"] == "down"
    assert to_forecast(row(decision="AVOID", h=5))["horizon"] == "t5"


def test_to_forecast_hold_mid_only_with_weak_call():
    assert to_forecast(row(decision="NO_TRADE", p_strong=0.3, call="weak"))["confidence"] == "mid"
    assert to_forecast(row(decision="NO_TRADE", p_strong=0.3, call=None))["confidence"] == "low"
    assert to_forecast(row(decision="LONG", call="strong"))["confidence"] == "low"     # direction is NO-GO: never above low


def test_to_forecast_skips_invalid_and_incomplete():
    assert to_forecast(row(decision="DATA_INVALID")) is None
    assert to_forecast(row(p_strong=None)) is None
    assert to_forecast(row(p_up=None, p_down=None)) is None


@pytest.mark.parametrize("ps,up,down", [(0.0, 0.5, 0.2), (1.0, 0.5, 0.2), (0.6, 0.0, 0.0), (0.123456789, 0.333333, 0.333333)])
def test_probs_edge_cases(ps, up, down):
    f = to_forecast(row(p_strong=ps, p_up=up, p_down=down))
    assert all(0 <= f[k] <= 1 for k in ("up", "hold", "down"))
    assert abs(f["up"] + f["hold"] + f["down"] - 1) < 1e-4
    assert all(round(f[k], 7) == f[k] for k in ("up", "hold", "down"))


# --- Postgres: the 4 tables the bridge touches, DDL copied from StockSenseVN db/migrations/001 (+ 023 analysis_enabled).
DDL = """
CREATE TABLE symbols (symbol varchar(12) PRIMARY KEY, analysis_enabled boolean NOT NULL DEFAULT false);
CREATE TABLE market_batches (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, data_date date NOT NULL, batch_completed_at timestamptz,
  model_version text NOT NULL, state text NOT NULL, source_status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (data_date, model_version));
CREATE TABLE market_summary (
  batch_id bigint PRIMARY KEY REFERENCES market_batches(id) ON DELETE CASCADE,
  advances integer NOT NULL CHECK (advances >= 0), nochanges integer NOT NULL CHECK (nochanges >= 0),
  declines integer NOT NULL CHECK (declines >= 0), labels_up integer NOT NULL CHECK (labels_up >= 0),
  labels_hold integer NOT NULL CHECK (labels_hold >= 0), labels_down integer NOT NULL CHECK (labels_down >= 0));
CREATE TABLE forecasts (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  batch_id bigint NOT NULL REFERENCES market_batches(id) ON DELETE CASCADE,
  symbol varchar(12) NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
  horizon text NOT NULL CHECK (horizon IN ('t2', 't5')),
  label text NOT NULL CHECK (label IN ('up', 'hold', 'down')),
  probability_up numeric(8, 7) NOT NULL CHECK (probability_up BETWEEN 0 AND 1),
  probability_hold numeric(8, 7) NOT NULL CHECK (probability_hold BETWEEN 0 AND 1),
  probability_down numeric(8, 7) NOT NULL CHECK (probability_down BETWEEN 0 AND 1),
  confidence text NOT NULL CHECK (confidence IN ('low', 'mid', 'high')),
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (batch_id, symbol, horizon),
  CHECK (abs(probability_up + probability_hold + probability_down - 1) < 0.0001));
CREATE TABLE forecast_ranges (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  batch_id bigint NOT NULL REFERENCES market_batches(id) ON DELETE CASCADE,
  symbol varchar(12) NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
  horizon text NOT NULL CHECK (horizon IN ('t2', 't5')),
  low_pct numeric(8, 6) NOT NULL CHECK (low_pct < 0), high_pct numeric(8, 6) NOT NULL CHECK (high_pct > 0),
  coverage numeric(4, 3) NOT NULL CHECK (coverage > 0 AND coverage < 1),
  risk_level text NOT NULL CHECK (risk_level IN ('low', 'normal', 'high')),
  has_news boolean NOT NULL DEFAULT false, sentiment_24h real, risk_flags text[] NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(), UNIQUE (batch_id, symbol, horizon));
ALTER TABLE market_summary ADD COLUMN vol_low integer NOT NULL DEFAULT 0, ADD COLUMN vol_normal integer NOT NULL DEFAULT 0,
  ADD COLUMN vol_high integer NOT NULL DEFAULT 0;
INSERT INTO symbols VALUES ('FPT', true), ('BID', false), ('VNM', true);
"""


@pytest.fixture
def pg():
    url = os.environ.get("STOCKSENSE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("STOCKSENSE_TEST_DATABASE_URL not set")
    import psycopg
    c = psycopg.connect(url, autocommit=True)
    c.execute("DROP SCHEMA IF EXISTS bridge_test CASCADE")
    c.execute("CREATE SCHEMA bridge_test")
    c.execute("SET search_path TO bridge_test")
    c.execute(DDL)
    c.autocommit = False
    yield c
    c.rollback()
    c.autocommit = True
    c.execute("DROP SCHEMA bridge_test CASCADE")
    c.close()


def sq_with(*rows):
    c = storage.connect(":memory:")
    for r in rows:
        c.execute(f"INSERT INTO fusion_decision ({','.join(r)}) VALUES ({','.join('?' * len(r))})", list(r.values()))
    return c


def one(pg, sql):
    return pg.execute(sql).fetchone()


def all_rows(pg, sql):
    return pg.execute(sql).fetchall()


def test_push_writes_batch_forecasts_summary(pg):
    sq = sq_with(row(h=2), row(h=5, decision="NO_TRADE", p_strong=0.3, call="weak"))
    assert push(sq, pg, VERSION) == 2
    assert one(pg, "SELECT data_date::text, model_version, state FROM market_batches") == ("2026-11-12", VERSION, "experimental")
    assert one(pg, "SELECT labels_up, labels_hold, labels_down FROM market_summary") == (0, 1, 0)   # T+5 counts
    assert one(pg, "SELECT label, confidence FROM forecasts WHERE horizon = 't5'") == ("hold", "mid")


def test_push_is_idempotent(pg):
    sq = sq_with(row(h=2), row(h=5))
    push(sq, pg, VERSION)
    before = all_rows(pg, "SELECT * FROM forecasts ORDER BY horizon")
    sq.execute("UPDATE fusion_decision SET p_strong = 0.99")          # rerun with different numbers
    assert push(sq, pg, VERSION) == 0
    assert all_rows(pg, "SELECT * FROM forecasts ORDER BY horizon") == before
    assert one(pg, "SELECT count(*) FROM market_batches") == (1,)


def test_push_skips_invalid_and_incomplete(pg):
    sq = sq_with(row(h=2, decision="DATA_INVALID", p_strong=None, p_up=None, p_down=None), row(h=5, p_strong=None),
                 row(base_date="2026-11-13", version="other-version"))
    assert push(sq, pg, VERSION) == 0
    assert one(pg, "SELECT count(*) FROM market_batches") == (0,)


def test_push_only_analysis_enabled(pg):
    sq = sq_with(row(h=2), row(h=5), row(symbol="BID"), row(symbol="NVL", h=5))
    assert push(sq, pg, VERSION) == 2
    assert one(pg, "SELECT array_agg(DISTINCT symbol) FROM forecasts") == (["FPT"],)


def test_push_batch_only_when_a_row_lands(pg):
    """Rows exist for the new date but none for an analysed symbol: no empty batch may become the 'latest' one."""
    sq = sq_with(row(symbol="BID"), row(symbol="NVL"))
    assert push(sq, pg, VERSION) == 0
    assert one(pg, "SELECT count(*) FROM market_batches") == (0,)


def test_main_bounds_connect_and_statement_time(monkeypatch):
    """A firewalled host or a held lock must fail fast: the push runs inside live_update.sh's live.lock, and a hang
    there makes every later live run skip (news collection stops)."""
    import psycopg
    from ai_core.fusion import stocksense
    seen = {}

    def fake_connect(url, **kw):
        seen.update(kw)
        raise psycopg.OperationalError("stop here")

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setenv("STOCKSENSE_DATABASE_URL", "postgresql://x@127.0.0.1/x")
    with pytest.raises(psycopg.OperationalError):
        stocksense.main()
    assert 0 < seen.get("connect_timeout", 0) <= 15
    assert "statement_timeout=" in seen.get("options", "") and "lock_timeout=" in seen.get("options", "")


def test_noop_rerun_keeps_batch_completed_at(pg):
    """The app header shows batch_completed_at: a rerun that inserts nothing must not move it to the last cron time."""
    sq = sq_with(row(h=2), row(h=5))
    push(sq, pg, VERSION)
    pg.execute("UPDATE market_batches SET batch_completed_at = '2000-01-01T00:00:00Z'")
    assert push(sq, pg, VERSION) == 0
    assert one(pg, "SELECT batch_completed_at::date::text FROM market_batches") == ("2000-01-01",)


def test_late_row_joins_existing_batch(pg):
    sq = sq_with(row(h=2))
    push(sq, pg, VERSION)
    pg.execute("UPDATE market_batches SET batch_completed_at = '2000-01-01T00:00:00Z'")
    sq.execute("INSERT INTO fusion_decision (symbol, base_date, h, decision, p_up, p_down, p_strong, call, has_news, "
               "risk_flags, version, created_at) VALUES ('FPT', '2026-11-12', 5, 'NO_TRADE', 0.4, 0.4, 0.3, 'weak', 0, "
               "'[]', ?, '2026-11-12T08:45:00Z')", (VERSION,))
    assert push(sq, pg, VERSION) == 1
    assert one(pg, "SELECT count(*) FROM market_batches") == (1,)
    assert one(pg, "SELECT labels_up, labels_hold, labels_down FROM market_summary") == (0, 1, 0)
    assert one(pg, "SELECT batch_completed_at::date::text FROM market_batches") != ("2000-01-01",)


def test_push_reports_stored_rows_that_differ(pg, capsys):
    """Stored forecasts stay as they are (immutable), but a later fusion.live REPLACE with other numbers is reported."""
    sq = sq_with(row(h=2), row(h=5))
    push(sq, pg, VERSION)
    capsys.readouterr()
    sq.execute("UPDATE fusion_decision SET p_strong = 0.99")
    assert push(sq, pg, VERSION) == 0
    out = capsys.readouterr().out
    assert "2 stored forecasts differ" in out
    push(sq, pg, VERSION)                                        # still differs: still reported, nothing rewritten
    assert one(pg, "SELECT max(probability_hold) FROM forecasts")[0] < 0.5


def test_push_since_skips_old_dates(pg):
    """The cron passes a recent `since`, so admin-deleted old batches are not recreated and newly enabled symbols are
    not back-filled into old batches."""
    sq = sq_with(row(base_date="2026-11-01"), row(base_date="2026-11-12"))
    assert push(sq, pg, VERSION, since="2026-11-05") == 1
    assert all_rows(pg, "SELECT data_date::text FROM market_batches") == [("2026-11-12",)]


# --- price ranges (rangecast): migration 025 --------------------------------------------------------------------------
from ai_core.fusion.stocksense import push_ranges
from ai_core.models.rangecast.rangecast import VERSION as RANGE_VERSION


def rng(**kw):
    r = dict(symbol="FPT", base_date="2026-11-12", h=5, close=122.5, lo_pct=-0.038, hi_pct=0.036, sigma=0.03, risk_level="high",
             model_version=RANGE_VERSION, created_at="2026-11-12T08:30:00Z")
    r.update(kw)
    return r


def sq_ranges(ranges, decisions=()):
    c = sq_with(*decisions)
    for r in ranges:
        c.execute(f"INSERT INTO range_forecast ({','.join(r)}) VALUES ({','.join('?' * len(r))})", list(r.values()))
    return c


def test_push_ranges_writes_batch_ranges_and_level_counts(pg):
    sq = sq_ranges([rng(h=2, risk_level="normal"), rng(h=5), rng(symbol="VNM", h=5, risk_level="low")],
                   [row(h=5, has_news=1, sentiment_mean_24h=-0.31, risk_flags='["sanction"]')])
    assert push_ranges(sq, pg) == 3
    assert one(pg, "SELECT data_date::text, model_version, state FROM market_batches") == ("2026-11-12", RANGE_VERSION, "experimental")
    got = one(pg, "SELECT low_pct::float, high_pct::float, coverage::float, risk_level, has_news, sentiment_24h, risk_flags "
                  "FROM forecast_ranges WHERE symbol = 'FPT' AND horizon = 't5'")
    assert got[:5] == (-0.038, 0.036, 0.8, "high", True) and got[5] == pytest.approx(-0.31) and got[6] == ["sanction"]
    assert one(pg, "SELECT vol_low, vol_normal, vol_high, labels_up, labels_hold, labels_down FROM market_summary") == (1, 0, 1, 0, 0, 0)
    assert one(pg, "SELECT count(*) FROM forecasts") == (0,)


def test_push_ranges_without_a_decision_row_has_no_news_context(pg):
    assert push_ranges(sq_ranges([rng()]), pg) == 1
    assert one(pg, "SELECT has_news, sentiment_24h, risk_flags FROM forecast_ranges") == (False, None, [])


def test_push_ranges_is_idempotent_and_keeps_stored_numbers(pg):
    sq = sq_ranges([rng(h=2), rng(h=5)])
    push_ranges(sq, pg)
    before = all_rows(pg, "SELECT * FROM forecast_ranges ORDER BY horizon")
    sq.execute("UPDATE range_forecast SET lo_pct = -0.09, risk_level = 'low'")
    assert push_ranges(sq, pg) == 0
    assert all_rows(pg, "SELECT * FROM forecast_ranges ORDER BY horizon") == before
    assert one(pg, "SELECT count(*) FROM market_batches") == (1,)


def test_push_ranges_skips_symbols_not_enabled_and_makes_no_empty_batch(pg):
    assert push_ranges(sq_ranges([rng(symbol="BID"), rng(symbol="ZZZ")]), pg) == 0
    assert one(pg, "SELECT count(*) FROM market_batches") == (0,)


def test_push_ranges_since_and_other_versions(pg):
    sq = sq_ranges([rng(base_date="2026-11-01"), rng(base_date="2026-11-12"), rng(symbol="VNM", model_version="rangecast-v0")])
    assert push_ranges(sq, pg, since="2026-11-10") == 1
    assert all_rows(pg, "SELECT data_date::text FROM market_batches") == [("2026-11-12",)]
