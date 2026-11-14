import os

import pytest

from ai_core.fusion.stocksense_news import MODEL, push_news
from ai_core.news import storage

# DDL copied from StockSenseVN db/migrations/024_nlp_news.sql (+ a minimal symbols table).
DDL = """
CREATE TABLE symbols (symbol varchar(12) PRIMARY KEY);
CREATE TABLE nlp_news (
  symbol varchar(12) NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
  document_id text NOT NULL,
  document_version_id text NOT NULL,
  source text NOT NULL,
  url text,
  title text NOT NULL,
  published_at timestamptz NOT NULL,
  relation text NOT NULL CHECK (relation IN ('direct', 'sector', 'market', 'ambiguous')),
  p_pos real NOT NULL, p_neu real NOT NULL, p_neg real NOT NULL,
  uncertain boolean NOT NULL,
  risk_flags text[] NOT NULL DEFAULT '{}',
  passage text NOT NULL,
  model_version text NOT NULL,
  processed_at timestamptz NOT NULL,
  PRIMARY KEY (symbol, document_id)
);
CREATE INDEX nlp_news_symbol_published_idx ON nlp_news (symbol, published_at DESC);
CREATE INDEX nlp_news_processed_idx ON nlp_news (processed_at);
INSERT INTO symbols VALUES ('FPT'), ('VNM');
"""


@pytest.fixture
def pg():
    url = os.environ.get("STOCKSENSE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("STOCKSENSE_TEST_DATABASE_URL not set")
    import psycopg
    # autocommit like stocksense.connect: NOTIFY is only delivered on commit, and pg.transaction() inside an open
    # transaction is just a savepoint. So no rollback cleanup: the schema is dropped instead.
    c = psycopg.connect(url, autocommit=True)
    c.execute("DROP SCHEMA IF EXISTS news_bridge_test CASCADE")
    c.execute("CREATE SCHEMA news_bridge_test")
    c.execute("SET search_path TO news_bridge_test")
    c.execute(DDL)
    yield c
    c.execute("DROP SCHEMA news_bridge_test CASCADE")
    c.close()


def add(sq, doc, symbol, version="v1", processed="2026-11-12T08:00:00Z", published="2026-11-12T07:00:00Z", title="T",
        uncertain=0, rep=1, model=MODEL, seen="2026-11-12T07:01:00Z", passage="FPT bị khởi tố", p_pos=0.1):
    sq.execute("INSERT OR IGNORE INTO document VALUES (?,?,?,?,?)", (doc, "cafef", f"https://x/{doc}", "news", seen))
    sq.execute("""INSERT OR IGNORE INTO document_version (document_version_id, document_id, content_hash, title,
                  original_text, normalized_text, published_at, version_first_seen_at, raw_object_key, is_representative)
                  VALUES (?,?,?,?,?,?,?,?,?,?)""",
               (f"{doc}-{version}", doc, f"h-{doc}-{version}", title, "o", "n", published, seen, "k", rep))
    sq.execute("""INSERT INTO article_company_result (result_id, document_version_id, target_entity_id, symbol, relation,
                  sentence_ids, passage_text, p_pos, p_neu, p_neg, uncertain, nlp_model_version, processed_at)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (f"r-{doc}-{version}-{symbol}-{model}", f"{doc}-{version}", symbol, symbol, "direct", "[]", passage,
                p_pos, 0.2, 0.7, uncertain, model, processed))
    sq.commit()


@pytest.fixture
def sq():
    return storage.connect(":memory:")


def one(pg, sql):
    return pg.execute(sql).fetchone()


def all_rows(pg, sql):
    return pg.execute(sql).fetchall()


def test_push_inserts_and_notifies(pg, sq):
    add(sq, "d1", "FPT")
    pg.execute("LISTEN nlp_news")
    assert push_news(sq, pg) == ["FPT"]
    assert one(pg, "SELECT title, uncertain, risk_flags FROM nlp_news") == ("T", False, ["prosecution"])
    assert one(pg, "SELECT url, source FROM nlp_news") == ("https://x/d1", "cafef")
    assert [n.payload for n in pg.notifies(timeout=1, stop_after=1)] == ["FPT"]


def test_notify_payload_is_sorted_symbols(pg, sq):
    add(sq, "d1", "VNM")
    add(sq, "d2", "FPT")
    pg.execute("LISTEN nlp_news")
    assert push_news(sq, pg) == ["FPT", "VNM"]
    assert [n.payload for n in pg.notifies(timeout=1, stop_after=1)] == ["FPT,VNM"]


def test_only_phobert_representative_titled(pg, sq):
    add(sq, "d1", "FPT")
    add(sq, "d2", "FPT", model="baseline-v1")
    add(sq, "d3", "FPT", rep=0)
    add(sq, "d4", "FPT", title=None)
    assert push_news(sq, pg) == ["FPT"] and one(pg, "SELECT count(*) FROM nlp_news") == (1,)


def test_latest_version_wins(pg, sq):
    add(sq, "d1", "FPT", version="v1")
    add(sq, "d1", "FPT", version="v2", title="T2", seen="2026-11-12T07:30:00Z")
    push_news(sq, pg)
    assert all_rows(pg, "SELECT document_version_id, title FROM nlp_news") == [("d1-v2", "T2")]


def test_watermark_boundary_and_quiet_rerun(pg, sq):
    add(sq, "d1", "FPT")
    push_news(sq, pg)
    pg.execute("LISTEN nlp_news")
    assert push_news(sq, pg) == []
    assert list(pg.notifies(timeout=0.3)) == []
    add(sq, "d9", "VNM", processed="2026-11-12T08:00:00Z")   # same second as the watermark
    assert push_news(sq, pg) == ["VNM"]


def test_rescored_row_updates(pg, sq):
    add(sq, "d1", "FPT")
    assert push_news(sq, pg) == ["FPT"] and one(pg, "SELECT round(p_pos::numeric, 2)::text FROM nlp_news") == ("0.10",)
    sq.execute("UPDATE article_company_result SET p_pos = 0.9, processed_at = '2026-11-12T09:00:00Z'")
    sq.commit()
    assert push_news(sq, pg) == ["FPT"] and one(pg, "SELECT round(p_pos::numeric, 2)::text FROM nlp_news") == ("0.90",)


def test_unknown_symbol_skipped(pg, sq):
    add(sq, "d1", "NVL")
    assert push_news(sq, pg) == [] and one(pg, "SELECT count(*) FROM nlp_news") == (0,)


def test_new_symbol_gets_history(pg, sq):
    add(sq, "d1", "FPT", processed="2026-11-12T09:00:00Z")
    add(sq, "d2", "NVL", processed="2026-11-10T08:00:00Z")    # older than the watermark
    assert push_news(sq, pg) == ["FPT"]
    pg.execute("INSERT INTO symbols VALUES ('NVL')")
    assert push_news(sq, pg) == ["NVL"] and one(pg, "SELECT count(*) FROM nlp_news WHERE symbol = 'NVL'") == (1,)
    assert push_news(sq, pg) == []
