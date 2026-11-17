"""NLP news results (SQLite article_company_result) -> StockSenseVN PostgreSQL nlp_news (migration 024).

Only PhoBERT results of representative, titled, dated articles; per (document, symbol) the newest document version.
Incremental: the watermark is max(processed_at) already in nlp_news and the comparison is >=, so a row finished in the
same second as the watermark is not lost; the upsert only rewrites a row whose processed_at moved forward, so a rerun
changes nothing and sends no notification. After a push that changed rows, pg_notify('nlp_news', 'FPT,VNM') (sorted
symbols) wakes StockSenseVN's realtime stream in the same transaction.
Run: STOCKSENSE_DATABASE_URL=postgresql://... .venv/bin/python -m ai_core.fusion.stocksense_news (after the news process).
"""
import os
from datetime import timezone

from ai_core.news import storage
from ai_core.news.risk import risk_flags
from .stocksense import connect

MODEL = "phobert-phobert-v1"

_UPSERT = """
INSERT INTO nlp_news (symbol, document_id, document_version_id, source, url, title, published_at, relation, p_pos, p_neu,
                      p_neg, uncertain, risk_flags, passage, model_version, processed_at)
VALUES (%(symbol)s, %(document_id)s, %(document_version_id)s, %(source)s, %(url)s, %(title)s, %(published_at)s,
        %(relation)s, %(p_pos)s, %(p_neu)s, %(p_neg)s, %(uncertain)s, %(risk_flags)s, %(passage)s, %(model_version)s,
        %(processed_at)s)
ON CONFLICT (symbol, document_id) DO UPDATE SET
  document_version_id = EXCLUDED.document_version_id, source = EXCLUDED.source, url = EXCLUDED.url,
  title = EXCLUDED.title, published_at = EXCLUDED.published_at, relation = EXCLUDED.relation, p_pos = EXCLUDED.p_pos,
  p_neu = EXCLUDED.p_neu, p_neg = EXCLUDED.p_neg, uncertain = EXCLUDED.uncertain, risk_flags = EXCLUDED.risk_flags,
  passage = EXCLUDED.passage, model_version = EXCLUDED.model_version, processed_at = EXCLUDED.processed_at
WHERE nlp_news.processed_at < EXCLUDED.processed_at
RETURNING symbol"""


def rows_since(sq, since: str | None, model: str = MODEL, symbols: list[str] | None = None) -> list[dict]:
    """Result rows with processed_at >= `since` (ISO 'YYYY-MM-DDTHH:MM:SSZ', None = all), keyed like nlp_news columns;
    `symbols` restricts to those symbols."""
    out = []
    only = "" if symbols is None else f" AND r.symbol IN ({','.join('?' * len(symbols))})"
    for r in sq.execute(
            """SELECT * FROM (
                 SELECT r.symbol, d.document_id, v.document_version_id, d.source, d.url_normalized AS url, v.title,
                        v.published_at, r.relation, r.p_pos, r.p_neu, r.p_neg, r.uncertain, r.passage_text AS passage,
                        r.nlp_model_version AS model_version, r.processed_at,
                        ROW_NUMBER() OVER (PARTITION BY d.document_id, r.symbol
                                           ORDER BY v.version_first_seen_at DESC, v.document_version_id DESC) AS rn
                 FROM article_company_result r
                 JOIN document_version v USING (document_version_id) JOIN document d USING (document_id)
                 WHERE r.nlp_model_version = ? AND v.is_representative = 1 AND v.title IS NOT NULL
                       AND v.published_at IS NOT NULL AND r.p_pos IS NOT NULL""" + only + """)
               WHERE rn = 1 AND (? IS NULL OR processed_at >= ?)""", (model, *(symbols or ()), since, since)):
        d = dict(r)
        del d["rn"]
        d["uncertain"] = bool(d["uncertain"])
        d["risk_flags"] = risk_flags(d["passage"])
        out.append(d)
    return out


def push_news(sq, pg, model: str = MODEL) -> list[str]:
    """Upsert new/rescored results into nlp_news and notify; returns the sorted symbols that changed."""
    (mark,) = pg.execute("SELECT max(processed_at) FROM nlp_news").fetchone()
    since = mark.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if mark else None
    rows = rows_since(sq, since, model)
    # symbols added to StockSenseVN after the watermark moved past their rows get their whole history
    fresh = [s for (s,) in pg.execute(
        "SELECT symbol FROM symbols s WHERE NOT EXISTS (SELECT 1 FROM nlp_news n WHERE n.symbol = s.symbol)")]
    if fresh and since:
        rows += rows_since(sq, None, model, fresh)
    if not rows:
        return []
    known = {s for (s,) in pg.execute("SELECT symbol FROM symbols WHERE symbol = ANY(%s)", ([r["symbol"] for r in rows],))}
    changed = set()
    with pg.transaction():
        for r in rows:
            if r["symbol"] in known and pg.execute(_UPSERT, r).fetchone():
                changed.add(r["symbol"])
        if changed:
            pg.execute("SELECT pg_notify('nlp_news', %s)", (",".join(sorted(changed)),))
    return sorted(changed)


def main() -> None:
    url = os.environ.get("STOCKSENSE_DATABASE_URL")
    if not url:
        print("stocksense news: STOCKSENSE_DATABASE_URL not set, skip")
        return
    with connect(url) as pg:
        changed = push_news(storage.connect(), pg)
    print(f"stocksense news: {len(changed)} symbols changed: {changed}")


if __name__ == "__main__":
    main()
