"""Read-only queries behind the app's decision board (the writer is `python -m ai_core.live`)."""
import pandas as pd

from .contract import WINDOW_H
from .risk import risk_flags

FUSION_VERSION = "decision-v0-nogo"   # mirrors ai_core.live.VERSION (NLP_Module cannot import ai_core)


def newest_decisions(con, version: str = FUSION_VERSION) -> pd.DataFrame:
    """Newest fusion_decision row per (symbol, h) of one version; empty if the table is missing."""
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='fusion_decision'").fetchone():
        return pd.DataFrame()
    return pd.read_sql_query(
        "SELECT * FROM fusion_decision WHERE version=? AND base_date=(SELECT MAX(base_date) FROM fusion_decision f "
        "WHERE f.symbol=fusion_decision.symbol AND f.h=fusion_decision.h AND f.version=fusion_decision.version) "
        "ORDER BY symbol, h", con, params=(version,))


def mark_stale(df: pd.DataFrame, latest_price=None) -> pd.DataFrame:
    """Rows older than the session the board should be on -> decision 'DATA_INVALID (stale)', call/p_strong blanked.

    Expected session = max(newest base_date on the board, `latest_price` = newest session in the price files, naive
    date or None). The board max catches one ticker left on yesterday's row (its sync failed, so ai_core.live wrote
    MISSING_SESSION, which never replaces the old row); the price date catches a live run that has not happened since
    the last sync (then every row is stale)."""
    if df.empty:
        return df
    ref = pd.Timestamp(df["base_date"].max())
    if latest_price is not None:
        ref = max(ref, pd.Timestamp(latest_price))
    stale = pd.to_datetime(df["base_date"]) < ref
    out = df.copy()
    out.loc[stale, "decision"] = "DATA_INVALID (stale)"
    out.loc[stale, "reason"] = [f"stale: phiên căn cứ {b} < {ref:%Y-%m-%d}" for b in out.loc[stale, "base_date"]]
    out["call"] = out["call"].where(~stale, None)
    out["p_strong"] = out["p_strong"].where(~stale)
    return out


def snapshot_passages(con, symbol: str, cutoff, model_version: str) -> pd.DataFrame:
    """The passages that fed the live snapshot at `cutoff` (tz-aware UTC): the selection of contract.build_snapshot in
    live mode, i.e. representative, relation != market, processed_at <= cutoff and within WINDOW_H hours of it.
    `flags` = risk_flags(passage_text), recomputed (as pipeline.load_results does), not stored."""
    df = pd.read_sql_query("""
        SELECT r.result_id, v.published_at, r.processed_at, v.title, r.relation, r.p_pos, r.p_neg, r.uncertain,
               r.passage_text
        FROM article_company_result r JOIN document_version v USING(document_version_id)
        WHERE r.nlp_model_version=? AND r.symbol=? AND v.is_representative=1 AND v.published_at IS NOT NULL
              AND r.relation != 'market'""", con, params=(model_version, symbol))
    for c in ("published_at", "processed_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df[(df["processed_at"] <= cutoff) & (cutoff - df["processed_at"] <= pd.Timedelta(hours=WINDOW_H))].copy()
    df["score"] = df["p_pos"] - df["p_neg"]
    df["uncertain"] = df["uncertain"].astype(bool)
    df["flags"] = [risk_flags(t) for t in df["passage_text"]]
    return df.sort_values("processed_at", ascending=False)
