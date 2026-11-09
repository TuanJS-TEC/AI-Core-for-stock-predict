from datetime import datetime, timezone
import pandas as pd
from nlp_module.contract import ArticleCompanyResult
from mmf import text as T

V = "v-test"
def art(rid, sym, pub, pos=0.9, neg=0.1, uncertain=False):
    t = datetime.fromisoformat(pub).replace(tzinfo=timezone.utc)
    return ArticleCompanyResult(rid, sym, "cafef", t, t, "direct", pos, 1 - pos - neg, neg, uncertain, False, True, V, ())

DAY = pd.DatetimeIndex(["2026-01-05"])   # cutoff 08:00 UTC

def one(results):
    return T.features_from_results(results, ["FPT"], DAY, V).iloc[0]

def test_article_inside_delay_or_after_cutoff_is_excluded():
    assert one([art("a", "FPT", "2026-01-05T07:45:00")]).has_news == 0   # 15 phút trước mốc, chưa qua 30 phút
    assert one([art("b", "FPT", "2026-01-05T09:00:00")]).has_news == 0   # sau mốc
    assert one([art("c", "FPT", "2026-01-05T07:30:00")]).has_news == 1

def test_uncertain_flag_is_ignored():
    r = one([art("d", "FPT", "2026-01-05T01:00:00", uncertain=True)])
    assert r.sentiment_mean > 0

def test_no_news_has_no_nan_and_other_symbol_not_counted():
    r = one([art("e", "VCB", "2026-01-05T01:00:00")])
    assert r.has_news == 0 and r[T.TEXT_FEATS].notna().all() and r.hours_since_last_news == 1.0
