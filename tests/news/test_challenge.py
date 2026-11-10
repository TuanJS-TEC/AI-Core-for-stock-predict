import csv
import tempfile

import pytest

from ai_core.news import annotate, storage
from ai_core.news.connectors.base import ParsedDoc
from ai_core.news.evaluate import challenge_report

MV = "baseline-tfidf-lr-v1"


def _result(c, key, body, probs, mv=MV):
    v, _ = storage.ingest(c, ParsedDoc("t", f"https://a.vn/{key}", "news", f"tin {key}", body, "2026-01-01T00:00:00Z",
                                       None, b"<html/>"), tempfile.mkdtemp())
    c.execute("""INSERT INTO article_company_result (result_id, document_version_id, target_entity_id, symbol, relation,
                 sentence_ids, passage_text, p_pos, p_neu, p_neg, nlp_model_version, processed_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?, '2026-01-01T00:00:00Z')""",
              (f"{key}:{mv}", v, "VHM", "VHM", "direct", "[0]", body, *probs, mv))
    return v


def _db():
    c = storage.connect(":memory:")
    _result(c, "neg", "Vinhomes lỗ nặng quý này.", (0.05, 0.05, 0.9))
    _result(c, "flag", "Cơ quan điều tra khởi tố lãnh đạo Vinhomes.", (0.3, 0.6, 0.1))
    _result(c, "neu", "Vinhomes họp đại hội cổ đông.", (0.2, 0.7, 0.1))
    return c


def test_sample_challenge_takes_negative_and_flagged_once():
    c = _db()
    assert annotate.sample_challenge(c, 60) == 2
    got = {r[0].split(":")[0] for r in c.execute("SELECT sample_id FROM annotation_sample WHERE stream='challenge'")}
    assert len(got) == 2
    assert annotate.sample_challenge(c, 60) == 0


def test_sample_challenge_one_per_document_and_respects_n():
    c = _db()
    assert annotate.sample_challenge(c, 1) == 1


def test_export_csv_accepts_challenge_stream():
    c = _db()
    annotate.sample_challenge(c, 60)
    p = tempfile.mkstemp(suffix=".csv")[1]
    assert annotate.export_csv(c, p, "challenge") == 2
    assert "label" in next(csv.reader(open(p, encoding="utf-8-sig")))


def test_challenge_report_f1_neg_and_ci():
    c = storage.connect(":memory:")
    # 6 labeled negatives: model gets 4 right; 2 neutrals: model calls one of them negative
    spec = [("n1", "negative", (0, 0, 1)), ("n2", "negative", (0, 0, 1)), ("n3", "negative", (0, 0, 1)),
            ("n4", "negative", (0, 0, 1)), ("n5", "negative", (0, 1, 0)), ("n6", "negative", (0, 1, 0)),
            ("u1", "neutral", (0, 1, 0)), ("u2", "neutral", (0, 0, 1))]
    for k, lab, p in spec:
        v = _result(c, k, f"passage {k}", p)
        c.execute("""INSERT INTO annotation_sample (sample_id, document_version_id, target_entity_id, passage_text,
                     sentence_ids, stream, sentiment_label, annotation_status) VALUES (?,?,?,?,?,?,?,?)""",
                  (f"{v}:VHM", v, "VHM", "x", "[0]", "challenge", lab, "labeled"))
    r = challenge_report(c, MV)
    assert (r["n"], r["n_neg"]) == (8, 6)
    assert r["f1_neg"] == pytest.approx(2 * 4 / (2 * 4 + 1 + 2))   # tp=4 fp=1 fn=2
    assert 0 <= r["ci_lo"] <= r["f1_neg"] <= r["ci_hi"] <= 1
    assert r == challenge_report(c, MV)


def test_cli_smoke_sample_export_report(tmp_path, capsys):
    from ai_core.news import cli
    db = str(tmp_path / "t.db")
    c = storage.connect(db)
    _result(c, "neg", "Vinhomes lỗ nặng quý này.", (0.05, 0.05, 0.9))
    c.commit(); c.close()
    cli.main(["--db", db, "challenge-sample", "-n", "5"])
    cli.main(["--db", db, "annotate-export", str(tmp_path / "x.csv"), "--stream", "challenge"])
    cli.main(["--db", db, "challenge-report", "--model", MV])
    out = capsys.readouterr().out
    assert "sampled 1" in out and "exported 1" in out and '"n": 0' in out
