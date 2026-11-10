"""Annotation workflow: sample -> CSV for the annotator -> import labels. Two streams:
dev (guideline, error analysis, training, model selection) and eval (frozen, time-based, never tuned on)."""
import csv
import random

from .dedup import jaccard, shingles
from .risk import risk_flags
from .storage import ROOT

LABELS = {"positive", "neutral", "negative"}
OTHER = {"unclear", "skip"}


def sample(c, model_version, n, eval_after, seed=0):
    """Random sample of direct, representative, not-yet-sampled article-company pairs.
    Stream is decided by time (published_at >= eval_after -> eval), never by random split.
    Model predictions are NOT shown to the annotator (avoids anchoring)."""
    rows = c.execute("""SELECT r.document_version_id vid, r.target_entity_id eid, r.passage_text, r.sentence_ids,
                        r.difficult, v.published_at FROM article_company_result r
                        JOIN document_version v USING(document_version_id)
                        WHERE r.nlp_model_version=? AND r.relation='direct' AND v.is_representative=1
                        AND v.published_at IS NOT NULL AND NOT EXISTS (SELECT 1 FROM annotation_sample a
                        WHERE a.document_version_id=r.document_version_id AND a.target_entity_id=r.target_entity_id)""",
                     (model_version,)).fetchall()
    random.Random(seed).shuffle(rows)
    for r in rows[:n]:
        stream = "eval" if r["published_at"][:10] >= eval_after else "dev"
        c.execute("""INSERT INTO annotation_sample (sample_id, document_version_id, target_entity_id, passage_text,
                     sentence_ids, difficult, stream) VALUES (?,?,?,?,?,?,?)""",
                  (f"{r['vid']}:{r['eid']}", r["vid"], r["eid"], r["passage_text"], r["sentence_ids"],
                   r["difficult"], stream))
    c.commit()
    return min(n, len(rows))


CHALLENGE_MODELS = ("baseline-tfidf-lr-v1", "phobert-phobert-v1")


def sample_challenge(c, n=60, seed=0):
    """Negative-class challenge set (stream 'challenge'): direct, representative, not-yet-sampled pairs where either
    CHALLENGE_MODELS member's p_neg is the largest of the three, or the passage has event-risk flags (recomputed from
    passage_text with risk.risk_flags, the same way pipeline.load_results does). One sample per document_id."""
    rows = c.execute("""SELECT r.document_version_id vid, v.document_id did, r.target_entity_id eid, r.passage_text,
                        r.sentence_ids, r.difficult, r.p_pos, r.p_neu, r.p_neg FROM article_company_result r
                        JOIN document_version v USING(document_version_id)
                        WHERE r.nlp_model_version IN (?,?) AND r.relation='direct' AND v.is_representative=1
                        AND v.published_at IS NOT NULL AND NOT EXISTS (SELECT 1 FROM annotation_sample a
                        WHERE a.document_version_id=r.document_version_id AND a.target_entity_id=r.target_entity_id)
                        ORDER BY r.document_version_id, r.target_entity_id, r.nlp_model_version""",
                     CHALLENGE_MODELS).fetchall()
    pool = {}
    for r in rows:   # a pair is scored by both models; it qualifies if either one calls it negative (strict argmax)
        neg = None not in (r["p_pos"], r["p_neu"], r["p_neg"]) and r["p_neg"] > max(r["p_pos"], r["p_neu"])
        if (r["vid"], r["eid"]) in pool:
            pool[r["vid"], r["eid"]][1] |= neg
        else:
            pool[r["vid"], r["eid"]] = [r, neg]
    cand = [r for r, neg in pool.values() if neg or risk_flags(r["passage_text"])]
    random.Random(seed).shuffle(cand)
    docs, n_new = set(), 0
    for r in cand:
        if n_new >= n:
            break
        if r["did"] in docs:
            continue
        docs.add(r["did"])
        c.execute("""INSERT INTO annotation_sample (sample_id, document_version_id, target_entity_id, passage_text,
                     sentence_ids, difficult, stream) VALUES (?,?,?,?,?,?,'challenge')""",
                  (f"{r['vid']}:{r['eid']}", r["vid"], r["eid"], r["passage_text"], r["sentence_ids"], r["difficult"]))
        n_new += 1
    c.commit()
    return n_new


def export_csv(c, path, stream=None, only_pending=True):
    q = """SELECT a.sample_id, a.target_entity_id symbol, v.published_at, d.url_normalized url, v.title, a.difficult,
           a.passage_text FROM annotation_sample a JOIN document_version v USING(document_version_id)
           JOIN document d USING(document_id) WHERE 1=1"""
    args = []
    if only_pending:
        q += " AND a.annotation_status='pending'"
    if stream:
        q += " AND a.stream=?"
        args.append(stream)
    rows = c.execute(q + " ORDER BY a.sample_id", args).fetchall()
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "symbol", "published_at", "url", "title", "difficult", "passage_text", "label", "note"])
        for r in rows:
            w.writerow([*tuple(r), "", ""])
    return len(rows)


def import_csv(c, path, annotator_id, guideline_version):
    n = bad = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        f.seek(0)
        delim = ";" if head.count(";") > head.count(",") else ","  # ponytail: some spreadsheet locales export ';'
        for r in csv.DictReader(f, delimiter=delim):
            lab = (r.get("label") or "").strip().lower()
            if not lab:
                continue
            if lab in LABELS:
                c.execute("""UPDATE annotation_sample SET sentiment_label=?, annotation_status='labeled', annotator_id=?,
                             guideline_version=? WHERE sample_id=?""", (lab, annotator_id, guideline_version, r["sample_id"]))
            elif lab in OTHER:
                c.execute("UPDATE annotation_sample SET sentiment_label=NULL, annotation_status=?, annotator_id=?, guideline_version=? WHERE sample_id=?",
                          ("skipped" if lab == "skip" else "unclear", annotator_id, guideline_version, r["sample_id"]))
            else:
                bad += 1
                continue
            n += 1
    c.commit()
    return n, bad


def dataset(c, stream):
    rows = c.execute("""SELECT a.sample_id, a.passage_text, a.sentiment_label, a.difficult, v.published_at
                        FROM annotation_sample a JOIN document_version v USING(document_version_id)
                        WHERE a.stream=? AND a.annotation_status='labeled' ORDER BY a.sample_id""", (stream,)).fetchall()
    return [r["passage_text"] for r in rows], [r["sentiment_label"] for r in rows], rows


def leak_check(c, threshold=0.8):
    """Near-identical passages in both dev and eval streams (reposts) would inflate eval scores."""
    dev = [(r[0], shingles(r[1])) for r in c.execute("SELECT sample_id, passage_text FROM annotation_sample WHERE stream='dev'")]
    hits = []
    for sid, text in c.execute("SELECT sample_id, passage_text FROM annotation_sample WHERE stream='eval'"):
        sh = shingles(text)
        hits += [(sid, d, round(jaccard(sh, s), 2)) for d, s in dev if jaccard(sh, s) >= threshold]
    return hits
