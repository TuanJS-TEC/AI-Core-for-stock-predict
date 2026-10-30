from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .sentiment.base import LABELS, entropy


def metrics(labels, probs):
    pred = [LABELS[int(np.argmax(p))] for p in probs]
    return {"n": len(labels), "macro_f1": f1_score(labels, pred, labels=LABELS, average="macro", zero_division=0),
            "report": classification_report(labels, pred, labels=LABELS, zero_division=0, output_dict=True),
            "confusion[pos,neu,neg]": confusion_matrix(labels, pred, labels=LABELS).tolist()}


def bootstrap_macro_f1(labels, probs, n=2000, seed=0):
    """(2.5, 97.5) percentiles of macro-F1 over `n` row resamples with replacement."""
    y = np.array(labels)
    pred = np.array([LABELS[int(np.argmax(p))] for p in probs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    # same macro-F1 over all 3 LABELS as metrics(): a class absent from a resample's truth and predictions scores 0
    f = [f1_score(y[i], pred[i], labels=LABELS, average="macro", zero_division=0) for i in idx]
    lo, hi = np.percentile(f, [2.5, 97.5])
    return float(lo), float(hi)


def challenge_report(c, model_version, n_boot=2000, seed=0):
    """Negative-class F1 of `model_version`'s STORED predictions (argmax) on the labeled 'challenge' stream, with the
    same 2.5/97.5 row-resampling bootstrap as bootstrap_macro_f1. Samples the model has no stored result for are
    counted in n_unscored, not in n."""
    rows = c.execute("""SELECT s.sentiment_label, r.p_pos, r.p_neu, r.p_neg FROM annotation_sample s
                        LEFT JOIN article_company_result r ON r.document_version_id=s.document_version_id
                        AND r.target_entity_id=s.target_entity_id AND r.nlp_model_version=?
                        WHERE s.stream='challenge' AND s.annotation_status='labeled'""", (model_version,)).fetchall()
    scored = [r for r in rows if r["p_neg"] is not None]
    y = np.array([r["sentiment_label"] == "negative" for r in scored])
    pred = np.array([LABELS[int(np.argmax((r["p_pos"], r["p_neu"], r["p_neg"])))] == "negative" for r in scored])

    def f1(t, p):
        d = 2 * (t & p).sum(-1) + (~t & p).sum(-1) + (t & ~p).sum(-1)
        return np.where(d > 0, 2 * (t & p).sum(-1) / np.maximum(d, 1), 0.0)

    out = {"n": len(scored), "n_neg": int(y.sum()), "n_unscored": len(rows) - len(scored)}
    if not len(scored):
        return {**out, "f1_neg": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    lo, hi = np.percentile(f1(y[idx], pred[idx]), [2.5, 97.5])
    return {**out, "f1_neg": float(f1(y, pred)), "ci_lo": float(lo), "ci_hi": float(hi)}


def abstention_curve(labels, probs):
    """-> [(entropy_threshold, coverage, accuracy_on_kept)] to choose the abstention threshold on VALIDATION data."""
    ent = np.array([entropy(p) for p in probs])
    ok = np.array([LABELS[int(np.argmax(p))] == l for p, l in zip(probs, labels)])
    out = []
    for t in np.unique(np.round(ent, 3)):
        keep = ent <= t
        out.append((float(t), float(keep.mean()), float(ok[keep].mean()) if keep.any() else float("nan")))
    return out


def choose_abstain(labels, probs, min_acc):
    """Largest coverage whose accuracy on kept items >= min_acc; None if the target is unreachable."""
    ok = [(t, cov) for t, cov, acc in abstention_curve(labels, probs) if acc >= min_acc]
    return max(ok, key=lambda x: x[1])[0] if ok else None


def link_changes(c, old, new, stream):
    """Counter[(group, change)] over a stream's annotation samples, re-linking each with `old` and `new` linkers.
    group = sentiment_label if labeled else annotation_status; change = same | dropped | demoted | other."""
    out = Counter()

    def pick(lk, eid, title, text, pub):
        return next((l for l in lk.link(title or "", text, pub or "") if l.entity_id == eid), None)

    for eid, label, status, title, text, pub in c.execute(
            "SELECT s.target_entity_id, s.sentiment_label, s.annotation_status, v.title, v.normalized_text, v.published_at"
            " FROM annotation_sample s JOIN document_version v USING (document_version_id) WHERE s.stream=?", (stream,)):
        o, n = pick(old, eid, title, text, pub), pick(new, eid, title, text, pub)
        if o is None and n is None:
            ch = "same"                       # never linked under either: nothing changed
        elif n is None:
            ch = "dropped"
        elif o is None:
            ch = "other"
        elif n.relation == o.relation:
            ch = "same"
        else:
            ch = "demoted" if o.relation == "direct" else "other"
        out[(label or status, ch)] += 1
    return out


def drop_stale_links(c, linker):
    """Delete article_company_result rows (every model version) that `linker` no longer links; -> rows deleted.
    Also deletes their snapshot_lineage rows (FK); feature_snapshot rows built from them stay stale until rebuilt."""
    linked = {}
    stale = []
    for rid, dv, eid, title, text, pub in c.execute(
            "SELECT r.result_id, r.document_version_id, r.target_entity_id, v.title, v.normalized_text, v.published_at"
            " FROM article_company_result r JOIN document_version v USING (document_version_id)").fetchall():
        if dv not in linked:
            linked[dv] = {l.entity_id for l in linker.link(title or "", text, pub or "")}
        if eid not in linked[dv]:
            stale.append((rid,))
    # snapshot_lineage is derived and FK-references the results: drop those rows, snapshots must be rebuilt afterwards
    c.executemany("DELETE FROM snapshot_lineage WHERE result_id=?", stale)
    c.executemany("DELETE FROM article_company_result WHERE result_id=?", stale)
    c.commit()
    return len(stale)
