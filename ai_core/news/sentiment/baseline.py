"""TF-IDF + Logistic Regression baseline, trained on labeled annotation samples."""
import math
from itertools import product

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from .. import evaluate
from .base import LABELS


def class_weights(labels, kind):
    if kind not in ("balanced", "sqrt"):
        raise ValueError(f"unknown class-weight kind {kind!r}")
    if kind == "balanced":
        return "balanced"
    n = len(labels)  # kind == "sqrt"
    return {c: (n / labels.count(c)) ** 0.5 for c in set(labels)}


def train(texts, labels, version, features="v1", weights="balanced", C=1.0, prior=(1.0, 1.0, 1.0)):
    if features not in ("v1", "v2"):
        raise ValueError(f"unknown features {features!r}")
    labels = list(labels)
    if features == "v2":
        vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.90, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True,
                                     lowercase=True))])
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
    m = make_pipeline(vec, LogisticRegression(max_iter=2000, C=C, class_weight=class_weights(labels, weights)))
    m.fit(texts, labels)
    return Baseline(m, version, prior)


def fit_prior(probs, labels, grid=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)):
    """(a_pos, 1, a_neg) with the best macro-F1 on (probs, labels); ties -> smallest |log a| sum, then first tried."""
    probs, best = np.asarray(probs, dtype=float), None
    for a_pos, a_neg in product(grid, grid):
        w = np.array([a_pos, 1.0, a_neg])
        key = (-evaluate.metrics(labels, probs * w)["macro_f1"], abs(math.log(a_pos)) + abs(math.log(a_neg)))
        if best is None or key < best[0]:
            best = (key, (a_pos, 1.0, a_neg))
    return best[1]


class Baseline:
    def __init__(self, model, version, prior=(1.0, 1.0, 1.0)):
        self.model, self.version, self.prior = model, version, tuple(prior)

    def predict(self, texts):
        proba = self.model.predict_proba(texts)
        idx = {c: i for i, c in enumerate(self.model.classes_)}
        out = [tuple(float(row[idx[l]]) if l in idx else 0.0 for l in LABELS) for row in proba]
        if self.prior == (1.0, 1.0, 1.0):  # skip: keeps v1 bundles bit-identical
            return out
        out = [tuple(p * w for p, w in zip(r, self.prior)) for r in out]
        return [tuple(p / sum(r) for p in r) for r in out]

    def save(self, path):
        joblib.dump((self.model, self.version, self.prior), path)

    @classmethod
    def load(cls, path):
        return cls(*joblib.load(path))
