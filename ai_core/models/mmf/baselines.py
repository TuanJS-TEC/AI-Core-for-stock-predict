"""Baselines: per-symbol class prior (fixed and adaptive), logistic regression and GBM on the last row."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import config as C
from .frame import GROUPS

ADAPT_N = 120
SUFFIX = {"price": ("price",), "text": ("text",), "all": ("price", "fund", "text")}


def _laplace(y):
    """(n_c + 1) / (n + 3) over classes 0, 1, 2."""
    return (np.bincount(np.asarray(y, dtype=int), minlength=3) + 1) / (len(y) + 3)


def _prior_fixed(train, rows):
    y = train["label"].dropna().astype(int)
    sym = train.loc[y.index, "symbol"].astype(str)
    per = {s: _laplace(g) for s, g in y.groupby(sym)}
    return np.array([per.get(s, np.full(3, 1 / 3)) for s in rows["symbol"].astype(str)])


def _prior_adaptive(frame, rows):
    """Last ADAPT_N labels of the symbol that matured by the session (label_end_date <= t)."""
    f = frame[(frame["status"].astype(str) == "ok") & frame["label"].notna()].sort_values("as_of_session")
    hist = {s: (g["label_end_date"].to_numpy(), g["label"].astype(int).to_numpy()) for s, g in f.groupby(f["symbol"].astype(str))}
    out = np.empty((len(rows), 3))
    for i, (s, t) in enumerate(zip(rows["symbol"].astype(str), rows["as_of_session"])):
        end, y = hist.get(s, (np.empty(0), np.empty(0, int)))
        out[i] = _laplace(y[end <= np.datetime64(t)][-ADAPT_N:])      # empty -> uniform
    return out


def _model(kind):
    if kind == "logit":
        return LogisticRegression(C=0.1, max_iter=2000)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=59)


def _proba(clf, x):
    """predict_proba with columns mapped to classes 0, 1, 2 (zero for a class absent from training)."""
    out = np.zeros((len(x), 3))
    out[:, clf.classes_.astype(int)] = clf.predict_proba(x)
    return out


def predict_baseline(name, blocks, scaled, frame):
    """Probabilities (n, 3) for blocks 'calib' and 'eval', in the block row order. blocks: usable rows only."""
    if name not in C.BASELINES:
        raise ValueError(f"unknown baseline {name!r}")
    outs = ("calib", "eval")
    if name == "prior_fixed":
        return {k: _prior_fixed(blocks["train"], blocks[k]) for k in outs}
    if name == "prior_adaptive":
        return {k: _prior_adaptive(frame, blocks[k]) for k in outs}
    kind, group = name.split("_")
    cols = [c for g in SUFFIX[group] for c in GROUPS[g]]
    # ponytail: NaN -> 0 (the train mean after scaling) for the logit; GBM takes NaN natively
    x = lambda k: scaled.loc[blocks[k].index, cols].to_numpy(float)
    xt, yt = x("train"), blocks["train"]["label"].to_numpy(dtype="float")
    ok = ~np.isnan(yt)
    clf = _model(kind).fit(np.nan_to_num(xt[ok]) if kind == "logit" else xt[ok], yt[ok].astype(int))
    return {k: _proba(clf, np.nan_to_num(x(k)) if kind == "logit" else x(k)) for k in outs}
