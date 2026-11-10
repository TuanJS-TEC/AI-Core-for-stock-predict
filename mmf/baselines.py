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


def predict_baseline(name, blocks, scaled, frame, price_cols=None):
    """Probabilities (n, 3) for blocks 'calib' and 'eval', in the block row order. blocks: usable rows only.

    price_cols replaces the columns of the price group (the ctx variant adds the context features)."""
    if name not in C.BASELINES and name not in C.HURDLE:
        raise ValueError(f"unknown baseline {name!r}")
    outs = ("calib", "eval")
    if name == "prior_fixed":
        return {k: _prior_fixed(blocks["train"], blocks[k]) for k in outs}
    if name == "prior_adaptive":
        return {k: _prior_adaptive(frame, blocks[k]) for k in outs}
    kind, group = name.split("_")[-2:]
    cols = [c for g in SUFFIX[group] for c in (price_cols if g == "price" and price_cols is not None else GROUPS[g])]
    # ponytail: NaN -> 0 (the train mean after scaling) for the logit; GBM takes NaN natively
    def x(k):
        v = scaled.loc[blocks[k].index, cols].to_numpy(float)
        return np.nan_to_num(v) if kind == "logit" else v

    xt, yt = x("train"), blocks["train"]["label"].to_numpy(dtype="float")
    ok = ~np.isnan(yt)
    xt, yt, xs = xt[ok], yt[ok].astype(int), {k: x(k) for k in outs}
    if name in C.HURDLE:      # stage 1: does the price leave the flat band; stage 2, on the movers only: up or down
        move = yt != 1
        pm, pu = _stage(kind, xt, move, xs), _stage(kind, xt[move], yt[move] == 2, xs)
        return {k: np.column_stack([pm[k] * (1 - pu[k]), 1 - pm[k], pm[k] * pu[k]]) for k in outs}
    clf = _model(kind).fit(xt, yt)
    return {k: _proba(clf, xs[k]) for k in outs}


def _stage(kind, x, target, xs):
    """P(target) per block from a binary model of the hurdle; a stage that sees one class returns its Laplace frequency."""
    if len(set(target)) < 2:
        return {k: np.full(len(v), (target.sum() + 1) / (len(target) + 2)) for k, v in xs.items()}
    clf = _model(kind).fit(x, target)
    return {k: clf.predict_proba(v)[:, 1] for k, v in xs.items()}
