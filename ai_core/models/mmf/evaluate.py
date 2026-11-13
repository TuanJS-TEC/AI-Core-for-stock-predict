"""Metrics, temperature/alpha calibration and paired block-bootstrap comparison."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support, roc_auc_score

from . import config as C
from ai_core.prices import config as PC
from ai_core.prices.stats import block_bootstrap_dates

PROB_COLS = ["p_down", "p_flat", "p_up"]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LABELS = [0, 1, 2]
NAMES = ["down", "flat", "up"]


def row_logloss(y, p):
    """Per-row negative log-likelihood of the true class."""
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def row_brier(y, p):
    """Per-row Brier score summed over the 3 classes (not divided)."""
    return ((p - np.eye(3)[y]) ** 2).sum(1)


def decide(p, alpha, prior):
    """Class = argmax(p / prior^alpha); prior floored at 1e-6."""
    return (p / np.maximum(prior, 1e-6) ** alpha).argmax(1)


def scores(y, p, alpha=0.0, prior=None):
    """Probability and decision metrics; decisions use `alpha`/`prior` (plain argmax when prior is None)."""
    y = np.asarray(y); p = np.asarray(p)
    pred = p.argmax(1) if prior is None else decide(p, alpha, prior)
    pr, rc, _, _ = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    m = y != 1
    auc_ud = roc_auc_score(y[m] == 2, p[m, 2] / (p[m, 2] + p[m, 0])) if len(set(y[m])) == 2 else np.nan
    out = {"logloss": row_logloss(y, p).mean(), "brier": row_brier(y, p).mean(), "accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
           "kappa": cohen_kappa_score(y, pred, labels=LABELS), "auc_ovr": roc_auc_score(y, p, multi_class="ovr", labels=LABELS) if len(set(y)) == 3 else np.nan, "auc_up_down": auc_ud}
    out.update({f"precision_{n}": v for n, v in zip(NAMES, pr)})
    out.update({f"recall_{n}": v for n, v in zip(NAMES, rc)})
    out["n"] = len(y)
    return out


def apply_temperature(logits, t):
    """Softmax of logits / t."""
    return softmax(np.asarray(logits) / t, axis=1)


def fit_temperature(logits, y):
    """Temperature T minimising log-loss; searched over beta = 1/T in [0.01, 10]."""
    z = np.asarray(logits); y = np.asarray(y)
    nll = lambda b: -log_softmax(z * b, axis=1)[np.arange(len(y)), y].mean()
    return 1.0 / minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded").x


def fit_alpha(y, p, prior):
    """Alpha from the grid with the best Macro-F1 (ties keep the smaller alpha)."""
    f1 = [f1_score(y, decide(p, a, prior), labels=LABELS, average="macro", zero_division=0) for a in ALPHAS]
    return ALPHAS[int(np.argmax(f1))]


def paired_diff(a, b, dates, n=2000, seed=0, block=PC.BOOTSTRAP_BLOCK):
    """Mean and 95% percentile CI of a - b: average per date, then block-bootstrap whole dates."""
    d = pd.Series(np.asarray(a) - np.asarray(b)).groupby(np.asarray(dates)).mean().to_numpy()
    idx = block_bootstrap_dates(d, block, n, np.random.default_rng(seed))
    boot = np.array([d[i].mean() for i in idx])
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def verdict(mean, hi, fold_wins, n_folds):
    """Pre-registered rule: evidence if hi < 0 and the model wins >= 2/3 of folds; weak if mean < 0."""
    if hi < 0 and 3 * fold_wins >= 2 * n_folds:
        return "Có bằng chứng"
    return "Có tín hiệu, bằng chứng yếu" if mean < 0 else "Chưa chứng minh được"
