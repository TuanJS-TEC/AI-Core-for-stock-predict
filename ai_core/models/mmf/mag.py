"""Scoring of the magnitude label "mag" (prereg addenda of 2026-11-12): R7 on folds A, B, C and R8 on the backward folds.

Every arm's probabilities are renormalised over the two classes (0 = Yếu, 1 = Mạnh) before any pick or score.
Use: python -m ai_core.models.mmf.mag --run-dir data/mmf/exp/mag/<id> --rule r7 --out data/mmf/exp/mag/r7.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import compare as CP, evaluate as E

BASE = ("logit_price", "gbm_price", "logit_all", "gbm_all")
DEEP = ("dl_price", "dl_text", "no_text", "fusion_concat", "fusion_attention")
R8_ARMS = ("prior_fixed", "logit_price", "dl_price", "no_text", "fusion_concat")
COLS = ["h", "so_sanh", "a", "b", "mean", "lo", "hi", "fold_wins", "n_folds", "n", "n_days", "verdict"]


def two_class(preds):
    """Copy with p_up = 0 and p_down, p_flat rescaled to sum to 1."""
    out = preds.copy()
    s = out["p_down"] + out["p_flat"]
    out["p_down"], out["p_flat"], out["p_up"] = out["p_down"] / s, out["p_flat"] / s, 0.0
    return out


def load(run_dir, h):
    return two_class(CP.load_h(run_dir, h))


def _same(arm, folds):
    return dict.fromkeys(folds, arm)


def _row(ph, h, name, a, b):
    return {"h": h, "so_sanh": name, "a": CP._fmt(a), "b": CP._fmt(b), **CP.compare(ph, a, b)}


def r7(ph, h):
    """Registered R7a-c, then the reported-only pairs (so_sanh starts with 'kèm')."""
    folds = sorted(ph["fold"].unique())
    best = lambda cands: {f: CP.pick(ph, f, cands) for f in folds}
    return [_row(ph, h, "R7a", best(BASE + DEEP), _same("prior_fixed", folds)),
            _row(ph, h, "R7b", best(DEEP), best(BASE)),
            _row(ph, h, "R7c", _same("fusion_concat", folds), _same("no_text", folds)),
            _row(ph, h, "kèm: attention - no_text", _same("fusion_attention", folds), _same("no_text", folds)),
            _row(ph, h, "kèm: dl_text - prior", _same("dl_text", folds), _same("prior_fixed", folds))]


def r8(ph, h):
    folds = sorted(ph["fold"].unique())
    return [_row(ph, h, n, _same("fusion_concat", folds), _same(b, folds))
            for n, b in (("R8a", "prior_fixed"), ("R8b", "logit_price"), ("R8c", "no_text"))]


def per_fold(ph, h, rows):
    """The same differences on each fold alone."""
    out = []
    for r in rows:
        a, b = (dict(x.split("=") for x in r[k].split()) for k in ("a", "b"))
        for f in a:
            c = CP.compare(ph, {f: a[f]}, {f: b[f]})
            out.append({"h": h, "so_sanh": r["so_sanh"], "fold": f, "a": a[f], "b": b[f], **{k: c[k] for k in ("mean", "lo", "hi", "n_days")}})
    return out


def arm_scores(ph, h):
    """Eval log-loss, Brier, AUC strong vs weak, accuracy and predicted strong share per (fold, arm)."""
    out = []
    for (f, a), g in ph[ph["block"] == "eval"].groupby(["fold", "arm"]):
        y, p = g["y_true"].to_numpy(), g[E.PROB_COLS].to_numpy()
        out.append({"h": h, "fold": f, "arm": a, "n": len(y), "n_days": g["as_of_session"].nunique(),
                    "logloss": E.row_logloss(y, p).mean(), "brier": ((p[:, 1] - y) ** 2).mean(),
                    "auc": roc_auc_score(y, p[:, 1]) if len(set(y)) == 2 else np.nan,
                    "accuracy": ((p[:, 1] > 0.5) == (y == 1)).mean(), "pred_strong": (p[:, 1] > 0.5).mean(), "true_strong": y.mean()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rule", choices=("r7", "r8"), required=True)
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    main_rows, fold_rows, scores = [], [], []
    for h in CP._h_dirs(a.run_dir):
        ph = load(a.run_dir, h)
        ph = ph[ph["fold"].isin(a.folds)]
        if a.rule == "r8":
            ph = ph[ph["arm"].isin(R8_ARMS)]
        rows = {"r7": r7, "r8": r8}[a.rule](ph, h)
        main_rows += rows; fold_rows += per_fold(ph, h, rows); scores += arm_scores(ph, h)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    md = lambda rows, cols=None: pd.DataFrame(rows)[cols or list(rows[0])].to_markdown(index=False, floatfmt=".4f")
    out.write_text(f"# {a.rule.upper()} (nhãn mag), folds {' '.join(a.folds)}\n\nRun: {a.run_dir}\n\n## So sánh (a - b, âm là a tốt hơn)\n\n"
                   f"{md(main_rows, COLS)}\n\n## Theo từng fold\n\n{md(fold_rows)}\n\n## Mọi arm trên eval\n\n{md(scores)}\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
