"""R12 (prereg addendum 2026-11-14): the financial branch fed with quarterly statements and their real publication dates
(fund_quarter.py) instead of annual ones, everything else as the locked v1 run: same folds, seeds, price encoder, rules.

Run once:  python -m ai_core.models.mmf.r12 --stock-dir data/prices/snapshots/<id>     then the same with --score.
"""
import argparse
import shutil

import pandas as pd
import torch

from . import compare as CP, config as C, evaluate as E, experiment as X
from .v3 import Scorer, _diff, _vs_chance
from ai_core.prices.data import build_panel
from ai_core.prices.splits import folds_from_config

ARMS = ("dl_fund", "no_text", "fusion_concat", "fusion_attention")
Q_BASE = ("logit_all", "gbm_all")                       # the baselines that read the fund group
OLD = "@nam"                                            # suffix of the locked run's arms (annual statements)
REF = ("prior_fixed", "logit_price", "dl_price")        # arms without a fund branch: taken from the locked run
# (name, a, b): paired log-loss a - b, negative favours a. The first three carry the registered verdicts.
COMPARE = (("R12a quý so với năm", "fusion_concat", "fusion_concat" + OLD),
           ("R12b fusion quý so với nhánh giá", "fusion_concat", "dl_price" + OLD),
           ("R12c nhánh tài chính quý so với tần suất", "dl_fund", "prior_fixed" + OLD),
           ("kèm: dl_fund", "dl_fund", "dl_fund" + OLD), ("kèm: no_text", "no_text", "no_text" + OLD),
           ("kèm: fusion_attention", "fusion_attention", "fusion_attention" + OLD),
           ("kèm: logit_all", "logit_all", "logit_all" + OLD), ("kèm: gbm_all", "gbm_all", "gbm_all" + OLD))


def copy_select(locked, run_dir):
    """The price encoder of every (h, fold) is the locked run's: its select.json is copied, never re-selected."""
    for src in sorted(locked.glob("h*/*/select.json")):
        dst = run_dir / src.relative_to(locked)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)


def load(run_dir, locked, h):
    """Eval and calib rows of the R12 arms (plain names) and of the locked run's arms (suffix OLD), one frame."""
    new = CP.load_h(run_dir, h)
    new = new[new["arm"].isin((*ARMS, *Q_BASE))]
    old = CP.load_h(locked, h)
    old = old[old["arm"].isin((*ARMS, *Q_BASE, *REF))].assign(arm=lambda d: d["arm"] + OLD)
    return pd.concat([new, old], ignore_index=True)


def score(run_dir, locked):
    cmp_rows, f1_rows, arm_rows, r0 = [], [], [], []
    for h in C.HORIZONS:
        d = load(run_dir, locked, h)
        folds = sorted(d["fold"].unique())
        for name, a, b in COMPARE:
            r = CP.compare(d, dict.fromkeys(folds, a), dict.fromkeys(folds, b))
            cmp_rows.append({"h": h, "so_sanh": name, "a": a, "b": b, **r})
        cols = ["symbol", "as_of_session", "y_true", "fold", "arm", "alpha", *E.PROB_COLS, *X.PRIOR_COLS]
        sc = Scorer(d[d["block"] == "eval"][cols])
        f1_rows += [_diff(sc, h, name, a, b) for name, a, b in COMPARE[:3]]
        arm_rows += [{"h": h, **sc.row(a)} for a in sorted(sc.arms)]
        f = next(r for r in arm_rows if r["h"] == h and r["arm"] == "fusion_concat")
        r0.append({"h": h, "arm": "fusion_concat", "macro_f1": f["macro_f1"], "f1_lo": f["f1_lo"], "auc_ovr": f["auc_ovr"],
                   "auc_lo": f["auc_lo"], "auc_up_down": f["auc_up_down"], "n": len(sc.y), "n_days": sc.n_days,
                   "verdict": _vs_chance(f["f1_lo"], f["auc_lo"])})
    md = lambda rows: pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")
    return (f"# R12: nhánh tài chính theo quý, fold A B C (vùng phát triển)\n\nRun: {run_dir}\nMốc (BCTC năm): {locked}\n\n"
            f"Arm không hậu tố dùng BCTC quý; hậu tố `{OLD}` là arm cùng tên của lượt đã khóa (BCTC năm).\n\n"
            f"## Chênh lệch log-loss ghép cặp (a - b, âm là a tốt hơn)\n\n{md(cmp_rows)}\n\n"
            f"## Chênh lệch Macro-F1, AUC, Accuracy theo quyết định chính thức (a - b, dương là a tốt hơn)\n\n{md(f1_rows)}\n\n"
            f"## fusion_concat (quý) so với ngẫu nhiên\n\n{md(r0)}\n\n## Mọi arm trên eval gộp ba fold\n\n{md(arm_rows)}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args(argv)
    panel = build_panel(stock_dir=a.stock_dir)
    ds = X._dataset_id(panel)[:16]
    run_dir, locked = C.DATA / "exp/r12" / ds, C.DATA / "runs" / ds
    if a.score:
        out = score(run_dir, locked)
        (run_dir.parent / "r12.md").write_text(out)
        print(out)
        return
    copy_select(locked, run_dir)
    X.run_all(panel, C.HORIZONS, folds_from_config(), run_dir, a.device, arms=ARMS, build=X.quarter_bundle)


if __name__ == "__main__":
    main()
