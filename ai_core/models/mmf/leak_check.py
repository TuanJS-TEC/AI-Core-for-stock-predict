"""Leak check: null distribution of dl_price AUC up-vs-down on eval when train labels are permuted between dates."""
import argparse
from pathlib import Path

import pandas as pd

from . import config as C   # first: puts `src` on sys.path
from . import evaluate as E, experiment as X, train as TR
from src.data import build_panel
from src.splits import folds_from_config

KIND = "lstm"; SEED = 59    # one encoder and one seed for every fold, not the locked kinds or the 5-seed ensemble
HEADER = ["h", "fold", "AUC Tăng/Giảm, nhãn thật", "AUC nhãn xáo trộn: trung bình", "độ lệch chuẩn", "min", "max", "log-loss nhãn xáo trộn"]


def _score(b, device, max_epochs):
    """(AUC up-vs-down, log-loss) on eval of dl_price trained on b, temperature fit on calib."""
    fit = TR.fit_net(X._net_maker(b, "dl_price", KIND), b.samples["train"], b.samples["tune"], SEED, device, max_epochs)
    z = {k: TR.predict_logits(fit["net"], b.samples[k], device) for k in ("calib", "eval")}
    t, _ = X.calibrate(b, z["calib"])
    sc = E.scores(b.samples["eval"].y, E.apply_temperature(z["eval"], t))
    return sc["auc_up_down"], sc["logloss"]


def run(panel, horizons, perms, device, build=X.fold_bundle, max_epochs=C.MAX_EPOCHS):
    """One row per (h, fold): true-label AUC, then mean, sd, min, max of the AUC and mean log-loss over `perms` permutations."""
    rows = []
    for h in horizons:
        for fold in folds_from_config():
            b = build(panel, h, fold)
            auc, _ = _score(b, device, max_epochs)
            null = pd.DataFrame([_score(X.shuffle_train_labels(b, s), device, max_epochs) for s in range(perms)], columns=["auc", "ll"])
            rows.append(dict(zip(HEADER, (h, fold.name, auc, null["auc"].mean(), null["auc"].std(), null["auc"].min(), null["auc"].max(), null["ll"].mean()))))
            print(" | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in rows[-1].values()), flush=True)
    return pd.DataFrame(rows)


def write(d, out):
    """Markdown table of run() at `out`."""
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(d.to_markdown(index=False, floatfmt=("", "", ".3f", ".3f", ".3f", ".3f", ".3f", ".4f")) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-dir", required=True)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--h", type=int, nargs="+", default=list(C.HORIZONS), choices=C.HORIZONS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=C.DATA / "leak_check.md")
    a = ap.parse_args(argv)
    print(write(run(build_panel(C.TICKERS, stock_dir=a.stock_dir), a.h, a.perms, a.device), a.out))


if __name__ == "__main__":
    main()
