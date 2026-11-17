"""R10 (prereg addendum 2026-11-13): the 3-class direction label under the v3 rules, plus two price-branch trials.

v3 rules: class-weighted cross-entropy, temperature and alpha fit on tune + calib, picks by Macro-F1 of the official
decision on tune + calib, price encoder kind taken from the locked run's select.json. Trials: E1 `short_price` (15-session
window of 6 step features + the session's 25 features), E2 `*_m` (train rows within 0.85-1.15 of their own threshold dropped).
Run once: python -m ai_core.models.mmf.v3 --stock-dir data/prices/snapshots/<id>   then the same with --score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch import nn

from . import baselines as B, compare as CP, config as C, evaluate as E, experiment as X, train as TR
from .frame import GROUPS
from ai_core.prices import config as PC
from ai_core.prices.data import build_panel
from ai_core.prices.splits import folds_from_config
from ai_core.prices.stats import block_bootstrap_dates

ARMS = ("dl_price", "no_text", "fusion_concat", "fusion_attention", "short_price", "dl_price_m", "fusion_concat_m")
BASE = ("prior_fixed", "logit_price", "gbm_price", "logit_all", "gbm_all")
FUSION = ("fusion_concat", "fusion_attention"); BASE_PICK = BASE[1:]
STEP = ["ret_1", "c_o", "h_l", "o_pc", "close_loc", "v_ma20"]; SHORT = 15
BAND = (0.85, 1.15)
FIT = ("tune", "calib"); BLOCKS = (*FIT, "eval")
KEY = ["as_of_session", "symbol"]
N_BOOT = 2000


class ShortPriceNet(nn.Module):
    """E1: GRU over the last SHORT sessions of the STEP features, joined with the session's own price features."""
    def __init__(self, n_tickers, f_price, step_idx):
        super().__init__()
        self.idx = list(step_idx)
        self.rnn = nn.GRU(len(self.idx), C.D, batch_first=True)
        self.flat = nn.Sequential(nn.Linear(f_price, C.D), nn.ReLU(), nn.Dropout(C.DROPOUT))
        self.ticker = nn.Embedding(n_tickers, C.EMB)
        self.drop = nn.Dropout(C.DROPOUT); self.out = nn.Linear(2 * C.D + C.EMB, 3)

    def forward(self, x_price, x_text, x_fund, ticker):
        z = torch.cat([self.rnn(x_price[:, -SHORT:, self.idx])[0][:, -1], self.flat(x_price[:, -1])], 1)
        return self.out(self.drop(torch.cat([z, self.ticker(ticker)], 1)))


def margin_weights(rows, h):
    """E2: 0 for a train row whose |h-session return| lies within BAND of its own label threshold, else 1."""
    theta = PC.K_ADAPT * rows["std_60"].to_numpy(float) * np.sqrt(h)
    r = rows["realized_return"].to_numpy(float)
    y = rows["label"].to_numpy(int)
    if not (((r > theta) == (y == 2)) & ((r < -theta) == (y == 0))).all():
        raise ValueError("threshold rebuilt from std_60 does not reproduce the labels")
    z = np.abs(r) / theta
    return (~((z >= BAND[0]) & (z <= BAND[1]))).astype(float)


def run_fold(panel, h, fold, kind, device, seeds=C.SEEDS, max_epochs=C.MAX_EPOCHS):
    """All R10 arms and baselines of one (h, fold): seed-mean probabilities on tune, calib, eval, alpha per arm."""
    b = X.fold_bundle(panel, h, fold)
    y_tr = b.samples["train"].y
    cw = len(y_tr) / (3 * np.bincount(y_tr, minlength=3))
    mw = margin_weights(b.blocks["train"], h)
    y_fit = np.concatenate([b.samples[k].y for k in FIT])
    prior = {k: X._prior_rows(b, k) for k in BLOCKS}
    prior_fit = np.concatenate([prior[k] for k in FIT])
    step_idx = [GROUPS["price"].index(c) for c in STEP]
    out = []

    def emit(arm, p):
        alpha = E.fit_alpha(y_fit, np.concatenate([p[k] for k in FIT]), prior_fit)
        for k in BLOCKS:
            s = b.samples[k]
            d = pd.DataFrame({"symbol": s.symbol, "as_of_session": s.as_of, "y_true": s.y, "block": k})
            d[E.PROB_COLS] = p[k]; d[X.PRIOR_COLS] = prior[k]
            out.append(d.assign(arm=arm, alpha=alpha, h=h, fold=fold.name))
        print(f"h{h} {fold.name} {arm} alpha={alpha}", flush=True)

    for arm in ARMS:
        make = (lambda: ShortPriceNet(len(b.tickers), len(GROUPS["price"]), step_idx)) if arm == "short_price" \
            else X._net_maker(b, arm.removesuffix("_m"), kind)
        ps = []
        for seed in seeds:
            fit = TR.fit_net(make, b.samples["train"], b.samples["tune"], seed, device, max_epochs, class_weight=cw,
                             sample_weight=mw if arm.endswith("_m") else None)
            z = {k: TR.predict_logits(fit["net"], b.samples[k], device) for k in BLOCKS}
            t = E.fit_temperature(np.concatenate([z[k] for k in FIT]), y_fit)
            ps.append({k: E.apply_temperature(z[k], t) for k in BLOCKS})
        emit(arm, {k: np.mean([p[k] for p in ps], axis=0) for k in BLOCKS})
    n_tune = len(b.blocks["tune"])      # baselines only know "calib" and "eval": hand them tune + calib as one block
    bl = {"train": b.blocks["train"], "calib": pd.concat([b.blocks["tune"], b.blocks["calib"]]), "eval": b.blocks["eval"]}
    for name in BASE:
        p = B.predict_baseline(name, bl, b.scaled, b.frame)
        emit(name, {"tune": p["calib"][:n_tune], "calib": p["calib"][n_tune:], "eval": p["eval"]})
    return pd.concat(out, ignore_index=True)


# --- scoring --------------------------------------------------------------------------------------------------------
def _decide(g):
    """Official decision per row; alpha is constant within a (fold, arm)."""
    pred = np.empty(len(g), dtype=int)
    for _, i in g.groupby("fold").indices.items():
        x = g.iloc[i]
        pred[i] = E.decide(x[E.PROB_COLS].to_numpy(), float(x["alpha"].iloc[0]), x[X.PRIOR_COLS].to_numpy())
    return pred


def _f1(y, pred):
    return f1_score(y, pred, labels=E.LABELS, average="macro", zero_division=0)


def _auc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", labels=E.LABELS) if len(set(y)) == 3 else np.nan


class Scorer:
    """Eval rows of every arm on one shared (session, symbol) order, with one set of bootstrap resamples of the days."""
    def __init__(self, ev):
        self.arms = {}
        ref = None
        ev = ev.assign(as_of_session=pd.to_datetime(ev["as_of_session"]).astype("datetime64[ns]"), symbol=ev["symbol"].astype(object),
                       fold=ev["fold"].astype(object), y_true=ev["y_true"].astype("int64"))
        for arm, g in ev.groupby("arm"):
            g = g.sort_values(KEY).reset_index(drop=True)
            keys = g[[*KEY, "y_true", "fold"]]
            if ref is None:
                ref = keys
            elif not keys.equals(ref):
                raise ValueError(f"{arm}: eval rows or labels differ from the other arms")
            self.arms[arm] = (g[E.PROB_COLS].to_numpy(), _decide(g), g)
        self.y, self.fold = ref["y_true"].to_numpy(), ref["fold"].to_numpy()
        days, pos = np.unique(ref["as_of_session"].to_numpy(), return_inverse=True)
        rows = [np.flatnonzero(pos == i) for i in range(len(days))]
        self.n_days = len(days)
        self.boot = [np.concatenate([rows[j] for j in idx])
                     for idx in block_bootstrap_dates(days, PC.BOOTSTRAP_BLOCK, N_BOOT, np.random.default_rng(0))]

    def add(self, name, by_fold):
        """Composite arm: per fold the rows of the arm picked for it."""
        p, pred = np.empty((len(self.y), 3)), np.empty(len(self.y), dtype=int)
        for f, arm in by_fold.items():
            m = self.fold == f
            p[m], pred[m] = self.arms[arm][0][m], self.arms[arm][1][m]
        self.arms[name] = (p, pred, None)

    def stat(self, arm, i=slice(None)):
        p, pred, _ = self.arms[arm]
        y = self.y[i]
        return np.array([_f1(y, pred[i]), _auc(y, p[i]), (pred[i] == y).mean()])

    def ci(self, arm, other=None):
        """(point, lo, hi) arrays for Macro-F1, AUC, accuracy of `arm`, or of arm - other."""
        f = (lambda i: self.stat(arm, i)) if other is None else (lambda i: self.stat(arm, i) - self.stat(other, i))
        boot = np.array([f(i) for i in self.boot])
        return f(slice(None)), np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)

    def fold_wins(self, a, b):
        fs = sorted(set(self.fold))
        return sum(_f1(self.y[self.fold == f], self.arms[a][1][self.fold == f]) > _f1(self.y[self.fold == f], self.arms[b][1][self.fold == f])
                   for f in fs), len(fs)

    def row(self, arm):
        p, pred, g = self.arms[arm]
        (f1, auc, acc), lo, hi = self.ci(arm)
        pr, rc, _, _ = precision_recall_fscore_support(self.y, pred, labels=E.LABELS, zero_division=0)
        ud = self.y != 1
        out = {"arm": arm, "accuracy": acc, "acc_lo": lo[2], "acc_hi": hi[2], "macro_f1": f1, "f1_lo": lo[0], "f1_hi": hi[0],
               "auc_ovr": auc, "auc_lo": lo[1], "auc_hi": hi[1],
               "auc_up_down": roc_auc_score(self.y[ud] == 2, p[ud, 2] / (p[ud, 2] + p[ud, 0])),
               "logloss": E.row_logloss(self.y, p).mean()}
        out.update({f"P_{n}": v for n, v in zip(E.NAMES, pr)}); out.update({f"R_{n}": v for n, v in zip(E.NAMES, rc)})
        out["pred_share"] = "/".join(f"{v:.2f}" for v in np.bincount(pred, minlength=3) / len(pred))
        return out


def picks(d, candidates):
    """Per fold, the candidate with the best Macro-F1 of the official decision on tune + calib (ties: first listed)."""
    out = {}
    for f, g in d[d["block"].isin(FIT)].groupby("fold"):
        score = {a: _f1(x["y_true"].to_numpy(), _decide(x.reset_index(drop=True))) for a in candidates if len(x := g[g["arm"] == a])}
        out[f] = max(candidates, key=lambda a: score.get(a, -1))
    return out


def _vs_chance(f1_lo, auc_lo):
    n = int(f1_lo > 1 / 3) + int(auc_lo > 0.5)
    return ("Chưa chứng minh được", "Có tín hiệu, bằng chứng yếu", "Có bằng chứng hơn ngẫu nhiên")[n]


def _diff(sc, h, name, a, b):
    (f1, auc, acc), lo, hi = sc.ci(a, b)
    w, n = sc.fold_wins(a, b)
    verdict = "Có bằng chứng" if lo[0] > 0 and 3 * w >= 2 * n else "Có tín hiệu, bằng chứng yếu" if f1 > 0 else "Chưa chứng minh được"
    return {"h": h, "so_sanh": name, "a": a, "b": b, "d_macro_f1": f1, "f1_lo": lo[0], "f1_hi": hi[0], "d_auc": auc, "auc_lo": lo[1],
            "auc_hi": hi[1], "d_accuracy": acc, "fold_wins": f"{w}/{n}", "verdict": verdict}


def score(run_dir, locked_dir):
    arms, r0, cmp_rows, pick_rows = [], [], [], []
    for h in C.HORIZONS:
        d = pd.concat([pd.read_parquet(p) for p in sorted(Path(run_dir).glob(f"h{h}_*.parquet"))], ignore_index=True)
        v1 = CP.load_h(locked_dir, h)
        v1 = v1[v1["arm"].isin(("dl_price", "fusion_concat")) & (v1["block"] == "eval")].assign(arm=lambda x: x["arm"] + "@v1")
        cols = ["symbol", "as_of_session", "y_true", "fold", "arm", "alpha", *E.PROB_COLS, *X.PRIOR_COLS]
        sc = Scorer(pd.concat([d[d["block"] == "eval"][cols], v1[cols]], ignore_index=True))
        pf, pb = picks(d, FUSION), picks(d, BASE_PICK)
        sc.add("fusion*", pf); sc.add("baseline*", pb)
        pick_rows.append({"h": h, **{f"fusion_{f}": a for f, a in pf.items()}, **{f"baseline_{f}": a for f, a in pb.items()}})
        for arm in ("fusion*", "baseline*", *ARMS, *BASE, "dl_price@v1", "fusion_concat@v1"):
            arms.append({"h": h, **sc.row(arm)})
        f = arms[-len(ARMS) - len(BASE) - 4]
        r0.append({"h": h, "arm": "fusion*", "macro_f1": f["macro_f1"], "f1_lo": f["f1_lo"], "auc_ovr": f["auc_ovr"], "auc_lo": f["auc_lo"],
                   "n": len(sc.y), "n_days": sc.n_days, "verdict": _vs_chance(f["f1_lo"], f["auc_lo"])})
        cmp_rows += [_diff(sc, h, *x) for x in (
            ("R1'", "fusion*", "baseline*"), ("R2'", "fusion*", "dl_price"),
            ("kèm: luật v3, dl_price", "dl_price", "dl_price@v1"), ("kèm: luật v3, fusion_concat", "fusion_concat", "fusion_concat@v1"),
            ("kèm: E1 chuỗi ngắn", "short_price", "dl_price"), ("kèm: E2 bỏ mẫu sát ngưỡng, dl_price", "dl_price_m", "dl_price"),
            ("kèm: E2 bỏ mẫu sát ngưỡng, fusion_concat", "fusion_concat_m", "fusion_concat"), ("kèm: nhánh tin", "fusion_concat", "no_text"))]
    md = lambda rows: pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")
    return (f"# R10: nhãn hướng 3 lớp theo luật v3, fold A B C (vùng phát triển)\n\nRun: {run_dir}\nMốc v1: {locked_dir}\n\n"
            f"## Arm được chọn (Macro-F1 trên tune + calib)\n\n{md(pick_rows)}\n\n## R0: fusion được chọn so với ngẫu nhiên\n\n{md(r0)}\n\n"
            f"## So sánh (a - b, dương là a tốt hơn)\n\n{md(cmp_rows)}\n\n## Mọi arm trên eval gộp ba fold (quyết định chính thức)\n\n{md(arms)}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args(argv)
    panel = build_panel(stock_dir=a.stock_dir)
    ds = X._dataset_id(panel)[:16]
    run_dir, locked = C.DATA / "exp/v3" / ds, C.DATA / "runs" / ds
    if a.score:
        out = score(run_dir, locked)
        (run_dir.parent / "r10.md").write_text(out)
        print(out)
        return
    for h in C.HORIZONS:
        for fold in folds_from_config():
            path = run_dir / f"h{h}_{fold.name}.parquet"
            if path.exists():
                continue
            kind = json.loads((locked / f"h{h}" / fold.name / "select.json").read_text())["price_kind"]
            d = run_fold(panel, h, fold, kind, a.device)
            X._atomic(path, lambda tmp: d.to_parquet(tmp, index=False))


if __name__ == "__main__":
    main()
