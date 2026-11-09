"""Ensemble over seeds, stacking, pre-registered comparisons (R1-R3), development report and lock file."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as C   # first: puts `src` on sys.path
from . import evaluate as E
from .experiment import PRIOR_COLS, _git_commit
from src.splits import folds_from_config

UNIMODAL = ("dl_price", "dl_fund", "dl_text")
FUSION = ("fusion_concat", "fusion_attention", "stack")
GROUPS = {"baseline": C.BASELINES, "unimodal": UNIMODAL, "fusion": FUSION}
ABLATIONS = ("no_price", "no_fund", "no_text")
KEYS = ["symbol", "as_of_session", "block"]
STACK_FOLDS = 5


def _h_dirs(run_dir):
    return sorted(int(p.name[1:]) for p in Path(run_dir).glob("h[0-9]*") if p.is_dir())


def load_h(run_dir, h):
    """Predictions of one horizon, probabilities averaged over seeds per (fold, arm, symbol, session, block)."""
    files = sorted(Path(run_dir).glob(f"h{h}/*/*/seed*.parquet"))
    if not files:
        raise ValueError(f"no predictions under {run_dir}/h{h}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in ("symbol", "block", "arm", "fold", "price_kind", "dataset_id", "nlp_version"):
        d[c] = d[c].astype(str)
    d["as_of_session"] = pd.to_datetime(d["as_of_session"])
    keys = ["fold", "arm", *KEYS]
    meta = [c for c in d.columns if c not in keys and c not in E.PROB_COLS and c not in ("alpha", "seed")]
    out = d.groupby(keys, sort=True).agg({**{c: "mean" for c in E.PROB_COLS}, **{c: "first" for c in meta}}).reset_index()
    out["y_true"] = out["y_true"].astype(int)
    out["seed"] = 0
    return _refit_alpha(out)


def _refit_alpha(d):
    """Alpha per (fold, arm), refit on the calib rows of the probabilities in `d`."""
    alpha = {}
    for (f, a), g in d[d["block"] == "calib"].groupby(["fold", "arm"]):
        alpha[(f, a)] = E.fit_alpha(g["y_true"].to_numpy(), g[E.PROB_COLS].to_numpy(), g[PRIOR_COLS].to_numpy())
    return d.assign(alpha=[alpha[k] for k in zip(d["fold"], d["arm"])])


def _fit_proba(x, y, x_new):
    """LogisticRegression(C=0.1) class probabilities with the 3 columns kept when a class is absent from y."""
    m = LogisticRegression(C=0.1, max_iter=1000).fit(x, y)
    out = np.zeros((len(x_new), 3))
    out[:, m.classes_] = m.predict_proba(x_new)
    return out


def stack(preds, fold):
    """Arm `stack` of one (h, fold): LR on the calib probabilities of dl_price/dl_fund/dl_text.

    Calib rows get 5-contiguous-block cross-fit probabilities (each block predicted by a model fit on the other
    four), eval rows a model fit on all calib, so selection on calib is not in-sample and eval labels are unused."""
    d = preds[preds["fold"] == fold]
    base = d[d["arm"] == UNIMODAL[0]].drop(columns=E.PROB_COLS)
    feats = []
    for arm in UNIMODAL:
        x = d[d["arm"] == arm][KEYS + E.PROB_COLS]
        base = base.merge(x.rename(columns={c: f"{arm}:{c}" for c in E.PROB_COLS}), on=KEYS, how="inner", validate="one_to_one")
        feats += [f"{arm}:{c}" for c in E.PROB_COLS]
    if len(base) != (d["arm"] == UNIMODAL[0]).sum():
        raise ValueError(f"fold {fold}: {UNIMODAL} do not share the same keys")
    base = base.sort_values(["block", "as_of_session", "symbol"]).reset_index(drop=True)
    x = base[feats].to_numpy(); y = base["y_true"].to_numpy()
    cal = np.flatnonzero((base["block"] == "calib").to_numpy()); ev = np.flatnonzero((base["block"] == "eval").to_numpy())
    p = np.zeros((len(base), 3))
    for part in np.array_split(cal, STACK_FOLDS):
        rest = np.setdiff1d(cal, part)
        p[part] = _fit_proba(x[rest], y[rest], x[part])
    p[ev] = _fit_proba(x[cal], y[cal], x[ev])
    out = base.drop(columns=feats).assign(arm="stack", seed=0, price_kind="none")
    out[E.PROB_COLS] = p
    return _refit_alpha(out)[list(preds.columns)]


def add_stack(preds):
    """`preds` plus the stack arm for every (h, fold) that has the three unimodal arms and no stack yet."""
    parts = [preds]
    for (h, f), g in preds.groupby(["h", "fold"]):
        have = set(g["arm"])
        if "stack" not in have and set(UNIMODAL) <= have:
            parts.append(stack(g, f))
    return pd.concat(parts, ignore_index=True)


def pick(preds, fold, candidates):
    """Candidate present in `preds` with the lowest mean calib log-loss in `fold` (ties: first listed)."""
    d = preds[(preds["fold"] == fold) & (preds["block"] == "calib")]
    ll = {a: E.row_logloss(g["y_true"].to_numpy(), g[E.PROB_COLS].to_numpy()).mean()
          for a in candidates if len(g := d[d["arm"] == a])}
    if not ll:
        raise ValueError(f"fold {fold}: none of {tuple(candidates)} in predictions")
    return min(ll, key=ll.get)


def _eval_ll(preds, arm, fold):
    d = preds[(preds["fold"] == fold) & (preds["arm"] == arm) & (preds["block"] == "eval")]
    return d.assign(ll=E.row_logloss(d["y_true"].to_numpy(), d[E.PROB_COLS].to_numpy()))[["symbol", "as_of_session", "ll"]]


def compare(preds, a_by_fold, b_by_fold):
    """Paired log-loss difference a - b over the eval blocks of the given folds (negative favours a)."""
    parts, wins = [], 0
    for f in a_by_fold:
        m = _eval_ll(preds, a_by_fold[f], f).merge(_eval_ll(preds, b_by_fold[f], f), on=["symbol", "as_of_session"], suffixes=("_a", "_b"))
        if m.empty:
            raise ValueError(f"fold {f}: no shared eval rows for {a_by_fold[f]} and {b_by_fold[f]}")
        wins += m["ll_a"].mean() < m["ll_b"].mean()
        parts.append(m)
    m = pd.concat(parts, ignore_index=True)
    mean, lo, hi = E.paired_diff(m["ll_a"], m["ll_b"], m["as_of_session"])
    return {"mean": mean, "lo": lo, "hi": hi, "fold_wins": int(wins), "n_folds": len(a_by_fold),
            "n": len(m), "n_days": int(m["as_of_session"].nunique()), "verdict": E.verdict(mean, hi, int(wins), len(a_by_fold))}


def _picks(preds, folds):
    """Per fold, the pick of each group that has a candidate in that fold (a group without one is left out)."""
    out = {}
    for f in folds:
        have = set(preds.loc[preds["fold"] == f, "arm"])
        out[f] = {g: pick(preds, f, c) for g, c in GROUPS.items() if have & set(c)}
    return out


def _fmt(by_fold):
    return " ".join(f"{f}={a}" for f, a in by_fold.items())


NOT_ENOUGH = "Chưa đủ fold để kết luận"


def _row(ph, h, a_by, b_by, all_folds):
    """compare() on the folds where both arms exist; `skipped` lists the others. Fewer folds than all_folds: no verdict."""
    ok = [f for f in a_by if f in b_by and all(len(ph[(ph["fold"] == f) & (ph["arm"] == x) & (ph["block"] == "eval")]) for x in (a_by[f], b_by[f]))]
    skipped = " ".join(f for f in all_folds if f not in ok)
    a, b = {f: a_by[f] for f in ok}, {f: b_by[f] for f in ok}
    r = compare(ph, a, b) if ok else {"n_folds": 0, "verdict": NOT_ENOUGH}
    if len(ok) < len(all_folds):
        r["verdict"] = NOT_ENOUGH
    return {"h": h, "a": _fmt(a), "b": _fmt(b), **r, "skipped": skipped}


def tables(preds):
    """arms / per_symbol / r1 / r2 / r3 / leak_alarms / picks as DataFrames, from the ensembled predictions.

    A comparison uses the folds where both sides exist; with fewer folds than the config has, its verdict is NOT_ENOUGH."""
    preds = add_stack(preds)
    all_folds = [f.name for f in folds_from_config()]
    ev = preds[preds["block"] == "eval"]
    arms, per_symbol = [], []
    for keys, g in ev.groupby(["h", "fold", "arm"]):
        arms.append({"h": keys[0], "fold": keys[1], "arm": keys[2], "alpha": g["alpha"].iloc[0], "n_days": g["as_of_session"].nunique(),
                     **E.scores(g["y_true"], g[E.PROB_COLS].to_numpy(), g["alpha"].iloc[0], g[PRIOR_COLS].to_numpy())})
        for s, gs in g.groupby("symbol"):
            per_symbol.append({"h": keys[0], "fold": keys[1], "arm": keys[2], "symbol": s, "alpha": gs["alpha"].iloc[0], "n_days": gs["as_of_session"].nunique(),
                               **E.scores(gs["y_true"], gs[E.PROB_COLS].to_numpy(), gs["alpha"].iloc[0], gs[PRIOR_COLS].to_numpy())})
    arms, per_symbol = pd.DataFrame(arms), pd.DataFrame(per_symbol)
    rows, picks = {"r1": [], "r2": [], "r3": []}, []
    for h, ph in preds.groupby("h"):
        folds = sorted(set(ph["fold"]))
        pk = _picks(ph, folds)
        picks += [{"h": h, "fold": f, **pk[f]} for f in folds]
        by = lambda g: {f: pk[f][g] for f in folds if g in pk[f]}
        for name, other in (("r1", "baseline"), ("r2", "unimodal")):
            rows[name].append(_row(ph, h, by("fusion"), by(other), all_folds))
        for ab in ABLATIONS:
            rows["r3"].append(_row(ph, h, dict.fromkeys(folds, "fusion_concat"), dict.fromkeys(folds, ab), all_folds))
    leak = arms[(arms["macro_f1"] >= C.LEAK_ALARM_F1) | (arms["auc_up_down"] >= C.LEAK_ALARM_AUC)]
    return {"arms": arms, "per_symbol": per_symbol, **{k: pd.DataFrame(v) for k, v in rows.items()},
            "leak_alarms": leak.reset_index(drop=True), "picks": pd.DataFrame(picks)}


def _all_preds(run_dir):
    hs = _h_dirs(run_dir)
    if not hs:
        raise ValueError(f"no h* directories under {run_dir}")
    return pd.concat([load_h(run_dir, h) for h in hs], ignore_index=True)


def _md(df, cols=None, floatfmt=".4f"):
    return "(không có)" if df.empty else (df if cols is None else df[cols]).to_markdown(index=False, floatfmt=floatfmt)


def write_report(run_dir, out=C.REPORTS / "mmf_dev_report.md"):
    """Markdown report (Vietnamese) of a run: all arms per (h, fold), R1-R3 with verdicts, per symbol, leak alarms."""
    preds = _all_preds(run_dir)
    t = tables(preds)
    ev = preds[preds["block"] == "eval"]
    metrics = ["n", "n_days", "alpha", "logloss", "brier", "accuracy", "macro_f1", "kappa", "auc_ovr", "auc_up_down",
               *[f"{k}_{n}" for k in ("precision", "recall") for n in E.NAMES]]
    cmp_cols = ["h", "a", "b", "mean", "lo", "hi", "fold_wins", "n_folds", "n", "n_days", "verdict", "skipped"]
    sym = t["per_symbol"].merge(t["picks"][["h", "fold", "fusion"]].rename(columns={"fusion": "arm"}).dropna(), on=["h", "fold", "arm"])
    lines = [
        "# Báo cáo phát triển: model đa phương thức T+2, T+5", "",
        "**Đây là vùng phát triển, chưa phải kết quả cuối.** Khối eval của fold A, B, C đã bị nhìn trong các thí nghiệm trước; "
        "vùng chấm cuối là các phiên sau 2026-10-30, chấm một lần cho mỗi h sau khi khóa cấu hình.", "",
        "Giới hạn dữ liệu (spec mục 6): 15 mã (6 mã ngân hàng thiếu biên gộp và thanh toán hiện hành, SSI thiếu BCTC, GAS có rất ít tin), "
        "tin chỉ từ 2021-10-21, BCTC theo năm không có ngày công bố nên năm Y coi là biết từ 01/04 năm Y+1, "
        "và AUC hướng giá trước đó chỉ khoảng 0.52 đến 0.54 nên kết quả khiêm tốn là kịch bản được kỳ vọng.", "",
        f"Dataset: {', '.join(sorted(set(ev['dataset_id'])))}. Phiên bản cảm xúc: {', '.join(sorted(set(ev['nlp_version'])))}. "
        f"Commit: {', '.join(sorted(set(ev['git_commit'].astype(str))))}. Xác suất là trung bình qua seed.", "",
        "Luật chọn, so sánh và diễn giải theo `docs/reports/mmf_preregistration.md`. Chọn theo log-loss trên calib; "
        "chênh lệch là log-loss(fusion) trừ log-loss(đối chứng), âm là fusion tốt hơn. "
        "Mỗi so sánh chỉ dùng các fold có đủ cả hai phía; cột `skipped` liệt kê fold bị bỏ, và thiếu fold thì kết luận là "
        f"\"{NOT_ENOUGH}\".", "",
        "## Arm được chọn theo (h, fold)", "", _md(t["picks"]), "",
        "## R1: fusion so với baseline tốt nhất", "", _md(t["r1"], cmp_cols), "",
        "## R2: fusion so với nhánh đơn tốt nhất", "", _md(t["r2"], cmp_cols), "",
        "## R3: đóng góp từng nhánh (fusion_concat so với arm bỏ nhánh)", "", _md(t["r3"], cmp_cols), "",
        "## Cảnh báo rò rỉ", "",
        f"Macro-F1 >= {C.LEAK_ALARM_F1} hoặc AUC Tăng so với Giảm >= {C.LEAK_ALARM_AUC} trên toàn mẫu eval của (h, fold, arm).", "",
        _md(t["leak_alarms"], ["h", "fold", "arm", "n", "macro_f1", "auc_up_down"]), "",
        "## Mọi arm theo (h, fold), khối eval, toàn mẫu", "", _md(t["arms"], ["h", "fold", "arm", *metrics]), "",
        "## Theo mã, fusion được chọn, toàn mẫu", "", _md(sym, ["h", "fold", "arm", "symbol", *metrics]), "",
    ]
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def _most_picked(picked, mean_ll):
    """Arm picked in most folds; ties go to the lowest mean calib log-loss."""
    n = Counter(picked.values())
    return min((a for a in n if n[a] == max(n.values())), key=mean_ll.get)


def write_lock(run_dir, out=C.REPORTS / "mmf_lock.json"):
    """Per h: the fusion, baseline and unimodal arms picked in most folds, plus per-fold picks. Needs every fold."""
    names = [f.name for f in folds_from_config()]
    if Path(out).exists():
        raise ValueError(f"{out} exists: the lock is written once, move it away to write again")
    if _h_dirs(run_dir) != sorted(C.HORIZONS):
        raise ValueError(f"lock needs horizons {sorted(C.HORIZONS)}, run has {_h_dirs(run_dir)}")
    need = [f"h{h}/{f}/{a}/seed{s}.parquet" for h in C.HORIZONS for f in names
            for a, seeds in [*((b, (0,)) for b in C.BASELINES), *((a, C.SEEDS) for a in C.ARMS)] for s in seeds]
    missing = [p for p in need if not (Path(run_dir) / p).exists()]
    if missing:
        raise ValueError(f"lock needs every fold, arm and seed: {len(missing)} files missing, e.g. {missing[:3]}")
    preds = _all_preds(run_dir)
    lock = {}
    for h, ph in preds.groupby("h"):
        ph = add_stack(ph)
        pk = _picks(ph, names)
        cal = ph[ph["block"] == "calib"]
        ll = {a: float(np.mean([E.row_logloss(g["y_true"].to_numpy(), g[E.PROB_COLS].to_numpy()).mean() for _, g in gg.groupby("fold")]))
              for a, gg in cal.groupby("arm")}
        kinds = Counter(ph[ph["arm"] == "dl_price"].groupby("fold")["price_kind"].first())
        ids = set(ph["dataset_id"])
        if len(ids) != 1:
            raise ValueError(f"h{h}: predictions from several datasets {sorted(ids)}")
        lock[str(h)] = {**{g: _most_picked({f: pk[f][g] for f in names}, ll) for g in GROUPS},
                        "price_kind": min(kinds, key=lambda k: (-kinds[k], C.PRICE_KINDS.index(k))), "dataset_id": ids.pop(),
                        "git_commit": _git_commit(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "picks": pk}
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=C.REPORTS / "mmf_dev_report.md", help="report path")
    ap.add_argument("--lock", action="store_true", help="also write the lock file (needs every fold of every h)")
    a = ap.parse_args(argv)
    print(write_report(a.run_dir, a.out))
    if a.lock:
        print(write_lock(a.run_dir))


if __name__ == "__main__":
    main()
