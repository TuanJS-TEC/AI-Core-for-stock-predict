"""Resumable runner over arms, folds and seeds: calibrated probabilities per (h, fold, arm, seed) as parquet."""
import argparse
import functools
import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C   # first: puts `src` on sys.path
from . import baselines as B, evaluate as E, train as TR
from .frame import GROUPS, build_frame
from .models import FusionNet
from .windows import Scalers, make_samples
from src.data import build_panel
from src.splits import Fold, folds_from_config, split_samples

OUT_BLOCKS = ("calib", "eval")
PRIOR_COLS = ["prior_down", "prior_flat", "prior_up"]


@dataclass
class Bundle:
    fold: Fold
    h: int
    nlp_version: str
    dataset_id: str | None
    frame: pd.DataFrame
    scaled: pd.DataFrame
    blocks: dict             # block -> usable rows kept by the windows, in Samples order
    samples: dict            # block -> Samples
    tickers: list
    prior: dict              # symbol -> class frequency on train (3,)


def _class_prior(train, tickers):
    """Per-symbol train class frequency, Laplace-smoothed like prior_fixed; a symbol without rows gets uniform."""
    return {s: B._laplace(train.loc[train["symbol"].astype(str) == s, "label"].dropna()) for s in tickers}


def bundle_from_frame(frame, h, fold, nlp_version, dataset_id, tickers):
    """Split usable rows, fit the scaler on train only, then restrict every block to the rows with full windows."""
    blocks = split_samples(frame[frame["usable"]], fold)
    scaled = Scalers.fit(blocks["train"]).apply(frame)
    samples = {k: make_samples(scaled, rows, tickers) for k, rows in blocks.items()}
    blocks = {k: rows.loc[samples[k].row_index] for k, rows in blocks.items()}
    return Bundle(fold, h, nlp_version, dataset_id, frame, scaled, blocks, samples, tickers, _class_prior(blocks["train"], tickers))


def fold_bundle(panel, h, fold):
    """Bundle of one (h, fold) from a price panel, with the fold's sentiment model."""
    nlp_version = C.FOLD_NLP[fold.name]
    return bundle_from_frame(build_frame(panel, h, nlp_version), h, fold, nlp_version, panel.dataset_id, sorted(panel.ohlcv))


def shuffle_train_labels(b, seed=0):
    """Copy of the bundle with train labels moved between dates, one permutation for all symbols; the rest is shared.

    A date the symbol has no row on gives -1 (dropped by fit_net, NaN for the baselines)."""
    tr = b.samples["train"]
    dates, pos = np.unique(tr.as_of, return_inverse=True)
    grid = np.full((len(b.tickers), len(dates)), -1, dtype=np.int64)
    grid[tr.ticker, pos] = tr.y
    y = grid[tr.ticker, np.random.default_rng(seed).permutation(len(dates))[pos]]
    train = b.blocks["train"].assign(label=np.where(y >= 0, y, np.nan))
    # ponytail: frame keeps true labels, only prior_adaptive reads them there and only once matured
    return replace(b, blocks={**b.blocks, "train": train}, samples={**b.samples, "train": replace(tr, y=y)},
                   prior=_class_prior(train, b.tickers))


def _atomic(path, write):
    """write(tmp) then os.replace, so a crash never leaves a partial file at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


@functools.lru_cache
def _git_commit():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=C.AC.ROOT, text=True).strip()


def _prior_rows(b, block):
    return np.array([b.prior[s] for s in b.samples[block].symbol])


def _write_preds(path, b, arm, seed, price_kind, probs, alpha):
    """Parquet of calib + eval rows in Samples order, metadata as plain columns."""
    parts = []
    for k in OUT_BLOCKS:
        s = b.samples[k]
        d = pd.DataFrame({"symbol": s.symbol, "as_of_session": s.as_of, "label_end_date": s.label_end, "y_true": s.y, "block": k})
        d[E.PROB_COLS] = probs[k]
        d[PRIOR_COLS] = _prior_rows(b, k)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True).assign(
        alpha=alpha, arm=arm, h=b.h, fold=b.fold.name, seed=seed, price_kind=price_kind,
        nlp_version=b.nlp_version, dataset_id=b.dataset_id, git_commit=_git_commit())
    _atomic(path, lambda tmp: out.to_parquet(tmp, index=False))


def _run_meta(b):
    return {"dataset_id": b.dataset_id, "nlp_version": b.nlp_version, "h": b.h, "fold": b.fold.name}


def _require_same(path, got, want):
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if bad:
        raise ValueError(f"{path}: existing file is from another run, (found, expected): {bad}")


def _reusable(path, b, price_kind):
    """True when `path` holds this run's output. A corrupt file or one from another run raises, never gets reused."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        d = pd.read_parquet(path, columns=["symbol", "as_of_session", "block", "price_kind", "git_commit", *_run_meta(b)])
    except Exception as e:
        raise ValueError(f"{path}: unreadable: {e!r}") from e
    want = {**_run_meta(b), "price_kind": price_kind}
    _require_same(path, {c: sorted(set(d[c].astype(str))) for c in want}, {c: [str(v)] for c, v in want.items()})
    keys = set(zip(d["symbol"].astype(str), pd.to_datetime(d["as_of_session"]), d["block"].astype(str)))
    expected = {(sym, day, k) for k in OUT_BLOCKS for sym, day in zip(b.samples[k].symbol, pd.to_datetime(b.samples[k].as_of))}
    if keys != expected:
        raise ValueError(f"{path}: existing file is from another run: keys differ, {len(keys ^ expected)} not shared")
    commits = sorted(set(d["git_commit"].astype(str)))
    if commits != [_git_commit()]:
        print(f"warning: {path} written by commit {commits}, now at {_git_commit()}", flush=True)
    return True


def _net_maker(b, arm, price_kind):
    branches, fusion = C.ARMS[arm]
    dims = [len(GROUPS[g]) for g in ("price", "text", "fund")]
    return lambda: FusionNet(branches, fusion, len(b.tickers), *dims, price_kind)


def calibrate(b, z_calib, z_eval=None):
    """(temperature, alpha), both fit on calib. z_eval is never read: eval takes no part in the fit."""
    y = b.samples["calib"].y
    t = E.fit_temperature(z_calib, y)
    return t, E.fit_alpha(y, E.apply_temperature(z_calib, t), _prior_rows(b, "calib"))


def run_baseline(b, name, out_path):
    """Baseline probabilities written unchanged; only alpha is fit (on calib). Existing file: checked, then skipped."""
    if _reusable(out_path, b, "none"):
        return
    start = time.time()
    p = B.predict_baseline(name, b.blocks, b.scaled, b.frame)
    alpha = E.fit_alpha(b.samples["calib"].y, p["calib"], _prior_rows(b, "calib"))
    _write_preds(out_path, b, name, 0, "none", p, alpha)
    print(f"h{b.h} {b.fold.name} {name} alpha={alpha} {time.time() - start:.0f}s", flush=True)


def run_arm(b, arm, seed, price_kind, out_path, device, max_epochs=C.MAX_EPOCHS):
    """Train one seed, write weights then calibrated probabilities. Existing parquet: checked, then skipped."""
    out_path = Path(out_path)
    kind = price_kind if "price" in C.ARMS[arm][0] else "none"
    if _reusable(out_path, b, kind):
        return
    start = time.time()
    fit = TR.fit_net(_net_maker(b, arm, price_kind), b.samples["train"], b.samples["tune"], seed, device, max_epochs)
    z = {k: TR.predict_logits(fit["net"], b.samples[k], device) for k in OUT_BLOCKS}
    t, alpha = calibrate(b, z["calib"], z["eval"])
    _atomic(out_path.with_suffix(".pt"), lambda tmp: torch.save(fit["net"].state_dict(), tmp))
    _write_preds(out_path, b, arm, seed, kind, {k: E.apply_temperature(z[k], t) for k in OUT_BLOCKS}, alpha)
    print(f"h{b.h} {b.fold.name} {arm} seed{seed} {kind} epoch={fit['best_epoch']} tune_nll={fit['tune_nll']:.4f} "
          f"t={t:.3f} alpha={alpha} {time.time() - start:.0f}s", flush=True)


def select_price_kind(b, run_dir, device, max_epochs=C.MAX_EPOCHS):
    """Encoder with the lowest mean tune NLL of dl_price over SELECT_SEEDS (tie: first listed), kept in select.json and reused only by the same run settings."""
    path = Path(run_dir) / f"h{b.h}" / b.fold.name / "select.json"
    meta = {**_run_meta(b), "max_epochs": max_epochs}
    if path.exists():
        saved = json.loads(path.read_text())
        _require_same(path, saved, meta)
        return saved["price_kind"]
    nll = {k: float(np.mean([TR.fit_net(_net_maker(b, "dl_price", k), b.samples["train"], b.samples["tune"], s, device, max_epochs)["tune_nll"]
                             for s in C.SELECT_SEEDS])) for k in C.PRICE_KINDS}
    kind = min(C.PRICE_KINDS, key=nll.get)
    _atomic(path, lambda tmp: tmp.write_text(json.dumps({"price_kind": kind, "tune_nll": nll, "seeds": C.SELECT_SEEDS, **meta})))
    print(f"h{b.h} {b.fold.name} select {kind} {nll}", flush=True)
    return kind


def _dataset_id(panel):
    if panel.dataset_id is None:
        raise ValueError("panel has no dataset_id: run on a stock dir with a manifest (a frozen snapshot)")
    return panel.dataset_id


def run_all(panel, horizons, folds, run_dir, device, arms=None, seeds=C.SEEDS, build=fold_bundle, max_epochs=C.MAX_EPOCHS):
    """Per (h, fold): every baseline, then the price encoder selection, then each arm x seed. `arms` narrows DL arms only."""
    if build is fold_bundle:
        _dataset_id(panel)
    for h in horizons:
        for fold in folds:
            b = build(panel, h, fold)
            base = Path(run_dir) / f"h{h}" / fold.name
            for name in C.BASELINES:
                run_baseline(b, name, base / name / "seed0.parquet")
            kind = select_price_kind(b, run_dir, device, max_epochs)
            for arm in (C.ARMS if arms is None else arms):
                for seed in seeds:
                    run_arm(b, arm, seed, kind, base / arm / f"seed{seed}.parquet", device, max_epochs)


def main(argv=None):
    folds = {f.name: f for f in folds_from_config()}
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=int, nargs="+", default=list(C.HORIZONS))
    ap.add_argument("--folds", nargs="+", choices=list(folds), default=list(folds))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stock-dir")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="1 epoch, 1 seed, first fold, under C.DATA/smoke")
    mode.add_argument("--shuffle-labels", action="store_true", help="leak check: train labels shuffled, fold C, 1 seed, under C.DATA/shuffle")
    a = ap.parse_args(argv)
    kind, names, kw = "runs", a.folds, {}
    if a.smoke:
        kind, names, kw = "smoke", a.folds[:1], {"max_epochs": 1, "seeds": (59,)}
    elif a.shuffle_labels:
        kind, names, kw = "shuffle", ["C"], {"seeds": (59,), "build": lambda p, h, f: shuffle_train_labels(fold_bundle(p, h, f))}
    panel = build_panel(stock_dir=a.stock_dir)
    run_all(panel, a.h, [folds[n] for n in names], C.DATA / kind / _dataset_id(panel)[:16], a.device, **kw)


if __name__ == "__main__":
    main()
