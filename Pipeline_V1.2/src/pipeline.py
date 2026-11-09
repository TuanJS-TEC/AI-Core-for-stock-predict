"""Một lệnh cho toàn bộ project: huấn luyện gộp 15 mã với embedding mã, K_THETA, rồi đánh giá test.

  python -m src.pipeline            lưới đầy đủ T x L x mô hình x wd x seed = 2 x 2 x 3 x 2 x 5 = 120 lượt
  python -m src.pipeline --quick    chạy thử nhanh với lưới C.QUICK_GRID (4 lượt, khoảng 1 phút)
  python -m src.pipeline --force    bỏ kết quả cũ, chạy lại từ đầu

Pipeline chỉ đọc dữ liệu đang có trong stock/. Cập nhật dữ liệu (ETL) là bước riêng, chạy trước khi cần:
  python sync_stock_parquet.py

Mỗi lượt là một mô hình cho cả 15 mã, trên CPU, xác định. Adam, MSE (đầu hồi quy) + CE không trọng số (đầu phân loại);
hai đầu không chia sẻ tham số nên early stopping riêng (MSE val gộp và NLL val gộp). T, alpha, lambda fit trên val
riêng từng mã, test chỉ để ghi nhận. Cấu hình (model, wd) cho từng (T, L) chọn bằng Macro-F1 val trung bình theo mã.

Ghi vào outputs/results/{C.RUN_NAME}/: experiment.json, runs.csv, preds_all.csv, states.pt, val_summary.csv,
final_config.csv, test_predictions.csv, train_class_freq.csv, final_table.csv, per_ticker.csv, confusion_*.
Hình ở outputs/figures/{C.RUN_NAME}/. Cuối cùng cập nhật kho model outputs/models/ (predict.update_models): mỗi
(mã, T, L) giữ model có Macro-F1 val của mã cao nhất giữa model đang lưu và các lượt vừa chạy; dự đoán bằng
`python -m src.predict`. Sau đó vẽ đồ thị dự đoán/thực tế (val và test) kèm MSE/MAE của mọi model trong kho
(report.plot_models) vào outputs/figures/models/ và outputs/results/models_regression.csv.
Chạy lại cùng lệnh sẽ tiếp tục từ các lượt chưa xong (kết quả được ghi sau mỗi (T, L)); khi dữ liệu đổi (số dòng,
ngày cuối) hoặc cấu hình đổi thì chạy lại từ đầu, vì mốc train/val/test = split_sizes(số dòng hiện tại).
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config as C, metrics as M, report
from .data import PooledData, PooledSplit, build_panel, build_pooled, make_loader, split_sizes
from .models import PooledMultiTask, arch_spec
from .predict import update_models

torch.use_deterministic_algorithms(C.DETERMINISTIC)

RUN_KEYS = ["T", "L", "model", "wd", "seed"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def forward_np(model: nn.Module, split: PooledSplit) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    reg, logits = model(torch.from_numpy(split.x), torch.from_numpy(split.ticker).long())
    return reg.numpy(), logits.numpy()


def fit(pdata: PooledData, name: str, seed: int, wd: float) -> nn.Module:
    set_seed(seed)
    model = PooledMultiTask(name, pdata.L, pdata.T, pdata.n_feat, len(pdata.tickers))
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=wd)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()      # không trọng số lớp
    loader = make_loader(pdata.train, shuffle=True, seed=seed)

    best_mse, best_reg, bad_reg = float("inf"), None, 0
    best_nll, best_cls, bad_cls = float("inf"), None, 0
    for _ in range(C.MAX_EPOCHS):
        model.train()
        for xb, yb, lb, tb in loader:
            opt.zero_grad()
            reg, logits = model(xb, tb)
            (mse(reg, yb) + ce(logits, lb)).backward()
            opt.step()
        reg_v, logits_v = forward_np(model, pdata.val)
        val_mse = float(np.mean((reg_v - pdata.val.y) ** 2))
        val_nll = M.nll(logits_v, pdata.val.labels)
        if val_mse < best_mse - C.MIN_DELTA:
            best_mse, best_reg, bad_reg = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad_reg += 1
        if val_nll < best_nll - C.MIN_DELTA:
            best_nll, best_cls, bad_cls = val_nll, copy.deepcopy(model.state_dict()), 0
        else:
            bad_cls += 1
        if bad_reg >= C.PATIENCE and bad_cls >= C.PATIENCE:
            break
    state = {k: v for k, v in best_reg.items() if k.startswith("reg.")}
    state.update({k: v for k, v in best_cls.items() if k.startswith("cls.")})
    model.load_state_dict(state)
    return model


def evaluate_run(pdata: PooledData, model: nn.Module, meta: dict) -> tuple[list[dict], pd.DataFrame]:
    """Một dòng chỉ số cho mỗi mã và bảng xác suất test của lượt này. T, alpha, lambda fit trên val riêng từng mã."""
    _, z_val = forward_np(model, pdata.val)
    reg_t, z_test = forward_np(model, pdata.test)
    pt, st = np.zeros((len(z_test), C.N_CLASSES)), np.zeros((len(z_test), C.N_CLASSES))
    rows = []
    for k, t in enumerate(pdata.tickers):
        mv, mt = pdata.val.of(k), pdata.test.of(k)
        yv, yt = pdata.val.labels[mv], pdata.test.labels[mt]
        row, pt[mt], st[mt] = M.ticker_metrics(z_val[mv], yv, z_test[mt], yt, pdata.prior[t])
        rows.append({"ticker": t, **meta, "theta": pdata.theta[t], "n_train": int(pdata.train.of(k).sum()),
                     "n_val": len(yv), "n_test": len(yt), **row,
                     **M.regression_scaled(reg_t[mt], pdata.test.y[mt])})   # thang chuẩn hóa của log close
    preds = pd.DataFrame({"ticker": np.asarray(pdata.tickers)[pdata.test.ticker], "T": pdata.T, "L": pdata.L,
                          "model": meta["model"], "wd": meta["wd"], "seed": meta["seed"],
                          "date": pdata.test.date, "y_true": pdata.test.labels,
                          **dict(zip(C.PROB_COLS, pt.T)), **dict(zip(C.SCORE_COLS, st.T)),
                          "y_pred": st.argmax(axis=1)})
    return rows, preds


class Store:
    """Kết quả của lần chạy, ghi lại sau mỗi (T, L) để có thể ngắt và chạy tiếp.

    Nếu experiment.json khác cấu hình hiện tại (K_THETA, kiến trúc, hoặc dữ liệu: số dòng, ngày cuối) thì kết quả
    cũ không còn đúng và chạy lại từ đầu.
    """

    def __init__(self, out_dir: Path, spec: dict, force: bool, log=print):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.dir / C.EXPERIMENT_FILE
        stale = False
        if spec_path.exists() and not force:
            old = json.loads(spec_path.read_text(encoding="utf-8"))
            stale = old != spec
            if stale:
                diff = {k: (old.get(k), v) for k, v in spec.items() if old.get(k) != v}
                log(f"{out_dir.name}: cấu hình/dữ liệu đã đổi {diff}; chạy lại từ đầu")
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        runs_p, preds_p, states_p = (self.dir / f for f in (C.RUNS_FILE, C.PREDS_FILE, C.STATES_FILE))
        fresh = force or stale or not runs_p.exists()
        self.runs = [] if fresh else pd.read_csv(runs_p).to_dict("records")
        self.preds = [] if fresh else [pd.read_csv(preds_p, parse_dates=["date"])]
        self.states = {} if fresh or not states_p.exists() else torch.load(states_p, weights_only=True)

    def done(self) -> set:
        return {tuple(r[k] for k in RUN_KEYS) for r in self.runs}

    def save(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        runs = pd.DataFrame(self.runs)
        preds = pd.concat(self.preds, ignore_index=True)
        runs.to_csv(self.dir / C.RUNS_FILE, index=False)
        preds.to_csv(self.dir / C.PREDS_FILE, index=False)
        torch.save(self.states, self.dir / C.STATES_FILE)
        return runs, preds


def run_experiment(out_dir: Path, horizons=C.HORIZONS, lookbacks=C.LOOKBACKS, models=C.MODELS,
                   wds=C.WEIGHT_DECAYS, seeds=C.SEEDS, tickers=C.TICKERS, panel=None, force: bool = False, log=print):
    panel = build_panel(tickers) if panel is None else panel
    spec = {**arch_spec(), "k_theta": C.K_THETA, "tickers": list(tickers),
            "n_rows": len(panel.dates), "last_date": str(panel.dates[-1].date()), "dataset_id": panel.dataset_id}
    store = Store(Path(out_dir), spec, force, log)
    done = store.done()
    todo = [(T, L, m, wd, s) for T in horizons for L in lookbacks for m in models for wd in wds for s in seeds]
    left = [k for k in todo if (k[0], k[1], k[2], float(k[3]), k[4]) not in done]
    log(f"{Path(out_dir).name}: còn {len(left)}/{len(todo)} lượt")
    t0, n_done = time.time(), 0
    for T in horizons:
        for L in lookbacks:
            keys = [k for k in left if k[:2] == (T, L)]
            if not keys:
                continue
            pdata = build_pooled(L, T, tickers, panel)
            log(f"  T={T} L={L}: train {len(pdata.train.labels)} / val {len(pdata.val.labels)} / "
                f"test {len(pdata.test.labels)} mẫu")
            for _, _, name, wd, seed in keys:
                model = fit(pdata, name, seed, wd)
                meta = {"T": T, "L": L, "model": name, "wd": float(wd), "seed": seed}
                rows, preds = evaluate_run(pdata, model, meta)
                store.runs.extend(rows)
                store.preds.append(preds)
                store.states[(T, L, name, float(wd), seed)] = copy.deepcopy(model.state_dict())
                n_done += 1
                el = time.time() - t0
                r = pd.DataFrame(rows)
                log(f"    {name:7s} wd={wd:g} seed={seed}: val F1 {r['val_f1_macro'].mean():.4f} "
                    f"kappa {r['val_kappa'].mean():.4f} | {el / n_done:.1f}s/lượt, "
                    f"còn ~{(len(left) - n_done) * el / n_done / 60:.0f} phút")
            store.save()
    runs, preds = store.save()
    return runs, preds, panel


def val_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Chỉ số val theo (T, L, model, wd): trung bình qua mã cho từng seed, rồi mean và std qua seed."""
    cols = ["val_f1_macro", "val_kappa", "val_f1_flat", "val_brier_skill", "val_nll_raw"]
    per_seed = runs.groupby(RUN_KEYS)[cols].mean().reset_index()
    g = per_seed.groupby(["T", "L", "model", "wd"])[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=C.STD_DDOF).add_suffix("_std")).reset_index()
    return out[["T", "L", "model", "wd"] + [f"{c}_{s}" for c in cols for s in ("mean", "std")]]


def select_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """(model, wd) có Macro-F1 val trung bình theo mã cao nhất cho từng (T, L). Chỉ dùng val."""
    out = []
    for (T, L), s in summary.groupby(["T", "L"]):
        best = s.sort_values(["model", "wd"]).loc[s["val_f1_macro_mean"].idxmax()]
        out.append({"T": T, "L": L, "model": best["model"], "wd": float(best["wd"]),
                    "val_f1_macro_mean_over_tickers": best["val_f1_macro_mean"],
                    "val_kappa_mean": best["val_kappa_mean"], "val_kappa_std": best["val_kappa_std"],
                    "val_brier_skill_mean": best["val_brier_skill_mean"]})
    return pd.DataFrame(out)


def finalize(out_dir: Path, fig_dir: Path, runs: pd.DataFrame, preds: pd.DataFrame, tickers, panel) -> pd.DataFrame:
    """Ghi val_summary / final_config / test_predictions / train_class_freq, rồi báo cáo test trên thư mục này."""
    summary = val_summary(runs)
    summary.to_csv(out_dir / C.VAL_SUMMARY_FILE, index=False)
    cfg = select_configs(summary)
    cfg.to_csv(out_dir / C.FINAL_CONFIG_FILE, index=False)
    parts, freqs = [], []
    for c in cfg.itertuples():
        k = (preds["T"] == c.T) & (preds["L"] == c.L) & (preds["model"] == c.model) & (preds["wd"] == c.wd)
        parts.append(preds[k])
        pdata = build_pooled(c.L, c.T, tickers, panel)
        freqs += [{"ticker": t, "T": c.T, "L": c.L, **dict(zip(C.PROB_COLS, pdata.prior[t]))} for t in tickers]
    pd.concat(parts, ignore_index=True).to_csv(out_dir / C.TEST_PREDICTIONS_FILE, index=False)
    pd.DataFrame(freqs).to_csv(out_dir / C.CLASS_FREQ_FILE, index=False)
    pd.set_option("display.width", C.DISPLAY_WIDTH)
    print("Cấu hình chọn bằng val:")
    print(cfg.round(C.DISPLAY_DECIMALS).to_string(index=False))
    print("Đánh giá test (chỉ ghi nhận):")
    return report.main(out_dir, fig_dir)


def check_data(panel) -> str:
    """Mọi mã cùng dãy ngày, mỗi phiên một dòng (build_panel đã kiểm tra). Dữ liệu do sync_stock_parquet.py chuẩn bị."""
    d = panel.dates
    return f"{len(d)} phiên ({d[0].date()} đến {d[-1].date()}), mốc chia {split_sizes(len(d))}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện gộp 15 mã với embedding mã và đánh giá test",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--quick", action="store_true", help="chạy thử nhanh với lưới nhỏ C.QUICK_GRID")
    ap.add_argument("--force", action="store_true", help="bỏ kết quả cũ, chạy lại từ đầu")
    a = ap.parse_args()

    start = time.time()
    panel = build_panel(C.TICKERS)
    print(f"Dữ liệu: {check_data(panel)}; K_THETA = {C.K_THETA}", flush=True)
    name = C.QUICK_NAME if a.quick else C.RUN_NAME
    out_dir, fig_dir = C.RESULTS_DIR / name, C.FIGURES_DIR / name
    runs, preds, panel = run_experiment(out_dir, panel=panel, force=a.force, **(C.QUICK_GRID if a.quick else {}))
    finalize(out_dir, fig_dir, runs, preds, C.TICKERS, panel)
    update_models(out_dir, panel)
    report.plot_models(panel)
    print(f"Kết quả: {out_dir}\nHình: {fig_dir}\nTổng thời gian: {(time.time() - start) / 60:.1f} phút", flush=True)


if __name__ == "__main__":
    main()
