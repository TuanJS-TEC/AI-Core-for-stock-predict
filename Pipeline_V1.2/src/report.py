"""Báo cáo đánh giá test từ test_predictions.csv: Macro-F1, Brier, hit rate, kappa, baseline hoán vị,
khoảng tin cậy block bootstrap, F1 theo từng mã và confusion matrix.

Metric tính theo (mã, seed), trung bình qua các mã cho từng seed, rồi mean và std qua 5 seed.
Hàng ensemble là trung bình xác suất 5 seed (một giá trị, không có std).
Baseline:
  repeat       lặp giá trị cuối của cửa sổ: r = 0 nên luôn đoán Đi ngang
  class_freq   xác suất không đổi bằng tần suất lớp ở train
  f1_perm_*    (cột) Macro-F1 khi hoán vị nhãn dự đoán của chính mô hình: cùng tỷ lệ nhãn, không có tín hiệu
Khoảng tin cậy: block bootstrap theo ngày (khối C.BOOTSTRAP_BLOCK phiên), mọi mã dùng chung các ngày được rút,
nên giữ được tương quan giữa các mã cùng ngày.
Confusion matrix cộng qua các mã, lấy trung bình qua seed. Nhãn tính từ log-return thật nên
không phụ thuộc thang giá.

plot_models (pipeline gọi sau khi cập nhật kho model, hoặc chạy riêng `python -m src.report`): với mỗi model trong
outputs/models/, vẽ đường dự đoán bước T của đầu hồi quy và đường giá thực tế trên val và test (không vẽ train),
kèm MSE/MAE của model và baseline repeat; ghi bảng C.MODEL_REGRESSION_FILE.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C, metrics as M, predict as PR
from .data import Panel, PooledData, build_panel, build_pooled, raw_windows

PROB = list(C.PROB_COLS)
SCORE = list(C.SCORE_COLS)
METRIC_COLS = ["f1_macro", "brier", "hit_rate", "directional_share", "kappa", "f1_perm"]


def date_range(preds: pd.DataFrame, T: int) -> tuple[str, str, int]:
    """Ngày đầu, ngày cuối và số mẫu mỗi mã của tập test ở horizon T. Mọi mã phải trùng nhau."""
    g = preds[preds["T"] == T]
    per = g.groupby("ticker")["date"].agg(["min", "max", "nunique"])
    if len(per.drop_duplicates()) != 1:
        raise AssertionError(f"Lịch test khác nhau giữa các mã ở T={T}")
    r = per.iloc[0]
    return str(pd.Timestamp(r["min"]).date()), str(pd.Timestamp(r["max"]).date()), int(r["nunique"])


def ensemble(g: pd.DataFrame) -> pd.DataFrame:
    """Trung bình xác suất và điểm quyết định qua các seed theo từng (mã, ngày). Cấu hình đã chọn bằng val,
    nên không dùng test để chọn gì. Trả về cùng định dạng với preds, seed = -1; nhãn = argmax điểm quyết định."""
    keys = ["ticker", "T", "L", "date"]
    e = g.groupby(keys, as_index=False).agg({"y_true": "first", **{c: "mean" for c in PROB + SCORE}})
    e["seed"] = -1
    e["y_pred"] = e[SCORE].to_numpy().argmax(axis=1)
    return e


def method_outputs(d: pd.DataFrame, method: str, freq: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Xác suất (N, N_CLASSES) và nhãn dự đoán (N,) của một (mã, seed) theo phương pháp."""
    n = len(d)
    if method == "model":
        return d[PROB].to_numpy(), d["y_pred"].to_numpy()
    if method == "repeat":      # lặp giá trị cuối: r = 0 nên luôn là Đi ngang
        pred = np.full(n, C.CLASS_FLAT, dtype=np.int64)
        return np.eye(C.N_CLASSES)[pred], pred
    t = d["ticker"].iloc[0]     # class_freq: xác suất không đổi bằng tần suất lớp ở train
    f = freq[freq["ticker"] == t][PROB].iloc[0].to_numpy()
    return np.tile(f, (n, 1)), np.full(n, int(f.argmax()))


def per_ticker_seed(g: pd.DataFrame, method: str, freq: pd.DataFrame | None = None) -> pd.DataFrame:
    """Macro-F1, Brier, hit rate, tỷ lệ dự đoán Giảm/Tăng, kappa và F1 hoán vị cho từng (mã, seed)."""
    rng = np.random.default_rng(C.SEED)
    rows = []
    for (t, seed), d in g.groupby(["ticker", "seed"]):
        y = d["y_true"].to_numpy()
        probs, pred = method_outputs(d, method, freq)
        cm = M.confusion(y, pred)
        rows.append({"ticker": t, "seed": seed, "f1_macro": M.f1_per_class(cm).mean(),
                     "brier": M.brier_score(probs, y),
                     "hit_rate": M.hit_rate(cm) if method == "model" else np.nan,
                     "directional_share": float(np.mean(pred != C.CLASS_FLAT)),
                     "kappa": M.cohen_kappa(cm),
                     "f1_perm": M.permutation_f1(y, pred, C.PERM_N, rng) if method == "model" else np.nan})
    return pd.DataFrame(rows)


def summarize(ps: pd.DataFrame) -> dict:
    """Trung bình qua mã theo từng seed, rồi mean và std (ddof = C.STD_DDOF) qua seed."""
    per_seed = ps.groupby("seed")[METRIC_COLS].mean()
    out = {}
    for col in per_seed:
        out[f"{col}_mean"] = float(per_seed[col].mean()) if per_seed[col].notna().any() else np.nan
        out[f"{col}_std"] = float(per_seed[col].std(ddof=C.STD_DDOF)) if per_seed[col].notna().any() else np.nan
    out["n_seeds"] = int(len(per_seed))
    out["n_hit_rate_nan"] = int(ps["hit_rate"].isna().sum()) if ps["hit_rate"].notna().any() else 0
    return out


def _batch_scores(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 và kappa của từng hàng. y, pred: (G, N) nhãn. Trả về hai mảng (G,)."""
    G, k = y.shape[0], C.N_CLASSES
    cm = np.zeros((G, k, k))
    np.add.at(cm, (np.repeat(np.arange(G), y.shape[1]), y.ravel(), pred.ravel()), 1)
    tp = np.einsum("gii->gi", cm)
    col, row, n = cm.sum(axis=1), cm.sum(axis=2), cm.sum(axis=(1, 2))
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    s = prec + rec
    f1 = np.divide(2 * prec * rec, s, out=np.zeros_like(tp), where=s > 0).mean(axis=1)
    po = tp.sum(axis=1) / n
    pe = (row * col).sum(axis=1) / n ** 2
    kappa = np.divide(po - pe, 1 - pe, out=np.zeros_like(po), where=pe < 1)
    return f1, kappa


def bootstrap_ci(g: pd.DataFrame) -> dict:
    """Khoảng tin cậy percentile của Macro-F1 và kappa (trung bình qua mọi (mã, seed)) bằng block bootstrap
    theo ngày. Mọi (mã, seed) dùng cùng một bộ ngày được rút."""
    wide_y = g.pivot_table(index=["ticker", "seed"], columns="date", values="y_true")
    wide_p = g.pivot_table(index=["ticker", "seed"], columns="date", values="y_pred")
    if wide_y.isna().any().any():
        raise AssertionError("Các (mã, seed) không có cùng dãy ngày test")
    y, p = wide_y.to_numpy().astype(np.int64), wide_p.to_numpy().astype(np.int64)
    rng = np.random.default_rng(C.SEED)
    f1s, kappas = [], []
    for idx in M.block_bootstrap_dates(wide_y.columns.to_numpy(), C.BOOTSTRAP_BLOCK, C.BOOTSTRAP_N, rng):
        f1, kappa = _batch_scores(y[:, idx], p[:, idx])
        f1s.append(f1.mean())
        kappas.append(kappa.mean())
    a = (1 - C.CI_LEVEL) / 2
    return {"f1_ci_lo": float(np.quantile(f1s, a)), "f1_ci_hi": float(np.quantile(f1s, 1 - a)),
            "kappa_ci_lo": float(np.quantile(kappas, a)), "kappa_ci_hi": float(np.quantile(kappas, 1 - a))}


def add_skill(table: pd.DataFrame) -> pd.DataFrame:
    """So với baseline tần suất lớp train của cùng (T, L): Brier skill score = 1 - Brier / Brier_baseline
    (dương là tốt hơn baseline) và chênh Macro-F1 (f1 - f1_baseline).
    f1_minus_perm: chênh Macro-F1 so với hoán vị nhãn dự đoán, tức phần F1 đến từ tín hiệu thật."""
    base = table[table["method"] == "class_freq"].set_index(["T", "L"])
    key = list(zip(table["T"], table["L"]))
    table["brier_skill"] = 1.0 - table["brier_mean"].to_numpy() / base.loc[key, "brier_mean"].to_numpy()
    table["f1_minus_class_freq"] = table["f1_macro_mean"].to_numpy() - base.loc[key, "f1_macro_mean"].to_numpy()
    if "f1_perm_mean" in table:
        table["f1_minus_perm"] = table["f1_macro_mean"] - table["f1_perm_mean"]
    return table


def confusion_summed(g: pd.DataFrame) -> np.ndarray:
    """Cộng qua mã, trung bình qua seed (3x3, hàng = thật, cột = dự đoán)."""
    cms = [M.confusion(d["y_true"].to_numpy(), d["y_pred"].to_numpy()) for _, d in g.groupby(["ticker", "seed"])]
    return np.sum(cms, axis=0) / g["seed"].nunique()


def fig_confusion(cms: dict, T: int, span: tuple[str, str, int], n_tickers: int, n_seeds: int, fig_dir: Path,
                  lookbacks=C.LOOKBACKS) -> None:
    fig, axes = plt.subplots(1, len(lookbacks), figsize=C.CONFUSION_FIGSIZE)
    names = list(C.LABEL_NAMES)
    k = C.N_CLASSES
    for ax, L in zip(np.atleast_1d(axes), lookbacks):
        cm = cms[(T, L)]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=C.CONFUSION_CMAP, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]:.0f})", ha="center", va="center",
                        fontsize=C.FIG_FONTSIZE,
                        color="white" if norm[i, j] > C.CONFUSION_TEXT_SWITCH else C.FIG_INK)
        ax.set_xticks(range(k), names)
        ax.set_yticks(range(k), names)
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"L={L}", fontsize=C.FIG_TITLE_FONTSIZE, loc="left")
    fig.suptitle(f"T+{T}: confusion matrix gộp {n_tickers} mã, test từ {span[0]} đến {span[1]} "
                 f"({span[2]} mẫu/mã). Hàng chuẩn hóa, trong ngoặc là số mẫu trung bình qua {n_seeds} seed",
                 fontsize=C.FIG_FONTSIZE, x=C.FIG_SUPTITLE_X, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / C.CONFUSION_FIG_FILE.format(T=T), dpi=C.FIG_DPI)
    plt.close(fig)


def per_ticker_table(ps: pd.DataFrame, T: int, L: int) -> pd.DataFrame:
    """F1, kappa, F1 hoán vị theo từng mã, trung bình qua seed."""
    t = ps.groupby("ticker")[["f1_macro", "kappa", "f1_perm", "brier"]].mean().reset_index()
    t.insert(0, "L", L)
    t.insert(0, "T", T)
    t["f1_minus_perm"] = t["f1_macro"] - t["f1_perm"]
    return t


def config_label(c: pd.Series) -> str:
    return f"{c['model']}/wd{c['wd']:g}"


def main(final_dir: Path, fig_dir: Path) -> pd.DataFrame:
    final_dir, fig_dir = Path(final_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(final_dir / C.TEST_PREDICTIONS_FILE, parse_dates=["date"])
    cfg = pd.read_csv(final_dir / C.FINAL_CONFIG_FILE).set_index(["T", "L"])
    freq = pd.read_csv(final_dir / C.CLASS_FREQ_FILE)
    n_tickers, n_seeds = preds["ticker"].nunique(), preds["seed"].nunique()

    rows, cms, spans, tick = [], {}, {}, []
    horizons = sorted(cfg.index.get_level_values("T").unique())
    for T in horizons:
        spans[T] = date_range(preds, T)
        best_L = cfg.loc[T]["val_f1_macro_mean_over_tickers"].idxmax()
        lookbacks = sorted(cfg.loc[T].index)
        for L in lookbacks:
            g = preds[(preds["T"] == T) & (preds["L"] == L)]
            c = cfg.loc[(T, L)]
            label = config_label(c)
            base = {"T": T, "L": L, "date_from": spans[T][0], "date_to": spans[T][1],
                    "n_test_per_ticker": spans[T][2], "n_tickers": n_tickers, "scale": C.SCALE_NOTE}
            for method, name in (("model", label),
                                 ("ensemble", f"{label}, trung bình xác suất {n_seeds} seed"),
                                 ("repeat", "repeat"), ("class_freq", "tần suất lớp train")):
                src = ensemble(g) if method == "ensemble" else g
                ps = per_ticker_seed(src, "model" if method == "ensemble" else method, freq)
                ci = bootstrap_ci(src) if method in ("model", "ensemble") else {}
                rows.append({**base, "method": method, "config": name,
                             "val_f1_macro": c["val_f1_macro_mean_over_tickers"] if method == "model" else np.nan,
                             "headline": method == "model" and L == best_L, **summarize(ps), **ci})
                if method == "model":
                    tick.append(per_ticker_table(ps, T, L))
            cm = confusion_summed(g)
            cms[(T, L)] = cm
            pd.DataFrame(cm, index=C.LABEL_NAMES, columns=C.LABEL_NAMES).to_csv(
                final_dir / C.CONFUSION_FILE.format(T=T, L=L))
            pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=C.LABEL_NAMES, columns=C.LABEL_NAMES).to_csv(
                final_dir / C.CONFUSION_NORM_FILE.format(T=T, L=L))
        fig_confusion(cms, T, spans[T], n_tickers, n_seeds, fig_dir, lookbacks)

    table = add_skill(pd.DataFrame(rows))
    table.to_csv(final_dir / C.FINAL_TABLE_FILE, index=False)
    per_ticker = pd.concat(tick, ignore_index=True)
    per_ticker.to_csv(final_dir / C.PER_TICKER_FILE, index=False)
    pd.set_option("display.width", C.DISPLAY_WIDTH)
    cols = ["T", "L", "method", "f1_macro_mean", "f1_macro_std", "f1_ci_lo", "f1_ci_hi", "f1_perm_mean",
            "kappa_mean", "kappa_std", "kappa_ci_lo", "kappa_ci_hi", "brier_mean", "brier_skill",
            "hit_rate_mean", "directional_share_mean", "f1_minus_class_freq", "headline"]
    print(table[cols].round(C.DISPLAY_DECIMALS).to_string(index=False))
    beat = per_ticker.groupby(["T", "L"])["f1_minus_perm"].apply(lambda s: f"{int((s > 0).sum())}/{len(s)}")
    print("Số mã có F1 > F1 hoán vị:", beat.to_dict())
    for T in horizons:
        print(f"T+{T}: test từ {spans[T][0]} đến {spans[T][1]}, {spans[T][2]} mẫu/mã")
    return table


# ---------------------------------------------------------------------------------------------------------------
# Đồ thị dự báo hồi quy của kho model (outputs/models)
# ---------------------------------------------------------------------------------------------------------------
REG_SPLITS = ("val", "test")


def to_price(a: dict, s: np.ndarray) -> np.ndarray:
    """Thang chuẩn hóa (scaler lưu cùng model) -> giá gốc."""
    return np.exp(s * a["scaler_scale"] + a["scaler_mean"])


def regression_curves(a: dict, pdata: PooledData, panel: Panel) -> dict:
    """Dự báo hồi quy của một model đã lưu trên val và test của dữ liệu hiện tại.

    Mỗi tập: pred, true, repeat (N, T) ở thang chuẩn hóa bằng scaler lưu cùng model (repeat lặp giá trị cuối của
    cửa sổ cho cả T bước) và date (N,) ngày của bước T.
    """
    model, L, T = PR.build(a), a["L"], a["T"]
    k = pdata.tickers.index(a["ticker"])
    logc = panel.log_close[a["ticker"]]
    scale = lambda v: (v - a["scaler_mean"]) / a["scaler_scale"]
    out = {}
    for name in REG_SPLITS:
        split = getattr(pdata, name)
        end = split.end_idx[split.of(k)]
        pred, _ = PR.forward(a, model, raw_windows(logc, end, L))
        out[name] = {"pred": pred, "true": scale(raw_windows(logc, end + T, T)),
                     "repeat": np.repeat(scale(logc[end])[:, None], T, axis=1), "date": panel.dates[end + T]}
    return out


def model_label(a: dict) -> str:
    return f"{a['model']}/wd{a['wd']:g}/seed{a['seed']}"


def regression_rows(a: dict, curves: dict) -> list[dict]:
    """MSE, MAE của model và baseline repeat trên val và test, trung bình qua T bước: thang chuẩn hóa (cùng thang
    với mse_scaled trong runs.csv) và thang giá gốc."""
    rows = []
    for name, c in curves.items():
        for method in ("model", "repeat"):
            p = c["pred"] if method == "model" else c["repeat"]
            err = to_price(a, p) - to_price(a, c["true"])
            rows.append({"ticker": a["ticker"], "T": a["T"], "L": a["L"], "config": model_label(a),
                         "split": name, "method": method, "date_from": str(c["date"][0].date()),
                         "date_to": str(c["date"][-1].date()), "n": len(p), **M.regression_scaled(p, c["true"]),
                         "mse_price": float(np.mean(err ** 2)), "mae_price": float(np.mean(np.abs(err)))})
    return rows


def fig_model(a: dict, curves: dict, rows: list[dict], pdata: PooledData, panel: Panel, path: Path) -> None:
    n_train, n_val, _ = pdata.sizes
    T = a["T"]
    fig, ax = plt.subplots(figsize=C.MODEL_FIGSIZE)
    ax.axvspan(panel.dates[n_train + n_val], panel.dates[-1], color=C.MODEL_TEST_SHADE, zorder=0)
    ax.plot(panel.dates[n_train:], np.exp(panel.log_close[a["ticker"]][n_train:]), color=C.MODEL_ACTUAL_COLOR,
            lw=C.MODEL_LINE_WIDTH, label="Thực tế")
    for i, name in enumerate(REG_SPLITS):
        c = curves[name]
        ax.plot(c["date"], to_price(a, c["pred"][:, -1]), color=C.MODEL_PRED_COLOR, lw=C.MODEL_LINE_WIDTH,
                label=f"Dự đoán t+{T} (đầu hồi quy)" if i == 0 else None)
    for x, text in ((panel.dates[n_train], "Val"), (panel.dates[n_train + n_val], "Test")):
        ax.text(x, 0.02, f" {text}", transform=ax.get_xaxis_transform(), va="bottom", fontsize=C.FIG_FONTSIZE,
                color=C.FIG_INK)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_ylabel("Giá đóng cửa (thang giá gốc)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=C.FIG_FONTSIZE)
    m = {(r["split"], r["method"]): r for r in rows}
    scaled = " · ".join(f"{s} {m[(s, 'model')]['mse_scaled']:.4f} / {m[(s, 'model')]['mae_scaled']:.4f} "
                        f"(repeat {m[(s, 'repeat')]['mse_scaled']:.4f} / {m[(s, 'repeat')]['mae_scaled']:.4f})"
                        for s in REG_SPLITS)
    price = " · ".join(f"{s} {m[(s, 'model')]['mae_price']:.4g} (repeat {m[(s, 'repeat')]['mae_price']:.4g})"
                       for s in REG_SPLITS)
    ax.set_title(f"{a['ticker']} · T={T}, L={a['L']} · {model_label(a)} · dữ liệu đến {panel.dates[-1].date()}\n"
                 f"MSE / MAE thang chuẩn hóa, trung bình {T} bước: {scaled}\nMAE thang giá gốc: {price}",
                 loc="left", fontsize=C.FIG_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=C.FIG_DPI)
    plt.close(fig)


def plot_models(panel: Panel, models_dir: Path = C.MODELS_DIR, fig_dir: Path = C.MODEL_FIGURES_DIR,
                out_file: Path = C.MODEL_REGRESSION_FILE, log=print) -> pd.DataFrame:
    """Một hình C.MODEL_FIG_FILE cho mỗi model trong kho và bảng MSE/MAE (model và repeat, val và test)."""
    paths = sorted(Path(models_dir).glob("*.pt"))
    if not paths:
        log(f"Chưa có model trong {models_dir}; chạy `python -m src.pipeline` trước")
        return pd.DataFrame()
    fig_dir, out_file = Path(fig_dir), Path(out_file)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pdatas, rows = {}, []
    for p in paths:
        a = PR.load_artifact(p)
        key = (a["T"], a["L"], tuple(a["tickers"]))
        if key not in pdatas:
            pdatas[key] = build_pooled(a["L"], a["T"], a["tickers"], panel)
        curves = regression_curves(a, pdatas[key], panel)
        r = regression_rows(a, curves)
        fig_model(a, curves, r, pdatas[key], panel,
                  fig_dir / C.MODEL_FIG_FILE.format(ticker=a["ticker"], T=a["T"], L=a["L"]))
        rows += r
    table = pd.DataFrame(rows).sort_values(["ticker", "T", "L", "split", "method"], ignore_index=True)
    table.to_csv(out_file, index=False)
    summary = table.groupby(["T", "L", "split", "method"])[["mse_scaled", "mae_scaled", "mae_price"]].mean()
    pd.set_option("display.width", C.DISPLAY_WIDTH)
    log(f"Đồ thị {len(paths)} model: {fig_dir}\nMSE/MAE từng model: {out_file}\n"
        f"Trung bình qua mã (thang chuẩn hóa và MAE thang giá gốc):\n{summary.round(C.DISPLAY_DECIMALS).to_string()}")
    return table


if __name__ == "__main__":
    plot_models(build_panel(C.TICKERS))

