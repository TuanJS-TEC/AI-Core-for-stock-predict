"""Model tốt nhất cho từng mã và dự đoán 3 nhãn kèm xác suất.

Kho model (outputs/models/): mỗi (mã, T, L) một file C.MODEL_FILE, 15 mã x 2 T x 2 L = 60 file. Mỗi file chứa trọng
số của mô hình gộp (dự đoán cho mã bằng embedding của mã đó), scaler của mã, tham số hiệu chỉnh (T, alpha, lambda),
prior, θ và chỉ số val/test.

Cập nhật (update_models, pipeline gọi sau mỗi lần chạy):
  1. ứng viên = lượt trong lưới vừa chạy có C.MODEL_SELECT_METRIC cao nhất trên val của chính mã đó;
  2. model đang lưu (nếu có) được chấm lại trên val của dữ liệu hiện tại: hiệu chỉnh lại trên val hiện tại, như mọi
     ứng viên, nên hai bên so trên cùng một tập và cùng một quy trình;
  3. ứng viên cao hơn (quá C.MODEL_MIN_GAIN) thì thay, bằng hoặc thấp hơn thì giữ model cũ (cập nhật hiệu chỉnh
     và chỉ số theo val mới).
     Model lưu với K_THETA hoặc kiến trúc khác thì nhãn/trọng số không so được nên luôn bị thay.
Chỉ dùng val để chọn; chỉ số test chỉ để ghi nhận. registry.csv liệt kê model đang lưu, registry_history.csv ghi
mọi lần cập nhật.

Dự đoán từ cửa sổ L phiên mới nhất của dữ liệu trong stock/ (chạy `python sync_stock_parquet.py` trước để có phiên
mới nhất):
  python -m src.predict              mọi mã
  python -m src.predict VNM FPT      một vài mã
Xác suất là xác suất đã hiệu chỉnh (đã co về tần suất lớp train, tối ưu Brier trên val). Nhãn dự đoán dùng đúng quy
tắc đã đánh giá: argmax(p_c / prior_c^alpha), nên có thể khác nhãn có xác suất lớn nhất khi alpha > 0.

Ngày: last_date là phiên cuối của dữ liệu (gốc của dự đoán), target_date là phiên giao dịch thứ T sau last_date (bỏ
thứ Bảy, Chủ Nhật và ngày nghỉ: ngày nghỉ đã qua suy ra từ dữ liệu, ngày nghỉ sắp tới lấy từ C.MARKET_HOLIDAYS),
predicted_at là lúc chạy. Dự đoán luôn tính từ phiên cuối của dữ liệu chứ không từ hôm nay: chạy vào cuối tuần hay
ngày nghỉ cho cùng kết quả với phiên giao dịch trước đó. Nếu dữ liệu thiếu phiên lẽ ra đã có (chưa sync) thì in cảnh
báo; target_note = "đã qua" khi phiên mục tiêu đã diễn ra, "dự kiến" khi nằm sau C.MARKET_CALENDAR_UNTIL (lịch nghỉ
chưa khai báo nên ngày có thể lùi).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C, metrics as M
from .data import Panel, build_panel, build_pooled, raw_windows
from .models import PooledMultiTask, arch_spec

SUMMARY_KEYS = ["val_f1_macro", "val_kappa", "val_brier_skill", "f1_macro", "kappa"]
WEEKMASK = "1111100"                        # thứ Hai .. thứ Sáu
WEEKDAY_NAMES = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")


def model_path(ticker: str, T: int, L: int, models_dir: Path = C.MODELS_DIR) -> Path:
    return Path(models_dir) / C.MODEL_FILE.format(ticker=ticker, T=T, L=L)


def load_artifact(path: Path) -> dict:
    return torch.load(path, weights_only=True)


def build(a: dict) -> PooledMultiTask:
    model = PooledMultiTask(a["model"], a["L"], a["T"], 1, len(a["tickers"]))
    model.load_state_dict(a["state_dict"])
    return model.eval()


@torch.no_grad()
def forward(a: dict, model: PooledMultiTask, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """windows: (N, L) log close gốc, chuẩn hóa bằng scaler lưu cùng model.
    Trả về (dự báo hồi quy (N, T) ở thang chuẩn hóa, logits (N, N_CLASSES))."""
    x = ((windows - a["scaler_mean"]) / a["scaler_scale"]).astype(np.float32)[:, None, :]
    tid = torch.full((len(x),), a["ticker_index"], dtype=torch.long)
    reg, z = model(torch.from_numpy(x), tid)
    return reg.numpy(), z.numpy()


def logits(a: dict, model: PooledMultiTask, windows: np.ndarray) -> np.ndarray:
    return forward(a, model, windows)[1]


def signature(panel: Panel) -> dict:
    return {"n_rows": len(panel.dates), "last_date": str(panel.dates[-1].date())}


def summary(row: dict | pd.Series) -> dict:
    """Chỉ số val (dùng để chọn) và test (chỉ ghi nhận) của một model."""
    return {k if k.startswith("val_") else f"test_{k}": float(row[k]) for k in SUMMARY_KEYS}


def calibration(row: dict | pd.Series, pdata, t: str) -> dict:
    return {"temperature": float(row["temperature"]), "alpha": float(row["alpha"]), "shrink": float(row["shrink"]),
            "prior": [float(v) for v in pdata.prior[t]], "theta": float(pdata.theta[t])}


def make_artifact(cand: pd.Series, state: dict, pdata, t: str, k: int, sig: dict, now: str) -> dict:
    s = pdata.scalers[t]
    return {"ticker": t, "ticker_index": k, "tickers": list(pdata.tickers), "T": pdata.T, "L": pdata.L,
            "model": str(cand["model"]), "wd": float(cand["wd"]), "seed": int(cand["seed"]),
            "state_dict": state, "scaler_mean": float(s.mean_[0]), "scaler_scale": float(s.scale_[0]),
            "k_theta": C.K_THETA, "arch": arch_spec(), "label_names": list(C.LABEL_NAMES),
            **calibration(cand, pdata, t), "metrics": summary(cand),
            "trained_on": sig, "evaluated_on": sig, "trained_at": now, "evaluated_at": now}


def rescore(a: dict, pdata, panel: Panel) -> dict:
    """Chấm lại model đang lưu trên val/test của dữ liệu hiện tại, hiệu chỉnh lại trên val hiện tại."""
    model, t = build(a), a["ticker"]
    k = pdata.tickers.index(t)
    z = {}
    for name, split in (("val", pdata.val), ("test", pdata.test)):
        m = split.of(k)
        z[name] = (logits(a, model, raw_windows(panel.log_close[t], split.end_idx[m], a["L"])), split.labels[m])
    row, _, _ = M.ticker_metrics(*z["val"], *z["test"], pdata.prior[t])
    return row


def registry_row(a: dict) -> dict:
    return {"ticker": a["ticker"], "T": a["T"], "L": a["L"], "model": a["model"], "wd": a["wd"], "seed": a["seed"],
            **a["metrics"], "trained_last_date": a["trained_on"]["last_date"],
            "evaluated_last_date": a["evaluated_on"]["last_date"], "trained_at": a["trained_at"]}


def update_models(run_dir: Path, panel: Panel, tickers=C.TICKERS, models_dir: Path = C.MODELS_DIR,
                  log=print) -> pd.DataFrame:
    """Cập nhật kho model từ runs.csv và states.pt của một lần chạy. Trả về bảng các quyết định."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(Path(run_dir) / C.RUNS_FILE)
    states = torch.load(Path(run_dir) / C.STATES_FILE, weights_only=True)
    sig, now = signature(panel), pd.Timestamp.now().isoformat(timespec="seconds")
    metric = C.MODEL_SELECT_METRIC
    out = []
    for (T, L), g in runs.groupby(["T", "L"]):
        pdata = build_pooled(int(L), int(T), tickers, panel)
        for k, t in enumerate(pdata.tickers):
            c = g[g["ticker"] == t].sort_values(["model", "wd", "seed"])
            cand = c.loc[c[metric].idxmax()]
            path = model_path(t, T, L, models_dir)
            old = load_artifact(path) if path.exists() else None
            if old is None:
                action, old_score = "new", np.nan
            elif old["k_theta"] != C.K_THETA or old["arch"] != arch_spec():
                action, old_score = "replaced (khác K_THETA/kiến trúc)", np.nan
            else:
                row = rescore(old, pdata, panel)
                old_score = row[metric]
                action = "replaced" if cand[metric] > old_score + C.MODEL_MIN_GAIN else "kept"
            if action == "kept":
                a = {**old, **calibration(row, pdata, t), "metrics": summary(row), "evaluated_on": sig,
                     "evaluated_at": now}
            else:
                key = (int(T), int(L), str(cand["model"]), float(cand["wd"]), int(cand["seed"]))
                a = make_artifact(cand, states[key], pdata, t, k, sig, now)
            torch.save(a, path)
            out.append({"updated_at": now, "ticker": t, "T": int(T), "L": int(L), "action": action,
                        f"candidate_{metric}": float(cand[metric]), f"previous_{metric}": old_score,
                        "candidate": f"{cand['model']}/wd{cand['wd']:g}/seed{int(cand['seed'])}",
                        "saved": f"{a['model']}/wd{a['wd']:g}/seed{a['seed']}"})
    hist = pd.DataFrame(out)
    hp = models_dir / C.MODEL_HISTORY_FILE
    hist.to_csv(hp, mode="a", header=not hp.exists(), index=False)
    reg = pd.DataFrame([registry_row(load_artifact(p)) for p in sorted(models_dir.glob("*.pt"))])
    reg.sort_values(["ticker", "T", "L"]).to_csv(models_dir / C.MODEL_REGISTRY_FILE, index=False)
    counts = hist["action"].str.split(" ").str[0].value_counts().to_dict()
    log(f"Kho model {models_dir}: {len(reg)} model; lần này {counts}")
    return hist


def market_holidays(panel: Panel) -> np.ndarray:
    """Ngày nghỉ (datetime64[D]): ngày thường không có phiên trong dữ liệu và ngày khai báo trong C.MARKET_HOLIDAYS."""
    past = pd.bdate_range(panel.dates[0], panel.dates[-1]).difference(panel.dates)
    return np.union1d(past.values.astype("datetime64[D]"), np.array(C.MARKET_HOLIDAYS, dtype="datetime64[D]"))


ONE_DAY = np.timedelta64(1, "D")


def day(d) -> np.datetime64:
    return np.datetime64(pd.Timestamp(d).date(), "D")


def session_after(d, n: int, holidays: np.ndarray) -> pd.Timestamp:
    """Phiên giao dịch thứ n sau ngày d (d là ngày nghỉ thì tính từ phiên kế tiếp)."""
    return pd.Timestamp(np.busday_offset(day(d), n, roll="forward", weekmask=WEEKMASK, holidays=holidays))


def latest_ready_session(now: pd.Timestamp, holidays: np.ndarray) -> pd.Timestamp:
    """Phiên mới nhất lẽ ra đã có dữ liệu lúc now: hôm nay nếu là ngày giao dịch và đã qua C.SESSION_DATA_READY,
    nếu không thì phiên giao dịch trước đó."""
    d = day(now) if now.strftime("%H:%M") >= C.SESSION_DATA_READY else day(now) - ONE_DAY
    return pd.Timestamp(np.busday_offset(d, 0, roll="backward", weekmask=WEEKMASK, holidays=holidays))


def missing_sessions(last_date, now: pd.Timestamp, holidays: np.ndarray) -> list[pd.Timestamp]:
    """Các phiên sau last_date lẽ ra đã có dữ liệu lúc now (rỗng nếu dữ liệu đã mới nhất)."""
    days = np.arange(day(last_date) + ONE_DAY, day(latest_ready_session(now, holidays)) + ONE_DAY)
    return [pd.Timestamp(d) for d in days[np.is_busday(days, weekmask=WEEKMASK, holidays=holidays)]]


def predict_latest(a: dict, panel: Panel, holidays: np.ndarray | None = None) -> dict:
    """Xác suất 3 nhãn cho T phiên sau phiên cuối của dữ liệu, từ L phiên mới nhất."""
    t, L = a["ticker"], a["L"]
    holidays = market_holidays(panel) if holidays is None else holidays
    z = logits(a, build(a), panel.log_close[t][-L:][None])
    probs, _, scores = M.apply_calibration(z, a["temperature"], np.asarray(a["prior"]), a["alpha"], a["shrink"])
    return {"ticker": t, "T": a["T"], "L": L, "last_date": str(panel.dates[-1].date()),
            "target_date": str(session_after(panel.dates[-1], a["T"], holidays).date()),
            **dict(zip(C.PROB_COLS, probs[0].round(4))), "label": a["label_names"][int(scores[0].argmax())],
            "up_above_pct": round(100 * float(np.exp(a["theta"]) - 1), 2),
            "down_below_pct": round(100 * float(np.exp(-a["theta"]) - 1), 2),
            "model": a["model"], "wd": a["wd"], "seed": a["seed"],
            "val_f1_macro": round(a["metrics"]["val_f1_macro"], 4),
            "trained_last_date": a["trained_on"]["last_date"]}


def target_note(target: pd.Timestamp, now: pd.Timestamp, holidays: np.ndarray) -> str:
    if target <= latest_ready_session(now, holidays):
        return "đã qua"
    return "dự kiến" if target > pd.Timestamp(C.MARKET_CALENDAR_UNTIL) else ""


def predict(tickers=C.TICKERS, panel: Panel | None = None, models_dir: Path = C.MODELS_DIR,
            now: pd.Timestamp | None = None) -> pd.DataFrame:
    panel = build_panel(C.TICKERS) if panel is None else panel
    now = pd.Timestamp.now() if now is None else now
    holidays = market_holidays(panel)
    rows, missing = [], []
    for t in tickers:
        for T in C.HORIZONS:
            for L in C.LOOKBACKS:
                p = model_path(t, T, L, models_dir)
                if p.exists():
                    rows.append(predict_latest(load_artifact(p), panel, holidays))
                else:
                    missing.append(p.name)
    if missing:
        print(f"Chưa có {len(missing)} model ({', '.join(missing[:4])}...); chạy `python -m src.pipeline` trước")
    table = pd.DataFrame(rows)
    if not table.empty:
        table["predicted_at"] = now.isoformat(timespec="seconds")
        table["target_note"] = [target_note(pd.Timestamp(d), now, holidays) for d in table["target_date"]]
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Dự đoán 3 nhãn kèm xác suất cho T phiên tới",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("tickers", nargs="*", default=C.TICKERS, help=f"mã cần dự đoán (mặc định: cả {len(C.TICKERS)} mã)")
    a = ap.parse_args()
    bad = sorted(set(a.tickers) - set(C.TICKERS))
    if bad:
        ap.error(f"mã không hợp lệ {bad}; chọn trong {C.TICKERS}")
    panel, now = build_panel(C.TICKERS), pd.Timestamp.now()
    table = predict(a.tickers, panel, now=now)
    if table.empty:
        return
    last = table["last_date"].iloc[0]
    C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PREDICTIONS_DIR / C.PREDICTIONS_FILE.format(date=last)
    table.to_csv(out, index=False)
    pd.set_option("display.width", C.DISPLAY_WIDTH)
    holidays = market_holidays(panel)
    lag = missing_sessions(last, now, holidays)
    if lag:
        print(f"CẢNH BÁO: dữ liệu mới đến phiên {last}, thiếu {len(lag)} phiên lẽ ra đã có "
              f"({', '.join(str(d.date()) for d in lag)}). Chạy `python sync_stock_parquet.py` rồi dự đoán lại; "
              f"nếu đó là ngày nghỉ chưa khai báo thì thêm vào MARKET_HOLIDAYS trong config.py.")
    print(f"Dự đoán lúc {table['predicted_at'].iloc[0]} từ phiên cuối {last} "
          f"({WEEKDAY_NAMES[pd.Timestamp(last).dayofweek]}): nhãn cho giá đóng cửa tại phiên mục tiêu so với phiên {last}.")
    for T, d in table.groupby("T")["target_date"].first().items():
        note = table.loc[table["T"] == T, "target_note"].iloc[0]
        print(f"  T = {T}: phiên mục tiêu {d} ({WEEKDAY_NAMES[pd.Timestamp(d).dayofweek]})"
              + (f" [{note}]" if note else ""))
    if (table["target_note"] == "dự kiến").any():
        print(f"  [dự kiến]: sau {C.MARKET_CALENDAR_UNTIL} chưa khai báo ngày nghỉ (MARKET_HOLIDAYS), ngày có thể lùi.")
    print("Tăng: giá thay đổi > up_above_pct %, Giảm: < down_below_pct %, Đi ngang: nằm giữa "
          "(ngưỡng θ riêng từng mã, từng T).")
    print(table.drop(columns=["last_date", "predicted_at", "target_note"]).to_string(index=False))
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
