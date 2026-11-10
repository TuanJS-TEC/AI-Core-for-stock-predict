"""Single-use final evaluation on the sessions after HOLDOUT_AFTER, with the locked configuration, once per h."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import compare as CP, explain as EXP, experiment as X, recommend as R
from ai_core.prices import config as PC
from ai_core.prices.data import build_panel
from ai_core.prices.splits import Fold

LOCK_FILE = C.REPORTS / "mmf_lock.json"
EARLY_VERDICT = "Chưa đủ phiên để kết luận (chạy sớm)"
NEWS_MIN_RATIO = 0.25      # eval share of rows with news, against calib, below which the news pipeline is suspect
STEP7 = (f"Step 7, the owner decides. h2 and h5 mature at different times: run each h once it has its own {C.HOLDOUT_MIN} matured sessions, "
         "e.g. `.venv_mmf/bin/python -m ai_core.models.mmf.holdout --h 2` first and `--h 5` a few sessions later. "
         "Eval predictions with y_true are written under C.DATA/holdout before the guard files; no other tool may score them. "
         "If writing the report fails after the guard exists, never delete the guard: in Python, with a panel of the same dataset "
         "(dataset_id in the guard), call `ai_core.models.mmf.holdout.write_report(h, panel)`, which scores the saved predictions of the existing run dir.")
CMP_COLS = ["so_sanh", "a", "b", "mean", "lo", "hi", "fold_wins", "n_folds", "n", "n_days", "verdict"]


def holdout_rows(frame):
    """Usable rows after HOLDOUT_AFTER whose label ended by the frame's last session."""
    return frame[frame["usable"] & (frame["as_of_session"] > pd.Timestamp(C.HOLDOUT_AFTER))
                 & (frame["label_end_date"] <= frame["as_of_session"].max())]


def _guards(guard_dir, h):
    """The guard of h in guard_dir (tracked) and under C.DATA/holdout (untracked): a checkout or clean of one leaves the other."""
    return [Path(guard_dir) / f"mmf_holdout_h{h}.lock", C.DATA / "holdout" / f"mmf_holdout_h{h}.lock"]


def check_ready(rows, h, guard_dir, early=False):
    """Raise when this h was already scored or has too few matured sessions."""
    if done := [g for g in _guards(guard_dir, h) if g.exists()]:
        raise ValueError(f"h{h} đã chấm ({done[0]}): chạy lại cần quyết định của chủ dự án")
    need, have = C.HOLDOUT_MIN_EARLY if early else C.HOLDOUT_MIN, rows["as_of_session"].nunique()
    if have < need:
        raise ValueError(f"h{h}: mới có {have} phiên có nhãn sau {C.HOLDOUT_AFTER}, cần {need} phiên")


def load_lock():
    if not LOCK_FILE.exists():
        raise ValueError(f"thiếu file khóa {LOCK_FILE}")
    return json.loads(LOCK_FILE.read_text())


def _arms(entry):
    """DL arms of one h: the locked ones (stack has no network), its unimodal inputs, and those of R3."""
    want = (entry["fusion"], entry["unimodal"], *CP.UNIMODAL, "fusion_concat", *CP.ABLATIONS)
    return [a for a in dict.fromkeys(want) if a != "stack"]


def _comparisons(preds, entry, early):
    """R1, R2, R3 on the single eval block with HOLDOUT_BLOCK, fold_wins and n_folds (0 or 1, 1). An early run gets no verdict."""
    fusion = entry["fusion"]
    pairs = [("R1", fusion, entry["baseline"]), ("R2", fusion, entry["unimodal"]), *(("R3", "fusion_concat", ab) for ab in CP.ABLATIONS)]
    out = pd.DataFrame([{"so_sanh": name, "a": a, "b": b, **CP.compare(preds, {"F": a}, {"F": b}, block=C.HOLDOUT_BLOCK)} for name, a, b in pairs])
    return out.assign(verdict=EARLY_VERDICT) if early else out


def _simulations(preds, b, fusion):
    """Rule at the margin picked on calib, next to buying every row. eval_end is the last session, so exits stay in the data."""
    row, ev = EXP._simulations(preds, b, fusion, block=C.HOLDOUT_BLOCK)
    sim = pd.DataFrame([{**row, "pham_vi": "theo luật"}, {**R.simulate(ev, -1.0, b.h, block=C.HOLDOUT_BLOCK), "fold": "F", "m": np.nan, "pham_vi": "toàn mẫu"}])
    return sim.assign(calib_n=[str(int(v)) if pd.notna(v) else "" for v in sim["calib_n"]])


def _markdown(path, b, entry, preds, cmp, early):
    """Markdown report (Vietnamese) of one h: sample, R1-R3, every arm, leak alarms, the recommendation rule, per symbol."""
    t = CP.tables(preds)        # verdicts of its R1-R3 are unused: they ask for every dev fold
    ev, fusion = b.blocks["eval"], entry["fusion"]
    n_days = ev["as_of_session"].nunique()
    lines = [
        f"# Chấm vùng cuối: T+{b.h}", "",
        f"**Kết quả chấm một lần, trên các phiên sau {C.HOLDOUT_AFTER}.** Cấu hình lấy từ file khóa, không chọn lại gì trên vùng này: nhiệt độ, alpha và ngưỡng `m` từ calib.", "",
        f"- Số phiên: {n_days} (từ {ev['as_of_session'].min():%Y-%m-%d} đến {ev['as_of_session'].max():%Y-%m-%d}). Số mẫu: {len(ev)}.",
        f"- Cờ `early`: {'có, ngưỡng tối thiểu ' + str(C.HOLDOUT_MIN_EARLY) + ' phiên thay vì ' + str(C.HOLDOUT_MIN) if early else 'không'}.",
        f"- Train đến {b.fold.train_end:%Y-%m-%d}, tune đến {b.fold.tune_end:%Y-%m-%d}, calib đến {b.fold.calib_end:%Y-%m-%d}. Mỗi arm học sâu 5 seed, xác suất là trung bình qua seed.",
        f"- Khóa: fusion `{fusion}`, baseline `{entry['baseline']}`, nhánh đơn `{entry['unimodal']}`, bộ mã hóa giá `{entry['price_kind']}`.",
        f"- Dataset chấm: {b.dataset_id}. Dataset lúc khóa: {entry['dataset_id']}. Commit: {X._git_commit()}.",
        f"- Điểm cảm xúc `{b.nlp_version}`; {ev['has_news'].mean():.1%} mẫu trên vùng này có tin.", "",
        "Giới hạn dữ liệu (spec mục 6): 15 mã (6 mã ngân hàng thiếu biên gộp và thanh toán hiện hành, SSI thiếu BCTC, GAS có rất ít tin), "
        "tin chỉ từ 2021-10-21, BCTC theo năm không có ngày công bố nên năm Y coi là biết từ 01/04 năm Y+1, "
        "và AUC hướng giá trước đó chỉ khoảng 0.52 đến 0.54 nên kết quả khiêm tốn là kịch bản được kỳ vọng.", "",
        f"Dự báo của khối eval kèm `y_true` đã nằm trên đĩa dưới `{C.DATA / 'holdout'}` từ trước khi ghi file khóa chạy; không công cụ nào khác được chấm các file này.", "",
        "## R1, R2, R3", "",
        f"Chênh lệch là log-loss(a) trừ log-loss(b), âm là a tốt hơn. CI 95% block bootstrap theo ngày, khối {C.HOLDOUT_BLOCK} phiên thay vì {PC.BOOTSTRAP_BLOCK} (phụ lục 2026-11-10 của luật đặt trước). Chỉ có một vùng nên điều kiện 2/3 fold thay bằng `fold_wins=1, n_folds=1` theo plan: "
        "kết luận chỉ phụ thuộc chênh lệch trung bình và cận trên CI." + (f" Chạy sớm (`early`): kết luận thay bằng \"{EARLY_VERDICT}\", số vẫn in." if early else ""), "",
        CP._md(cmp, CMP_COLS), "",
        "## Cảnh báo rò rỉ", "",
        f"Macro-F1 >= {C.LEAK_ALARM_F1} hoặc AUC Tăng so với Giảm >= {C.LEAK_ALARM_AUC}: phải kiểm tra trước khi tin.", "",
        CP._md(t["leak_alarms"], ["arm", "n", "macro_f1", "auc_up_down"]), "",
        "## Mọi arm, toàn mẫu", "", CP._md(t["arms"].drop(columns=["h", "fold"])), "",
        "## Luật khuyến nghị kèm độ phủ", "",
        f"**Giả định chưa xác nhận:** lợi suất tính theo thanh toán T+2 và không bán khống (chỉ LONG), chủ dự án chưa xác nhận giả định này. "
        f"Mua ở giá mở cửa t+1, bán ở giá đóng cửa t+h+1, trừ phí {C.AC.COST:.4f}. Không tính trượt giá, kẹt trần sàn, hay giới hạn vốn; các lệnh của phiên liền nhau chồng lấn thời gian.", "",
        f"Fusion khóa `{fusion}`. `m` chọn trên calib; không có `m` đạt 50 lệnh thì không có lệnh, và `m` có trung bình calib âm thì luật vẫn giao dịch (xem `calib_mean_net`). "
        f"Hàng có phiên bán t+h+1 vượt dữ liệu bị bỏ. `coverage` là tỷ lệ hàng có lợi suất được gắn \"Ưu tiên mua\"; `toàn mẫu` là mua mọi hàng, để so sánh. `n` dưới khoảng 30 thì CI gần như không đọc được. CI 95% block bootstrap phiên, khối {C.HOLDOUT_BLOCK} phiên như phần so sánh.", "",
        CP._md(_simulations(preds, b, fusion), EXP.SIM_COLS), "",
        "## Theo mã, fusion khóa, toàn mẫu", "", CP._md(t["per_symbol"].query("arm == @fusion").drop(columns=["h", "fold", "arm"])), "",
    ]
    path.write_text("\n".join(lines))
    return path


def _rows(panel, h):
    return holdout_rows(X.build_frame(panel, h, C.FOLD_NLP["F"]))


def _bundle(panel, h, rows):
    """Fold F bundle whose eval block holds every matured holdout session."""
    # label_end_date.max(), not the last as_of: split_samples purges a label ending after the fold, which would drop h sessions
    fold = Fold("F", *map(pd.Timestamp, C.FINAL_FOLD[1:]), rows["label_end_date"].max())
    return X.fold_bundle(panel, h, fold)


def _run_dir(dataset_id):
    return C.DATA / "holdout" / dataset_id[:16]


def run(h, panel, guard_dir=C.REPORTS, early=False, device="cuda"):
    """Train on fold F, then guard, then score. Refused when not ready; once a guard exists a failed run stays refused."""
    rows = _rows(panel, h)
    check_ready(rows, h, guard_dir, early)
    entry = load_lock()[str(h)]
    b = _bundle(panel, h, rows)
    news = {k: b.blocks[k]["has_news"].mean() for k in ("calib", "eval")}
    print(f"h{h}: {news['eval']:.1%} mẫu eval có tin, calib {news['calib']:.1%}", flush=True)
    if news["eval"] < NEWS_MIN_RATIO * news["calib"]:
        raise ValueError(f"h{h}: tỷ lệ mẫu eval có tin {news['eval']:.1%} dưới {NEWS_MIN_RATIO:.0%} của calib ({news['calib']:.1%}): kiểm tra pipeline tin trước khi chấm")
    base = _run_dir(X._dataset_id(panel)) / f"h{h}" / "F"
    X.run_baseline(b, entry["baseline"], base / entry["baseline"] / "seed0.parquet")
    for arm in _arms(entry):
        for seed in C.SEEDS:
            X.run_arm(b, arm, seed, entry["price_kind"], base / arm / f"seed{seed}.parquet", device)
    body = json.dumps({"dataset_id": b.dataset_id, "git_commit": X._git_commit(), "n_sessions": int(b.blocks["eval"]["as_of_session"].nunique()),
                       "early": early, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2)
    for g in _guards(guard_dir, h):
        g.parent.mkdir(parents=True, exist_ok=True)
        with open(g, "x") as f:   # "x": a concurrent run stops here too
            f.write(body)
    return write_report(h, panel, guard_dir)


def write_report(h, panel, guard_dir=C.REPORTS):
    """Score the saved predictions of a guarded h into its report; also the recovery call when a run failed after its guard."""
    found = [g for g in _guards(guard_dir, h) if g.exists()]
    if not found:
        raise ValueError(f"h{h} chưa có file khóa chạy: chỉ run() được chấm")
    guard = json.loads(found[-1].read_text())
    if guard["dataset_id"] != panel.dataset_id:
        raise ValueError(f"h{h}: dữ liệu {panel.dataset_id} khác dữ liệu lúc chạy {guard['dataset_id']}: dùng đúng dataset đó")
    entry, early = load_lock()[str(h)], guard["early"]
    b = _bundle(panel, h, _rows(panel, h))   # ponytail: rebuilds the frames so run and recovery share one path, text features are cached
    preds = CP.add_stack(CP.load_h(_run_dir(guard["dataset_id"]), h))
    return _markdown(Path(guard_dir) / f"mmf_holdout_h{h}.md", b, entry, preds, _comparisons(preds, entry, early), early)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, epilog=STEP7)
    ap.add_argument("--h", type=int, nargs="+", required=True, choices=C.HORIZONS)
    ap.add_argument("--early", action="store_true", help=f"minimum {C.HOLDOUT_MIN_EARLY} sessions instead of {C.HOLDOUT_MIN}")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    panel = build_panel()
    for h in a.h:
        print(run(h, panel, early=a.early, device=a.device))


if __name__ == "__main__":
    main()
