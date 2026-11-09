# Model đa phương thức dự báo xu hướng giá T+2, T+5 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây model fusion giá + BCTC + cảm xúc tin tức dự báo 3 lớp T+2, T+5 cho 15 mã, có đối chứng, ablation, SHAP/LIME, và chấm một lần trên dữ liệu sau 2026-10-30.

**Architecture:** Package mới `mmf/` ở gốc AI_Core, đọc lớp dữ liệu `Pipeline_V1.2/src` và điểm cảm xúc trong `NLP_Module/data/nlp.db` (chỉ đọc). Mỗi lượt là một arm (tập nhánh + kiểu fusion) trên một fold, lưu xác suất ra parquet và chạy tiếp được khi bị ngắt. Chọn arm trên khối calib, so sánh bằng bootstrap ghép cặp theo ngày, rồi khóa cấu hình trước khi chấm vùng cuối.

**Tech Stack:** Python 3.14, PyTorch (CUDA khi có), scikit-learn, pandas 2.3.3, shap, lime, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-multimodal-trend-forecast-design.md`

## Global Constraints

- Chạy bằng `.venv_mmf/bin/python` từ gốc AI_Core (`/home/administrator/TLS/tuanhm@linux/AI_Core`). Không cài gì vào `NLP_Module/.venv`, đó là môi trường cron.
- Không sửa `NLP_Module/`, `Pipeline_V1.2/src/`, `ai_core/`. `NLP_Module/` bị gitignore ở repo gốc và có `.git` riêng. Được phép duy nhất: thêm dòng điểm cảm xúc vào `nlp.db` ở Task 3.
- `string[python]` so sánh `==` trả toàn False trên pandas 2.3.3 + Python 3.14. Luôn `.astype(str)` cột chuỗi đọc từ parquet trước khi lọc.
- Mọi nhánh dùng cùng tập mẫu: `as_of_session >= "2022-01-01"`, `status == "ok"`, đủ cửa sổ.
- Không dùng khối `eval` hay vùng cuối để chọn model, ngưỡng hay siêu tham số. Scaler, nhiệt độ, alpha chỉ fit trên train/tune/calib.
- Commit chỉ các đường dẫn của task (`git add <path>`), không `git add -A`: `.gitignore` và `data/finhay/*` đang có thay đổi không thuộc plan này. Commit message kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Docstring và comment viết như `ai_core/` (tiếng Anh ngắn). Văn bản báo cáo viết tiếng Việt.

## Review Focus

1. BCTC đọc bằng so sánh chuỗi trên cột `string[python]` cho ra toàn 0 mà không báo lỗi: phải lấy đúng giá trị thật của FPT, VCB (Task 2).
2. Năm tài chính dùng trước ngày công bố giả định, hoặc năm quá cũ vẫn được dùng: FY2024 không được thấy trước 2025-04-01, quá 400 ngày thì `has_bctc=0` (Task 2).
3. Bài báo đăng sau mốc chốt hoặc trong vòng 30 phút trước mốc lọt vào đặc trưng của phiên đó: phải bị loại (Task 3).
4. Cửa sổ chứa phiên thiếu: mẫu phải bị bỏ, không được trượt sang phiên khác (Task 4).
5. Chấm vùng cuối lần hai, hoặc khi chưa đủ phiên, hoặc gộp hàng chưa đến hạn nhãn: phải từ chối và không ghi gì (Task 13).

---

## Cấu trúc file

```
mmf/__init__.py        rỗng
mmf/config.py          hằng số, đường dẫn, sys.path
mmf/fundamentals.py    BCTC theo năm -> chỉ số theo phiên, point-in-time
mmf/text.py            điểm cảm xúc -> đặc trưng văn bản theo phiên, cache parquet
mmf/frame.py           ghép giá + BCTC + văn bản thành bảng dài theo (mã, phiên)
mmf/windows.py         scaler theo fold, cắt cửa sổ thành Samples
mmf/evaluate.py        chỉ số, nhiệt độ, alpha, bootstrap ghép cặp
mmf/baselines.py       prior cố định, prior thích ứng, logistic, GBM
mmf/models.py          encoder từng nhánh và FusionNet
mmf/train.py           vòng huấn luyện, early stopping, dự đoán
mmf/experiment.py      chạy arm theo (h, fold, seed), lưu xác suất, chạy tiếp được
mmf/compare.py         ensemble, bảng so sánh, stacking, verdict, khóa cấu hình
mmf/explain.py         hoán vị theo nhóm, SHAP, LIME
mmf/recommend.py       quy tắc khuyến nghị và mô phỏng phụ
mmf/holdout.py         chấm vùng cuối một lần
docs/reports/mmf_preregistration.md   luật đặt trước
tests/test_mmf_*.py    một file test cho mỗi module
```

Dữ liệu sinh ra nằm ở `data/mmf/` (gitignore).

---

### Task 1: Môi trường, đặt trước luật, config

**Files:**
- Create: `mmf/__init__.py`, `mmf/config.py`, `docs/reports/mmf_preregistration.md`
- Modify: `.git/info/exclude` (thêm `.venv_mmf/`, `data/mmf/`; dùng file này để không đụng `.gitignore` đang có thay đổi)
- Test: `tests/test_mmf_config.py`

**Interfaces:**
- Produces: `mmf.config` với các hằng số dưới đây.

- [ ] **Step 1: Tạo venv kế thừa gói của môi trường cron mà không sửa nó**

```bash
printf '.venv_mmf/\ndata/mmf/\n' >> .git/info/exclude
python3 -m venv .venv_mmf
echo "$PWD/NLP_Module/.venv/lib/python3.14/site-packages" > .venv_mmf/lib/python3.14/site-packages/nlp_base.pth
.venv_mmf/bin/python -m pip install shap lime
.venv_mmf/bin/python -c "import torch, sklearn, shap, lime, pandas; print(torch.cuda.is_available(), pandas.__version__)"
```
Expected: in `True 2.3.3`. Nếu `import shap` lỗi trên Python 3.14 thì dừng và báo BLOCKED, không tự đổi sang gói khác.

- [ ] **Step 2: Viết test thất bại**

```python
from mmf import config as C
from src.splits import folds_from_config

def test_fold_nlp_maps_every_fold_and_labels_precede_train_end():
    assert {f.name for f in folds_from_config()} == set(C.FOLD_NLP) - {"F"}
    assert C.FOLD_NLP == {"A": "phobert-foldB", "B": "phobert-foldB", "C": "phobert-foldC", "F": "phobert-phobert-v1"}

def test_final_fold_ends_where_dev_ends():
    assert C.FINAL_FOLD == ("F", "2026-05-04", "2026-07-30", "2026-10-30")
    assert C.HOLDOUT_AFTER == "2026-10-30"

def test_arms_cover_every_ablation():
    assert C.ARMS["fusion_concat"] == (("price", "fund", "text"), "concat")
    assert C.ARMS["fusion_attention"] == (("price", "fund", "text"), "attention")
    assert C.ARMS["no_price"][0] == ("fund", "text")
    assert C.ARMS["no_fund"][0] == ("price", "text")
    assert C.ARMS["no_text"][0] == ("price", "fund")
    assert {"dl_price", "dl_fund", "dl_text"} <= set(C.ARMS)
```

- [ ] **Step 3: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_config.py -q`
Expected: FAIL, `No module named 'mmf'`.

- [ ] **Step 4: Viết `mmf/config.py`**

`from ai_core import config as AC` ở đầu để có `sys.path` và `AC.ROOT`, `AC.NLP`, `AC.COST`. Các giá trị:

```python
TICKERS = src.config.TICKERS; HORIZONS = (2, 5); START = "2022-01-01"
BANKS = ("ACB", "MBB", "STB", "TCB", "VCB", "VPB")
LP = 60; LT = 10                     # cửa sổ nhánh giá, nhánh văn bản
D = 32; EMB = 4; DROPOUT = 0.2; PRICE_KINDS = ("lstm", "gru", "transformer")
LR = 1e-3; WD = 1e-2; BATCH = 256; MAX_EPOCHS = 60; PATIENCE = 10
SEEDS = (59, 60, 61, 62, 63); SELECT_SEEDS = SEEDS[:2]
FOLD_NLP = {"A": "phobert-foldB", "B": "phobert-foldB", "C": "phobert-foldC", "F": "phobert-phobert-v1"}
FINAL_FOLD = ("F", "2026-05-04", "2026-07-30", "2026-10-30")
HOLDOUT_AFTER = "2026-10-30"; HOLDOUT_MIN = 40; HOLDOUT_MIN_EARLY = 20
ARMS = {"dl_price": (("price",), "none"), "dl_fund": (("fund",), "none"), "dl_text": (("text",), "none"),
        "fusion_concat": (("price", "fund", "text"), "concat"), "fusion_attention": (("price", "fund", "text"), "attention"),
        "no_price": (("fund", "text"), "concat"), "no_fund": (("price", "text"), "concat"), "no_text": (("price", "fund"), "concat")}
BASELINES = ("prior_fixed", "prior_adaptive", "logit_price", "gbm_price", "logit_text", "gbm_text", "logit_all", "gbm_all")
LEAK_ALARM_F1 = 0.50; LEAK_ALARM_AUC = 0.65
DATA = AC.ROOT / "data/mmf"; REPORTS = AC.ROOT / "docs/reports"
NLP_DB = AC.NLP / "data/nlp.db"; BCTC_FILE = AC.NLP / "data/bctc.parquet"
```

- [ ] **Step 5: Viết `docs/reports/mmf_preregistration.md` với đúng nội dung sau**

```markdown
# Luật đặt trước: model đa phương thức T+2, T+5

Viết trước khi có kết quả thật. Mọi thay đổi sau này phải ghi thêm ở cuối file kèm ngày và lý do, không sửa phần trên.

## Vùng dữ liệu
- Vùng phát triển: khối eval của fold A, B, C trong `DATE_FOLDS`. Các khối này đã bị nhìn trong thí nghiệm trước, nên kết quả ở đây chỉ để phát triển.
- Vùng chấm cuối: phiên sau 2026-10-30, chấm một lần cho mỗi h sau khi khóa cấu hình.

## Quy trình chọn
Trong mỗi (h, fold): early stopping trên tune; nhiệt độ và alpha trên calib; chọn baseline tốt nhất, nhánh đơn tốt nhất và fusion tốt nhất bằng log-loss trên calib.

## Tiêu chí, theo từng h
Dùng chênh lệch log-loss ghép cặp, 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần).
- R1, thắng baseline: fusion tốt hơn baseline tốt nhất ở ít nhất 2/3 fold và cận trên CI gộp < 0.
- R2, lợi ích fusion: fusion tốt hơn nhánh đơn tốt nhất ở ít nhất 2/3 fold và cận trên CI gộp < 0.
- R3, đóng góp nhánh: với mỗi nhánh, so `fusion_concat` với arm bỏ nhánh đó bằng cùng phép đo.

Diễn giải cho mỗi so sánh:
- "Có bằng chứng": đạt điều kiện trên.
- "Có tín hiệu, bằng chứng yếu": chênh lệch trung bình có lợi nhưng CI chứa 0, hoặc CI đạt nhưng không đủ 2/3 fold.
- "Chưa chứng minh được": còn lại.

## Chỉ số báo cáo, không dùng để chọn
Accuracy, Precision, Recall, Macro-F1, AUC-ROC, Brier, kappa, theo từng h, fold và mã. Không có ngưỡng tuyệt đối cho Macro-F1.

## Cảnh báo rò rỉ
Macro-F1 toàn mẫu >= 0.50 hoặc AUC Tăng so với Giảm >= 0.65 thì coi là nghi rò rỉ, phải kiểm tra trước khi tin.

## Công khai
Mọi kết quả chọn lọc theo độ tin cậy phải báo kèm độ phủ, số mẫu, số ngày, và đi cùng kết quả toàn mẫu.
```

- [ ] **Step 6: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_config.py -q`
Expected: PASS.

- [ ] **Step 7: Commit spec, plan, luật đặt trước và config**

```bash
git add mmf/__init__.py mmf/config.py tests/test_mmf_config.py docs/reports/mmf_preregistration.md docs/superpowers/specs/2026-11-09-multimodal-trend-forecast-design.md docs/superpowers/plans/2026-11-09-multimodal-trend-forecast.md
git commit -m "feat(mmf): pre-registered rules and config for the multimodal trend forecast"
```

---

### Task 2: Chỉ số tài chính point-in-time

**Files:**
- Create: `mmf/fundamentals.py`
- Test: `tests/test_mmf_fundamentals.py`

**Interfaces:**
- Consumes: `C.BCTC_FILE`, `C.BANKS`, `nlp_module.forecast.fundamentals.annual_wide`.
- Produces: `FUND_FEATS = ["ep", "eps_yoy", "roe", "roa", "de", "gross_margin", "rev_yoy", "current_ratio", "has_bctc", "is_bank"]`; `annual_ratios(path=C.BCTC_FILE) -> DataFrame` (symbol, year, available_at, eps, eps_yoy, roe, roa, de, gross_margin, rev_yoy, current_ratio); `add_fundamentals(rows: DataFrame) -> DataFrame` nhận cột `symbol, as_of_session, close`, trả về thêm `FUND_FEATS`.

Mã khoản mục: EPS `is_lai_co_ban_tren_co_phieu`. Lợi nhuận `is_lai_lo_thuan_sau_thue` (khác), `is_loi_nhuan_sau_thue` (ngân hàng). Vốn `bs_von_chu_so_huu_4d280b22`, `bs_von_chu_so_huu_6cda78ae`. Tài sản `bs_tong_tai_san`. Nợ `bs_no_phai_tra`, `bs_tong_no_phai_tra`. Doanh thu `is_doanh_so_thuan`, `is_tong_thu_nhap_hoat_dong`. Lãi gộp `is_lai_gop`. Thanh toán `bs_tai_san_ngan_han` / `bs_no_ngan_han`.

- [ ] **Step 1: Viết test thất bại**

```python
import numpy as np, pandas as pd, pytest
from mmf import fundamentals as F

def row(df, s, y): return df[(df.symbol == s) & (df.year == y)].iloc[0]

def test_real_values_nonbank_and_bank():
    a = F.annual_ratios()
    fpt, vcb = row(a, "FPT", 2024), row(a, "VCB", 2024)
    assert fpt.eps == pytest.approx(4292)
    assert fpt.roe == pytest.approx(9.427423e12 / 3.572754e13, rel=1e-3)
    assert fpt.gross_margin == pytest.approx(2.369835e13 / 6.284879e13, rel=1e-3)
    assert fpt.de == pytest.approx(3.627246e13 / 3.572754e13, rel=1e-3)
    assert fpt.current_ratio == pytest.approx(4.553594e13 / 3.483618e13, rel=1e-3)
    assert fpt.available_at == pd.Timestamp("2025-04-01")
    assert vcb.eps == pytest.approx(5571)
    assert vcb.roe == pytest.approx(3.385312e13 / 1.962092e14, rel=1e-3)
    assert vcb.de == pytest.approx(1.889664e15 / 1.962092e14, rel=1e-3)
    assert np.isnan(vcb.gross_margin) and np.isnan(vcb.current_ratio)

def rows(sym, day, close=100.0):
    return pd.DataFrame({"symbol": [sym], "as_of_session": [pd.Timestamp(day)], "close": [close]})

def test_fiscal_year_not_visible_before_available_at():
    before = F.add_fundamentals(rows("FPT", "2025-03-31")).iloc[0]
    after = F.add_fundamentals(rows("FPT", "2025-04-01")).iloc[0]
    assert after.ep == pytest.approx(4292 / (100.0 * 1000))
    assert before.ep != after.ep

def test_stale_year_dropped_and_missing_symbol_flagged():
    stale = F.add_fundamentals(rows("FPT", "2031-01-02")).iloc[0]
    ssi = F.add_fundamentals(rows("SSI", "2025-06-02")).iloc[0]
    for r in (stale, ssi):
        assert r.has_bctc == 0 and r.ep == 0 and r.roe == 0

def test_bank_flag_and_nan_free():
    out = F.add_fundamentals(pd.concat([rows("VCB", "2025-06-02"), rows("FPT", "2025-06-02")]))
    assert out.set_index("symbol").is_bank.to_dict() == {"VCB": 1.0, "FPT": 0.0}
    assert out[F.FUND_FEATS].notna().all().all()
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_fundamentals.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/fundamentals.py`**

Đọc parquet, ép `ticker`, `statement`, `item_code` về `str` trước khi `annual_wide`. Giá `close` theo nghìn đồng nên `ep = eps / (close * 1000)`. Tăng trưởng so với năm liền trước, chia `|năm trước|`, năm thiếu thì NaN, cắt `[-2, 2]`. `add_fundamentals` dùng `merge_asof` backward theo `symbol`, `tolerance=400 ngày`, giữ thứ tự dòng vào. `has_bctc = 1` khi có năm khớp và `eps`, `roe`, `roa`, `de` đều hữu hạn, ngược lại mọi cột số về 0. Các cột còn lại NaN về 0. Cắt tỷ số về `[-5, 5]`.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_fundamentals.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/fundamentals.py tests/test_mmf_fundamentals.py
git commit -m "feat(mmf): point-in-time financial ratios from annual statements"
```

---

### Task 3: Đặc trưng văn bản theo phiên

**Files:**
- Create: `mmf/text.py`
- Test: `tests/test_mmf_text.py`

**Interfaces:**
- Consumes: `nlp_module.pipeline.load_results(c, version)`, `nlp_module.contract.build_snapshot(results, symbol, cutoff, mode, version)`, `.features`.
- Produces: `TEXT_FEATS = ["has_news", "news_count_24h", "news_count_72h", "unique_source_count", "hours_since_last_news", "difficult_ratio", "direct_news_ratio", "risk_flag_72h", "sentiment_mean", "sentiment_std", "sentiment_mean_24h", "sentiment_abs_max", "sentiment_shock_24h", "sentiment_decay"]`; `features_from_results(results, symbols, sessions, version) -> DataFrame` (symbol, as_of_session, *TEXT_FEATS); `text_features(version, symbols, sessions) -> DataFrame` có cache `C.DATA / f"text_{version}_{n_results}_{last_session:%Y%m%d}.parquet"`; `score_missing(version, model_path)`, CLI `python -m mmf.text score`.

- [ ] **Step 1: Viết test thất bại**

```python
from datetime import datetime, timezone
import pandas as pd
from nlp_module.contract import ArticleCompanyResult
from mmf import text as T

V = "v-test"
def art(rid, sym, pub, pos=0.9, neg=0.1, uncertain=False):
    t = datetime.fromisoformat(pub).replace(tzinfo=timezone.utc)
    return ArticleCompanyResult(rid, sym, "cafef", t, t, "direct", pos, 1 - pos - neg, neg, uncertain, False, True, V, ())

DAY = pd.DatetimeIndex(["2026-01-05"])   # cutoff 08:00 UTC

def one(results):
    return T.features_from_results(results, ["FPT"], DAY, V).iloc[0]

def test_article_inside_delay_or_after_cutoff_is_excluded():
    assert one([art("a", "FPT", "2026-01-05T07:45:00")]).has_news == 0   # 15 phút trước mốc, chưa qua 30 phút
    assert one([art("b", "FPT", "2026-01-05T09:00:00")]).has_news == 0   # sau mốc
    assert one([art("c", "FPT", "2026-01-05T07:30:00")]).has_news == 1

def test_uncertain_flag_is_ignored():
    r = one([art("d", "FPT", "2026-01-05T01:00:00", uncertain=True)])
    assert r.sentiment_mean > 0

def test_no_news_has_no_nan_and_other_symbol_not_counted():
    r = one([art("e", "VCB", "2026-01-05T01:00:00")])
    assert r.has_news == 0 and r[T.TEXT_FEATS].notna().all() and r.hours_since_last_news == 1.0
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_text.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/text.py`**

Mốc chốt mỗi phiên là ngày + `CLOSE_UTC` (`nlp_module.forecast.trial_forecast.CLOSE_UTC`), mode `"backtest"`. Trước khi gọi `build_snapshot`, dùng `dataclasses.replace(r, uncertain=False)`. `build_snapshot` quét toàn bộ input mỗi lần nên phải truyền trước danh sách đã lọc theo mã. Giá trị None về 0. `hours_since_last_news` chuẩn hóa `min(h, 240) / 240`, None về 1.0. Đếm tin dùng `log1p`. Kết nối DB chỉ đọc: `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`, `row_factory = sqlite3.Row`.

`score_missing` nạp `nlp_module.sentiment.phobert.PhoBert(str(path), version)` và `nlp_module.entities.Linker(load_aliases())`, gọi `pipeline.process(storage.connect(), model, linker)` với `abstain_entropy=None`.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_text.py -q`
Expected: PASS.

- [ ] **Step 5: Chấm các bài còn thiếu cho model theo fold**

`phobert-foldB`, `phobert-foldC` hiện dừng ở bài đăng 2026-10-21. Khóa chung với cron để không ghi đồng thời:

```bash
cd NLP_Module && export PYTHONPATH=.. && flock data/live.lock ../.venv_mmf/bin/python -m mmf.text score --version phobert-foldB --path data/models/phobert-foldB && flock data/live.lock ../.venv_mmf/bin/python -m mmf.text score --version phobert-foldC --path data/models/phobert-foldC
```
Phải chạy từ `NLP_Module/` vì `storage.connect` và `load_aliases` đọc đường dẫn tương đối. Expected: mỗi lệnh in số bài đã chấm. Kiểm tra `max(published_at)` của hai phiên bản đạt ít nhất 2026-10-30. Chạy lại lệnh là an toàn, vì `process` bỏ qua bài đã chấm.

- [ ] **Step 6: Commit**

```bash
git add mmf/text.py tests/test_mmf_text.py
git commit -m "feat(mmf): per-session text features from fold-specific sentiment scores"
```

---

### Task 4: Bảng ghép và cửa sổ

**Files:**
- Create: `mmf/frame.py`, `mmf/windows.py`
- Test: `tests/test_mmf_frame.py`

**Interfaces:**
- Consumes: `src.data.build_panel`, `src.dataset.build_dataset(panel, h)`, `src.price_features.FEATURES`, Task 2, Task 3.
- Produces:
  - `frame.GROUPS = {"price": FEATURES, "fund": FUND_FEATS, "text": TEXT_FEATS}`
  - `frame.build_frame(panel, h, nlp_version, text_fn=text.text_features) -> DataFrame`: một dòng mỗi (symbol, phiên trên lịch), kể cả dòng không dùng được, có `open`, `close`, `flag_hl_equal`, các cột nhóm, `label`, `status`, `label_end_date`, `realized_return`, `usable` (bool).
  - `windows.Scalers.fit(train_rows) -> Scalers`, `.apply(frame) -> DataFrame`: chuẩn hóa từng cột, cắt `[-5, 5]`, fit chỉ trên hàng đưa vào. Không chuẩn hóa `has_news`, `has_bctc`, `is_bank`.
  - `windows.Samples` dataclass: `x_price (N, LP, Fp)`, `x_text (N, LT, Ft)`, `x_fund (N, Ff)` (float32); `ticker (N,)` (int64); `y (N,)` (int64); `symbol`, `as_of`, `label_end` (mảng), với `__len__`.
  - `windows.make_samples(scaled_frame, block_rows, tickers, lp=C.LP, lt=C.LT) -> Samples`.

- [ ] **Step 1: Viết test thất bại**

```python
import numpy as np, pandas as pd, pytest
from src import config as PC
from src.data import build_panel, load_raw
from src.splits import folds_from_config, split_samples
from mmf import config as C, frame as FR, windows as W
from mmf.text import TEXT_FEATS

def no_text(version, symbols, sessions):
    idx = pd.MultiIndex.from_product([symbols, sessions], names=["symbol", "as_of_session"])
    out = pd.DataFrame(0.0, index=idx, columns=TEXT_FEATS).reset_index()
    return out

@pytest.fixture(scope="module")
def panel():
    return build_panel(PC.TICKERS, load_raw())

@pytest.fixture(scope="module")
def frame(panel):
    return FR.build_frame(panel, 5, "x", text_fn=no_text)

def test_usable_rows_start_2022_and_hl_equal_is_kept(frame):
    u = frame[frame.usable]
    assert u.as_of_session.min() >= pd.Timestamp(C.START)
    assert (u.status == "ok").all()
    assert frame.close_loc[frame.flag_hl_equal & (frame.as_of_session >= C.START)].eq(0.5).all()

def test_future_change_does_not_touch_past_features(panel):
    base = FR.build_frame(panel, 5, "x", text_fn=no_text)
    cut = pd.Timestamp("2025-06-02")
    panel2 = build_panel(PC.TICKERS, {k: v.assign(close=v.close.where(v.index <= cut, v.close * 2),
                                                  high=v.high.where(v.index <= cut, v.high * 2)) for k, v in load_raw().items()})
    other = FR.build_frame(panel2, 5, "x", text_fn=no_text)
    cols = [c for g in FR.GROUPS.values() for c in g]
    a, b = (f[f.as_of_session <= cut].sort_values(["symbol", "as_of_session"])[cols].reset_index(drop=True) for f in (base, other))
    pd.testing.assert_frame_equal(a, b)

def test_scaler_fits_on_given_rows_only(frame):
    tr = frame[frame.usable & (frame.as_of_session < "2024-01-01")]
    s = W.Scalers.fit(tr).apply(frame)
    got = s[s.index.isin(tr.index)]["ret_5"]
    assert abs(got.mean()) < 0.02 and abs(got.std() - 1) < 0.02

def test_window_with_missing_session_is_dropped(frame):
    f = frame.copy()
    hole = pd.Timestamp("2024-03-14")
    f.loc[(f.symbol == "FPT") & (f.as_of_session == hole), FR.GROUPS["price"]] = np.nan
    rows = f[f.usable & f.as_of_session.between("2024-03-01", "2024-04-30")]
    s = W.make_samples(W.Scalers.fit(f[f.usable]).apply(f), rows, PC.TICKERS)
    fpt = pd.to_datetime(s.as_of[s.symbol == "FPT"])
    cal = sorted(f.as_of_session.unique())
    blocked = set(cal[cal.index(hole):cal.index(hole) + C.LP])     # mọi cửa sổ chứa phiên thiếu
    assert not blocked & set(fpt)
    assert (fpt < hole).any()                                      # cửa sổ trước phiên thiếu vẫn còn
    assert np.isfinite(s.x_price).all()
    vcb = pd.to_datetime(s.as_of[s.symbol == "VCB"])
    assert hole in set(vcb)

def test_window_ends_at_sample_session(frame):
    f = frame[frame.symbol == "FPT"].sort_values("as_of_session")
    row = f[f.as_of_session == "2024-03-14"]
    s = W.make_samples(frame, row, PC.TICKERS)
    assert s.x_price.shape == (1, C.LP, len(FR.GROUPS["price"]))
    j = FR.GROUPS["price"].index("ret_5")
    assert s.x_price[0, -1, j] == pytest.approx(float(row.ret_5.iloc[0]))

def test_real_retention_at_least_99_percent(frame):
    u = frame[frame.usable]
    s = W.make_samples(frame, u, PC.TICKERS)
    assert len(s) >= 0.99 * len(u)
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_frame.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/frame.py` và `mmf/windows.py`**

`close_loc` đặt 0.5 khi `flag_hl_equal` (ngày trần/sàn, biết tại t). `usable = status == "ok"`, `as_of_session >= C.START`, nhóm giá hữu hạn. `make_samples` dựng mảng theo mã và vị trí trên lịch, kiểm tra `isfinite` toàn cửa sổ giá và văn bản, bỏ mẫu không đạt, không điền. Mẫu hoàn toàn nằm trong `build_dataset`, không tính lại nhãn.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_frame.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/frame.py mmf/windows.py tests/test_mmf_frame.py
git commit -m "feat(mmf): joined frame and leak-safe sample windows"
```

---

### Task 5: Chỉ số và so sánh ghép cặp

**Files:**
- Create: `mmf/evaluate.py`
- Test: `tests/test_mmf_evaluate.py`

**Interfaces:**
- Consumes: `src.stats.block_bootstrap_dates`, `src.config.BOOTSTRAP_BLOCK`.
- Produces: `PROB_COLS = ["p_down", "p_flat", "p_up"]`; `row_logloss(y, p) -> ndarray`; `row_brier(y, p) -> ndarray` (tổng 3 lớp, không chia); `scores(y, p, alpha=0.0, prior=None) -> dict` với khóa `logloss, brier, accuracy, macro_f1, kappa, auc_ovr, auc_up_down, precision_down, precision_flat, precision_up, recall_down, recall_flat, recall_up, n`; `decide(p, alpha, prior) -> ndarray`; `fit_temperature(logits, y) -> float`; `apply_temperature(logits, t) -> ndarray`; `fit_alpha(y, p, prior) -> float` từ lưới `(0.0, 0.25, 0.5, 0.75, 1.0)` theo Macro-F1; `paired_diff(a, b, dates, n=2000, seed=0) -> (mean, lo, hi)` cho từng ngày lấy trung bình rồi bootstrap chênh lệch `a - b`; `verdict(mean, hi, fold_wins, n_folds) -> str` trả đúng 3 câu trong luật đặt trước.

- [ ] **Step 1: Viết test thất bại**

```python
import numpy as np, pytest
from mmf import evaluate as E

P = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
Y = np.array([0, 1, 2])

def test_logloss_and_brier_by_hand():
    assert E.row_logloss(Y, P) == pytest.approx(-np.log(.8) * np.ones(3))
    assert E.row_brier(Y, P) == pytest.approx(np.ones(3) * (.04 + .01 + .01))

def test_perfect_scores():
    s = E.scores(Y, P)
    assert s["macro_f1"] == 1 and s["accuracy"] == 1 and s["auc_up_down"] == 1 and s["n"] == 3

def test_temperature_recovers_scale_and_keeps_argmax():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 4000)
    z = np.eye(3)[y] * 1.0 + rng.normal(0, 1, (4000, 3))
    t = E.fit_temperature(z * 5, y)
    assert t > 2
    assert (E.apply_temperature(z, t).argmax(1) == z.argmax(1)).all()

def test_alpha_shifts_toward_rare_class():
    p = np.tile([.45, .4, .15], (9, 1)); y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert (E.decide(p, 1.0, np.array([.45, .45, .1])) == 2).all()
    assert (E.decide(p, 0.0, np.array([.45, .45, .1])) == 0).all()

def test_paired_diff_sign_and_ci():
    d = np.repeat(np.arange(200), 5)
    m, lo, hi = E.paired_diff(np.full(1000, .9), np.full(1000, 1.0), d)
    assert m == pytest.approx(-.1) and hi < 0
    rng = np.random.default_rng(1)
    m, lo, hi = E.paired_diff(rng.normal(1, .3, 1000), rng.normal(1, .3, 1000), d)
    assert lo < 0 < hi

def test_verdict_three_outcomes():
    assert E.verdict(-.01, -.001, 2, 3) == "Có bằng chứng"
    assert E.verdict(-.01, .001, 3, 3) == "Có tín hiệu, bằng chứng yếu"
    assert E.verdict(-.01, -.001, 1, 3) == "Có tín hiệu, bằng chứng yếu"
    assert E.verdict(.01, .02, 0, 3) == "Chưa chứng minh được"
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_evaluate.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/evaluate.py`**

`decide = argmax(p / prior^alpha)`, sàn prior `1e-6`. `auc_up_down` chỉ trên mẫu thật sự Tăng hoặc Giảm, điểm `p_up / (p_up + p_down)`, NaN nếu thiếu lớp. `fit_temperature` dùng `scipy.optimize.minimize_scalar` có chặn trên `beta = 1/T` trong `[0.01, 10]`. `paired_diff` lấy CI percentile 95%.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/evaluate.py tests/test_mmf_evaluate.py
git commit -m "feat(mmf): metrics, calibration and paired block-bootstrap comparison"
```

---

### Task 6: Baseline

**Files:**
- Create: `mmf/baselines.py`
- Test: `tests/test_mmf_baselines.py`

**Interfaces:**
- Consumes: Task 4 (`frame`), `FR.GROUPS`, `E.PROB_COLS`, `src.splits`.
- Produces: `predict_baseline(name, blocks: dict[str, DataFrame], scaled: DataFrame, frame: DataFrame) -> dict[str, ndarray]` với `blocks` là kết quả `split_samples` chỉ gồm `usable`, trả xác suất `(n, 3)` cho `"calib"` và `"eval"`, cùng thứ tự dòng. Tên hợp lệ là `C.BASELINES`.

Định nghĩa:
- `prior_fixed`: tần suất lớp theo mã trên train, làm trơn Laplace `(n_c + 1) / (n + 3)`.
- `prior_adaptive`: tại phiên t, tần suất trong 120 nhãn gần nhất của mã có `label_end_date <= t`, làm trơn Laplace.
- `logit_*`: `LogisticRegression(C=0.1, max_iter=2000)` trên dòng cuối, `C` cố định.
- `gbm_*`: `HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=59)`.
- Hậu tố `price` là nhóm giá, `text` là nhóm văn bản, `all` là cả 3 nhóm.

- [ ] **Step 1: Viết test thất bại**

```python
import numpy as np, pandas as pd, pytest
from mmf import baselines as B

def toy():
    days = pd.bdate_range("2024-01-01", periods=300)
    f = pd.DataFrame({"symbol": "FPT", "as_of_session": days, "status": "ok", "label": np.resize([0, 1, 2, 2], 300)})
    f["label_end_date"] = f.as_of_session.shift(-5).fillna(days[-1] + pd.Timedelta(days=7))
    return f

def test_prior_fixed_is_laplace_from_train_only():
    f = toy()
    blocks = {"train": f.iloc[:100], "calib": f.iloc[100:200], "eval": f.iloc[200:]}
    p = B.predict_baseline("prior_fixed", blocks, f, f)
    n = np.bincount(f.label.iloc[:100], minlength=3)
    assert p["eval"][0] == pytest.approx((n + 1) / 103)

def test_prior_adaptive_uses_only_matured_labels():
    f = toy()
    blocks = {"train": f.iloc[:100], "calib": f.iloc[100:200], "eval": f.iloc[200:]}
    g = f.copy()
    t = f.as_of_session.iloc[200]
    g.loc[g.as_of_session >= t - pd.Timedelta(days=3), "label"] = 0   # nhãn chưa đến hạn tại t
    a = B.predict_baseline("prior_adaptive", blocks, f, f)["eval"][0]
    b = B.predict_baseline("prior_adaptive", {**blocks, "eval": g.iloc[200:]}, g, g)["eval"][0]
    assert a == pytest.approx(b)
    assert a.sum() == pytest.approx(1)
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_baselines.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/baselines.py`**

Tính `prior_adaptive` từ `frame` theo mã, dùng mọi nhãn `status == "ok"` đã đến hạn, không giới hạn trong khối train, vì đó là thông tin có được lúc chạy thật. Logit và GBM fit trên `blocks["train"]` với các cột trong `scaled`.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_baselines.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/baselines.py tests/test_mmf_baselines.py
git commit -m "feat(mmf): prior, adaptive prior, logistic and GBM baselines"
```

---

### Task 7: Mạng đa phương thức

**Files:**
- Create: `mmf/models.py`
- Test: `tests/test_mmf_models.py`

**Interfaces:**
- Produces: `FusionNet(branches: tuple[str, ...], fusion: str, n_tickers: int, f_price: int, f_text: int, f_fund: int, price_kind: str = "lstm")` với `forward(x_price, x_text, x_fund, ticker) -> logits (B, 3)`; thuộc tính `last_weights` `(B, K)` sau mỗi lượt attention, ngược lại `None`.

Kiến trúc:
- Giá: `lstm` hoặc `gru` 1 tầng `C.D`, lấy trạng thái cuối; `transformer` là chiếu tuyến tính, vị trí học được, 1 tầng `nn.TransformerEncoderLayer(C.D, nhead=4, dim_feedforward=64, dropout=C.DROPOUT, batch_first=True)`, lấy token cuối.
- Văn bản: GRU 1 tầng `C.D`, lấy trạng thái cuối.
- Tài chính: `Linear(f_fund, C.D)`, ReLU, Dropout.
- `concat`: nối các vector nhánh. `attention`: một query học được, `nn.MultiheadAttention(C.D, 1, batch_first=True)` trên các token nhánh. `none`: dùng thẳng một nhánh.
- Đầu ra: nối thêm embedding mã `C.EMB`, Dropout, `Linear(..., 3)`.

- [ ] **Step 1: Viết test thất bại**

```python
import pytest, torch
from mmf import config as C
from mmf.models import FusionNet

def inputs(n=4):
    return torch.randn(n, C.LP, 25), torch.randn(n, C.LT, 14), torch.randn(n, 10), torch.zeros(n, dtype=torch.long)

@pytest.mark.parametrize("kind", C.PRICE_KINDS)
@pytest.mark.parametrize("arm", list(C.ARMS))
def test_every_arm_and_kind_gives_three_logits(arm, kind):
    br, fu = C.ARMS[arm]
    assert FusionNet(br, fu, 15, 25, 14, 10, kind)(*inputs()).shape == (4, 3)

def test_absent_branch_is_never_read():
    br, fu = C.ARMS["no_text"]
    net = FusionNet(br, fu, 15, 25, 14, 10).eval()
    p, t, f, k = inputs()
    assert torch.equal(net(p, t, f, k), net(p, t * 0 + 99, f, k))

def test_attention_weights_sum_to_one():
    br, fu = C.ARMS["fusion_attention"]
    net = FusionNet(br, fu, 15, 25, 14, 10).eval()
    net(*inputs())
    assert net.last_weights.shape == (4, 3)
    assert torch.allclose(net.last_weights.sum(1), torch.ones(4))

def test_last_step_reaches_the_output():
    net = FusionNet(("price",), "none", 15, 25, 14, 10).eval()
    p, t, f, k = inputs()
    q = p.clone(); q[:, -1] += 5
    assert not torch.equal(net(p, t, f, k), net(q, t, f, k))
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_models.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/models.py` theo mô tả trên**

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/models.py tests/test_mmf_models.py
git commit -m "feat(mmf): branch encoders and concat/attention FusionNet"
```

---

### Task 8: Huấn luyện

**Files:**
- Create: `mmf/train.py`
- Test: `tests/test_mmf_train.py`

**Interfaces:**
- Consumes: `FusionNet`, `Samples`, `E.fit_temperature`.
- Produces: `fit_net(make_net: Callable[[], FusionNet], train: Samples, tune: Samples, seed: int, device: str, max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE) -> dict` (khóa `net` đã nạp trọng số tốt nhất, `best_epoch`, `tune_nll`); `predict_logits(net, s: Samples, device, batch=1024) -> ndarray (N, 3)`.

Dùng AdamW `lr=C.LR, weight_decay=C.WD`, `CrossEntropyLoss` không trọng số, batch `C.BATCH`, clip norm gradient 1.0. Early stopping trên NLL của tune, trả trọng số tốt nhất.

- [ ] **Step 1: Viết test thất bại**

```python
import numpy as np, torch
from mmf import config as C, train as TR
from mmf.models import FusionNet
from mmf.windows import Samples

def toy(n, seed, signal=True):
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n)
    p = r.normal(0, 1, (n, C.LP, 25)).astype("float32")
    if signal:
        p[:, -1, 0] += (y - 1) * 3
    return Samples(p, np.zeros((n, C.LT, 14), "float32"), np.zeros((n, 10), "float32"), np.zeros(n, "int64"),
                   y.astype("int64"), np.array(["FPT"] * n), np.arange(n), np.arange(n))

def net():
    return FusionNet(("price",), "none", 15, 25, 14, 10, "gru")

def test_learns_planted_signal_better_than_prior():
    tr, tu = toy(600, 0), toy(200, 1)
    out = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=15)
    assert out["tune_nll"] < 0.8 * np.log(3)

def test_same_seed_reproduces_on_cpu():
    tr, tu = toy(200, 0), toy(50, 1)
    x = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=3)["tune_nll"]
    y = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=3)["tune_nll"]
    assert x == y

def test_returns_best_epoch_not_last():
    tr, tu = toy(60, 0, signal=False), toy(60, 1, signal=False)
    out = TR.fit_net(net, tr, tu, 59, "cpu", max_epochs=30, patience=30)
    z = TR.predict_logits(out["net"], tu, "cpu")
    nll = torch.nn.functional.cross_entropy(torch.tensor(z), torch.tensor(tu.y)).item()
    assert abs(nll - out["tune_nll"]) < 1e-5
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_train.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/train.py`**

`fit_net` gọi `torch.manual_seed(seed)` trước `make_net()`, nên khởi tạo cũng theo seed; loader dùng `torch.Generator().manual_seed(seed)`. Không dùng `torch.use_deterministic_algorithms` trên CUDA. Test tái lập chỉ áp dụng cho CPU.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_train.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/train.py tests/test_mmf_train.py
git commit -m "feat(mmf): training loop with early stopping on tune NLL"
```

---

### Task 9: Bộ chạy thí nghiệm

**Files:**
- Create: `mmf/experiment.py`
- Test: `tests/test_mmf_experiment.py`

**Interfaces:**
- Consumes: mọi task trước, `src.splits.folds_from_config`, `split_samples`, `src.data.build_panel`.
- Produces: `fold_bundle(panel, h, fold) -> Bundle` (frame, scaled, blocks, samples theo khối); `run_arm(bundle, arm, seed, price_kind, out_path, device)`; `select_price_kind(bundle, run_dir, device) -> str`, ghi `select.json`; `calibrate(bundle, z_calib, z_eval) -> (t, alpha)`; `shuffle_train_labels(bundle, seed=0) -> Bundle`; `run_all(panel, horizons, folds, run_dir, device, arms=None, seeds=C.SEEDS, build=fold_bundle, max_epochs=C.MAX_EPOCHS)`; CLI `python -m mmf.experiment [--h 2 5] [--folds A B C] [--device cuda] [--stock-dir PATH] [--smoke] [--shuffle-labels]`.

Bố cục: `C.DATA/runs/{dataset_id[:16]}/h{h}/{fold}/{arm}/seed{seed}.parquet`, baseline ghi `.../{name}/seed0.parquet`. Arm học sâu ghi thêm trọng số `seed{seed}.pt` cạnh đó để Task 12 nạp lại. Mỗi parquet gồm `symbol, as_of_session, label_end_date, y_true, block ("calib"|"eval"), p_down, p_flat, p_up, prior_down, prior_flat, prior_up, alpha`, và metadata arm, h, fold, seed, price_kind, nlp_version, dataset_id, git_commit. Cột `prior_*` là tần suất lớp của mã trên train. Ghi nguyên tử (file tạm rồi `os.replace`), file đã có thì bỏ qua. Nhiệt độ và alpha fit trên calib, áp cho cả hai khối. Mọi arm trong một (h, fold) phải có đúng cùng tập khóa `(symbol, as_of_session)` ở eval.

- [ ] **Step 1: Viết test thất bại**

Test dùng `build=` bơm Bundle nhỏ dựng từ dữ liệu giả 3 mã, `max_epochs=1`, một fold, một seed.

```python
from mmf import experiment as X

def run(tmp_path, b):
    X.run_all(None, (5,), [b.fold], tmp_path, "cpu", seeds=(59,), build=lambda *a: b, max_epochs=1)

def test_all_arms_write_same_eval_keys(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    files = list(tmp_path.glob("h5/*/*/seed*.parquet"))
    keys = {f.parent.name: _eval_keys(f) for f in files}
    first = next(iter(keys.values()))
    assert set(keys) >= set(C.ARMS) | set(C.BASELINES)
    assert all(k == first for k in keys.values())

def test_probabilities_are_valid(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    d = pd.read_parquet(next(tmp_path.glob("h5/*/fusion_attention/seed59.parquet")))
    p = d[E.PROB_COLS].to_numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1)

def test_rerun_skips_existing_files(tmp_path, tiny_bundle):
    run(tmp_path, tiny_bundle)
    before = {f: f.stat().st_mtime_ns for f in tmp_path.glob("h5/*/*/seed*.parquet")}
    run(tmp_path, tiny_bundle)
    assert before == {f: f.stat().st_mtime_ns for f in tmp_path.glob("h5/*/*/seed*.parquet")}

def test_eval_labels_do_not_change_calibration(tiny_bundle):
    a = X.calibrate(tiny_bundle, *_logits(tiny_bundle))
    tiny_bundle.samples["eval"].y[:] = 0
    assert a == X.calibrate(tiny_bundle, *_logits(tiny_bundle))

def test_shuffle_changes_only_train_labels(tiny_bundle):
    s = X.shuffle_train_labels(tiny_bundle)
    assert not np.array_equal(s.samples["train"].y, tiny_bundle.samples["train"].y)
    for b in ("tune", "calib", "eval"):
        np.testing.assert_array_equal(s.samples[b].y, tiny_bundle.samples[b].y)
```

`tiny_bundle` là fixture phạm vi function trong cùng file: 3 mã, 400 phiên giả, đủ cột `FR.GROUPS`, khối train/tune/calib/eval lần lượt 200/60/60/60 phiên, `fold` là `Fold("A", ...)`. `_eval_keys` đọc `block == "eval"` và trả tập khóa, `_logits` trả logits cố định theo seed cho calib và eval. `shuffle_train_labels` hoán vị nhãn giữa các ngày, cùng một phép hoán vị cho mọi mã, để giữ tương quan giữa các mã.

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_experiment.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/experiment.py`**

`select_price_kind`: với `SELECT_SEEDS`, huấn luyện `dl_price` cho mỗi kiểu trong `C.PRICE_KINDS`, chọn NLL tune trung bình thấp nhất, hòa thì lấy kiểu đầu danh sách. Mọi arm có nhánh giá trong (h, fold) đó dùng kiểu đã chọn. `run_all` chạy baseline trước, sau đó `select_price_kind`, rồi từng arm và seed. Khi `dataset_id is None` thì từ chối chạy, trừ khi có Bundle bơm vào. `--smoke` đặt `max_epochs=1`, `seeds=(59,)`, một fold, ghi vào `C.DATA/smoke`. `--shuffle-labels` đặt `seeds=(59,)`, fold C, giữ số epoch bình thường, áp `shuffle_train_labels`, ghi vào `C.DATA/shuffle`.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_experiment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/experiment.py tests/test_mmf_experiment.py
git commit -m "feat(mmf): resumable experiment runner over arms, folds and seeds"
```

---

### Task 10: So sánh, stacking, verdict và khóa cấu hình

**Files:**
- Create: `mmf/compare.py`
- Test: `tests/test_mmf_compare.py`

**Interfaces:**
- Consumes: file do Task 9 ghi, `E.paired_diff`, `E.verdict`, `E.scores`.
- Produces: `load_h(run_dir, h) -> DataFrame` (ensemble: trung bình xác suất qua seed theo khóa, thêm cột `arm, fold, block`); `stack(preds, fold) -> DataFrame` (arm `stack`: `LogisticRegression(C=0.1)` trên xác suất calib của `dl_price, dl_fund, dl_text`, dự đoán eval); `pick(preds, fold, candidates) -> str` theo log-loss trên calib; `compare(preds, a_by_fold: dict, b_by_fold: dict) -> dict` (`mean, lo, hi, fold_wins, verdict`); `tables(preds) -> dict[str, DataFrame]` (`arms`, `per_symbol`, `r1`, `r2`, `r3`, `leak_alarms`); `write_report(run_dir, out=C.REPORTS/"mmf_dev_report.md")`; `write_lock(run_dir, out=C.REPORTS/"mmf_lock.json")`; CLI `python -m mmf.compare --run-dir PATH [--lock]`.

Nhóm ứng viên: baseline `C.BASELINES`; nhánh đơn `dl_price, dl_fund, dl_text`; fusion `fusion_concat, fusion_attention, stack`. R3 so `fusion_concat` với `no_price`, `no_fund`, `no_text`.

File khóa ghi cho mỗi h: `fusion` (arm được chọn nhiều fold nhất, hòa thì log-loss calib trung bình thấp nhất), `baseline`, `unimodal`, `price_kind`, `dataset_id`, `git_commit`, `created_at`.

- [ ] **Step 1: Viết test thất bại**

```python
def fake(arm, fold, offset, n=300):
    """Preds giả: offset càng lớn càng chắc đúng, cùng khóa cho mọi arm."""

def test_pick_uses_calib_only():
    p = pd.concat([fake("a", "A", 2.0).query("block == 'calib'"), fake("a", "A", 0.0).query("block == 'eval'"),
                   fake("b", "A", 0.0).query("block == 'calib'"), fake("b", "A", 2.0).query("block == 'eval'")])
    assert C.pick(p, "A", ["a", "b"]) == "a"

def test_clear_winner_gets_evidence_verdict():
    p = pd.concat([fake("good", f, 2.0) for f in "ABC"] + [fake("bad", f, 0.0) for f in "ABC"])
    r = C.compare(p, dict.fromkeys("ABC", "good"), dict.fromkeys("ABC", "bad"))
    assert r["verdict"] == "Có bằng chứng" and r["fold_wins"] == 3 and r["hi"] < 0

def test_stack_trains_on_calib_never_eval():
    p = pd.concat([fake(a, "A", 1.0) for a in ("dl_price", "dl_fund", "dl_text")])
    a = C.stack(p, "A")
    q = p.copy(); q.loc[q.block == "eval", "y_true"] = 0
    b = C.stack(q, "A")
    np.testing.assert_array_equal(a[E.PROB_COLS].to_numpy(), b[E.PROB_COLS].to_numpy())

def test_leak_alarm_fires_on_suspicious_score():
    t = C.tables(pd.concat([fake(a, f, 8.0) for a in ALL_ARMS for f in "ABC"]))
    assert len(t["leak_alarms"]) > 0

def test_lock_refuses_without_every_fold(tmp_path):
    with pytest.raises(ValueError):
        C.write_lock(_run_dir_with_only_fold_A(tmp_path), tmp_path / "lock.json")
```

`fake` dựng logits từ nhãn ngẫu nhiên cố định seed, cộng `offset` vào lớp đúng, softmax, cho 2 khối calib và eval, có đủ cột `prior_*`. `ALL_ARMS` lấy từ `mmf.config.ARMS` và `mmf.config.BASELINES`. `_run_dir_with_only_fold_A` ghi các parquet của `fake` vào đúng bố cục Task 9, chỉ fold A. Test này import `from mmf import compare as C, evaluate as E`.

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_compare.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/compare.py`**

`compare` gộp eval của 3 fold, ghép cặp theo khóa, dùng `row_logloss` và ngày `as_of_session`. `fold_wins` đếm fold có log-loss trung bình thấp hơn. Macro-F1 của ensemble dùng alpha fit lại trên calib của xác suất ensemble với `prior_*`, không dùng alpha từng seed. Báo cáo Markdown tiếng Việt gồm: bảng mọi arm theo (h, fold), bảng R1, R2, R3 kèm verdict, bảng theo mã, cảnh báo rò rỉ, và dòng ghi rõ "vùng phát triển, chưa phải kết quả cuối".

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_compare.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmf/compare.py tests/test_mmf_compare.py
git commit -m "feat(mmf): stacking, pre-registered comparisons, dev report and lock file"
```

---

### Task 11: Chạy thật trên vùng phát triển

**Files:**
- Create: `docs/reports/mmf_dev_report.md`, `docs/reports/mmf_lock.json` (sinh tự động)
- Modify: không sửa code; lỗi thật phát hiện ở đây sửa bằng test mới trong module liên quan.

- [ ] **Step 1: Đóng băng dữ liệu**

`stock/manifest.json` chưa có vì sync bản mới chưa chạy sau khi merge; cron 15:20 ngày thường tiếp theo sẽ tạo nó. Nếu vẫn chưa có, chạy đúng lệnh của cron. Sau đó đóng băng:

```bash
cd Pipeline_V1.2 && VNSTOCK_TELEMETRY=0 VNSTOCK_DISABLE_AGENT_SETUP=1 VNSTOCK_AGENT_TARGETS=none ../NLP_Module/.venv/bin/python sync_stock_parquet.py && ../NLP_Module/.venv/bin/python -m src.snapshot freeze
```
Expected: có `stock/manifest.json` và in đường dẫn `stock/snapshots/<id>`. Ghi lại đường dẫn đó làm `SNAP`.

- [ ] **Step 2: Chạy thử nhanh**

Run: `.venv_mmf/bin/python -m mmf.experiment --smoke --stock-dir Pipeline_V1.2/$SNAP --device cuda`
Expected: xong trong vài phút, có đủ file cho mọi arm trong `data/mmf/smoke`.

- [ ] **Step 3: Kiểm tra rò rỉ bằng nhãn xáo trộn**

Run: `.venv_mmf/bin/python -m mmf.experiment --shuffle-labels --h 5 --stock-dir Pipeline_V1.2/$SNAP --device cuda`, rồi `.venv_mmf/bin/python -m mmf.compare --run-dir data/mmf/shuffle/<id>`
Expected: mọi arm học sâu có `auc_up_down` trong 0.5 ± 0.05 và log-loss không thấp hơn `prior_fixed` quá 0.005. Nếu vượt, dừng và tìm rò rỉ trước khi đi tiếp. Đây là bước kiểm tra, không commit báo cáo của nó.

- [ ] **Step 4: Chạy đầy đủ ở nền**

Run: `nohup .venv_mmf/bin/python -m mmf.experiment --h 2 5 --folds A B C --stock-dir Pipeline_V1.2/$SNAP --device cuda > data/mmf/dev.log 2>&1 &`
Expected: chạy được tiếp nếu bị ngắt, đủ `8 arm x 5 seed + 8 baseline` cho mỗi (h, fold). Theo dõi bằng số file.

- [ ] **Step 5: Tạo báo cáo vùng phát triển**

Run: `.venv_mmf/bin/python -m mmf.compare --run-dir data/mmf/runs/<id>`
Expected: `docs/reports/mmf_dev_report.md`. Nếu có cảnh báo rò rỉ thì điều tra trước. Ghi kết quả kèm `git_commit`, không tinh chỉnh chỉ để cải thiện số ở eval.

- [ ] **Step 6: Khóa cấu hình và commit**

```bash
.venv_mmf/bin/python -m mmf.compare --run-dir data/mmf/runs/<id> --lock
git add docs/reports/mmf_dev_report.md docs/reports/mmf_lock.json
git commit -m "exp(mmf): development results and locked configuration"
```

---

### Task 12: Giải thích và khuyến nghị

**Files:**
- Create: `mmf/explain.py`, `mmf/recommend.py`
- Test: `tests/test_mmf_explain.py`, `tests/test_mmf_recommend.py`

**Interfaces:**
- Consumes: `Bundle`, `FusionNet`, file khóa.
- Produces:
  - `explain.group_permutation(predict_fn, samples, groups=("price", "fund", "text"), seed=0) -> DataFrame` (group, delta_logloss, lo, hi): hoán vị vector một nhóm giữa các mã **trong cùng phiên**, qua `explain._shuffle_group(samples, group, rng) -> Samples`.
  - `explain.shap_by_group(net, background: Samples, samples: Samples, device="cpu") -> DataFrame` (group, class, mean_abs_shap, share) bằng `shap.GradientExplainer` trên CPU, tối đa 200 mẫu nền và 500 mẫu.
  - `explain.lime_cases(gbm, X, cols, rows) -> list[dict]`: `lime.lime_tabular.LimeTabularExplainer`, cho arm `gbm_all`.
  - `recommend.RULES = ("Ưu tiên mua", "Tránh", "Không đủ tin cậy")`; `recommend.pick_margin(p_up, p_down, net_ret, grid=np.arange(0, 0.51, 0.02), min_n=50) -> float | None`, chọn trên calib; `recommend.label(p_up, p_down, m) -> ndarray`; `recommend.trade_return(frame, h, cost=AC.COST) -> Series`: `log(close[t+h+1] / open[t+1]) - cost`; `recommend.simulate(rows, m, h, seed=0) -> dict` gồm `coverage, n, n_days, mean_net, rand_median, ci_lo, ci_hi`.

- [ ] **Step 1: Viết test thất bại**

```python
# test_mmf_explain.py
def test_permutation_detects_informative_group(toy_samples):
    fn = lambda s: _probs_from_fund_only(s)     # mô hình giả chỉ đọc nhóm fund
    t = X.group_permutation(fn, toy_samples).set_index("group")
    assert t.loc["fund", "delta_logloss"] > 0.05
    assert t.loc["text", "delta_logloss"] == pytest.approx(0, abs=1e-9)

def test_permutation_stays_within_same_session(toy_samples):
    perm = X._shuffle_group(toy_samples, "fund", np.random.default_rng(0))
    for d in np.unique(toy_samples.as_of):
        k = toy_samples.as_of == d
        assert sorted(map(tuple, perm.x_fund[k])) == sorted(map(tuple, toy_samples.x_fund[k]))

def test_shap_shares_sum_to_one_per_class(toy_net, toy_samples):
    t = X.shap_by_group(toy_net, toy_samples, toy_samples)
    assert np.allclose(t.groupby("class")["share"].sum(), 1)

# test_mmf_recommend.py
def test_trade_return_uses_next_open_and_exit_close():
    f = _toy_prices()   # open = 10..., close = 20...
    r = R.trade_return(f, 2, cost=0.0045)
    assert r.iloc[0] == pytest.approx(np.log(f.close.iloc[3] / f.open.iloc[1]) - 0.0045)
    assert r.iloc[-1:].isna().all()

def test_margin_needs_min_trades():
    assert R.pick_margin(np.array([.9] * 10), np.array([.1] * 10), np.ones(10), min_n=50) is None

def test_labels_three_way():
    assert list(R.label(np.array([.6, .1, .4]), np.array([.2, .6, .35]), 0.2)) == ["Ưu tiên mua", "Tránh", "Không đủ tin cậy"]

def test_simulate_reports_coverage_with_result():
    out = R.simulate(_toy_rows(), 0.1, 2)
    assert {"coverage", "n", "n_days", "mean_net", "rand_median", "ci_lo", "ci_hi"} <= set(out)
```

Fixture: `toy_samples` là `Samples` giả 5 mã, 40 phiên, nhãn phụ thuộc `x_fund[:, 0]`. `_probs_from_fund_only` là softmax từ `x_fund[:, 0]`. `toy_net` là `FusionNet` đủ 3 nhánh chưa huấn luyện. `_toy_prices` có `open`, `close` tăng dần cho một mã. `_toy_rows` có `symbol, as_of_session, p_up, p_down, net_ret`.

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_explain.py tests/test_mmf_recommend.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết hai module**

Chạy `GradientExplainer` ở chế độ train cho tầng RNN nếu backend đòi, trên CPU để tránh lỗi backward của cuDNN. Gộp `|SHAP|` theo nhóm bằng tổng qua thời gian và đặc trưng, rồi chia tổng ba nhóm. `simulate` chỉ tính LONG, CI dùng `block_bootstrap_dates`, `rand_median` là trung vị của 1000 lần rút ngẫu nhiên cùng số lệnh. Lợi suất tính theo giả định T+2 và không bán khống, ghi rõ giả định này trong báo cáo vì chủ dự án chưa xác nhận.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_explain.py tests/test_mmf_recommend.py -q`
Expected: PASS.

- [ ] **Step 5: Thêm phần giải thích và khuyến nghị vào báo cáo vùng phát triển**

Thêm CLI `python -m mmf.explain --run-dir PATH`. Với mỗi (h, fold), nó nạp trọng số `seed59.pt` của arm fusion trong file khóa, refit `gbm_all` cho LIME, và chọn ngưỡng `m` trên calib từ xác suất ensemble. Lệnh ghi `docs/reports/mmf_explain.md` và hình vào `docs/reports/mmf_figures/`: bảng permutation, tỷ trọng SHAP theo nhóm và lớp, 3 ca LIME, reliability diagram, ma trận nhầm lẫn, kết quả mô phỏng khuyến nghị kèm độ phủ.

Run: `.venv_mmf/bin/python -m mmf.explain --run-dir data/mmf/runs/<id>`
Expected: có báo cáo và hình cho cả T+2, T+5.

- [ ] **Step 6: Commit**

```bash
git add mmf/explain.py mmf/recommend.py tests/test_mmf_explain.py tests/test_mmf_recommend.py docs/reports/mmf_explain.md docs/reports/mmf_figures
git commit -m "feat(mmf): group permutation, SHAP, LIME and recommendation rule"
```

---

### Task 13: Chấm vùng cuối một lần

**Files:**
- Create: `mmf/holdout.py`
- Test: `tests/test_mmf_holdout.py`

**Interfaces:**
- Consumes: `docs/reports/mmf_lock.json`, `experiment.fold_bundle`, `C.FINAL_FOLD`, `compare.compare`, `recommend.simulate`.
- Produces: `holdout_rows(frame) -> DataFrame` (dòng `usable`, `as_of_session > C.HOLDOUT_AFTER`, `label_end_date <= phiên cuối`); `check_ready(rows, h, guard_dir, early=False) -> None` (raise `ValueError` chứa "đã chấm" nếu đã chạy, chứa "phiên" nếu chưa đủ phiên); `LOCK_FILE = C.REPORTS / "mmf_lock.json"`; `load_lock() -> dict` (raise `ValueError` chứa "khóa" nếu thiếu file); `run(h, panel, guard_dir=C.REPORTS, early=False, device="cuda")`; CLI `python -m mmf.holdout --h 2 5 [--early]`; file `docs/reports/mmf_holdout_h{h}.md` và `docs/reports/mmf_holdout_h{h}.lock`.

Huấn luyện với fold `F`: train đến 2026-05-04, tune đến 2026-07-30, calib đến 2026-10-30, điểm cảm xúc `phobert-phobert-v1`. Chạy các arm trong file khóa, cùng `fusion_concat`, `no_price`, `no_fund`, `no_text` cho R3, mỗi arm đủ 5 seed. So sánh R1, R2, R3 không yêu cầu 2/3 fold vì chỉ có một vùng, verdict dùng `fold_wins=1, n_folds=1`.

- [ ] **Step 1: Viết test thất bại**

```python
def frame_with(n_days):
    """Frame giả, 15 mã: 10 phiên trước HOLDOUT_AFTER, n_days phiên đã có nhãn sau đó, và 5 phiên cuối
    status='not_matured', label_end_date=NaT. Cột usable = status == 'ok'."""

def test_refuses_below_minimum_sessions(tmp_path):
    with pytest.raises(ValueError, match="phiên"):
        H.check_ready(H.holdout_rows(frame_with(39)), 5, tmp_path)
    H.check_ready(H.holdout_rows(frame_with(40)), 5, tmp_path)

def test_early_flag_lowers_minimum_only_to_twenty(tmp_path):
    H.check_ready(H.holdout_rows(frame_with(20)), 5, tmp_path, early=True)
    with pytest.raises(ValueError):
        H.check_ready(H.holdout_rows(frame_with(19)), 5, tmp_path, early=True)

def test_second_run_is_refused(tmp_path):
    (tmp_path / "mmf_holdout_h5.lock").write_text("{}")
    with pytest.raises(ValueError, match="đã chấm"):
        H.check_ready(H.holdout_rows(frame_with(50)), 5, tmp_path)

def test_rows_exclude_dev_and_unmatured():
    f = frame_with(50)
    r = H.holdout_rows(f)
    assert r.as_of_session.min() > pd.Timestamp("2026-10-30")
    assert r.label_end_date.notna().all() and r.status.eq("ok").all()
    assert r.as_of_session.nunique() == f[f.status == "ok"].as_of_session.nunique() - f[f.as_of_session <= "2026-10-30"].as_of_session.nunique()

def test_refuses_without_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "LOCK_FILE", tmp_path / "nope.json")
    with pytest.raises(ValueError, match="khóa"):
        H.load_lock()
```

- [ ] **Step 2: Chạy test để thấy thất bại**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_holdout.py -q`
Expected: FAIL, thiếu module.

- [ ] **Step 3: Viết `mmf/holdout.py`**

`run` gọi `check_ready` trước mọi việc. Tạo file `.lock` chứa `dataset_id`, `git_commit`, `n_sessions`, `early`, thời điểm chạy **ngay trước khi tính chỉ số trên vùng cuối**, để một lần chạy lỗi giữa chừng cũng không được chạy lại mà không có quyết định của chủ dự án. Báo cáo Markdown tiếng Việt, ghi rõ số phiên, số mẫu, cờ `early`, và giới hạn trong spec mục 6.

- [ ] **Step 4: Chạy test**

Run: `.venv_mmf/bin/python -m pytest tests/test_mmf_holdout.py -q`
Expected: PASS.

- [ ] **Step 5: Chạy toàn bộ test của repo**

Run: `.venv_mmf/bin/python -m pytest tests -q` và `cd Pipeline_V1.2 && ../.venv_mmf/bin/python -m pytest -q`
Expected: toàn bộ PASS, kể cả 42 test `ai_core` và 81 test Pipeline.

- [ ] **Step 6: Commit code**

```bash
git add mmf/holdout.py tests/test_mmf_holdout.py
git commit -m "feat(mmf): single-use final evaluation on sessions after 2026-10-30"
```

- [ ] **Step 7: Việc của chủ dự án, không tự chạy**

Khi đủ 40 phiên có nhãn (khoảng cuối 12/2026 đến đầu 01/2027), hoặc 20 phiên nếu cần gấp, chủ dự án quyết định chạy `.venv_mmf/bin/python -m mmf.holdout --h 2 5`. Lệnh này chỉ chạy được một lần cho mỗi h. Thực thi plan dừng ở Step 6 và báo lại mốc này.
