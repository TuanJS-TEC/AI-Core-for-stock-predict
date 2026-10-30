# Hệ thống quyết định đa phương thức — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Sửa dữ liệu giá, mở rộng ra 20 mã, chạy thí nghiệm hướng giá có quy tắc GO/NO-GO đặt trước, vá các lỗi NLP
đã biết, và dựng màn hình quyết định chạy shadow live (thiết kế Bước 0–3 và Bước 5).

**Kiến trúc:**
- `Pipeline_V1.2/stock` là nguồn giá duy nhất, được `sync_stock_parquet.py` giữ đúng một chuẩn điều chỉnh.
- `ai_core/` dùng lại `volcast.add_features`, `walk_forward._split` và `src.metrics.block_bootstrap_dates`. Nó thêm
  phần nhãn và thí nghiệm hướng (`direction.py`) và phần ghi quyết định live (`live.py`).
- NLP được vá tại chỗ (`entities.py`, `contract.py`, `risk.py` mới).
- **Bước 4 (RuleGate) không nằm trong kế hoạch này.** Nó chỉ được lập kế hoạch riêng nếu Task 10 cho kết quả GO.

**Tech stack:** Python 3.14 (`NLP_Module/.venv`), pandas, scikit-learn 1.9, pytest, SQLite, Streamlit, vnstock 4.0.9.

**Spec:** `docs/superpowers/specs/2026-10-29-multimodal-decision-system-design.md`

## Global Constraints

- Mốc chốt dữ liệu: 15:00 giờ Việt Nam (08:00 UTC). Dùng lại `trial_forecast.CLOSE_UTC`, không định nghĩa lại.
- Ngưỡng biên độ ngày: `RET_BAND = 0.072` (`volcast.RET_BAND`).
- Chi phí khứ hồi `COST = 0.0045`, dùng cho cả sàn của τ lẫn phần trừ mỗi lệnh. `KAPPA = 0.5`. Cả hai cố định.
- Cửa sổ test đóng băng: fold C kết thúc ở `TEST_END = "2026-10-30"`. `walk_forward.FOLDS` không đổi.
- Backtest dùng kết quả sentiment `phobert-fold{A,B,C}` (đã chấm đủ 27 mã). Live dùng `baseline-tfidf-lr-v1`.
- Không sửa `Pipeline_V1.2/src/{data,models,pipeline}.py`, và không train lại Pipeline_V1.2.
- Mọi quyết định ship/GO phải có quy tắc ghi ra trước khi chạy và được ghi vào báo cáo.
- Chạy vnstock luôn kèm `VNSTOCK_TELEMETRY=0 VNSTOCK_DISABLE_AGENT_SETUP=1 VNSTOCK_AGENT_TARGETS=none`.
- Với symbol trong `NLP_Module`: chạy `gitnexus_impact` trước khi sửa và `gitnexus_detect_changes` trước khi commit
  (`NLP_Module/CLAUDE.md`). Nếu công cụ không có, ghi rõ trong commit message.
- Lệnh chạy:
  - Python: `NLP_Module/.venv/bin/python` (gọi tắt là `$PY` trong các bước bên dưới).
  - Test AI_Core: chạy từ `AI_Core/` bằng `$PY -m pytest tests`.
  - Test NLP: chạy từ `NLP_Module/` bằng `.venv/bin/python -m pytest tests`.
  - Test Pipeline: chạy từ `Pipeline_V1.2/` bằng `$PY -m pytest tests`.

## Review Focus

1. **Nguồn điều chỉnh ex-date trễ một ngày.** Phiên chồng lấn vẫn khớp, nhưng phiên mới gãy so với lịch sử. Sync
   phải bắt được qua bước nhảy > `RET_BAND` sau khi ghép, rồi tải lại toàn bộ (Task 1,
   `test_big_move_after_merge_triggers_full_refetch`).
2. **vnstock trả rỗng hoặc lỗi.** Sync giữ nguyên dữ liệu và không tải lại toàn bộ. Live đánh dấu dữ liệu cũ
   (stale) (Task 1 `test_empty_fetch_keeps_data`, Task 15 `test_status_missing_session`).
3. **Ngày nghỉ đã khai báo trong `MARKET_HOLIDAYS`** không bị coi là thiếu phiên. Ngày thường không có dữ liệu thì bị
   coi là thiếu (Task 15 `test_status_declared_holiday_ok`).
4. **Mã không có tin nào** (GAS, nhiều ngày): snapshot vẫn có `risk_flag_72h = 0` và `has_news = False`, không lỗi
   (Task 12 `test_risk_flag_zero_without_news`).
5. **Dòng khởi động thiếu `vol20`** (khoảng 20 phiên đầu) hoặc thiếu giá tương lai: nhãn NaN và bị loại, không bị
   gán NONE (Task 8 `test_label_nan_without_vol_or_future`).

---

### Task 0: Quản lý phiên bản cho AI_Core

**Files:**
- Create: `AI_Core/.gitignore`

- [ ] **Step 1:** Tạo `.gitignore` với nội dung `NLP_Module/` (đã có repo riêng), `Pipeline_V1.2/stock/`,
  `Pipeline_V1.2/outputs/`, `.pytest_cache/`, `__pycache__/`.
- [ ] **Step 2:** `cd AI_Core && git init && git add -A && git commit -m "chore: track ai_core, tests, docs, Pipeline_V1.2 code"`.
  Kiểm tra: `git status` sạch, và `git ls-files | grep -c NLP_Module` ra 0.

Quy ước commit trong các task sau: file trong `NLP_Module/` commit ở repo `NLP_Module`; file còn lại commit ở repo `AI_Core`.

---

## Bước 0 — Vết nối giá

### Task 1: Sync tải lại toàn bộ khi chuẩn điều chỉnh thay đổi

Đã xác nhận ngày 2026-10-29: vnstock trả chuỗi đã điều chỉnh. TCB tải mới cho ra 22.38 ở ngày 27/10, bản lưu là 33.55.

**Files:**
- Modify: `Pipeline_V1.2/src/config.py` (mục 9)
- Modify: `Pipeline_V1.2/sync_stock_parquet.py` (`process_stock_file`, thêm `basis_changed`)
- Test: `Pipeline_V1.2/tests/test_sync.py`

**Interfaces:**
- Produces:
  - `C.SYNC_BASIS_TOL = 0.005` (nghìn đồng; giá có 2 chữ số thập phân).
  - `C.SYNC_RET_BAND = 0.072`.
  - `basis_changed(stored: pd.DataFrame, merged: pd.DataFrame) -> bool`: cả hai đầu vào đã qua `normalize_sessions`.

- [ ] **Step 1: Viết test hỏng.** Dùng `frame()` và `monkeypatch` có sẵn, thay `S.fetch_latest_ohlcv` bằng một hàm ghi
  lại `start_date` mỗi lần gọi.

```python
def test_overlap_mismatch_triggers_full_refetch(tmp_path, monkeypatch):
    csv = tmp_path / "AAA_stock.csv"
    frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00"], [33.8, 33.55]).to_csv(csv, index=False)
    full = frame(["2026-10-26 07:00:00", "2026-10-27 07:00:00", "2026-10-28 07:00:00"], [22.55, 22.38, 22.35])
    calls = []
    def fake(symbol, start_date, end_date, max_retries=3):
        calls.append(start_date)
        return full[full.time >= start_date].reset_index(drop=True)
    monkeypatch.setattr(S, "fetch_latest_ohlcv", fake)
    S.process_stock_file(str(csv), str(tmp_path), delay_between_calls=0)
    pq = pd.read_parquet(tmp_path / C.STOCK_FILE.format(ticker="AAA"))
    assert calls == ["2026-10-27", "2026-10-26"]                  # incremental, then full from the first stored day
    assert list(pq.close) == [22.55, 22.38, 22.35]

def test_big_move_after_merge_triggers_full_refetch(...):  # overlap 2026-10-27 matches (33.55) but 10-28 = 22.35
    ...  # assert len(calls) == 2 and the stored history is the fetched full series

def test_consistent_update_fetches_once(...):              # overlap matches, no big move
    ...  # assert len(calls) == 1

def test_old_gap_does_not_refetch(...):                    # stored history already holds a >RET_BAND jump (missing sessions)
    ...  # overlap matches, new rows are normal: assert len(calls) == 1

def test_empty_fetch_keeps_data(...):                      # fake returns pd.DataFrame()
    ...  # assert len(calls) == 1 and parquet closes == stored closes
```

- [ ] **Step 2:** `cd Pipeline_V1.2 && $PY -m pytest tests/test_sync.py -v`. Kỳ vọng 3 test mới FAIL;
  `test_empty_fetch_keeps_data` và `test_old_gap_does_not_refetch` có thể đã PASS.
- [ ] **Step 3:** Cài đặt `basis_changed`. Trả True khi một trong hai điều kiện đúng:
  - Một phiên có trong `stored` mà giá đóng cửa trong `merged` lệch quá `SYNC_BASIS_TOL`.
  - Bước nhảy `|pct_change| > SYNC_RET_BAND` của `merged.close` xảy ra giữa hai phiên mà phiên sau **mới hơn
    `stored.time.max()`** (tính từ phiên cuối của `stored` trở đi). Bước nhảy đã nằm sẵn trong lịch sử (phiên bị
    thiếu thật như 22–23/10) không được tính lại mỗi ngày; một khoảng trống hợp lệ trong phiên mới chỉ gây đúng một
    lần tải lại, rồi nằm trong `stored` từ hôm sau.
    <!-- ponytail: không phân biệt được bước nhảy do phiên thiếu với corporate action trong lần chạy đầu tiên; chấp nhận một lần tải lại. -->

  Trong `process_stock_file`, sau khi ghép: nếu `basis_changed`, gọi lại `fetch_latest_ohlcv(symbol,
  df_csv.time.min() as "%Y-%m-%d", end_date_str)`. Nếu kết quả không rỗng thì nó thay toàn bộ `df_merged`, kèm log
  `logger.warning("[SYM] chuẩn điều chỉnh đổi, tải lại toàn bộ lịch sử")`. Nếu sau khi tải lại vẫn còn bước nhảy thì
  `logger.error(...)` và vẫn ghi.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Kỳ vọng tất cả PASS, kể cả các test cũ.
- [ ] **Step 5:** Commit vào repo AI_Core: `fix(sync): refetch full history when the adjustment basis changes`.

### Task 2: Sửa dữ liệu TCB và chặn vết nối bằng test

**Files:**
- Modify: `tests/test_alignment.py`
- Modify: `Pipeline_V1.2/sync_stock_parquet.py` (`--full-refresh`, `drop_open_session`)
- Test: `Pipeline_V1.2/tests/test_sync.py`

**Bổ sung sau khi Task 2 bị chặn (2026-10-30):** vết nối TCB nằm sẵn trong dữ liệu đã lưu (27/10 = 33.55 cũ, 28/10 và
29/10 đã theo chuẩn mới), còn lần fetch tăng dần bắt đầu từ 29/10 và khớp, nên `basis_changed` của Task 1 không thể
bắt được. Cần một cách tải lại tường minh. Ngoài ra chạy sync giữa phiên (trước 15:00 giờ VN) ghi thêm một dòng phiên
hôm nay chưa chốt, làm lệch lịch giữa các mã (`test_prices_aligned_and_unique` hỏng).

**Interfaces:**
- Produces:
  - `process_stock_file(..., full_refresh: bool = False)`. Khi True, bỏ bước tăng dần: gọi `fetch_latest_ohlcv(symbol,
    df_csv.time.min() as "%Y-%m-%d", end_date_str)` và, nếu kết quả không rỗng, nó thay toàn bộ `df_merged` (kèm
    `logger.warning("[SYM] tải lại toàn bộ lịch sử (--full-refresh)")`); rỗng thì giữ nguyên dữ liệu.
  - CLI `--full-refresh SYM...` (các mã cần tải lại toàn bộ; mã không có trong danh sách chạy như cũ).
  - `drop_open_session(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame`: bỏ các dòng có ngày bằng ngày của `now`
    (giờ Việt Nam) khi `now` trước 15:00 giờ Việt Nam. `fetch_latest_ohlcv` gọi nó trên kết quả trước khi trả, nên
    cả sync, `--full-refresh` lẫn `init_stock_file` (Task 5) không bao giờ ghi phiên chưa chốt.

- [ ] **Step 0 (làm trước Step 1):** Trong `Pipeline_V1.2/tests/test_sync.py` viết test hỏng rồi cài đặt phần Interfaces
  ở trên: `test_full_refresh_replaces_history` (lưu 33.8/33.55/22.35 liền kề, fake trả chuỗi đã điều chỉnh, kiểm tra
  `calls` bắt đầu ở ngày đầu tiên đã lưu và parquet bằng chuỗi fetch), `test_full_refresh_empty_keeps_data`,
  `test_drop_open_session_before_close` và `test_drop_open_session_after_close` (hai test dùng `now` cố định). Chạy
  `cd Pipeline_V1.2 && $PY -m pytest tests -v` kỳ vọng tất cả PASS, rồi commit
  `feat(sync): --full-refresh and drop the open session` (file CSV/parquet thật chưa bị đụng ở bước này).
- [ ] **Step 1:** Thêm test vào `tests/test_alignment.py`:

```python
def test_no_corporate_action_seam():
    from nlp_module.forecast.trial_forecast import check_prices
    assert check_prices(load_prices())["n_big_moves"] == 0
```

- [ ] **Step 2:** `$PY -m pytest tests/test_alignment.py -v`. Kỳ vọng FAIL với `n_big_moves == 1`, chính là TCB 28/10.
- [ ] **Step 3:** Sao lưu `TCB_stock.csv` và `.parquet` ra scratchpad, rồi chạy sync cho TCB (kèm các biến vnstock ở
  Global Constraints): `cd Pipeline_V1.2 && $PY sync_stock_parquet.py --symbols TCB --full-refresh TCB`. Log phải có
  dòng "tải lại toàn bộ" và không có dòng phiên hôm nay nếu chạy trước 15:00.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Kỳ vọng `test_no_corporate_action_seam` PASS.
  `test_pipeline_prices_cover_nlp_copy` vẫn PASS vì FPT không đổi; test này được viết lại ở Task 3.
- [ ] **Step 5:** Commit vào repo AI_Core: `test: no corporate-action seam in canonical prices`.

### Task 3: Một nguồn giá duy nhất cho volcast

**Files:**
- Modify: `NLP_Module/nlp_module/forecast/volcast.py` (`load_history`, `bars`, thêm `STOCK_DIR`)
- Modify: `NLP_Module/nlp_module/forecast/trial_forecast.py` (default `--stock-dir`)
- Modify: `tests/test_alignment.py` (thay `test_pipeline_prices_cover_nlp_copy`)

**Interfaces:**
- Produces:
  - `volcast.STOCK_DIR = Path(__file__).resolve().parents[3] / "Pipeline_V1.2" / "stock"`.
  - `volcast.SYMBOLS` = 15 mã Pipeline cũ (ACB FPT GAS HPG MBB MSN MWG SSI STB TCB VCB VIC VNM VPB VRE). Đây là tập
    volcast live dùng cho tới khi Task 7 quyết định mở rộng; việc thêm file mã mới vào `stock/` ở Task 5 không được
    làm đổi tập này.
  - `load_history(stock_dir=STOCK_DIR, symbols=None)`; `symbols=None` nghĩa là mọi file trong thư mục (giữ test dùng
    thư mục tạm chạy được).
  - `bars(c, stock_dir=STOCK_DIR, symbols=None)`.
  - CLI `volcast eval|train|predict --symbols fixed|all|A,B,...`. `fixed` (mặc định) = `SYMBOLS`; `all` = mọi file
    (`None`). `main` truyền kết quả xuống `bars`/`load_history`.

- [ ] **Step 1:** Thay `test_pipeline_prices_cover_nlp_copy` bằng:

```python
def test_volcast_reads_canonical_prices():
    from nlp_module.forecast import volcast
    h = volcast.load_history(symbols=volcast.SYMBOLS).set_index(["symbol", "date"]).close.sort_index()
    p = load_prices().set_index(["symbol", "date"]).close.sort_index()
    assert set(h.index.get_level_values(0)) == set(volcast.SYMBOLS) and set(C.UNIVERSE) <= set(volcast.SYMBOLS)
    assert (h.loc[p.index] == p).all()
```

  Thêm test cho `load_history` với `symbols=["AAA"]` trên thư mục tạm có hai mã: chỉ trả AAA.
- [ ] **Step 2:** Chạy `$PY -m pytest tests/test_alignment.py -v`. Kỳ vọng FAIL, vì volcast vẫn đọc `data/stock` (bản
  sao cũ còn vết nối TCB).
- [ ] **Step 3:** Cài đặt như mục Interfaces. Default `--stock-dir` của `trial_forecast` đổi thành
  `str(volcast.STOCK_DIR)`; import muộn bên trong `main`, như `prices_from_stock` đang làm. Thư mục
  `NLP_Module/data/stock` giữ nguyên trên đĩa nhưng không còn code nào đọc. Ghi một dòng vào phần Limitations của
  `NLP_Module/docs/model_report.md`.
- [ ] **Step 4:** Chạy test AI_Core và test NLP. Kỳ vọng tất cả PASS.
- [ ] **Step 5:** Commit cả hai repo: `refactor(volcast): read prices from Pipeline_V1.2/stock`.

### Task 4: Đóng băng fold C, có mốc so sánh mới, volcast live sạch

**Files:**
- Modify: `ai_core/config.py`, `ai_core/splits.py`
- Modify: `NLP_Module/nlp_module/forecast/walk_forward.py` (`_split`)
- Modify: `NLP_Module/nlp_module/forecast/volcast.py` (`walk`, `main`)
- Test: `NLP_Module/tests/forecast/test_walk_forward.py`, `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `walk_forward._split(d, h, train_start, val_start, test_start, test_end, purge=None)`: `purge=None` nghĩa là bằng `h`.
  - `ai_core.config`: `TEST_END = "2026-10-30"`, `COST = 0.0045`, `KAPPA = 0.5`, và
    `FOLDS = [(n, a, b, c, TEST_END if n == "C" else e) for n, a, b, c, e in walk_forward.FOLDS]`.
  - `ai_core.splits.folds(df, h, purge=None, folds=C.FOLDS)` yield `(name, tr, va, trva, te)`.
  - `volcast.walk(df, h, folds=FOLDS)`. CLI `volcast eval|train --test-end YYYY-MM-DD` thay ngày cuối của fold C.

- [ ] **Step 1: Viết test hỏng.**
  - `test_split_purge_overrides_h`: với `purge=h+1`, ngày cuối của train cách ngày đầu val đúng `h+1` phiên.
  - `tests/test_splits.py::test_frozen_fold_c_ends_before_test_end`: `max(te.date) < TEST_END` cho fold C.
  - `test_walk_honours_folds`: dữ liệu tổng hợp giống `test_walk_reports_none_when_unreachable`, truyền
    `folds=[FOLDS[0]]`, kết quả có đúng 1 dòng.
  - `test_pick_threshold_keeps_wilson_bound` (test số 4 của spec): với 20 seed ngẫu nhiên (`conf ~ U(0, 0.5)`,
    `ok ~ Bernoulli(0.5 + conf)`), nếu `t = pick_threshold(conf, ok)` khác None thì
    `wilson(ok[conf >= t].sum(), (conf >= t).sum())[0] >= 0.65` và `(conf >= t).sum() >= 50`. Test này PASS ngay; nó
    là test hồi quy, không phải test hỏng.
- [ ] **Step 2:** Chạy các test trên. Kỳ vọng FAIL.
- [ ] **Step 3:** Cài đặt theo Interfaces. `_split` thay mọi chỗ purge đang dùng `h` bằng `p = h if purge is None else purge`.
- [ ] **Step 4:** Chạy test AI_Core và test NLP. Kỳ vọng tất cả PASS, các test `_split` cũ vẫn đúng.
- [ ] **Step 5:** Chạy eval làm mốc:
  `cd NLP_Module && $PY -m nlp_module.forecast.volcast eval --test-end 2026-10-30 | tee data/reports/volcast_eval_fixed15.txt`.
  Tính `mean(test_acc - rule_acc)` trên 6 ô (ô không có lệnh tính 0) và ghi vào `model_report.md` dưới tiêu đề
  "Fixed-15 baseline (2026-10-29)".
- [ ] **Step 6:** `volcast train --test-end 2026-10-30`, rồi `volcast predict`. Kiểm tra: `vol_forecast` của TCB có
  `base_date` mới nhất.
- [ ] **Step 7:** Commit cả hai repo: `feat: frozen fold C and fixed-15 volcast baseline`.

---

## Bước 1 — Universe 20 mã

### Task 5: Khởi tạo lịch sử cho mã mới

**Files:**
- Modify: `Pipeline_V1.2/src/config.py` (thêm `SYNC_INIT_START = "2021-01-04"`)
- Modify: `Pipeline_V1.2/sync_stock_parquet.py` (thêm `init_stock_file`, CLI `--init`)
- Test: `Pipeline_V1.2/tests/test_sync.py`

**Interfaces:**
- Produces: `init_stock_file(symbol: str, stock_dir: str, start: str = C.SYNC_INIT_START) -> str | None`. Hàm tải từ
  `start` tới hôm nay, ghi `{SYM}_stock.csv` đã qua `clean_types` và `normalize_sessions`, trả đường dẫn; trả None nếu
  tải rỗng. Nếu file đã tồn tại thì không ghi đè.

- [ ] **Step 1:** Viết test hỏng.
  - `test_init_writes_csv_from_start`: fake fetch, kiểm tra `start_date == "2021-01-04"`, CSV có cột `C.RAW_COLS`, giờ
    07:00 UTC.
  - `test_init_does_not_overwrite`: file đã có thì giữ nguyên.
- [ ] **Step 2:** Chạy, kỳ vọng FAIL. **Step 3:** Cài đặt. Với `--init SYM...`, `main` gọi `init_stock_file` cho từng
  mã trước khi glob CSV. **Step 4:** Chạy, kỳ vọng PASS.
- [ ] **Step 5:** `$PY sync_stock_parquet.py --init BID CTG HDB VHM NVL`. Kiểm tra log cuối không có cảnh báo "không
  cùng dãy ngày" và có 20 file parquet.
- [ ] **Step 6:** Commit vào repo AI_Core: `feat(sync): --init bootstraps a new ticker's history`.

### Task 6: Mở rộng universe ra 20 mã ở mọi nơi

**Files:**
- Modify: `ai_core/config.py`, `NLP_Module/nlp_module/forecast/trial_forecast.py`, `NLP_Module/app.py:971`

**Interfaces:**
- Produces: `ai_core.config.UNIVERSE` = 20 mã sắp xếp theo chữ cái (ACB BID CTG FPT GAS HDB HPG MBB MSN MWG NVL SSI STB
  TCB VCB VHM VIC VNM VPB VRE). `trial_forecast.UNIVERSE` cùng danh sách; app `default_syms` cũng vậy.

- [ ] **Step 1:** Sửa `UNIVERSE`, rồi chạy `$PY -m pytest tests`. Kỳ vọng `test_prices_aligned_and_unique` và
  `test_no_corporate_action_seam` PASS với 20 mã. Nếu một mã mới có bước nhảy, chạy lại sync cho mã đó; Task 1 sẽ tự
  tải lại.
- [ ] **Step 2:** Sửa chú thích "BID CTG HDB VHM NVL lack prices..." trong `ai_core/config.py` thành mô tả mới.
- [ ] **Step 3:** Dựng lại snapshot cho app:
  `cd NLP_Module && .venv/bin/python -m nlp_module.cli snapshot --model baseline:data/models/baseline.joblib --symbols <20 mã> --start 2021-10-21 --end 2026-10-29 --out data/snapshots_baseline_backtest.parquet`.
  Làm tương tự với `phobert:data/models/phobert-v1` vào `data/snapshots_phobert_backtest.parquet`.
- [ ] **Step 4:** Commit cả hai repo: `feat: 20-ticker universe`.

### Task 7: Lần thử volcast 20 mã (quy tắc đặt trước)

**Files:**
- Modify: `NLP_Module/nlp_module/forecast/volcast.py` (`SYMBOLS`, `VERSION` nếu ship)
- Modify: `NLP_Module/docs/model_report.md`

`volcast.SYMBOLS` (15 mã) và `--symbols` đã có từ Task 3, nên volcast live vẫn dùng 15 mã suốt Task 5–7.

**Quy tắc ship (đã chép từ spec):**
- (a) `mean(test_acc - rule_acc)` ≥ mốc fixed-15 của Task 4.
- (b) Fold C `test_n ≥ 100` ở cả h=2 và h=5.

- [ ] **Step 1:** Ghi quy tắc vào `model_report.md` **trước khi chạy**.
- [ ] **Step 2:** `volcast eval --symbols all --test-end 2026-10-30 | tee data/reports/volcast_eval_u20.txt`. Thư mục
  giá lúc này đã có 20 mã. Kiểm tra đầu ra liệt kê đủ 20 mã.
- [ ] **Step 3:** Ghi kết quả và quyết định.
  - Nếu ship: đặt `volcast.SYMBOLS` = 20 mã (cùng danh sách `ai_core.config.UNIVERSE` ở Task 6), `VERSION =
    "volcast-v2.2"`, rồi `train` và `predict`.
  - Nếu không: `SYMBOLS` giữ 15 mã, `VERSION` giữ `"volcast-v2.1"`; không cần chạy lại `train`/`predict`.
- [ ] **Step 4:** Commit vào repo NLP: `exp(volcast): 20-ticker basket — <shipped|kept 15>`.

---

## Bước 2 — Thí nghiệm hướng giá (cổng GO/NO-GO)

### Task 8: Nhãn giao dịch được và khung dữ liệu

**Files:**
- Create: `ai_core/direction.py`
- Test: `tests/test_direction.py`

**Interfaces:**
- Consumes: `ai_core.prices.load_prices`, `volcast.add_features`, `fundamentals.bctc_features(C.NLP / "data/bctc.parquet")`
  (truyền vào để `add_features` không đọc theo đường dẫn tương đối), `C.COST`, `C.KAPPA`.
- Produces:
  - `FEATS: list[str]` = các cột giá trong `volcast.FEATS` (bỏ `fundamentals.BCTC`) cộng `["exret1", "exret5", "exret20"]`.
  - `add_labels(df: pd.DataFrame, h: int, cost: float = C.COST, kappa: float = C.KAPPA) -> pd.DataFrame`: thêm các
    cột `r_trade{h}`, `r_ex{h}`, `tau{h}`, `y{h}`. `y{h}` có giá trị -1.0 / 0.0 / 1.0 hoặc NaN.
  - `build_frame(h: int) -> pd.DataFrame`: gồm `load_prices`, `add_features`, `exret_k = ret_k - mean theo ngày`, và
    `add_labels`.

- [ ] **Step 1: Viết test hỏng** trên khung tổng hợp 3 mã × 40 phiên:

```python
def test_tradable_return_uses_next_open_to_close_t1h():
    d = _frame()                                       # helper: known open/close per (symbol, date)
    out = add_labels(d, 2)
    i = ...                                            # a row with enough future
    assert out.r_trade2[i] == pytest.approx(np.log(close[t + 3] / open[t + 1]))

def test_excess_and_tau():
    # r_ex sums to 0 per date; tau = max(0.0045, 0.5 * vol20 * sqrt(h)); y is 1 above tau, -1 below -tau, else 0

def test_label_nan_without_vol_or_future():
    # rows inside vol20 warm-up and the last h+1 rows per symbol have y NaN (not 0)
```

- [ ] **Step 2:** `$PY -m pytest tests/test_direction.py -v`. Kỳ vọng FAIL (module chưa có).
- [ ] **Step 3:** Cài đặt. `r_trade = log(close.shift(-(h+1)) / open.shift(-1))`, tính theo từng symbol.
- [ ] **Step 4:** Chạy lại, kỳ vọng PASS. **Step 5:** Commit: `feat(ai_core): tradable direction labels`.

### Task 9: Mô hình, chọn trên val, chỉ số và quy tắc GO

**Files:**
- Modify: `ai_core/direction.py`
- Test: `tests/test_direction.py`

**Interfaces:**
- Consumes: `ai_core.splits.folds(df, h, purge=h + 1)`, `src.metrics.block_bootstrap_dates`,
  `volcast.MODELS["gbm"]` (cấu hình E4).
- Produces:
  - `MODELS = {"logit": C ∈ {0.01, 0.1, 1.0} (StandardScaler + LogisticRegression đa lớp), "gbm": volcast.MODELS["gbm"]}`.
  - `select_on_val(tr, va, h) -> dict`. Fit trên `tr`, hiệu chỉnh sigmoid (`CalibratedClassifierCV(method="sigmoid",
    cv=3)` trên `tr`), rồi chọn `m` trên `va` từ lưới `np.round(np.arange(0.0, 0.51, 0.02), 2)` sao cho lợi suất ròng
    trung bình mỗi lệnh LONG lớn nhất với ≥ 50 lệnh. Trả `{"kind", "C", "m", "val_net", "val_n"}`, hoặc `kind=None`
    nếu không cấu hình nào đạt 50 lệnh.
  - `long_mask(p_up, p_down, m) -> np.ndarray[bool]` (`p_up - p_down >= m`).
  - `run_fold(tr, va, trva, te, h, rng) -> dict`. Gồm `n`, `hit`, `net` (mean `r_trade - COST`), `net_ex`
    (mean `r_ex - COST`), `ci_lo`, `ci_hi` (của `net_ex`, block 20, n=500), `rand_median` (random-same-count, 1000
    lần), `turnover`, `max_dd` (danh mục đều trọng số các lệnh LONG theo ngày), và `e0_prior`.
  - `go(rows: list[dict], pooled_ci_lo: float) -> bool`. True khi `net > 0` ở ≥ 2/3 fold, **và** `net > rand_median` ở
    ≥ 2/3 fold, **và** `pooled_ci_lo > 0`. Fold có `kind=None` tính là thua.

- [ ] **Step 1: Viết test hỏng.**
  - `test_long_mask_threshold`.
  - `test_go_rule`: 3 trường hợp. (1) Cả ba điều kiện đúng thì True. (2) Chỉ 1/3 fold có `net > 0` thì False.
    (3) `pooled_ci_lo = -0.001` thì False.
  - `test_purged_split_no_leakage` (test số 3 của spec): đổi `close` của mọi dòng trong test fold C (nhân 1.5). Kết
    quả `select_on_val(tr, va, 2)` phải không đổi: cùng `kind`, `C`, `m`, và `val_net` khớp tới 1e-12.
- [ ] **Step 2:** Chạy, kỳ vọng FAIL. **Step 3:** Cài đặt theo Interfaces. **Step 4:** Chạy, kỳ vọng PASS.
- [ ] **Step 5:** Commit: `feat(ai_core): direction experiment E0/E3/E4 with GO rule`.

### Task 10: Chạy thí nghiệm và quyết định GO/NO-GO

**Files:**
- Modify: `ai_core/direction.py` (`main()`)
- Create: `docs/reports/direction_report.md`

- [ ] **Step 1:** Chép quy tắc GO (Task 9) vào đầu `docs/reports/direction_report.md`, ghi ngày, **trước khi chạy**.
- [ ] **Step 2:** `cd AI_Core && $PY -m ai_core.direction`. Lệnh in và nối vào báo cáo: bảng E0/E3/E4 × fold × h với
  `n, hit, net, net_ex, CI, rand_median, turnover, max_dd`; dòng E1 lấy từ
  `Pipeline_V1.2/outputs/results/pooled_embed_k0.43/final_table.csv` kèm chú thích "split 70/15/15, dữ liệu trước khi
  sửa"; và quyết định GO/NO-GO cho từng h.
- [ ] **Step 3:** Commit: `exp: direction GO/NO-GO — <GO h=…|NO-GO>`.
  - Nếu GO: làm tiếp Task 11–15 (Bước 3 và bảng `fusion_decision`), bỏ Task 16–17, và lập kế hoạch Bước 4 riêng
    (RuleGate sẽ mở rộng `decide` ở Task 15).
  - Nếu NO-GO: tiếp tục như viết bên dưới.

---

## Bước 3 — NLP (chạy song song với Bước 2 được)

### Task 11: Bỏ link địa danh và tài khoản ngân hàng

**Files:**
- Modify: `NLP_Module/nlp_module/entities.py` (`Linker.find`)
- Modify: `NLP_Module/nlp_module/evaluate.py` (thêm `drop_stale_links`)
- Test: `NLP_Module/tests/test_pipeline_parts.py` (`TextTest`)

**Interfaces:**
- Produces:
  - Hai regex `_PLACE_BEFORE` và `_ACCOUNT_BEFORE`, áp lên 40 ký tự đứng trước mention. Trong `find`, bỏ mention
    khi gặp một trong hai trường hợp:
    - `_ACCOUNT_BEFORE` khớp, ví dụ "(số )?tài khoản( mở)?( tại)?", "chuyển (tiền|khoản)( vào| qua)?".
    - `_PLACE_BEFORE` khớp ("tại|ở" + tùy chọn "biệt thự|căn hộ|dự án|khu đô thị|nhà") **và** ngay sau alias là một
      từ viết hoa (tên khu, ví dụ Riverside, Ocean Park, Smart City, Grand Park).
  - `drop_stale_links(c, linker) -> int`: xóa các dòng `article_company_result` (mọi model version) mà linker mới không
    còn link `(document_version_id, target_entity_id)`; trả về số dòng đã xóa.

- [ ] **Step 1: Viết test hỏng.**

```python
def test_landmark_is_not_a_company(self):
    self.assertNotIn("VHM", [l.symbol for l in L.link("", "BIDV thu giữ biệt thự tại Vinhomes Riverside của bị cáo.")])
def test_bank_account_is_not_a_company(self):
    self.assertNotIn("VCB", [l.symbol for l in L.link("", "Nạn nhân chuyển tiền vào tài khoản mở tại Vietcombank.")])
def test_real_vinhomes_news_still_links(self):
    self.assertIn("VHM", [l.symbol for l in L.link("Vinhomes báo lãi quý 3", "Vinhomes ghi nhận doanh thu tăng.")])
```

  Thêm một test cho `drop_stale_links` trên DB `:memory:`: 1 kết quả có link cũ bị bỏ, 1 kết quả còn link; sau khi gọi
  hàm còn 1 dòng.
- [ ] **Step 2:** Chạy, kỳ vọng FAIL. **Step 3:** Cài đặt. **Step 4:** Chạy toàn bộ test NLP, kỳ vọng PASS (kể cả
  `test_currency_and_city_not_company`).
- [ ] **Step 5:** Ghi `link_changes(c, old, new, "eval")` và `"dev"` vào `data/reports/link_check.txt`, rồi chạy
  `drop_stale_links` trên `data/nlp.db` và ghi số dòng đã xóa vào `model_report.md`.
- [ ] **Step 6:** Commit vào repo NLP: `fix(entities): skip landmark and bank-account mentions`.

### Task 12: Cờ rủi ro sự kiện và FEATURE_VERSION v4

**Files:**
- Create: `NLP_Module/nlp_module/risk.py`
- Modify: `NLP_Module/nlp_module/contract.py`, `NLP_Module/nlp_module/pipeline.py` (`load_results`)
- Test: `NLP_Module/tests/test_snapshot.py`, `NLP_Module/tests/test_risk.py`

**Interfaces:**
- Produces:
  - `risk.FLAGS: dict[str, re.Pattern]` gồm các khóa `guarantee`, `seizure`, `prosecution`, `sanction`, `bond_default`.
  - `risk.risk_flags(text: str) -> list[str]` theo thứ tự khóa của `FLAGS`.
  - `ArticleCompanyResult.risk_flags: tuple = ()` (trường cuối, có mặc định).
  - `contract.FEATURE_VERSION = "v4"` và feature `risk_flag_72h` = số kết quả trong `win` có `risk_flags` không rỗng.

- [ ] **Step 1: Viết test hỏng.**
  - `test_guarantee_flag`: "MWG bảo lãnh vô điều kiện cho Bách Hóa Xanh vay 2.000 tỷ" cho ra `["guarantee"]`.
  - `test_no_flag`: "MWG khai trương 50 cửa hàng" cho ra `[]`.
  - `test_risk_flag_72h_counts_window`: 2 kết quả có cờ, 1 cái quá 72 giờ, cho ra 1.
  - `test_risk_flag_zero_without_news`: không có kết quả nào thì `risk_flag_72h == 0` và `has_news is False`.
- [ ] **Step 2:** Chạy, kỳ vọng FAIL. **Step 3:** Cài đặt. `load_results` đọc thêm `r.passage_text` và điền
  `risk_flags=tuple(risk_flags(...))`. **Step 4:** Chạy toàn bộ test NLP, kỳ vọng PASS.
- [ ] **Step 5:** Commit vào repo NLP: `feat: event risk flags, snapshot v4`.

### Task 13: Challenge set cho lớp negative

**Files:**
- Modify: `NLP_Module/nlp_module/annotate.py` (thêm `sample_challenge`)
- Modify: `NLP_Module/nlp_module/evaluate.py` (thêm `challenge_report`)
- Modify: `NLP_Module/nlp_module/cli.py` (lệnh `challenge-sample`, `challenge-report`)
- Modify: `NLP_Module/nlp_module/schema.sql` (comment `stream`: dev | eval | challenge)

**Interfaces:**
- Produces:
  - `sample_challenge(c, n: int = 60, seed: int = 0) -> int`. Chọn các kết quả mà `baseline-tfidf-lr-v1` hoặc
    `phobert-phobert-v1` có `p_neg` cao nhất trong ba lớp, hoặc có `risk_flags` không rỗng. Mỗi `document_id` lấy một
    mẫu, bỏ qua `(document_version_id, target_entity_id)` đã có trong `annotation_sample`. Ghi vào stream `challenge`
    và trả số mẫu.
  - `challenge_report(c, model_version) -> dict` gồm `n`, `n_neg`, `f1_neg`, `ci_lo`, `ci_hi` (bootstrap 2.000 lần,
    như eval).

- [ ] **Step 1:** Viết test `:memory:`: 3 kết quả, gồm 1 negative, 1 có cờ, 1 neutral không cờ. `sample_challenge`
  phải lấy đúng 2 mẫu, và gọi lần hai thì lấy 0.
- [ ] **Step 2 → 4:** FAIL → cài đặt → PASS.
- [ ] **Step 5:** `cli challenge-sample -n 60`, rồi `cli annotate-export data/challenge_to_label.csv --stream challenge`.
  **Dừng ở đây cho chủ dự án gán nhãn** (guideline v1-strict, cần ≥ 30 negative độc lập).
- [ ] **Step 6:** Khi đã có nhãn: `cli annotate-import ... --guideline v1-strict`, rồi
  `cli challenge-report --model <mỗi model>`. Ghi kết quả vào `model_report.md`.
- [ ] **Step 7:** Commit vào repo NLP: `feat: negative challenge set`.

### Task 14: Kiểm định BCTC "học vẹt mã"

**Files:**
- Modify: `NLP_Module/nlp_module/forecast/volcast.py` (thêm `bctc_check`, CLI `bctc-check`)
- Test: `NLP_Module/tests/forecast/test_volcast.py`

**Interfaces:**
- Produces: `bctc_check(df, h, folds, n_perm=20, seed=0) -> dict` với các khóa:
  - `real`: mean `test_acc` của `walk`.
  - `perm`: list n_perm giá trị mean `test_acc`, mỗi lần xáo ánh xạ mã → giá trị BCTC trước `add_bctc`.
  - `loto`: `{symbol: test_acc trên mã đó khi bỏ nó khỏi train}`.

- [ ] **Step 1:** Viết test trên `_bars` tổng hợp với `n_perm=2`: kiểm tra đủ khóa, `len(perm) == 2`, và `loto` có đủ mã.
- [ ] **Step 2 → 4:** FAIL → cài đặt → PASS.
- [ ] **Step 5:** Ghi quy tắc vào `model_report.md` trước khi chạy: nếu `mean(perm) ≥ real - 0.01` thì kết luận BCTC
  chỉ mã hóa danh tính mã. Sau đó chạy
  `volcast bctc-check --test-end 2026-10-30 | tee data/reports/volcast_bctc_check.txt` và ghi kết luận. Volcast live
  không đổi.
- [ ] **Step 6:** Commit vào repo NLP: `exp(volcast): BCTC permutation and leave-one-ticker-out`.

---

## Bước 5 — Màn hình và shadow (theo kết quả NO-GO; nếu GO, kế hoạch Bước 4 mở rộng Task 15)

### Task 15: Bảng `fusion_decision` và trạng thái dữ liệu

**Files:**
- Modify: `NLP_Module/nlp_module/schema.sql`, `NLP_Module/nlp_module/storage.py` (`connect`)
- Create: `ai_core/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Produces:
  - Bảng `fusion_decision(symbol, base_date, h, decision, p_up, p_down, margin, p_strong, call, has_news,
    sentiment_mean_24h, risk_flags, reason, version, created_at)` với `PRIMARY KEY (symbol, base_date, h, version)`.
    `decision` nhận `LONG | AVOID | NO_TRADE | DATA_INVALID`; `risk_flags` là chuỗi JSON. DDL viết ở cả hai nơi,
    như `vol_forecast`.
  - `ai_core.live.data_status(hist: pd.DataFrame, symbol: str, now: pd.Timestamp, holidays: np.ndarray) -> str | None`
    trả `"SEAM"` (có `|ret1| > RET_BAND` trong 21 phiên cuối), `"MISSING_SESSION"` (`src.predict.missing_sessions`
    không rỗng), hoặc None.
  - `ai_core.live.holidays(hist) -> np.ndarray`: ngày thường không có phiên trong `hist`, hợp với `MARKET_HOLIDAYS`.
  - `ai_core.live.decide(status: str | None) -> tuple[str, str]`: `("DATA_INVALID", status)` hoặc
    `("NO_TRADE", "no direction model (NO-GO)")`.
  - `ai_core.live.main()`: với mỗi mã trong `UNIVERSE` và mỗi h ∈ {2, 5}, đọc `vol_forecast` mới nhất (version volcast
    hiện hành) và snapshot live (`build_snapshot` với `baseline-tfidf-lr-v1`, cutoff 08:00 UTC của `base_date`), rồi
    ghi `INSERT OR REPLACE` với `VERSION = "decision-v0-nogo"`.

- [ ] **Step 1: Viết test hỏng.**
  - `test_status_ok`: lịch sử liền mạch tới phiên sẵn sàng gần nhất cho ra None.
  - `test_status_missing_session`: `now` = thứ Năm 16:00, dữ liệu dừng ở thứ Ba cho ra `"MISSING_SESSION"`.
  - `test_status_declared_holiday_ok`: thứ Tư nằm trong `holidays` thì cho ra None.
  - `test_status_seam`: ret1 = -0.33 ở 5 phiên trước cho ra `"SEAM"`.
  - `test_decide_data_invalid_never_trades` (test số 8 của spec, bản NO-GO): `decide("SEAM")[0] == "DATA_INVALID"`.
  - `test_fusion_decision_table_exists`: `storage.connect(":memory:")` có bảng này.
- [ ] **Step 2 → 4:** FAIL → cài đặt → PASS (test AI_Core và test NLP).
- [ ] **Step 5:** Chạy tay `cd AI_Core && $PY -m ai_core.live`. Kiểm tra có 40 dòng (20 mã × 2 h) cho `base_date`
  mới nhất. Lưu ý: từ 15:00 tới lúc sync 15:20 xong, phiên hôm nay chưa có trong lịch sử, nên các lần chạy lúc đó ghi
  `MISSING_SESSION`. Các lần chạy sau sync (15:30–16:00) ghi đè bằng `INSERT OR REPLACE`. Đây là hành vi mong muốn
  (fail-closed).
- [ ] **Step 6:** Commit cả hai repo: `feat: fusion_decision table and live data status`.

### Task 16: Đưa vào live và thêm tab màn hình

**Files:**
- Modify: `NLP_Module/scripts/live_update.sh`
- Modify: `NLP_Module/app.py` (thêm tab)

- [ ] **Step 1:** Trong `live_update.sh`, ngay sau dòng `volcast predict` và trong cùng khung 15:00–16:00, thêm:
  `(cd .. && NLP_Module/.venv/bin/python -m ai_core.live) || echo "decision write failed, continuing"`.
- [ ] **Step 2:** Thêm tab "Bảng quyết định" vào `st.tabs(...)` ở `app.py:376`. Tab đọc `fusion_decision` mới nhất cho
  mỗi `(symbol, h)` và hiển thị: decision và reason, volcast call và p_strong, `has_news`, `sentiment_mean_24h`, cờ
  rủi ro. Bấm vào một mã thì mở danh sách passage 72 giờ của mã đó (dùng `db_results` có sẵn) kèm cờ rủi ro. Đọc DB ở
  chế độ chỉ đọc, như `db_vol_forecast`.
- [ ] **Step 3:** Kiểm tra: `cd NLP_Module && .venv/bin/python -m streamlit run app.py` mở được tab mới, không có lỗi
  trong console. Hôm sau, `live.log` có dòng của `ai_core.live`.
- [ ] **Step 4:** Commit vào repo NLP: `feat(app): decision board tab; live writes decisions`.

### Task 17: Báo cáo shadow sau 40 phiên

**Files:**
- Create: `ai_core/shadow.py`
- Test: `tests/test_shadow.py`

**Interfaces:**
- Produces: `shadow_report(c, hist) -> dict` gồm:
  - `n_sessions` và tỷ lệ `DATA_INVALID`.
  - Accuracy live của volcast: các `call` khác NULL đã đủ h phiên tương lai, so với nhãn absmove tính từ `hist`.
  - Accuracy test của fold C trong `volcast_report.json` để so sánh.

- [ ] **Step 1:** Viết test với 2 mã × 10 phiên tổng hợp, 3 call đã đủ tương lai (2 đúng), 1 call chưa đủ. Kỳ vọng
  accuracy 2/3 và `n_matured == 3`.
- [ ] **Step 2 → 4:** FAIL → cài đặt → PASS. **Step 5:** Commit: `feat(ai_core): shadow report`.
- [ ] **Step 6 (sau ≥ 40 phiên kể từ Task 16):** `$PY -m ai_core.shadow >> docs/reports/shadow_report.md`, rồi
  commit báo cáo.
