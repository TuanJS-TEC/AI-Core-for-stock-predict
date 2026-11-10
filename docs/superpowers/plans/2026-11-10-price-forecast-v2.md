# Price Forecast v2 (thêm mã, bối cảnh thị trường, hai tầng) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tăng sức dự báo ba lớp Giảm / Đi ngang / Tăng (kèm xác suất) của nhánh giá `mmf` bằng ba hướng: huấn luyện trên ~100 mã, thêm đặc trưng bối cảnh thị trường thật, và mô hình hai tầng; rồi khóa một cấu hình v2 để chấm cùng lượt với v1 trên vùng chấm cuối.

**Architecture:** Mọi thứ mới đi qua một cơ chế "variant" chạy song song với v1: `base` (dữ liệu v1, chỉ thêm mô hình hai tầng), `ext` (train thêm mã ngoài, tune/calib/eval vẫn chỉ 15 mã), `ext_ctx` (`ext` cộng 19 đặc trưng bối cảnh). Dữ liệu phụ (mã ngoài, VNINDEX, chuỗi toàn cầu) được đóng băng thành một thư mục `aux` có `aux_id`. Kết quả ghi thành arm tên `<model>@<variant>` trong cùng run dir để `mmf.compare` ghép cặp với arm v1. Arm, baseline, đặc trưng, nhãn, fold và file khóa của v1 không đổi.

**Tech Stack:** Python trong `.venv_mmf` (torch, scikit-learn, pandas 2.3.3, numpy 2.3.5), vnstock 4.0.9 nguồn `kbs`, `urllib` của stdlib cho Yahoo chart API. Không thêm dependency.

**Spec:** Hội thoại 2026-11-10 (không lưu thành file): chủ dự án muốn dự báo ba lớp kèm xác suất tốt hơn và yêu cầu làm cả ba hướng. Mọi quyết định cần dùng đã chép vào Global Constraints; nếu mâu thuẫn, Global Constraints thắng.

## Global Constraints

- `MAIN=/home/administrator/TLS/tuanhm@linux/AI_Core`. Python cho mmf: `.venv_mmf/bin/python`; test mmf chạy từ gốc repo: `.venv_mmf/bin/python -m pytest tests -q`. Test Pipeline: `cd Pipeline_V1.2 && $MAIN/NLP_Module/.venv/bin/python -m pytest tests -q`.
- Làm trong git worktree, nhánh `price-v2` từ `multimodal-decision` (cron chạy từ cây chính). Trong worktree tạo symlink tới cây chính cho các đường dẫn bị gitignore: `NLP_Module`, `.venv_mmf`, `Pipeline_V1.2/stock`, `Pipeline_V1.2/stock_ext`, `data/mmf`. Chạy hai bộ test ở trên trước khi sửa gì và ghi số test PASS làm baseline.
- **Không sửa:** `FEATURES` và mọi hàm trong `Pipeline_V1.2/src/price_features.py`, `labels.py`, `C.DATE_FOLDS`, `mmf.config.ARMS`, `mmf.config.BASELINES`, `docs/reports/mmf_lock.json`, `docs/reports/mmf_dev_report.md`, phần đã có của `docs/reports/mmf_preregistration.md` (chỉ được nối thêm ở cuối), `Pipeline_V1.2/stock/`, crontab, `NLP_Module/`, `ai_core/`.
- **Không bao giờ chạy `python -m mmf.holdout`.** Thí nghiệm phát triển chỉ dùng snapshot `Pipeline_V1.2/stock/snapshots/24251acc1a5695bd`.
- v1 phải giữ nguyên hành vi: với mã mới, `fold_bundle` cho ra `Samples` giống hệt trước (test ở Task 4), và `mmf.holdout` khi không có `mmf_lock_v2.json` chạy đúng như cũ.
- Mã ngoài chỉ nằm trong khối `train`. Các khối `tune`, `calib`, `eval` chỉ chứa 15 mã `mmf.config.TICKERS`.
- Không forward-fill, backfill, clip hay winsorize giá. Phiên thiếu là hàng NaN; cửa sổ chứa NaN bị bỏ. Riêng chuỗi toàn cầu dùng "giá trị đã biết gần nhất": thanh có ngày **nhỏ hơn hẳn** ngày phiên, cũ quá 7 ngày lịch thì NaN.
- Rolling luôn trailing, `min_periods` bằng độ dài cửa sổ.
- Gọi vnstock cách nhau 3 giây (giới hạn khách 20 lượt/phút). Request ra ngoài chỉ mang mã chứng khoán và `User-Agent: Mozilla/5.0`, không mang thông tin dự án hay người dùng.
- `Pipeline_V1.2/src/config.py` dùng CRLF; sau khi sửa chạy `file` để chắc vẫn là CRLF. File mới dùng LF.
- Mọi commit kết thúc bằng `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Ứng viên v2 cố định trước khi xem kết quả, đúng 12 cái: `hurdle_logit_price@base`, `hurdle_gbm_price@base`, và với mỗi variant `ext`, `ext_ctx`: `logit_price`, `gbm_price`, `hurdle_logit_price`, `hurdle_gbm_price`, `dl_price`. Không thêm ứng viên sau khi đã chạy Task 7.

**Đã kiểm ngày 2026-11-10, quyết định không làm (kèm lý do):**

| Ý tưởng | Kết quả kiểm | Quyết định |
|---|---|---|
| Khối ngoại, tự doanh theo mã | CafeF chỉ trả ~248 phiên gần nhất; Finhay có từ 2026-07-30 | Bỏ: không phủ khối train (2022–2025). Cron Finhay tiếp tục tích lũy. |
| Chênh lệch VN30F1M | `kbs` và `vci` chỉ trả 37 phiên | Bỏ |
| Chỉ số VN30 | Thiếu phiên 2025-04-04, sẽ làm rơi 60 cửa sổ của mọi mã | Bỏ, chỉ dùng VNINDEX (đủ 1457 phiên) |
| Chỉ số ngành (VNFIN, VNREAL) | `kbs` không hỗ trợ | Bỏ |
| S&P 500, USD/VND qua MSN | Tối đa 365 thanh | Dùng Yahoo chart API (đủ từ 2021-01-04) |
| FRED | Không kết nối được | Bỏ |
| Lãi suất liên ngân hàng | Công bố trễ nhiều ngày, không có ngày công bố đáng tin | Bỏ |
| Mô hình sâu hai đầu ra | Softmax ba lớp đã biểu diễn được cùng phân phối | Hai tầng chỉ làm cho logit và GBM |

**Giới hạn phải ghi trong báo cáo:** danh sách VN100 lấy tại 2026-11-10 nên tập train của mã ngoài có thiên lệch sống sót (không ảnh hưởng hàng chấm điểm); Yahoo chart API không chính thức; chính sách điều chỉnh giá vẫn "unknown"; fold A/B/C đã bị nhìn nhiều lần.

## Review Focus

1. Mã ngoài niêm yết muộn, bị đình chỉ hoặc thiếu phiên: hàng NaN, cửa sổ bị bỏ, không điền; mã dưới 500 phiên bị loại hẳn. (Task 1)
2. Thanh toàn cầu có ngày bằng hoặc sau ngày phiên, `close` null (Yahoo trả null cho thanh cuối của `VND=X`), kỳ nghỉ Mỹ dài: không lọt vào đặc trưng của phiên đó; cũ quá 7 ngày thì NaN. (Task 2, Task 3)
3. Thư mục `aux` bị sửa sau khi đóng băng, hoặc cũ hơn dữ liệu giá: từ chối, không chạy tiếp. (Task 2, Task 4, Task 8)
4. Mã mới làm đổi v1: `Samples` của `fold_bundle` và tên file của arm v1 phải y nguyên; holdout không có khóa v2 phải y nguyên. (Task 4, Task 8)
5. Mặt cắt ngang quá mỏng (ít hơn 30 mã có giá trị): `breadth_ma20`, `adv_share`, `xs_rank*` là NaN chứ không phải một con số lệch. (Task 3)

---

### Task 1: Dữ liệu mã ngoài và VNINDEX, bộ nạp `mmf/aux.py`

**Files:**
- Create: `Pipeline_V1.2/data/ext_universe.csv`, `mmf/aux.py`, `tests/test_mmf_aux.py`
- Modify: `.gitignore` (thêm dòng `Pipeline_V1.2/stock_ext/`), `mmf/config.py`

**Interfaces:**
- Produces (`mmf/config.py`): `EXT_DIR = AC.ROOT / "Pipeline_V1.2/stock_ext"`, `EXT_UNIVERSE = AC.ROOT / "Pipeline_V1.2/data/ext_universe.csv"`, `INDEX = "VNINDEX"`, `EXT_MIN_SESSIONS = 500`.
- Produces (`mmf/aux.py`):
  - `load_frames(ext_dir, calendar: pd.DatetimeIndex, exclude=C.TICKERS) -> dict[str, pd.DataFrame]`: mỗi `<SYM>_stock.parquet` trong `ext_dir` trừ `exclude` và `C.INDEX`; đọc bằng `src.data.load_ohlcv(sym, ext_dir)`, giữ `open, high, low, close, volume` float64, `reindex(calendar)` không điền; bỏ mã có ít hơn `C.EXT_MIN_SESSIONS` hàng `close` hữu hạn; khóa sắp theo tên.
  - `load_index(ext_dir, calendar) -> pd.DataFrame`: cùng dạng cho `C.INDEX`; thiếu file → `ValueError`.

- [ ] **Step 1: Viết test thất bại** `tests/test_mmf_aux.py` (parquet tổng hợp trong `tmp_path`, lịch 600 phiên ngày thường; ghi bằng cột `time, open, high, low, close, volume`, `time` = ngày + 07:00 UTC):
  - `test_load_frames_reindexes_without_filling`: mã `AAA` đủ 600 phiên nhưng bỏ phiên thứ 300 → `frames["AAA"].loc[calendar[300]].isna().all()` và hàng 301 là giá thật.
  - `test_short_history_symbol_is_dropped`: mã `BBB` chỉ 499 phiên cuối → `"BBB" not in frames`; mã `CCC` 500 phiên → có, và 100 hàng đầu là NaN.
  - `test_thesis_tickers_and_index_are_excluded`: file `FPT_stock.parquet` và `VNINDEX_stock.parquet` có trong thư mục → không nằm trong `frames`; `load_index` trả bảng cùng chỉ số `calendar`.
  - `test_load_index_missing_raises`.
- [ ] **Step 2: Chạy, thấy FAIL:** `.venv_mmf/bin/python -m pytest tests/test_mmf_aux.py -v`.
- [ ] **Step 3: Cài đặt** bốn hằng và hai hàm.
- [ ] **Step 4: Chạy lại** → PASS.
- [ ] **Step 5: Sinh `ext_universe.csv`** (một lần, cột `symbol`): `Listing(source="kbs").symbols_by_group("VN100")` trừ 15 mã `C.TICKERS`, sắp theo tên. Kỳ vọng 85 dòng. File này đóng băng, không sinh lại.
- [ ] **Step 6: Tải dữ liệu** vào cây chính (khoảng 10 phút: 86 mã, mỗi mã một lượt tải lịch sử và một lượt tải chồng; thư mục này phải có trước khi tạo symlink `stock_ext` trong worktree): `mkdir -p $MAIN/Pipeline_V1.2/stock_ext && cd $MAIN/Pipeline_V1.2 && $MAIN/NLP_Module/.venv/bin/python sync_stock_parquet.py --stock-dir stock_ext --init VNINDEX $(tail -n +2 data/ext_universe.csv)`. Cảnh báo "Các mã không cùng dãy ngày" là bình thường ở thư mục này. Mã tải lỗi: chạy lại đúng lệnh (file đã có được giữ nguyên).
- [ ] **Step 7: Kiểm tra trên dữ liệu thật:** với `cal = build_panel(stock_dir=<snapshot>).calendar`, in `len(load_frames(C.EXT_DIR, cal))` (kỳ vọng 70–85) và `load_index(C.EXT_DIR, cal).close.isna().sum()` (kỳ vọng 0). Nếu VNINDEX thiếu phiên, dừng và báo.
- [ ] **Step 8: Commit** `feat(mmf): external training universe and VNINDEX loaders`.

---

### Task 2: Chuỗi toàn cầu và đóng băng `aux`

**Files:**
- Create: `mmf/globaldata.py`, `tests/test_mmf_globaldata.py`
- Modify: `mmf/aux.py`, `mmf/config.py`, `tests/test_mmf_aux.py`

**Interfaces:**
- Produces (`mmf/config.py`): `GLOBAL_FILE = AC.ROOT / "data/global/global.parquet"`, `GLOBAL_SERIES = {"spx": "^GSPC", "vix": "^VIX", "brent": "BZ=F", "usdvnd": "VND=X"}`, `GLOBAL_STALE_DAYS = 7`, `AUX = DATA / "aux"`.
- Produces (`mmf/globaldata.py`):
  - `parse_chart(payload: dict) -> pd.DataFrame`: cột `date` (ngày của thanh theo múi giờ sàn: `timestamp + meta["gmtoffset"]`, chuẩn hóa về ngày), `close`; bỏ thanh `close` null; mỗi ngày một dòng (giữ dòng cuối).
  - `fetch(symbol: str, start="2021-01-01") -> pd.DataFrame`: GET `https://query1.finance.yahoo.com/v8/finance/chart/<symbol đã quote>?period1=..&period2=..&interval=1d`, timeout 30 giây, rồi `parse_chart`.
  - `sync(out=C.GLOBAL_FILE) -> pd.DataFrame`: bảng dài `series, date, close` của mọi `C.GLOBAL_SERIES`, ghi qua file tạm + `os.replace`; một chuỗi lỗi → raise, không ghi gì. CLI: `python -m mmf.globaldata`.
  - `asof_before(s: pd.Series, calendar: pd.DatetimeIndex, stale_days=C.GLOBAL_STALE_DAYS) -> pd.Series`: `s` có chỉ số ngày; tại phiên `t` trả giá trị của thanh cuối có ngày `< t`; NaN nếu không có hoặc thanh đó cũ hơn `stale_days` ngày lịch.
- Produces (`mmf/aux.py`):
  - `@dataclass Aux`: `frames: dict[str, pd.DataFrame]`, `index: pd.DataFrame`, `glob: pd.DataFrame` (bảng dài `series, date, close`), `aux_id: str | None`.
  - `aux_id_of(d: Path) -> str`: sha256 hex của `"<dataset_id trong d/manifest.json>:<sha256 bytes của d/global.parquet>"`.
  - `freeze(ext_dir=C.EXT_DIR, global_file=C.GLOBAL_FILE, out_root=C.AUX) -> Path`: chép mọi `*_stock.parquet`, `manifest.json` và `global.parquet` vào `out_root/<aux_id[:16]>/`; thư mục đã có và `aux_id_of` khớp → trả về; khác → `ValueError`. Thiếu `manifest.json` → `ValueError`. CLI: `python -m mmf.aux freeze` in đường dẫn.
  - `load(aux_dir, calendar) -> Aux`: `aux_id_of(aux_dir)[:16] != Path(aux_dir).name` → `ValueError` ("aux bị sửa sau khi đóng băng"); `index.close` NaN ở phiên cuối của `calendar` → `ValueError` ("aux cũ hơn dữ liệu giá").

- [ ] **Step 1: Viết test thất bại:**
  - `test_parse_chart_drops_null_close_and_uses_exchange_date`: payload với `timestamp=[1762785000, 1762871400]` (14:30 UTC hai ngày liền), `gmtoffset=-18000`, `close=[10.0, None]` → một dòng, `date == 2025-11-10`, `close == 10.0`.
  - `test_asof_before_is_strict`: `s` có thanh `2026-10-19: 1.0`, `2026-10-20: 2.0`; lịch `[2026-10-20, 2026-10-21]` → `[1.0, 2.0]`.
  - `test_asof_before_stale_is_nan`: thanh cuối `2026-10-09`, phiên `2026-10-19` → NaN; phiên `2026-10-16` → có giá trị.
  - `test_sync_writes_nothing_when_one_series_fails`: `monkeypatch` `fetch` ném lỗi ở chuỗi thứ hai → file `out` không tồn tại, không còn `*.tmp`.
  - `test_freeze_is_idempotent_and_load_detects_tampering` (trong `test_mmf_aux.py`): `freeze` hai lần cùng đường dẫn; ghi đè `global.parquet` trong thư mục đóng băng → `load` ném `ValueError`.
  - `test_load_refuses_aux_older_than_calendar`: lịch dài hơn dữ liệu VNINDEX một phiên → `ValueError`.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Chạy** `.venv_mmf/bin/python -m pytest tests/test_mmf_globaldata.py tests/test_mmf_aux.py -v` → PASS.
- [ ] **Step 5: Tải và đóng băng trên cây chính:** `cd $MAIN && .venv_mmf/bin/python -m mmf.globaldata && .venv_mmf/bin/python -m mmf.aux freeze`. Ghi lại đường dẫn in ra làm `$AUX`. Kiểm tra mỗi chuỗi có thanh đầu ≤ 2021-01-08 và hơn 1400 dòng; không đạt thì dừng và báo.
- [ ] **Step 6: Commit** `feat(mmf): global series with strict as-of join and frozen aux directory` (kèm `data/global/global.parquet`, theo cách `data/finhay/*.parquet` đang được theo dõi).

---

### Task 3: Đặc trưng bối cảnh `mmf/context.py`

**Files:**
- Create: `mmf/context.py`, `tests/test_mmf_context.py`

**Interfaces:**
- Consumes: `globaldata.asof_before`; bảng OHLCV trên lịch (từ `Panel.on_calendar` và `Aux.frames`).
- Produces:
  - `CTX_FEATS` đúng 19 tên theo thứ tự: `ix_ret1 ix_ret5 ix_ret20 ix_std20 ix_dd60 breadth_ma20 adv_share spx_ret1 spx_ret5 vix_log vix_chg5 brent_ret5 usdvnd_ret5 usdvnd_ret20 rel_ix5 rel_ix20 beta60 xs_rank5 xs_rank20`.
  - `MIN_XS = 30`.
  - `context_features(frames: dict[str, pd.DataFrame], index: pd.DataFrame, glob: pd.DataFrame) -> pd.DataFrame`: bảng dài `symbol, as_of_session, *CTX_FEATS`, một dòng mỗi (mã trong `frames`, phiên của lịch).
- Định nghĩa (`C` là close, `r_k = log(C_t / C_{t-k})`, theo hàng lịch):
  - `ix_ret1/5/20` = `r_k` của VNINDEX; `ix_std20` = std (ddof=1) của `ix_ret1` trên 20; `ix_dd60 = C_ix / max_60(C_ix) − 1`.
  - `breadth_ma20` = tỷ lệ mã trong `frames` có `C > MA20`; `adv_share` = tỷ lệ mã có `r_1 > 0`; mẫu số là số mã có giá trị hữu hạn, nhỏ hơn `MIN_XS` → NaN.
  - Toàn cầu: lợi suất tính trên thanh của chính chuỗi đó (`log(c / c.shift(k))`), rồi mới `asof_before` lên lịch. `spx_ret1`, `spx_ret5`, `vix_log = log(vix)`, `vix_chg5 = log(vix / vix.shift(5))`, `brent_ret5`, `usdvnd_ret5`, `usdvnd_ret20`.
  - Theo mã: `rel_ix5 = r_5 − ix_ret5`, `rel_ix20 = r_20 − ix_ret20`; `beta60 = cov_60(r_1, ix_ret1) / var_60(ix_ret1)` (NaN khi phương sai 0); `xs_rank5`, `xs_rank20` = hạng phần trăm (`rank(pct=True)`) của `r_5`, `r_20` giữa các mã trong `frames` tại phiên đó, NaN nếu ít hơn `MIN_XS` mã có giá trị.
  - `±inf` → NaN.

- [ ] **Step 1: Viết test thất bại** `tests/test_mmf_context.py` (bước đi ngẫu nhiên 40 mã × 300 phiên, seed cố định; `index` là trung bình của chúng; `glob` bốn chuỗi cùng ngày lịch):
  - `test_names_and_count`: `len(CTX_FEATS) == 19`, không trùng, không trùng tên nào trong `src.price_features.FEATURES`.
  - `test_prefix_invariance`: kết quả trên 200 phiên đầu bằng kết quả đầy đủ cắt tới phiên 200 (`assert_frame_equal`).
  - `test_future_perturbation`: nhân giá mọi mã, chỉ số và chuỗi toàn cầu sau phiên `t` với 1.7 → hàng `≤ t` không đổi.
  - `test_same_day_global_bar_is_not_used`: đổi `close` của thanh `spx` có ngày đúng bằng phiên `t` → `spx_ret1` tại `t` không đổi, tại phiên kế tiếp thì đổi.
  - `test_thin_cross_section_is_nan`: chỉ 29 mã → `breadth_ma20`, `adv_share`, `xs_rank5`, `xs_rank20` toàn NaN; 30 mã → `xs_rank5` nằm trong `(0, 1]` ở các hàng hữu hạn.
  - `test_missing_session_stays_nan`: một mã có hàng `g` NaN → `rel_ix5` của mã đó NaN tại `g`; các mã khác vẫn hữu hạn tại `g`.
  - `test_no_inf`.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt** `context_features`.
- [ ] **Step 4: Chạy** → PASS.
- [ ] **Step 5: Commit** `feat(mmf): causal market, breadth, global and cross-sectional context features`.

---

### Task 4: Variant bundle và bộ chạy

**Files:**
- Modify: `mmf/config.py`, `mmf/windows.py`, `mmf/baselines.py`, `mmf/experiment.py`
- Test: `tests/test_mmf_variants.py` (mới); `tests/test_mmf_experiment.py`, `tests/test_mmf_frame.py`, `tests/test_mmf_baselines.py` phải PASS không sửa.

**Interfaces:**
- Consumes: `aux.Aux`, `aux.load`, `context.CTX_FEATS`, `context.context_features`, `src.price_features.{compute, market_features, FEATURES}`, `src.labels.make_labels`.
- Produces (`mmf/config.py`):
  - `VARIANTS = {"base": (False, False), "ext": (True, False), "ext_ctx": (True, True)}` (giá trị: train thêm mã ngoài, thêm đặc trưng bối cảnh).
  - `HURDLE = ("hurdle_logit_price", "hurdle_gbm_price")`.
  - `V2_BASELINES = {"base": HURDLE, "ext": ("logit_price", "gbm_price", *HURDLE), "ext_ctx": ("logit_price", "gbm_price", *HURDLE)}`.
  - `V2_DL = {"base": (), "ext": ("dl_price",), "ext_ctx": ("dl_price",)}`.
- Produces (`mmf/windows.py`): `Scalers.fit(train_rows, cols=None)` (`None` → `COLS`); `Scalers.apply` chuẩn hóa đúng các cột trong `self.mean.index`; `make_samples(scaled_frame, block_rows, tickers, lp=C.LP, lt=C.LT, price_cols=None)` (`None` → `GROUPS["price"]`).
- Produces (`mmf/baselines.py`): `predict_baseline(name, blocks, scaled, frame, price_cols=None)`: nhóm `price` dùng `price_cols` khi được truyền. (Tên `hurdle_*` thêm ở Task 5.)
- Produces (`mmf/experiment.py`):
  - `Bundle` thêm ba trường cuối có mặc định: `price_cols: list | None = None`, `variant: str | None = None`, `aux_id: str | None = None`.
  - `variant_bundle(panel, aux: Aux, h: int, fold: Fold, variant: str) -> Bundle`.
  - `out_name(b: Bundle, name: str) -> str`: `name` khi `b.variant is None`, ngược lại `f"{name}@{b.variant}"`. `_write_preds` ghi cột `arm` bằng tên này; bundle có `aux_id` thì ghi thêm cột `aux_id` và `_run_meta` kiểm thêm khóa đó (bundle v1 không có cột, không kiểm).
  - `select_price_kind`: file là `select.json` khi `b.variant is None`, ngược lại `select@<variant>.json`.
  - `run_variants(panel, aux, horizons, folds, run_dir, device, variants=tuple(C.VARIANTS), seeds=C.SEEDS, max_epochs=C.MAX_EPOCHS, build=variant_bundle)`: với mỗi (h, fold, variant): từng tên trong `C.V2_BASELINES[variant]` qua `run_baseline`, rồi nếu `C.V2_DL[variant]` khác rỗng thì `select_price_kind` và `run_arm` cho từng seed. Thư mục: `run_dir/h<h>/<fold>/<out_name>/seed<k>.parquet`.
  - CLI: `--variants` (`nargs="+"`, `choices=list(C.VARIANTS)`) và `--aux-dir`. Có `--variants` thì chỉ chạy `run_variants` (arm v1 chạy bằng lệnh riêng vào cùng `--name`); thiếu `--aux-dir` → lỗi argparse. `--shuffle-labels` kết hợp được với `--variants`: `build` bọc `shuffle_train_labels`.
- Cách dựng `variant_bundle` (thứ không suy ra được từ chữ ký):
  1. `frame = build_frame(panel, h, C.FOLD_NLP[fold.name])` (15 mã, y như v1).
  2. Variant có mã ngoài: với mỗi mã trong `aux.frames`, đặc trưng giá bằng `compute(f, market)` với `market = market_features({s: panel.on_calendar(s) for s in sorted(panel.ohlcv)})` (đúng proxy thị trường của v1, không tính lại trên 100 mã), nhãn bằng `make_labels(f, h)`, `close_loc = 0.5` khi `flag_hl_equal`, mọi cột nhóm `fund` và `text` bằng `0.0`; nối vào `frame`.
  3. Variant có bối cảnh: merge `context_features({15 mã + mã ngoài}, aux.index, aux.glob)` theo `symbol, as_of_session`; `price_cols = FEATURES + CTX_FEATS`. Không có bối cảnh: `price_cols = FEATURES`.
  4. `usable` = trạng thái `ok`, phiên `>= C.START`, mọi `price_cols` hữu hạn.
  5. `split_samples(frame[usable], fold)`; ở `tune`, `calib`, `eval` chỉ giữ hàng có `symbol` thuộc `sorted(panel.ohlcv)`.
  6. Scaler fit trên `train` (cả mã ngoài) với `cols = price_cols` cộng các cột `fund`, `text` không nhị phân.
  7. `tickers = sorted(panel.ohlcv) + sorted(mã ngoài)`; `aux_id = aux.aux_id`; variant `base` không dùng mã ngoài lẫn bối cảnh nhưng vẫn mang `variant="base"` và `aux_id`.

- [ ] **Step 1: Viết test thất bại** `tests/test_mmf_variants.py`. Dùng dữ liệu thật như `tests/test_mmf_frame.py` (`build_panel(PC.TICKERS, load_raw())`, `text_fn` giả qua `monkeypatch` của `mmf.frame.text.text_features` nếu cần). `Aux` dựng trong bộ nhớ: 30 mã giả `X00..X29`, mỗi mã là OHLC của một mã thật nhân `1 + 0.01·i` với 100 phiên đầu là NaN; `index` = trung bình close của 15 mã (đủ cột OHLCV); `glob` bốn chuỗi bước đi ngẫu nhiên trên ngày thường; `aux_id="test"`. Fold C từ `folds_from_config()`, `h=5`.
  - `test_v1_samples_unchanged`: `fold_bundle` trước và sau khi gọi `variant_bundle` cho cùng `x_price`, `y`, `symbol`, `as_of` (so với một bản dựng tay theo đường cũ: `make_samples(Scalers.fit(train).apply(frame), rows, tickers)`).
  - `test_ext_rows_only_in_train`: variant `ext` → `set(b.samples["train"].symbol)` chứa ít nhất một `X..`; với `tune`, `calib`, `eval`: `set(symbol) <= set(PC.TICKERS)`.
  - `test_ext_eval_rows_equal_v1_eval_rows`: `(symbol, as_of)` của `eval` ở variant `ext` và `base` trùng với `fold_bundle`.
  - `test_base_variant_matches_v1_inputs`: `x_price` của `eval` ở variant `base` bằng của `fold_bundle` (`np.array_equal`).
  - `test_ctx_widens_price_input`: variant `ext_ctx` → `b.samples["train"].x_price.shape[2] == len(FEATURES) + len(CTX_FEATS)`, mọi giá trị hữu hạn, `b.price_cols` đúng thứ tự.
  - `test_ext_train_labels_are_purged`: ở `train` của variant `ext`, `label_end.max() <= fold.train_end`.
  - `test_out_name_and_paths`: `run_variants(..., variants=("ext",), seeds=(59,), max_epochs=1)` vào `tmp_path` tạo `hurdle_gbm_price@ext/seed0.parquet` (sau Task 5; ở task này kiểm `logit_price@ext/seed0.parquet` và `dl_price@ext/seed59.parquet`), cột `arm` là tên có `@ext`, có cột `aux_id == "test"`, và `select@ext.json` tồn tại. Tạm đặt `C.V2_BASELINES["ext"]` qua `monkeypatch` thành `("logit_price", "gbm_price")` cho tới Task 5.
  - `test_reuse_refuses_other_aux`: chạy lại với `aux_id="khac"` → `ValueError`.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy toàn bộ** `.venv_mmf/bin/python -m pytest tests -q` → PASS, số test ≥ baseline cộng test mới.
- [ ] **Step 5: Commit** `feat(mmf): variants base/ext/ext_ctx running beside the locked arms`.

---

### Task 5: Mô hình hai tầng (hurdle)

**Files:**
- Modify: `mmf/baselines.py`
- Test: `tests/test_mmf_baselines.py`, `tests/test_mmf_variants.py` (bỏ `monkeypatch` tạm ở Task 4)

**Interfaces:**
- Produces: `predict_baseline` nhận thêm tên trong `C.HURDLE` (tên ngoài `C.BASELINES` và `C.HURDLE` vẫn `ValueError`). Với `hurdle_<kind>_price`: tầng 1 là `_model(kind)` nhị phân dự báo `label != 1` trên mọi hàng train có nhãn → `pm`; tầng 2 là `_model(kind)` nhị phân dự báo `label == 2` chỉ trên hàng train có `label != 1` → `pu`; đầu ra `[pm·(1−pu), 1−pm, pm·pu]`. Cùng cột và cùng cách xử lý NaN như `<kind>_price`. Tầng nào chỉ thấy một lớp trong train thì xác suất của tầng đó là tần suất lớp (0 hoặc 1 được làm trơn Laplace `(n_c+1)/(n+2)`).

- [ ] **Step 1: Viết test thất bại:**
  - `test_hurdle_rows_sum_to_one_and_shape`: dữ liệu tổng hợp 600 hàng, 5 cột → `(n, 3)`, mỗi hàng tổng 1 (`allclose`), mọi giá trị trong `(0, 1)`.
  - `test_hurdle_learns_two_separate_signals`: cột 0 quyết định có biến động hay không (`|x0| > 1` → nhãn khác 1), cột 1 quyết định hướng (`x1 > 0` → 2, ngược lại 0) → trên tập kiểm tra, `argmax` đúng hơn 80% với `hurdle_gbm_price`.
  - `test_hurdle_single_class_stage_does_not_crash`: train không có hàng `label == 1` → chạy được, `p_flat` trong `(0, 0.05)`.
  - `test_unknown_name_still_raises`.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Chạy** `.venv_mmf/bin/python -m pytest tests -q` → PASS.
- [ ] **Step 5: Commit** `feat(mmf): two-stage move/direction baselines`.

---

### Task 6: So sánh v2, luật đặt trước, khóa v2

**Files:**
- Modify: `mmf/compare.py`, `docs/reports/mmf_preregistration.md` (chỉ nối thêm)
- Test: `tests/test_mmf_compare.py`

**Interfaces:**
- Produces (`mmf/compare.py`):
  - `V2 = tuple(f"{n}@{v}" for v in C.VARIANTS for n in (*C.V2_BASELINES[v], *C.V2_DL[v]))` (12 tên).
  - `LADDER`: 10 cặp `(a, b)` cố định: với `m` trong `logit_price, gbm_price, dl_price`: `(m@ext, m)`, `(m@ext_ctx, m@ext)`; với `k` trong `logit, gbm`: `(hurdle_k_price@base, k_price)`, `(hurdle_k_price@ext_ctx, k_price@ext_ctx)`.
  - `tables(preds)` trả thêm `"r4"` và `"ladder"` (DataFrame rỗng khi `preds` không có arm nào thuộc `V2`). `r4`, mỗi h hai dòng qua `_row`: `a` = pick theo calib trong `V2` của từng fold; `b` = pick `baseline` của v1, rồi `b` = `dl_price`; thêm cột `so_sanh` (`"v2 so với baseline"`, `"v2 so với dl_price"`). `ladder`: mỗi h mười dòng qua `_row` với arm cố định ở mọi fold. Bảng R1–R3 và `picks` không đổi.
  - `write_report`: khi `r4` không rỗng, thêm hai mục "R4: v2 so với v1" và "Từng bước cải tiến" sau R3, kèm bốn dòng giới hạn ở Global Constraints.
  - `write_lock_v2(run_dir, out=C.REPORTS / "mmf_lock_v2.json")`: mỗi h một mục `{"model": tên V2 được pick nhiều fold nhất (hòa: log-loss calib trung bình thấp hơn, dùng `_most_picked`), "price_kind": giá trị xuất hiện nhiều nhất trong các `select@<variant>.json` của variant đó (hòa: đứng trước trong `C.PRICE_KINDS`; model không phải `dl_price` → `"none"`), "aux_id", "dataset_id", "git_commit", "created_at", "picks": {fold: tên}}`. `out` đã tồn tại → `ValueError`. Thiếu fold nào của h nào → `ValueError`. CLI: `--lock-v2`.

- [ ] **Step 1: Nối phụ lục vào `mmf_preregistration.md`** (trước khi chạy Task 7; **chủ dự án đọc và đồng ý nội dung này trước khi commit**):

  ```markdown
  ## Phụ lục 2026-11-10 (v2 nhánh giá, viết trước khi chạy bất kỳ ứng viên v2 nào)
  Ứng viên v2 là 12 cấu hình cố định: `hurdle_logit_price@base`, `hurdle_gbm_price@base`, và với mỗi variant `ext`, `ext_ctx`: `logit_price`, `gbm_price`, `hurdle_logit_price`, `hurdle_gbm_price`, `dl_price`. `ext` huấn luyện thêm trên các mã VN100 ngoài 15 mã; tune, calib, eval vẫn chỉ 15 mã. `ext_ctx` thêm 19 đặc trưng bối cảnh (VNINDEX, độ rộng thị trường, S&P 500, VIX, Brent, USD/VND, hạng cắt ngang).
  Chọn: trong mỗi (h, fold), ứng viên v2 có log-loss calib thấp nhất. Khóa: mỗi h lấy ứng viên được chọn ở nhiều fold nhất, ghi vào `mmf_lock_v2.json`.
  R4, theo từng h, cùng phép đo và thang diễn giải như R1: v2 so với baseline tốt nhất của v1, và v2 so với `dl_price` của v1.
  Trên vùng chấm cuối, v2 được chấm trong cùng lượt chạy với v1 (khối 5 phiên, 2000 lần), một lần cho mỗi h. Kết quả R1–R3 của v1 không đổi cách tính.
  Giới hạn biết trước: fold A, B, C đã bị nhìn nhiều lần, nên R4 trên vùng phát triển chỉ để chọn; danh sách VN100 lấy ngày 2026-11-10 (thiên lệch sống sót ở tập train của mã ngoài). Vùng chấm cuối chưa được đọc khi viết phụ lục này.
  ```
- [ ] **Step 2: Viết test thất bại** trong `tests/test_mmf_compare.py` (dùng hàm dựng `preds` tổng hợp có sẵn của file, thêm arm tên `@`):
  - `test_v2_names`: `len(V2) == 12`, `len(LADDER) == 10`, mọi tên trong `LADDER` thuộc `V2` hoặc `C.BASELINES` hoặc là `dl_price`.
  - `test_tables_without_v2_arms_are_unchanged`: `tables(preds)["r4"].empty`, và `r1`, `r2`, `r3`, `picks` bằng kết quả trước khi sửa (so với giá trị test cũ đang kiểm).
  - `test_r4_picks_v2_by_calib_and_compares_on_eval`: hai ứng viên v2, một cái tốt hơn trên calib nhưng tệ hơn trên eval → `a` của `r4` là cái tốt trên calib; `mean` có dấu đúng theo eval.
  - `test_ladder_row_missing_arm_gives_not_enough`: thiếu `dl_price@ext` ở fold B → dòng đó có verdict `NOT_ENOUGH` và `skipped == "B"`.
  - `test_write_lock_v2_refuses_overwrite_and_missing_fold`.
- [ ] **Step 3: Chạy, thấy FAIL.**
- [ ] **Step 4: Cài đặt.**
- [ ] **Step 5: Chạy** `.venv_mmf/bin/python -m pytest tests -q` → PASS.
- [ ] **Step 6: Commit** `feat(mmf): v2 comparison tables, pre-registered R4 and v2 lock`.

---

### Task 7: Chạy vùng phát triển và báo cáo

**Files:**
- Create: `docs/reports/mmf_v2_dev_report.md`

**Interfaces:**
- Consumes: CLI của Task 4 và Task 6; `$AUX` từ Task 2 Step 5.

- [ ] **Step 1: Chạy arm v1 vào run dir mới** (khoảng 7 phút trên GPU): `SNAP=Pipeline_V1.2/stock/snapshots/24251acc1a5695bd; .venv_mmf/bin/python -m mmf.experiment --name v2 --stock-dir $SNAP`.
- [ ] **Step 2: Chạy variant:** `.venv_mmf/bin/python -m mmf.experiment --name v2 --stock-dir $SNAP --variants base ext ext_ctx --aux-dir $AUX`. Ước lượng 40–60 phút (train lớn gấp khoảng 6 lần). Bị ngắt thì chạy lại đúng lệnh, bộ chạy tự bỏ qua file đã xong.
- [ ] **Step 3: Kiểm rò rỉ bằng nhãn xáo trộn:** `.venv_mmf/bin/python -m mmf.experiment --stock-dir $SNAP --shuffle-labels --variants ext_ctx --aux-dir $AUX`, rồi chấm thư mục `data/mmf/shuffle/...` bằng `mmf.compare --run-dir ... --out <scratchpad>/shuffle.md`. Kỳ vọng: mọi arm `@ext_ctx` có AUC Tăng/Giảm trong khoảng 0.47–0.53 và log-loss không thấp hơn `prior_fixed`. Ngoài khoảng đó: dừng, tìm rò rỉ (skill `superpowers:systematic-debugging`), không viết báo cáo.
- [ ] **Step 4: Sinh báo cáo:** `.venv_mmf/bin/python -m mmf.compare --run-dir data/mmf/exp/v2/24251acc1a5695bd --out docs/reports/mmf_v2_dev_report.md`. Bảng "Cảnh báo rò rỉ" có dòng nào thuộc `V2` thì quay lại Step 3.
- [ ] **Step 5: Viết thêm ở đầu báo cáo** một đoạn tóm tắt tối đa 10 dòng: verdict của R4 theo từng h; trong 10 dòng của bảng từng bước, bước nào có lợi ở ít nhất 2/3 fold; AUC Tăng/Giảm của ứng viên được chọn so với `dl_price` v1. Không có cải thiện thì viết đúng như vậy, không gọi tín hiệu yếu là thành công.
- [ ] **Step 6: Commit** `exp(mmf): v2 dev-region run and report`.
- [ ] **Step 7: (chủ dự án) Đọc báo cáo và quyết định** có khóa v2 không. Đồng ý: `.venv_mmf/bin/python -m mmf.compare --run-dir data/mmf/exp/v2/24251acc1a5695bd --out <scratchpad>/tmp.md --lock-v2`, commit `docs/reports/mmf_lock_v2.json`. Không đồng ý: dừng plan ở đây, Task 8 không cần làm.

---

### Task 8: Chấm v2 trong lượt holdout, tài liệu

**Files:**
- Modify: `mmf/holdout.py`, `docs/mmf_thu_nghiem.md`
- Test: `tests/test_mmf_holdout.py`

**Interfaces:**
- Consumes: `X.variant_bundle`, `X.out_name`, `X.run_baseline`, `X.run_arm`, `aux.load`, `mmf_lock_v2.json` (dạng ở Task 6).
- Produces (`mmf/holdout.py`):
  - `LOCK_V2_FILE = C.REPORTS / "mmf_lock_v2.json"`; `load_lock_v2() -> dict | None` (`None` khi không có file).
  - `run(h, panel, guard_dir=C.REPORTS, early=False, device="cuda", aux=None)`: có khóa v2 mà `aux is None` → `ValueError` **trước** mọi bước huấn luyện và trước guard. Có khóa v2: dựng `X.variant_bundle(panel, aux, h, <cùng Fold F như _bundle>, variant)`, huấn luyện model đã khóa (tên `hurdle_*`, `logit_price`, `gbm_price` qua `run_baseline`; `dl_price` qua `run_arm` với mọi `C.SEEDS` và `price_kind` trong khóa) vào `base/<model@variant>/`, **xong hết rồi mới ghi guard**. Không có khóa v2: hành vi cũ.
  - `_comparisons(preds, entry, early, v2=None)`: có `v2` thì thêm hai dòng `("R4", v2, entry["baseline"])` và `("R4", v2, entry["unimodal"])`, cùng `C.HOLDOUT_BLOCK`.
  - Báo cáo holdout in thêm `aux_id` và tên model v2.
  - CLI: `--aux-dir` (truyền qua `aux.load(a.aux_dir, panel.calendar)`).

- [ ] **Step 1: Viết test thất bại** trong `tests/test_mmf_holdout.py` (theo mẫu test tổng hợp có sẵn của file; `LOCK_V2_FILE` trỏ vào `tmp_path` qua `monkeypatch`):
  - `test_without_v2_lock_report_has_no_r4`: không có file khóa v2 → bảng so sánh chỉ có R1, R2, R3 (các test cũ PASS không sửa).
  - `test_v2_lock_without_aux_raises_before_guard`: có khóa v2, `aux=None` → `ValueError`, không guard nào được tạo, không file dự báo nào được ghi.
  - `test_v2_lock_adds_two_r4_rows`: khóa v2 `{"model": "hurdle_logit_price@base", "price_kind": "none", ...}` và một `Aux` tổng hợp → bảng so sánh có đúng hai dòng `R4`, `a == "hurdle_logit_price@base"`.
  - `test_v2_failure_leaves_no_guard`: `monkeypatch` `X.variant_bundle` ném lỗi → không guard nào tồn tại sau khi `run` raise.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt.**
- [ ] **Step 4: Cập nhật `docs/mmf_thu_nghiem.md`:** thêm mục "Variant v2" (lệnh `--variants`, `--aux-dir`, bảng sửa ở đâu: `mmf/context.py`, `mmf/aux.py`, `mmf/globaldata.py`), và mục "Trước lượt chấm cuối" gồm đúng ba lệnh theo thứ tự, do chủ dự án chạy: cập nhật `stock_ext` (`sync_stock_parquet.py --stock-dir stock_ext`), `python -m mmf.globaldata`, `python -m mmf.aux freeze`; rồi `python -m mmf.holdout --h 2 --aux-dir <đường dẫn vừa in>`. Sửa câu "Hiện `mmf.holdout` chỉ đọc file khóa v1" cho đúng hiện trạng.
- [ ] **Step 5: Chạy toàn bộ:** `.venv_mmf/bin/python -m pytest tests -q` và `cd Pipeline_V1.2 && $MAIN/NLP_Module/.venv/bin/python -m pytest tests -q` → 0 FAIL, số test ≥ baseline cộng test mới. `git status` không có thay đổi ở `docs/reports/mmf_lock.json`, `mmf_dev_report.md`.
- [ ] **Step 6: Commit** `feat(mmf): score the locked v2 model in the single holdout run`, rồi dùng skill `superpowers:finishing-a-development-branch` để merge vào `multimodal-decision`. Phải merge trước khi chủ dự án chạy `mmf.holdout` (dự kiến cuối tháng 12/2026).
