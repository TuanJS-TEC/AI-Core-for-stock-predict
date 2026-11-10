# Price Branch Overhaul (P0–P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Làm nhánh dữ liệu giá của `Pipeline_V1.2` có thể tái lập và không rò rỉ: sync có nguồn/snapshot/hash, lịch phiên kiểm chứng, feature–nhãn–split nhân quả dùng chung train/inference, và `predict` từ chối khi dữ liệu không hợp lệ.

**Architecture:** P0 sửa `sync_stock_parquet.py` và thêm `src/validation.py`, `src/snapshot.py`. P1 thêm `src/{price_features,labels,splits,dataset}.py` chạy trên bảng **theo lịch phiên, reindex không điền** (phiên thiếu = hàng NaN, nên `shift`/`rolling` không bao giờ trượt sang phiên khác). P2 nối `dataset_id` vào `pipeline.py`/`predict.py`. Đường cũ (`build_pooled`, `models.py`, model đã lưu, `ai_core`, `volcast`) giữ nguyên hành vi: các module P1 là đường mới chạy song song, **không** đổi bảng live `decision-v0-nogo`.

**Tech Stack:** Python (`NLP_Module/.venv`, môi trường thật của cron): pandas **2.3.3**, numpy **2.2.6**, pyarrow 26.0.0, pytest 9.1.1, scikit-learn 1.9.1, vnstock 4.0.9. (`requirements.txt` ghi pandas 3.0.6/numpy 2.5.4 nhưng venv thực tế không phải vậy; mã mới phải chạy đúng trên bản đã cài, không dùng tính năng chỉ có ở pandas 3, và không dùng `pct_change`.)

**Spec:** Đề xuất "Đề xuất cải tổ nhánh dữ liệu giá" do người dùng dán trong phiên 2026-11-09 (không lưu thành file) cộng phần đối chiếu mã trong cùng phiên. Mọi giá trị cần dùng đã chép vào Global Constraints bên dưới; nếu có mâu thuẫn, Global Constraints thắng.

## Global Constraints

- Phạm vi: dữ liệu giá ngày, horizon `h ∈ {2, 5}`, ba lớp 0 Giảm / 1 Đi ngang / 2 Tăng. Không đụng NLP, BCTC, hợp nhất đa phương thức.
- **Không sửa** `NLP_Module/` (repo riêng), `ai_core/`, và không đổi `ai_core.live.VERSION = "decision-v0-nogo"`. Test của `ai_core` (`cd AI_Core && $PY -m pytest tests`) phải vẫn PASS ở cuối.
- `$PY = /home/administrator/TLS/tuanhm@linux/AI_Core/NLP_Module/.venv/bin/python`. Test Pipeline chạy từ `Pipeline_V1.2/`: `$PY -m pytest tests -q`.
- Cron `20 15 * * 1-5` chạy `sync_stock_parquet.py` và `*/15` chạy `ai_core.live` **từ cây làm việc chính**. Làm trong git worktree (Task 1 Step 1), chỉ merge khi toàn bộ test PASS.
- `src/config.py`, `src/predict.py`, `src/pipeline.py` dùng **CRLF**; `data.py`, `sync_stock_parquet.py`, `tests/*` dùng LF; file mới dùng LF. Sau khi sửa file CRLF chạy `file <path>` để chắc vẫn là CRLF.
- Nguồn dữ liệu vnstock: `Quote(source=C.SYNC_SOURCE, symbol=...)` với `SYNC_SOURCE = "kbs"`. Đã kiểm 2026-11-09 trên vnstock 4.0.9 (bản mới nhất trên index vnstocks.com): `Quote.__init__(source=DataSource.KBS, symbol='', ...)`, `history(symbol, start, end, interval)`; `DataSource` gồm kbs, vci, msn, dnse; ACB 2026-10-12..11-09 qua mặc định và `source="kbs"` cho dữ liệu giống hệt nhau và khớp parquet đã lưu. `vci` cũng khớp giá nhưng trả thêm phiên trước `start` và giờ `00:00`, nên không dùng.
- Tải chồng `SYNC_OVERLAP_SESSIONS = 20` phiên gần nhất. Ngưỡng so giá giữ `SYNC_BASIS_TOL = 0.005` (nghìn đồng), áp cho open/high/low/close.
- Volume lỗi/NaN **không** thành 0; volume 0 thật được giữ. Không forward-fill/backfill, không clip, không winsorize.
- `adjustment_policy` ghi `"unknown"`; không tự gọi là total-return.
- Nhãn: `θ = k · σ60 · √h`, `k = K_ADAPT = 0.43`, `σ60` = std (ddof=1) của log-return 1 phiên trong 60 phiên kết thúc tại `t` (`min_periods=60`); `|R| ≤ θ` là Đi ngang (đúng ±θ cũng là Đi ngang); `θ` NaN hoặc ≤ 0 → không nhãn; nhãn chưa tới hạn là `<NA>`, không phải Đi ngang. `LABEL_VERSION = "adaptive-k0.43-w60-v1"`.
- Split: khối `BLOCK_SESSIONS = 63` phiên (tune, calib, eval), 3 fold A/B/C, eval cuối của fold C là `2026-10-30`; mẫu thuộc khối chỉ khi `label_end_date ≤ cuối khối`.
- Rolling luôn trailing, `min_periods` khai báo tường minh, không `center`.
- Mọi commit kết thúc bằng dòng `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

**Không làm trong plan này** (đã cân nhắc, để sau): bảng thí nghiệm A→E trên nhãn ba lớp; VNINDEX (không có nguồn, dùng trung bình đều của vũ trụ làm proxy thị trường); `available_at` theo từng dòng (không có bằng chứng vintage; chỉ lưu `fetched_at` ở mức snapshot); volume điều chỉnh; retrain `volcast`.

## Review Focus

1. Một mã mất phiên cuối (API lỗi) trong khi mã khác có đủ → chỉ mã đó `features_ok=False`; không điền, không trượt. (Task 8)
2. Volume NaN/âm, `H == L`, volume 0: không `inf`/NaN ngầm, không thành 0 giả. (Task 2, Task 5)
3. File ngày nghỉ thiếu một ngày thật nghỉ, hoặc liệt kê nhầm một ngày có dữ liệu: ngày thiếu → NaN và `missing_session` quanh đó, ngày có dữ liệu luôn thắng danh sách. (Task 1, Task 4, Task 6)
4. Corporate action làm lệch open/high/low nhưng close khớp; tải lại toàn bộ thất bại: giữ nguyên dữ liệu đã lưu. (Task 2)
5. Ghi parquet/CSV bị ngắt giữa chừng, `stock/` chưa có `manifest.json` (thư mục cũ, test): không để hai file lệch nhau, không crash. (Task 2, Task 3)

---

### Task 1: Validation, lịch ngày nghỉ, worktree (P0)

**Files:**
- Create: `Pipeline_V1.2/src/validation.py`, `Pipeline_V1.2/data/hose_holidays.csv`, `Pipeline_V1.2/tests/test_validation.py`
- Modify: `Pipeline_V1.2/src/config.py` (mục 9; CRLF)

**Interfaces:**
- Produces (`src/config.py`): `SYNC_SOURCE = "kbs"`, `SYNC_OVERLAP_SESSIONS = 20`, `HOLIDAYS_FILE = ROOT / "data" / "hose_holidays.csv"`.
- Produces (`src/validation.py`):
  - `check_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`: `df` có cột `time, open, high, low, close, volume` (số, có thể NaN). Trả `(good, bad)`; `bad` giữ chỉ số gốc và thêm cột `reason` ∈ {`"nonpositive_price"` (OHLC NaN hoặc ≤ 0), `"ohlc_order"` (high < max(open, close, low) hoặc low > min(open, close, high)), `"volume_invalid"` (NaN hoặc < 0)}, kiểm theo đúng thứ tự đó. Volume 0 hợp lệ.
  - `load_holidays(path=C.HOLIDAYS_FILE) -> np.ndarray` (`datetime64[D]`): ngày trong CSV (cột `date,verified`) ∪ `C.MARKET_HOLIDAYS`.
  - `trading_calendar(first, last, holidays: np.ndarray, observed=()) -> pd.DatetimeIndex`: ngày thường trong `[first, last]` trừ `holidays`, hợp với `observed` (ngày có dữ liệu thật), sắp xếp.
  - `unexplained_gaps(dates: pd.DatetimeIndex, holidays: np.ndarray) -> pd.DatetimeIndex`: ngày thường trong `[dates.min(), dates.max()]` không có trong `dates` và không nằm trong `holidays`.

- [ ] **Step 1: Tạo worktree.** Dùng skill `superpowers:using-git-worktrees`: nhánh `price-branch-overhaul` từ `multimodal-decision`. Trong worktree: `ln -s /home/administrator/TLS/tuanhm@linux/AI_Core/Pipeline_V1.2/stock Pipeline_V1.2/stock` (chỉ đọc; `stock/` bị gitignore nên test dùng dữ liệu thật cần symlink này). Chạy `cd Pipeline_V1.2 && $PY -m pytest tests -q`, ghi lại số test PASS làm baseline.
- [ ] **Step 2: Viết test thất bại** `tests/test_validation.py`:
  - `test_check_ohlcv_quarantines_with_reason`: 5 dòng, `open=[10,10,10,10,-1]`, `high=[11,9,11,11,1]`, `low=[9,8,9,9,1]`, `close=[10,10,10,10,1]`, `volume=[100,100,nan,-5,0]`. Kỳ vọng `good.index.tolist() == [0]` và `dict(zip(bad.index, bad.reason)) == {1: "ohlc_order", 2: "volume_invalid", 3: "volume_invalid", 4: "nonpositive_price"}`.
  - `test_zero_volume_is_valid`: một dòng hợp lệ với `volume=0` nằm trong `good`.
  - `test_unexplained_gaps`: ngày `2026-10-19, 20, 22, 23` → `unexplained_gaps(d, np.array([], "datetime64[D]"))` là `[2026-10-21]`; với `holidays=["2026-10-21"]` rỗng; cuối tuần không bao giờ là gap.
  - `test_trading_calendar_observed_day_beats_holiday_list`: `first=2026-10-19, last=2026-10-23, holidays=["2026-10-21"]` → không chứa 10-21; thêm `observed=[2026-10-21]` → chứa.
  - `test_load_holidays_unions_csv_and_config`: CSV tạm có `2026-04-30` → kết quả chứa `2026-04-30` và `2027-01-01`.
- [ ] **Step 3: Chạy, thấy FAIL:** `cd Pipeline_V1.2 && $PY -m pytest tests/test_validation.py -v` → ImportError/AttributeError.
- [ ] **Step 4: Cài đặt** bốn hàm trên trong `src/validation.py` và ba hằng trong `src/config.py` (mục 9). `load_holidays` dùng `pd.read_csv`.
- [ ] **Step 5: Sinh danh sách ứng viên.** Chạy một lần bằng `$PY` từ `Pipeline_V1.2/` trên dữ liệu thật: lấy ngày thường nằm giữa phiên đầu và cuối mà **không mã nào** trong 20 mã có dữ liệu; ghi `data/hose_holidays.csv` với cột `date,verified`, mọi dòng `verified=false`. Kỳ vọng ~69 dòng.
- [ ] **Step 6: (người dùng) Đối chiếu** từng dòng với thông báo nghỉ lễ của HOSE rồi đổi `verified=true`. Agent không tự xác nhận được; để nguyên `false` nếu chưa kiểm.
- [ ] **Step 7: Xác nhận** `$PY -m pytest tests/test_validation.py -v` PASS và kiểm tra một lần: `unexplained_gaps` của trục ngày chung 20 mã (với `load_holidays()`) rỗng. Không biến kiểm tra này thành test (một ngày API lỗi sẽ làm test hỏng ngẫu nhiên).
- [ ] **Step 8: Commit** `feat(price): validation helpers and holiday calendar` (kiểm `file src/config.py` vẫn là CRLF trước khi commit).

---

### Task 2: Sync cứng hóa (P0)

**Files:**
- Modify: `Pipeline_V1.2/sync_stock_parquet.py`
- Test: `Pipeline_V1.2/tests/test_sync.py`

**Interfaces:**
- Consumes: `validation.check_ohlcv`, `C.SYNC_SOURCE`, `C.SYNC_OVERLAP_SESSIONS`, `C.SYNC_BASIS_TOL`.
- Produces:
  - `clean_types(df) -> pd.DataFrame` (chữ ký cũ): ép số, chạy `check_ohlcv`, **chỉ trả dòng hợp lệ**, `volume` là `int64`; dòng bị loại ghi `logger.error` kèm `reason`.
  - `basis_changed(stored, merged) -> bool` (chữ ký cũ): so cả `open/high/low/close` trên phiên chung.
  - `overlap_start(stored: pd.DataFrame, n: int = C.SYNC_OVERLAP_SESSIONS) -> str`: ngày `YYYY-MM-DD` của phiên thứ `n` tính từ cuối (hoặc phiên đầu nếu ít hơn `n`).
  - `save_raw(df_raw: pd.DataFrame, stock_dir: str, symbol: str) -> Path`: ghi `stock_dir/raw/<YYYYMMDD hôm nay>/<SYM>.parquet`.
  - `write_pair(df: pd.DataFrame, parquet_path: str, csv_path: str | None) -> None`: ghi file tạm cùng thư mục, rồi `os.replace` parquet rồi CSV; lỗi bất kỳ → xóa file tạm, raise, **không** thay file nào.

- [ ] **Step 1: Viết test thất bại** trong `tests/test_sync.py` (dùng `frame()` và mẫu `_run` có sẵn):
  - `test_clean_types_drops_nan_volume_instead_of_zeroing`: 2 dòng, volume dòng 2 là `None` → `len(out) == 1`, `out.close.tolist() == [1.0]`, `out.volume.dtype == "int64"`.
  - `test_basis_changed_sees_open_high_low`: `merged` = `stored` với `open[0] += 0.5`, close giữ nguyên → `True`; `basis_changed(stored, stored)` → `False`.
  - `test_quote_gets_explicit_source`: thay `S.Quote` bằng lớp ghi `**kw` của `__init__`, `history` trả `DataFrame()`; sau `fetch_latest_ohlcv("AAA", "2026-10-01", "2026-10-02")` thì `kw == {"source": C.SYNC_SOURCE, "symbol": "AAA"}`.
  - `test_overlap_start_is_20th_last_session`: 30 phiên → `overlap_start(stored) == stored.time.iloc[-20]` dạng `%Y-%m-%d`; với 5 phiên → phiên đầu.
  - `test_process_fetches_from_overlap_start`: `fetch` giả ghi `start_date`; `calls == [overlap_start(stored)]`.
  - `test_raw_response_saved_before_merge`: sau khi chạy với fetch trả một phiên mới, `len(list((tmp_path/"raw").glob("*/AAA.parquet"))) == 1`.
  - `test_failed_csv_write_keeps_both_old_files`: chạy một lần tạo parquet+CSV, đọc bytes cả hai; sau đó `monkeypatch` `pd.DataFrame.to_csv` ném `OSError`, chạy lại với fetch trả phiên mới → bytes parquet và CSV không đổi, không còn `*.tmp` trong thư mục.
- [ ] **Step 2: Chạy, thấy FAIL:** `$PY -m pytest tests/test_sync.py -v`.
- [ ] **Step 3: Cài đặt** các hàm trên trong `sync_stock_parquet.py`: `Quote(source=C.SYNC_SOURCE, symbol=symbol)`; `start_date_str = overlap_start(df_csv)` thay cho ngày cuối; `save_raw` gọi ngay sau khi `fetch_latest_ohlcv` trả dữ liệu (trước `clean_types`); thay hai khối ghi ở bước 4 và 5 bằng một lần `write_pair` bọc `try/except` (lỗi → `logger.error`, dữ liệu cũ giữ nguyên). `basis_changed` giữ nguyên nhánh bước nhảy.
- [ ] **Step 4: Chạy toàn bộ** `$PY -m pytest tests/test_sync.py -v`; kỳ vọng PASS, gồm các test cũ.
- [ ] **Step 5: Kiểm tra trên dữ liệu thật, không gọi API:** sao `stock/` sang `/tmp/claude-1000/.../scratchpad/stock_copy` (đường dẫn scratchpad của phiên), chạy `$PY sync_stock_parquet.py --stock-dir <copy> --no-fetch`; kỳ vọng mọi parquet có `len` và `close` bằng bản gốc (so bằng pandas), log không có dòng loại bỏ.
- [ ] **Step 6: Commit** `fix(sync): explicit source, 20-session overlap, OHLC basis check, no zero volume, paired atomic write`.
- [ ] **Step 7: (sau khi merge, trước phiên cron kế tiếp)** sao lưu `cp -a Pipeline_V1.2/stock Pipeline_V1.2/stock_backup_2026-11-09` (ngoài git), chạy tay `--symbols ACB` và đọc log: nếu `basis_changed` kích hoạt tải lại toàn bộ thì đó là hành vi đúng (open/high/low cũ lệch), nhưng cần biết trước.

---

### Task 3: Manifest, dataset_id, snapshot bất biến (P0)

**Files:**
- Create: `Pipeline_V1.2/src/snapshot.py`, `Pipeline_V1.2/tests/test_snapshot.py`
- Modify: `Pipeline_V1.2/sync_stock_parquet.py` (`main`), `Pipeline_V1.2/src/data.py`

**Interfaces:**
- Consumes: `validation.load_holidays`, `validation.unexplained_gaps`.
- Produces (`src/snapshot.py`):
  - `content_hash(df: pd.DataFrame) -> str`: sha256 hex của CSV chuẩn hóa (`RAW_COLS`, sắp theo `time`, `time` dạng ISO UTC, `float_format="%.6f"`).
  - `build_manifest(stock_dir: Path, source: str, fetched_at: str, fetched: bool) -> dict`: khóa `dataset_id` (sha256 của các dòng `SYM:content_sha256` sắp theo mã), `fetched_at`, `fetched`, `source`, `adjustment_policy: "unknown"`, `libs` (`vnstock`, `pandas`, `pyarrow`; `None` nếu không cài), `tickers` (`{SYM: {rows, first, last, content_sha256, file_sha256}}`), `quality` (`{SYM: {"unexplained_gaps": [ISO date,...]}}`), `holidays_unverified: int`.
  - `write_manifest(stock_dir: Path, manifest: dict) -> None`: ghi `stock_dir/manifest.json` qua file tạm + `os.replace`.
  - `snapshot_dir(stock_dir: Path, dataset_id: str) -> Path`: `stock_dir/snapshots/<dataset_id[:16]>`.
  - `freeze(stock_dir: Path) -> str`: đọc `manifest.json`, sao parquet + manifest vào `snapshot_dir`; đã tồn tại và nội dung khớp → trả `dataset_id`, khác → `ValueError`. CLI: `python -m src.snapshot freeze` in `dataset_id`.
- Produces (`src/data.py`): `load_ohlcv(ticker, stock_dir=None)`, `load_raw(tickers, stock_dir=None)`, `build_panel(tickers, raw=None, stock_dir=None)` nhận thư mục dữ liệu (mặc định `C.STOCK_DIR`); `Panel` thêm trường cuối `dataset_id: str | None = None`, đọc từ `stock_dir/manifest.json` nếu có.

- [ ] **Step 1: Viết test thất bại** `tests/test_snapshot.py` (dữ liệu tổng hợp 2 mã trong `tmp_path`, 12 phiên):
  - `test_content_hash_changes_when_one_ohlc_value_changes_with_same_rows_and_last_date`; ổn định khi đảo thứ tự dòng.
  - `test_dataset_id_changes_with_content_not_with_fetched_at`: hai manifest cùng dữ liệu, khác `fetched_at` → cùng `dataset_id`; đổi một `high` → khác.
  - `test_manifest_records_policy_unknown_and_gaps`: `adjustment_policy == "unknown"`; bỏ một ngày thường không trong holidays → `quality[sym]["unexplained_gaps"]` chứa ngày đó.
  - `test_write_manifest_atomic_leaves_no_tmp`.
  - `test_freeze_is_idempotent_and_refuses_overwrite`: `freeze` hai lần cùng id; sửa file trong snapshot rồi `freeze` lại với dữ liệu gốc → `ValueError`.
  - `test_revision_isolation`: `freeze`, sau đó ghi đè parquet sống bằng giá khác và cập nhật manifest; `build_panel(tickers, stock_dir=snapshot_dir(...))` vẫn trả `log_close` cũ và `dataset_id` cũ.
  - `test_panel_without_manifest_has_none_dataset_id`.
- [ ] **Step 2: Chạy, thấy FAIL:** `$PY -m pytest tests/test_snapshot.py -v`.
- [ ] **Step 3: Cài đặt** `snapshot.py` và tham số `stock_dir` ở `data.py` (`stock_path(ticker, stock_dir=None)`). Trong `sync_stock_parquet.main`, sau `check_alignment`, gọi `write_manifest(stock_dir, build_manifest(stock_dir, C.SYNC_SOURCE, <ISO UTC now>, fetched=not args.no_fetch))`.
- [ ] **Step 4: Chạy** `$PY -m pytest tests -q`; kỳ vọng PASS toàn bộ (test cũ không đổi vì `Panel` thêm trường có mặc định).
- [ ] **Step 5: Commit** `feat(price): content-hash manifest, dataset_id and immutable snapshots`.

---

### Task 4: Panel giữ OHLCV chuẩn và bảng theo lịch (P1)

**Files:**
- Modify: `Pipeline_V1.2/src/data.py`
- Test: `Pipeline_V1.2/tests/test_data_canonical.py`

**Interfaces:**
- Consumes: `validation.{load_holidays, trading_calendar, unexplained_gaps}`.
- Produces: `Panel` thêm ba trường mặc định `None` sau `dataset_id`: `ohlcv: dict[str, pd.DataFrame] | None` (chỉ số `session_date` tz-naive, cột `open, high, low, close, volume` float64), `calendar: pd.DatetimeIndex | None`, `gaps: pd.DatetimeIndex | None`; method `Panel.on_calendar(ticker: str) -> pd.DataFrame` = `ohlcv[ticker].reindex(calendar)` **không điền**. `build_panel(tickers, raw=None, stock_dir=None, holidays=None)` điền các trường (`holidays=None` → `load_holidays()`); khi `gaps` không rỗng gọi `warnings.warn` nêu số ngày. `log_close` giữ nguyên giá trị cũ.

- [ ] **Step 1: Viết test thất bại** `tests/test_data_canonical.py` (parquet tổng hợp 3 mã trong `tmp_path`, 80 phiên, bỏ phiên thứ 40 ở cả 3 mã):
  - `test_gap_not_in_holidays_is_reported_and_left_nan`: `holidays=[]` → `panel.gaps == [ngày bỏ]`, `panel.on_calendar("AAA").loc[ngày bỏ].isna().all()`, và hàng liền sau vẫn là giá thật (không điền).
  - `test_gap_listed_as_holiday_is_not_a_gap`: `holidays=[ngày bỏ]` → `panel.gaps` rỗng, `ngày bỏ not in panel.calendar`.
  - `test_data_day_beats_holiday_list`: `holidays` chứa một ngày **có** dữ liệu → ngày đó vẫn có trong `calendar` và `on_calendar` có giá.
  - `test_log_close_unchanged`: `panel.log_close[t]` bằng `np.log(close)`.
  - `test_legacy_panel_constructor_still_works`: `Panel(dates, stamps, {...})` không truyền trường mới, `ohlcv is None`.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt** trong `data.py`; `calendar = trading_calendar(dates[0], dates[-1], holidays, observed=dates)`, `gaps = unexplained_gaps(dates, holidays)`.
- [ ] **Step 4: Chạy** `$PY -m pytest tests -q` (gồm `test_pipeline.py` dùng dữ liệu thật) → PASS; `verified=false` trong `hose_holidays.csv` chỉ gây cảnh báo; nếu `panel.gaps` khác rỗng với dữ liệu thật thì quay lại Task 1 Step 5–6.
- [ ] **Step 5: Commit** `feat(price): Panel carries canonical OHLCV, trading calendar and gaps`.

---

### Task 5: Đặc trưng nhân quả (P1)

**Files:**
- Create: `Pipeline_V1.2/src/price_features.py`, `Pipeline_V1.2/tests/test_price_features.py`

**Interfaces:**
- Consumes: bảng theo lịch từ `Panel.on_calendar` (cột `open, high, low, close, volume`, chỉ số lịch, hàng thiếu là NaN).
- Produces:
  - `FEATURES: list[str]` đúng 25 tên theo thứ tự: `ret_1 ret_2 ret_5 ret_10 ret_20 ret_60 d_ma20 d_ma60 dd_60 std_10 std_20 std_60 std_ratio_20_60 downside_20 c_o h_l o_pc close_loc v_ma20 logv_rel60 mret1 mret5 mvol20 ex_ret1 ex_ret5`.
  - `market_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame`: cột `mret1, mret5, mvol20`; `mret1` = trung bình (bỏ NaN) log-return 1 phiên của các mã, NaN nếu ít hơn `ceil(n/2)` mã có giá trị; `mret5` = tổng trượt 5 của `mret1`, `mvol20` = std trượt 20 của `mret1`, đều `min_periods` = cửa sổ.
  - `compute(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame`: chỉ số của `frame`, cột `FEATURES + ["flag_hl_equal"]`.
- Định nghĩa: `ret_k = log(C_t/C_{t-k})`; `d_ma_w = C/MA_w − 1`; `dd_60 = C/max_60(C) − 1`; `std_w` = std (ddof=1) của `ret_1` trên `w`; `std_ratio_20_60 = std_20/std_60`; `downside_20 = sqrt(mean(min(ret_1,0)²))` trên 20; `c_o = log(C/O)`; `h_l = log(H/L)`; `o_pc = log(O_t/C_{t-1})`; `close_loc = (C−L)/(H−L)`, NaN và `flag_hl_equal=True` khi `H==L`; `v_ma20 = V_t / mean(V_{t-20..t-1})` (NaN nếu mẫu số 0); `logv_rel60 = log1p(V_t) − mean(log1p(V_{t-60..t-1}))`; `ex_ret1 = ret_1 − mret1`, `ex_ret5 = ret_5 − mret5`. Mọi rolling `min_periods` = độ dài cửa sổ; `shift` theo hàng của bảng lịch.

- [ ] **Step 1: Viết test thất bại** `tests/test_price_features.py` (hàm dựng bước đi ngẫu nhiên OHLCV 250 phiên × 3 mã, seed cố định):
  - `test_prefix_invariance`: với `t` ở hàng 150, `compute(full, market_full).loc[:t]` bằng `compute(prefix, market_prefix).loc[:t]` (`assert_frame_equal`, `market_prefix` dựng từ prefix của cả 3 mã).
  - `test_future_perturbation`: nhân giá và volume mọi hàng sau `t` với 1.7 → đặc trưng và `market_features` tại các hàng `≤ t` không đổi.
  - `test_scale_invariance`: nhân open/high/low/close với 7.3 (volume giữ) → toàn bộ `FEATURES` `allclose` với bản gốc.
  - `test_equal_high_low_flagged_not_inf`: đặt `H=L=C` một hàng → `close_loc` NaN, `flag_hl_equal` True, không có `±inf` nào trong cả bảng.
  - `test_missing_session_row_is_nan_not_filled`: đặt cả hàng `g` là NaN → `ret_1` NaN tại `g` và `g+1`, `std_20` NaN từ `g` tới `g+20` và hợp lệ lại ở `g+21` (`ret_1` đã hợp lệ từ `g+2`; `ret_60` NaN tới `g+60`, hợp lệ ở `g+61`).
  - `test_volume_nan_stays_nan_zero_volume_is_valid`: volume NaN tại `g` → `v_ma20` NaN tại `g`; volume 0 (không phải 0 do NaN) → `logv_rel60` hữu hạn tại hàng đó.
  - `test_feature_count_and_names`: `len(FEATURES) == 25`, không trùng.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt** `FEATURES`, `market_features`, `compute` trong `src/price_features.py` theo định nghĩa trên.
- [ ] **Step 4: Chạy** `$PY -m pytest tests/test_price_features.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(price): causal price/volume/market features on the session calendar`.

---

### Task 6: Nhãn thích ứng có phiên bản (P1)

**Files:**
- Create: `Pipeline_V1.2/src/labels.py`, `Pipeline_V1.2/tests/test_labels.py`
- Modify: `Pipeline_V1.2/src/config.py` (mục 3; CRLF)

**Interfaces:**
- Consumes: bảng theo lịch (cột `close`); `data.assign_labels(r: np.ndarray, theta: float) -> np.ndarray` (đã có; ±θ là Đi ngang).
- Produces: `C.K_ADAPT = 0.43`, `C.SIGMA_WINDOW = 60`, `C.LABEL_VERSION = "adaptive-k0.43-w60-v1"`; `labels.make_labels(frame: pd.DataFrame, h: int, k: float = C.K_ADAPT, window: int = C.SIGMA_WINDOW) -> pd.DataFrame` với chỉ số = `as_of_session` (chỉ số của `frame`) và cột `target_session` (phiên thứ `h` kế tiếp **trên lịch**, `NaT` nếu vượt), `label_end_date` (= `target_session`), `realized_return`, `theta_used`, `label` (`Int64`, `<NA>` nếu không hợp lệ), `status` ∈ {`"ok"`, `"not_matured"`, `"missing_session"`, `"bad_sigma"`}, `label_version`. Thứ tự ưu tiên status: `not_matured` (vượt cuối lịch) → `missing_session` (có NaN `close` trong `[t, t+h]`) → `bad_sigma` (`θ` NaN hoặc ≤ 0) → `ok`.

- [ ] **Step 1: Viết test thất bại** `tests/test_labels.py`:
  - `test_target_follows_calendar_not_rows`: 120 phiên, hàng `g=70` là NaN. Với `as_of = g−2`, `h=5` → `target_session == frame.index[g+3]`, `status == "missing_session"`, `label` là `<NA>`; với `as_of = g+61` (cửa sổ σ đã qua gap), `h=5` → `status == "ok"`.
  - `test_theta_formula_and_prefix_invariance`: chuỗi lợi suất xen kẽ `+0.01/−0.012` → tại hàng 100, `theta_used == K_ADAPT * np.std(ret_1[41:101], ddof=1) * sqrt(h)`; `make_labels(prefix).theta_used` bằng bản đầy đủ tại các hàng của prefix.
  - `test_boundary_and_sign`: `assign_labels(np.array([-θ, θ, θ+1e-9, -θ-1e-9]), θ).tolist() == [1, 1, 2, 0]`.
  - `test_not_matured_is_na_never_flat`: `h` hàng cuối có `status == "not_matured"`, `label.isna()`, `realized_return` NaN.
  - `test_constant_price_gives_bad_sigma`: giá hằng → `status == "bad_sigma"`, `label.isna()`.
  - `test_label_version_and_columns`: cột đúng danh sách trên, `label_version` bằng `C.LABEL_VERSION` ở mọi hàng.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt** `make_labels`: log-return 1 phiên → `σ = rolling(window, min_periods=window).std()`; `target = close.shift(-h)`; NaN `close` trong `[t, t+h]` phát hiện bằng `isna().rolling(h+1).sum().shift(-h) > 0`; chỉ gọi `assign_labels` trên hàng `ok`.
- [ ] **Step 4: Chạy** `$PY -m pytest tests/test_labels.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(price): versioned volatility-adaptive labels with target/end dates`.

---

### Task 7: Split theo ngày và purge (P1)

**Files:**
- Create: `Pipeline_V1.2/src/splits.py`, `Pipeline_V1.2/tests/test_splits.py`
- Modify: `Pipeline_V1.2/src/config.py` (mục 5; CRLF)

**Interfaces:**
- Produces: `C.BLOCK_SESSIONS = 63`; `C.DATE_FOLDS: tuple[tuple[str, str, str, str, str], ...]` literal `(name, train_end, tune_end, calib_end, eval_end)` cho A, B, C (điền ở Step 4); `splits.Fold` (NamedTuple cùng 5 trường, ngày là `pd.Timestamp`); `splits.make_date_folds(calendar: pd.DatetimeIndex, last: str, block: int = 63, n: int = 3) -> list[Fold]`: fold C có `eval_end = last`, `calib_end/tune_end/train_end` lùi 63/126/189 phiên trên lịch; fold trước dịch thêm 63 phiên mỗi lần (các khối eval liền nhau, không chồng); trả theo thứ tự thời gian A, B, C; `splits.folds_from_config() -> list[Fold]`; `splits.split_samples(samples: pd.DataFrame, fold: Fold) -> dict[str, pd.DataFrame]` (khóa `train, tune, calib, eval`; `samples` có `as_of_session`, `label_end_date`; khối train: `as_of ≤ train_end`; các khối sau: `as_of ∈ (cuối khối trước, cuối khối]`; mọi khối chỉ giữ `label_end_date ≤ cuối khối`, `NaT` bị loại); `splits.split_manifest(samples, folds) -> pd.DataFrame` cột `fold, block, start, end, n_kept, n_purged`.

- [ ] **Step 1: Viết test thất bại** `tests/test_splits.py`:
  - `test_make_date_folds_blocks_are_63_sessions_and_eval_disjoint`: trên lịch 800 phiên tổng hợp, `last` là phiên 780 → `eval_end` của C là `last`; số phiên giữa các mốc liên tiếp là 63; `eval_end(B) == calib_end(C)`, `eval_end(A) == calib_end(B)`.
  - `test_purge_no_train_label_crosses_cutoff`: mẫu hằng ngày với `label_end = as_of + 5 phiên`; với mọi khối, `label_end_date.max() ≤ cuối khối` và `as_of_session.min() > cuối khối trước`; mẫu `as_of` cách `train_end` 2 phiên không nằm trong `train`.
  - `test_blocks_are_disjoint_and_cover_only_kept`: tập chỉ số các khối không giao nhau; `n_kept + n_purged` của manifest bằng số mẫu có `as_of` trong khoảng khối.
  - `test_appending_new_data_does_not_move_folds`: `split_samples` trên mẫu cắt tới `eval_end` và trên mẫu dài hơn cho cùng tập chỉ số ở cả bốn khối.
  - `test_config_folds_are_ordered_and_end_at_frozen_test_end`: `folds_from_config()` tăng dần theo ngày trong từng fold và giữa các fold; `eval_end` của C là `2026-10-30`.
- [ ] **Step 2: Chạy, thấy FAIL** (các test dùng `make_date_folds`; test config sẽ chờ Step 4).
- [ ] **Step 3: Cài đặt** `splits.py` và `BLOCK_SESSIONS`.
- [ ] **Step 4: Sinh `DATE_FOLDS`:** chạy một lần `make_date_folds(panel.calendar, "2026-10-30")` trên `build_panel(...)` thật (từ `Pipeline_V1.2/`), dán ba bộ ngày thành literal vào `C.DATE_FOLDS`. Không tính lại lúc chạy, để dữ liệu mới append không dời mốc.
- [ ] **Step 5: Chạy** `$PY -m pytest tests/test_splits.py -v` → PASS.
- [ ] **Step 6: Commit** `feat(price): frozen date folds with label_end_date purge`.

---

### Task 8: Dataset dùng chung train/inference (P1)

**Files:**
- Create: `Pipeline_V1.2/src/dataset.py`, `Pipeline_V1.2/tests/test_dataset.py`

**Interfaces:**
- Consumes: `Panel` (đã có `ohlcv`, `calendar`), `price_features.{FEATURES, market_features, compute}`, `labels.make_labels`.
- Produces: `dataset.LABEL_COLS = ["target_session", "label_end_date", "realized_return", "theta_used", "label", "status", "label_version"]`; `dataset.build_dataset(panel: Panel, h: int) -> pd.DataFrame`: một dòng mỗi `(symbol, as_of_session)` trên lịch, cột `symbol, as_of_session, h, *FEATURES, flag_hl_equal, features_ok, *LABEL_COLS`; `features_ok` = mọi `FEATURES` hữu hạn; `dataset.inference_features(panel: Panel) -> pd.DataFrame`: một dòng mỗi mã tại phiên cuối của lịch, cột `symbol, as_of_session, *FEATURES, flag_hl_equal, features_ok`, **không có** cột nhãn. Cả hai gọi chung một hàm nội bộ `_features(panel) -> dict[str, pd.DataFrame]`.

- [ ] **Step 1: Viết test thất bại** `tests/test_dataset.py` (panel tổng hợp 3 mã, 300 phiên, `ohlcv`/`calendar` dựng trực tiếp):
  - `test_offline_online_parity`: với mỗi mã, hàng `inference_features` bằng hàng `build_dataset(panel, 2)` tại `as_of_session` cuối (`assert_series_equal` trên `FEATURES`).
  - `test_inference_has_no_target_columns`: `set(inference_features(p).columns).isdisjoint(LABEL_COLS)`.
  - `test_missing_last_session_marks_only_that_symbol`: đặt hàng cuối của mã `BBB` là NaN → `features_ok` False cho `BBB`, True cho hai mã còn lại; mã còn lại không đổi giá trị so với panel đầy đủ.
  - `test_last_rows_not_matured`: `build_dataset(p, 5)` có đúng 5 hàng cuối mỗi mã với `status == "not_matured"`.
  - `test_training_rows_require_features_and_ok_label`: số hàng `features_ok & (status == "ok")` bằng số hàng có `label.notna()` và đủ đặc trưng.
- [ ] **Step 2: Chạy, thấy FAIL.**
- [ ] **Step 3: Cài đặt** `dataset.py`.
- [ ] **Step 4: Chạy** `$PY -m pytest tests -q` → PASS.
- [ ] **Step 5: Commit** `feat(price): shared training/inference dataset with feature/label separation`.

---

### Task 9: `pipeline.py`/`predict.py` nhận dataset_id và từ chối khi dữ liệu không hợp lệ (P2)

**Files:**
- Modify: `Pipeline_V1.2/src/pipeline.py`, `Pipeline_V1.2/src/predict.py` (CRLF cả hai), `Pipeline_V1.2/docs/pipeline.md`
- Test: `Pipeline_V1.2/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Panel.dataset_id`, `Panel.gaps`.
- Produces:
  - `pipeline.run_experiment`: `spec` thêm khóa `"dataset_id": panel.dataset_id`.
  - `predict.signature(panel) -> dict` thêm `"dataset_id"`.
  - `predict.invalid_reason(panel: Panel, L: int, now: pd.Timestamp, holidays: np.ndarray) -> str | None`: `"MISSING_SESSION"` nếu `missing_sessions(panel.dates[-1], now, holidays)` khác rỗng hoặc có phần tử `panel.gaps ≥ panel.dates[-L]` (`gaps is None` → bỏ qua điều kiện sau); ngược lại `None`.
  - `predict.predict(...)` trả bảng có thêm cột `status` (`"OK"` / `"DATA_INVALID"`) và `reason`; hàng không hợp lệ có xác suất NaN và `label == "DATA_INVALID"`.

- [ ] **Step 1: Viết test thất bại** trong `tests/test_pipeline.py`:
  - `test_store_treats_new_dataset_id_as_stale`: tạo `P.Store` hai lần trên cùng thư mục với `spec` khác nhau chỉ ở `dataset_id` → lần hai in thông báo "chạy lại từ đầu" (qua `log`) và `runs` rỗng.
  - `test_predict_data_invalid_when_session_missing`: dùng `small_run`; `now = last + 3 ngày làm việc 18:00` (lỡ phiên) → mọi hàng `status == "DATA_INVALID"`, `reason == "MISSING_SESSION"`, xác suất NaN, `label == "DATA_INVALID"`.
  - `test_predict_data_invalid_when_gap_inside_window`: gán `panel.gaps = pd.DatetimeIndex([panel.dates[-10]])` với `L=30` → mọi hàng `L=30` không hợp lệ.
  - Test cũ `test_predict_gives_probabilities_and_label` giữ nguyên và PASS (hàng hợp lệ có `status == "OK"`).
- [ ] **Step 2: Chạy, thấy FAIL:** `$PY -m pytest tests/test_pipeline.py -k "stale or data_invalid" -v`.
- [ ] **Step 3: Cài đặt** các thay đổi ở `pipeline.py` và `predict.py`; trong `predict.main` giữ thông điệp hướng dẫn chạy sync, nay đi kèm cột `status`. Cập nhật `docs/pipeline.md` một mục ngắn về `manifest.json`, `snapshot`, các module P1 và việc chúng chưa dùng cho model hiện hành.
- [ ] **Step 4: Chạy toàn bộ:** `cd Pipeline_V1.2 && $PY -m pytest tests -q` (kỳ vọng ≥ baseline Step 1 Task 1 cộng test mới, 0 FAIL) và `cd .. && $PY -m pytest tests -q` (test `ai_core`, 0 FAIL). `file src/config.py src/predict.py src/pipeline.py` đều CRLF.
- [ ] **Step 5: Kiểm tra cuối trên dữ liệu thật, không gọi API:** `$PY -c "from src.data import build_panel; from src.dataset import inference_features as f; p=build_panel(); print(f(p).features_ok.all(), len(p.gaps))"` từ `Pipeline_V1.2/` → `True 0` khi danh sách ngày nghỉ đã đủ; nếu `gaps` khác 0, liệt kê ngày và bổ sung `hose_holidays.csv` (Task 1 Step 6) trước khi merge.
- [ ] **Step 6: Commit** `feat(price): dataset_id in experiments and DATA_INVALID in predict`, rồi dùng skill `superpowers:finishing-a-development-branch` để merge; sao lưu `stock/` trước phiên cron đầu tiên chạy mã mới (Task 2 Step 7).
