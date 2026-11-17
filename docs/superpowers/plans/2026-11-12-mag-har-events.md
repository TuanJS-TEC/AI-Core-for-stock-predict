# Baseline HAR và lịch sự kiện cho nhãn mag (R9): kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** đo xem lịch sự kiện doanh nghiệp (biết trước, khác nhau giữa các mã) có thêm thông tin cho nhãn `mag-median-v1` ngoài một baseline HAR tương đối hay không (R9a, R9b), trên 6 fold A–Fb, chạy đúng một lần.

**Architecture:** lớp dữ liệu `ai_core/prices/events.py` tải lịch sự kiện VCI và lưu ảnh chụp theo ngày. `ai_core/models/mmf/har.py` dựng 7 đặc trưng HAR (trừ trung bình cắt ngang) và 4 đặc trưng sự kiện point-in-time, fit hai hồi quy logistic lồng nhau trên đúng các hàng của R7/R8 (qua `experiment.fold_bundle(..., label="mag")`), ghi dự báo bằng `experiment._write_preds`, rồi chấm R9 bằng `mag.two_class` + `compare.compare` cùng các mốc đọc từ run mag.

**Tech Stack:** Python, pandas, scikit-learn (đã có), vnstock 4.x nguồn `vci` (đã có trong `.venv`). Không thêm dependency.

**Spec:** `docs/superpowers/specs/2026-11-12-mag-har-events-design.md` (phụ lục R9 nháp nằm trong đó).

## Global Constraints

- Không chạy `ai_core.models.mmf.holdout`. Không sửa `docs/reports/mmf_lock.json`, `DATE_FOLDS`, `BACK_FOLDS`, `FEATURES`, `config.ARMS`, `config.BASELINES`, `make_mag_labels`.
- Snapshot giá duy nhất: `data/prices/snapshots/24251acc1a5695bd`. Không đọc phiên sau 2026-10-30 để phát triển.
- Kết quả mới vào `data/mmf/exp/har/`; run mag `data/mmf/exp/mag/24251acc1a5695bd` chỉ đọc, không ghi.
- Không có arm nào được chạy trên dữ liệu thật trước khi chủ dự án xác nhận phụ lục R9 (Task 5). Task 1–4 chỉ chạy trên dữ liệu giả, trừ phần tải và kiểm toán lịch sự kiện (Task 2) vốn không chạm nhãn.
- Không commit: chủ dự án chưa cho commit nhánh này (bước "Commit" của skill được thay bằng "chạy bộ test").
- Test giá/dữ liệu chạy bằng `.venv`; test mmf bằng `.venv_mmf`; luôn từ gốc repo. Kiến trúc: `prices` không import `models`; `tests/test_architecture.py` phải xanh.
- vnstock guest: 20 lệnh/phút → nghỉ 3.5 s giữa các lệnh.

## Review Focus

1. Sự kiện công bố đúng phiên t (`public_date == t`) phải vô hình tại t; công bố t−1 phải thấy. → test `test_event_known_strictly_before_t` (Task 3).
2. Sự kiện có `exright_date`/ngày họp trống (NaN) chỉ được tính vào `ev_new5`, không làm lỗi hay tạo cờ giả. → test `test_missing_event_date_counts_only_as_announcement` (Task 3).
3. Mã không có sự kiện nào trong file: `ev_next = 1`, các cờ = 0, không NaN. → test `test_symbol_without_events` (Task 3).
4. Ngày sự kiện rơi vào ngày nghỉ/cuối tuần: ánh xạ sang phiên giao dịch kế tiếp trên `panel.calendar` khi đếm t+1..t+h. → test `test_event_on_holiday_maps_to_next_session` (Task 3).
5. Hàng eval của run har khác run mag (ví dụ build lại khác): bộ chấm phải dừng, không ghép thiếu. → test `test_r9_refuses_mismatched_rows` (Task 4).

---

### Task 1: lớp dữ liệu lịch sự kiện

**Files:**
- Create: `ai_core/prices/events.py`
- Modify: `ai_core/paths.py` (thêm `EVENTS = PRICES / "events"`)
- Test: `tests/prices/test_events.py`

**Interfaces:**
- Produces:
  - `CODES = ("DIV", "ISS", "AGME", "AGMR", "EGME")`, `COLUMNS = ["symbol", "event_id", "event_code", "public_date", "record_date", "exright_date", "display_date1"]`
  - `normalize(raw: list[dict], symbol: str) -> pd.DataFrame`: bản ghi thô VCI (khóa camelCase `id, eventCode, publicDate, recordDate, exrightDate, displayDate1`) → `COLUMNS`, các cột ngày là `datetime64[ns]` chuẩn hóa về ngày (bỏ giờ, bỏ tz), mã ngoài `CODES` bị bỏ, trùng `event_id` bị bỏ.
  - `fetch(symbols, since="2020-01-01", pause=3.5) -> pd.DataFrame`: phân trang `Company(symbol=s, source="vci")._fetch_events(event_codes=",".join(CODES), from_date=..., page=p, size=100)` tới khi trang < 100.
  - `save(df, day: str) -> Path` ghi `paths.EVENTS / f"{day}.parquet"`; `load(path) -> pd.DataFrame`.
  - `main()`: `python -m ai_core.prices.events` tải `config.TICKERS`, lưu theo ngày hôm nay (giờ VN), in số dòng theo mã.

- [ ] **Step 1: Viết test đỏ** `tests/prices/test_events.py`:
  - `test_normalize_keeps_known_codes_and_parses_dates`: hai bản ghi `DIV` (`publicDate="2026-11-02T00:00:00"`, `exrightDate="2026-11-11T00:00:00"`) và một `DDINS` → 1 dòng, `public_date == Timestamp("2026-11-02")`, `exright_date == Timestamp("2026-11-11")`, `list(df.columns) == COLUMNS`.
  - `test_normalize_drops_duplicate_ids`: hai bản ghi cùng `id` → 1 dòng.
  - `test_normalize_missing_dates_are_nat`: `exrightDate=None` → `NaT`.
  - `test_fetch_paginates`: monkeypatch `Company` bằng giả trả 100 rồi 3 bản ghi, `pause=0` → 103 dòng, gọi đúng 2 trang.
- [ ] **Step 2:** `.venv/bin/python -m pytest -q tests/prices/test_events.py` → FAIL (module chưa có).
- [ ] **Step 3: Cài** `events.py` theo Interfaces; import vnstock bên trong `fetch` (để import module không kéo vnstock).
- [ ] **Step 4:** chạy lại test → PASS; `tests/test_architecture.py tests/test_paths.py` xanh.

### Task 2: tải, kiểm toán lịch sự kiện (cổng trước mọi lượt chạy)

**Files:**
- Create: `docs/reports/mag_events_audit.md`
- Dữ liệu: `data/prices/events/<ngày>.parquet`

**Interfaces:**
- Consumes: `events.main`, `events.load` (Task 1).
- Produces: một file sự kiện cố định cho R9; đường dẫn và sha256 ghi trong audit.

- [ ] **Step 1:** `VNSTOCK_TELEMETRY=0 .venv/bin/python -m ai_core.prices.events` → file `data/prices/events/2026-11-1x.parquet`, 15 mã đều có ≥ 1 dòng. Ghi `sha256sum`.
- [ ] **Step 2: Xác minh nghĩa các trường** trên ít nhất 3 cuộc họp `AGME` và 3 cổ tức `DIV` (đối chiếu tiêu đề trong `nlp.db`, bảng `document_version`, hoặc trang HOSE/VSDC): `display_date1` của `AGME/EGME` có phải ngày họp không; `exright_date` của `DIV/ISS` có phải ngày GDKHQ không; `AGMR` là gì. Nếu nghĩa khác spec, sửa ánh xạ ngày trong Task 3 (được phép vì chưa thấy kết quả nào) và ghi lý do vào audit.
- [ ] **Step 3: Đối chiếu ngày công bố** cho ≥ 10 sự kiện thuộc ≥ 5 mã: `public_date` của VCI so với ngày bài sớm nhất cùng nội dung trong `nlp.db` (hoặc VSDC). Ghi bảng (mã, sự kiện, public_date VCI, ngày nguồn khác, chênh). **Nếu có ca nào VCI sớm hơn ngày công bố thật: dừng, báo chủ dự án.**
- [ ] **Step 4: Đo độ phủ** (chưa có nhãn, chỉ đếm): số sự kiện theo mã và năm 2022–2026; tỷ lệ (mã, phiên) từ 2022 có `ev_exr_in_h` hoặc `ev_mtg_in_h` = 1 cho h = 2 và 5 (dùng hàm Task 3 khi xong; có thể điền sau Task 3). Ghi vào audit, thay con số ước đoán của spec.

### Task 3: đặc trưng HAR và sự kiện

**Files:**
- Create: `ai_core/models/mmf/har.py`
- Test: `tests/models/mmf/test_mmf_har.py`

**Interfaces:**
- Consumes: `panel.on_calendar(s)` (cột `open, high, low, close`), `panel.calendar`, bảng `events.COLUMNS`.
- Produces:
  - `HAR_FEATS = ["har_rv_d", "har_rv_w", "har_rv_m", "har_pk_d", "har_pk_w", "har_pk_m", "har_gap_w"]`, `EV_FEATS = ["ev_exr_in_h", "ev_mtg_in_h", "ev_next", "ev_new5"]`, `EPS = 1e-8`, `NEXT_CAP = 20`, `NEW_WINDOW = 5`.
  - `har_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame` index `(symbol, as_of_session)`, cột `HAR_FEATS`, đã trừ trung bình cắt ngang theo phiên (chỉ trên mã có giá trị hữu hạn).
  - `event_features(events: pd.DataFrame, calendar: pd.DatetimeIndex, symbols: list[str], h: int) -> pd.DataFrame` index `(symbol, as_of_session)` trên mọi phiên của `calendar`, cột `EV_FEATS`, không NaN.

Định nghĩa đúng như spec, mục "Đặc trưng HAR" và "Đặc trưng sự kiện". Ngày sự kiện: `exright_date` cho `DIV, ISS`; `display_date1` cho `AGME, EGME` (hoặc ánh xạ đã sửa ở Task 2); ngày không phải phiên → phiên kế tiếp (`calendar.searchsorted(d)`). "Biết tại t" ⇔ `public_date < t`. Đếm phiên theo vị trí trong `calendar`.

- [ ] **Step 1: Viết test đỏ** trên khung giả 4 mã × 80 phiên (`pd.bdate_range`), seed cố định:
  - `test_har_is_cross_sectionally_demeaned`: tổng mỗi cột theo phiên ≈ 0 (khi đủ 4 mã).
  - `test_har_uses_data_up_to_t_only`: đổi `close/high/low/open` ở phiên t+1 không đổi hàng t.
  - `test_har_rv_w_matches_formula`: với một mã, trước khi trừ trung bình, `har_rv_w` tại t = `ln(mean(r[t-4..t]²) + 1e-8)` (kiểm qua chênh lệch giữa hai mã để triệt tiêu phần trung bình, hoặc gọi hàm nội bộ `_raw_har`).
  - `test_event_known_strictly_before_t`: một `DIV` có `exright_date` = phiên 50; `public_date` = phiên 40 → `ev_exr_in_h` (h=5) = 1 tại các phiên 45..49 nhưng = 0 tại phiên 40; đổi `public_date` thành phiên 39 → tại phiên 40 bằng 1.
  - `test_event_window_is_t_plus_1_to_t_plus_h`: với `exright_date` = phiên 50, h=2: cờ = 1 tại phiên 48, 49; = 0 tại 47 và 50.
  - `test_ev_next_capped_and_one_when_none`: sự kiện cách 3 phiên → `ev_next = 3/20`; cách 30 phiên → 1.0; không có → 1.0.
  - `test_ev_new5_counts_announcements_in_t_minus_5_to_t_minus_1`: hai thông báo có `public_date` ở phiên 32 và 35 → `ev_new5` = 0 tại t=32 (cửa sổ 27..31), 1 tại t=33, 2 tại t=36 (cửa sổ 31..35), 0 tại t=41 (cửa sổ 36..40).
  - Ba test của Review Focus 1–4: `test_missing_event_date_counts_only_as_announcement`, `test_symbol_without_events`, `test_event_on_holiday_maps_to_next_session` (ngày sự kiện thứ Bảy → phiên thứ Hai kế tiếp).
- [ ] **Step 2:** `.venv_mmf/bin/python -m pytest -q tests/models/mmf/test_mmf_har.py` → FAIL.
- [ ] **Step 3: Cài** `har.py` (phần đặc trưng). Vector hóa theo ma trận (phiên × mã) như `make_mag_labels`; với sự kiện, lặp theo từng sự kiện là đủ (vài trăm dòng).
- [ ] **Step 4:** test → PASS.
- [ ] **Step 5:** điền độ phủ vào `docs/reports/mag_events_audit.md` (Task 2 Step 4) bằng `event_features` trên file sự kiện thật và `build_panel(stock_dir=SNAP).calendar`. Không đọc nhãn.

### Task 4: chạy hai arm, giả dược, và chấm R9

**Files:**
- Modify: `ai_core/models/mmf/har.py` (thêm phần chạy và chấm, CLI)
- Test: `tests/models/mmf/test_mmf_har.py`

**Interfaces:**
- Consumes: `experiment.fold_bundle(panel, h, fold, label="mag")`, `experiment._write_preds(path, b, arm, seed, price_kind, probs, alpha)`, `experiment._prior_rows`, `baselines._model("logit")`, `baselines._proba`, `evaluate.fit_alpha`, `mag.two_class`, `compare.load_h`, `compare.compare`, `compare._fmt`, Task 3.
- Produces:
  - `ARMS = {"logit_har": HAR_FEATS, "logit_har_ev": HAR_FEATS + EV_FEATS}`
  - `predict(b, extra: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]`: lấy `extra` (index `(symbol, as_of_session)`) cho các hàng `b.blocks[k]`; mean/std trên hàng train, clip ±5, NaN → 0; fit `_model("logit")` trên nhãn train hữu hạn; trả xác suất (n, 3) cho `"calib"`, `"eval"` theo thứ tự hàng của block.
  - `run(panel, events, horizons, folds, run_dir, placebo_seed=None)`: mỗi (h, fold): bundle, `extra = har_features ⨝ event_features(h)`, mỗi arm: `predict` → `alpha = fit_alpha(y_calib, p_calib, _prior_rows(b, "calib"))` → `_write_preds(run_dir/f"h{h}"/fold.name/arm/"seed0.parquet", b, arm, 0, "none", p, alpha)`. `placebo_seed` khác None: trước khi tính đặc trưng sự kiện, đổi cột `symbol` của `events` theo một hoán vị không điểm cố định của 15 mã (rng = `default_rng(placebo_seed)`), chỉ chạy `logit_har_ev`, ghi dưới tên arm `logit_har_ev_p{seed}`.
  - `r9(ph: pd.DataFrame, h: int) -> list[dict]`: hai hàng chính `R9a` (`logit_har_ev − logit_har`), `R9b` (`logit_har − prior_adaptive`), rồi các hàng "kèm:" (`logit_har − prior_fixed`, `logit_har_ev − prior_fixed`, `logit_har − dl_price`, `logit_har_ev − dl_price`, `logit_har_ev − fusion_concat`), cùng định dạng hàng như `mag._row`.
  - `load_joint(har_dir, mag_dir, h) -> pd.DataFrame`: `two_class(concat(load_h(mag_dir, h), load_h(har_dir, h)))`; với mỗi fold, tập (symbol, as_of_session) của block eval của `logit_har` phải bằng của `prior_fixed`, nếu không thì `raise ValueError`.
  - `main()`: `--stock-dir`, `--events`, `--h`, `--folds` (mặc định `A B C D E Fb`), `--placebo N` (chạy N giả dược, seed 0..N−1), `--score --out PATH` (viết bảng R9, từng fold, mọi arm trên eval như `mag.main`, thêm bảng giả dược: R9a thật và phân vị 5/50/95 của `logit_har_ev_p* − logit_har`, và hệ số 4 đặc trưng sự kiện theo fold). Run dir: `C.DATA / "exp/har" / dataset_id[:16]`.

- [ ] **Step 1: Viết test đỏ** (dùng lại fixture `tiny_bundle` / `fake_frame` từ `tests/models/mmf/test_mmf_experiment.py`):
  - `test_predict_returns_valid_probs_in_block_order`: `predict` trên bundle giả + `extra` ngẫu nhiên → hàng mỗi block khớp `len(b.blocks[k])`, tổng xác suất = 1, cột lớp 2 bằng 0.
  - `test_predict_scaler_from_train_only`: đổi `extra` trên hàng eval không đổi xác suất calib.
  - `test_nested_arm_recovers_informative_feature`: `extra` có một cột = nhãn + nhiễu nhỏ trong `EV_FEATS`, HAR là nhiễu → log-loss eval của `logit_har_ev` < `logit_har`.
  - `test_r9_rows_and_verdict`: preds giả 6 fold (như `_preds` của `test_mmf_mag.py`, mở rộng ra 6 fold và các arm cần) với `logit_har_ev` biết nhãn → `R9a` "Có bằng chứng", `n_folds == 6`.
  - `test_r9_refuses_mismatched_rows`: bỏ một hàng eval của `logit_har` ở run har → `load_joint` raise `ValueError`.
  - `test_placebo_permutation_has_no_fixed_point`: với seed 0..19, không mã nào giữ lại sự kiện của chính nó.
- [ ] **Step 2:** chạy test → FAIL.
- [ ] **Step 3: Cài** theo Interfaces.
- [ ] **Step 4:** `.venv_mmf/bin/python -m pytest -q tests/models/mmf` → PASS; cả hai lệnh test của `CLAUDE.md` xanh (cập nhật số test kỳ vọng trong `CLAUDE.md`).
- [ ] **Step 5: Thử khói không đọc kết quả**: không có lệnh khói trên dữ liệu thật trước Task 5. Chỉ kiểm `python -m ai_core.models.mmf.har --help`.

### Task 5: chủ dự án xác nhận, rồi chạy R9 một lần

**Files:**
- Modify: `docs/reports/mmf_preregistration.md` (chỉ nối thêm ở cuối)
- Create: `docs/reports/mmf_har_ev_report.md`

- [ ] **Step 1 (chủ dự án):** đọc spec và audit; xác nhận phụ lục R9. Trợ lý nối phụ lục (bản nháp trong spec, điền tên file sự kiện, sha256, ngày giờ soạn) vào cuối `mmf_preregistration.md`. Không sửa phần trên.
- [ ] **Step 2:** `SNAP=data/prices/snapshots/24251acc1a5695bd; .venv_mmf/bin/python -m ai_core.models.mmf.har --stock-dir $SNAP --events data/prices/events/<file> --h 2 5 2>&1 | tee data/mmf/exp/har/run.log` → 2 h × 6 fold × 2 arm = 24 file.
- [ ] **Step 3:** `... har --stock-dir $SNAP --events ... --h 2 5 --placebo 20` → 240 file giả dược.
- [ ] **Step 4:** `... har --score --out data/mmf/exp/har/r9.md` → bảng R9a, R9b, kèm, từng fold, giả dược. Kiểm ngưỡng AUC nhãn thật ≥ 0.75 trước khi đọc R9.
- [ ] **Step 5:** viết `docs/reports/mmf_har_ev_report.md` (tiếng Việt, cùng khuôn `mmf_mag_report.md`): kết luận theo thang đã đăng ký, bảng R9, giả dược, hệ số sự kiện, độ phủ, giới hạn (vùng phát triển, dữ liệu nhà cung cấp, thiếu lực). Áp đúng quy tắc dừng của phụ lục; không thử biến thể nào.

### Task 6: ảnh chụp sự kiện hằng ngày (cho kiểm chứng tiến)

**Files:**
- Modify: `CLAUDE.md` (một dòng về cron sự kiện)

- [ ] **Step 1 (chủ dự án thêm crontab):** `50 15 * * 1-5 cd "<repo>" && VNSTOCK_TELEMETRY=0 .venv/bin/python -m ai_core.prices.events >> logs/events.log 2>&1`.
- [ ] **Step 2:** sau phiên kế tiếp, kiểm có file mới trong `data/prices/events/` và `logs/events.log` không lỗi.
- [ ] **Step 3:** nếu R9a "Có bằng chứng" ở ít nhất một h và chủ dự án quyết định khóa: kế hoạch riêng cho `har_ev_lock.json` và chấm tiến (≥ 40 phiên đã trưởng thành sau ngày khóa, dùng ảnh chụp của từng ngày). Ngoài phạm vi plan này.

## Ngoài phạm vi

- Dòng tiền khối ngoại/tự doanh (Finhay đang tích lũy từ 2026-07-28; xem lại khi có ≥ 1 năm), collector sổ lệnh SSI, phái sinh, vĩ mô, lịch BCTC quý, fusion head nơ-ron.
