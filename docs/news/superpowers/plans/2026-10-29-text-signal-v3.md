# Text signal v3: đặc trưng cường độ/bất ngờ, hồi quy Text→Return, giá 5 mã — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai Phần 3 của đề xuất và đo xem văn bản có thêm giá trị dự báo so với giá hay không:
(3.1) đặc trưng cường độ/bất ngờ của tin, (3.2) hồi quy TF-IDF + Ridge trực tiếp từ passage sang lợi suất,
(3.3) công cụ dựng `data/prices.csv` khi có giá của 5 mã thiếu. Mỗi thí nghiệm có tiêu chí kết luận đặt trước khi chạy.

**Architecture:** Đặc trưng mới nằm trong `contract.build_snapshot` (FEATURE_VERSION v3) cùng một nhóm `--news3` trong
`trial_forecast` và `walk_forward`, theo đúng khuôn của `--fund`/`--bctc`. Hồi quy Text→Return là module mới
`nlp_module/text_return.py`. Ridge huấn luyện trên cửa sổ train của từng fold, rồi forecaster học trên cửa sổ
validation (nơi dự báo Ridge là out-of-sample) và chấm trên test. Đây là nghiên cứu: không ship gì vào hệ thống live.

**Tech Stack:** Python 3.14, pandas, scikit-learn (TfidfVectorizer, Ridge, LogisticRegression). Không thêm dependency.

**Spec:** `docs/superpowers/specs/2026-10-29-improvement-proposal.md` (Phần 3)

## Global Constraints
- Không thêm thư viện ngoài `sklearn/pandas/numpy`. `contract.py` giữ stdlib-only (dùng `math`).
- Không rò rỉ dữ liệu qua mốc 15:00: bài chỉ được dùng ở phiên t khi `published_at + BACKTEST_DELAY` ≤ close t
  (`CLOSE_UTC` = 08:00 UTC). Nhãn đọc giá từ close t đến close t+h. Mọi ranh giới fold purge h phiên
  (`walk_forward._split`).
- Fold: `walk_forward.FOLDS`. h ∈ (2, 5). Target `absmove` và `excess` (`trial_forecast.add_targets`).
- Mỗi cửa sổ test chỉ chấm một lần. Kết luận đọc theo tiêu chí ghi sẵn trong task, không chỉnh tham số sau khi xem test.
- Không ghi vào `data/nlp.db`, không đổi hệ thống live (`scripts/live_update.sh`, `volcast`, bundle trong `data/models/`).
- Repo có git nhưng người dùng chưa yêu cầu commit: bỏ các bước commit. Mỗi task kết thúc bằng
  `.venv/bin/python -m pytest -q tests` xanh.

## Sự thật đã kiểm tra
- `contract.build_snapshot` đã tính `sentiment_std` và `hours_since_last_news`, nhưng `trial_forecast.TEXT` không
  dùng hai đặc trưng này (đúng như spec nói).
- Báo cáo (mục "Magnitude target"): AUC đơn biến trên nhãn T+5 `absmove` là `vol10` 0.582, `news_count_72h` 0.558,
  `sentiment_mean_24h` 0.534. Thêm toàn bộ text vào `vol10` không tăng AUC (0.582 → 0.581).
- `add_text` hiện điền mọi giá trị thiếu bằng 0.0. Với `hours_since_last_news` (None khi chưa từng có tin), 0 nghĩa là
  "vừa có tin", sai nghĩa. Vì vậy cần chính sách điền riêng.
- **3.3 bị chặn bởi dữ liệu:** không có nguồn giá lịch sử nào hợp lệ theo `docs/source_terms.md`. vnstock bị cách ly
  trên PyPI, TCBS bị Cloudflare chặn, Vietcap đòi giả header, SSI iBoard chỉ có giá live. `data/stock/*.csv` là file
  người dùng cung cấp (cột `time,open,high,low,close,volume`, nghìn VND, đã điều chỉnh cổ tức). Task 1 chỉ dựng công
  cụ. Việc thêm giá BID, CTG, HDB, VHM, NVL là **hành động của người dùng**.

## Bỏ khỏi phạm vi (có lý do)
- **Không đưa đặc trưng text vào volcast live.** Kế hoạch này chỉ đo. Nếu Task 4 hoặc Task 5 kết luận dương tính theo
  tiêu chí của nó, việc tích hợp vào volcast (cần snapshot live lúc dự báo) sẽ có kế hoạch riêng.
- **Trọng số suy giảm (3.1):** spec cho chọn "half-life 24h **hoặc** ưu tiên cửa sổ 24h". Cửa sổ 24h đã có
  (`sentiment_mean_24h`), nên kế hoạch thêm bản suy giảm half-life 24h làm đặc trưng mới và giữ nguyên các đặc trưng cũ.

## Review Focus
1. Bài đăng sau 15:00 (sau close) phải tính cho phiên **kế tiếp**, không phải phiên cùng ngày. Test ở Task 5.
2. Bài đăng ngày nghỉ (cuối tuần, lễ) rơi vào phiên giao dịch đầu tiên sau đó. Test ở Task 5.
3. Tập train của Ridge không được chứa bài có cửa sổ nhãn (t → t+h) chồng lên validation. Test ở Task 5.
4. Mã chưa từng có tin: `hours_since_last_news` điền 720 (không phải 0). Test ở Task 3.
5. Cửa sổ mà mọi bài đều `uncertain`: `sentiment_abs_max` và `sentiment_decay` là None, không phải 0. Test ở Task 2.

---

### Task 1: Công cụ dựng `data/prices.csv` từ `data/stock`

**Files:**
- Modify: `nlp_module/trial_forecast.py`
- Test: `tests/test_trial_forecast.py`

**Interfaces:**
- Produces: `prices_from_stock(stock_dir: str, symbols: set[str]) -> pd.DataFrame` với các cột `date, symbol, close`.
  Đọc `*_stock.csv` qua `volcast.load_history(stock_dir)`, import `volcast` **bên trong hàm** (volcast đã import
  trial_forecast ở top-level, import ngược ở top-level sẽ thành vòng import), giữ mã nằm trong `symbols`, `date` theo định dạng
  `YYYY-MM-DD`, sắp theo `symbol, date`.
- Produces: `check_prices(df) -> dict` với các khóa `n_symbols`, `same_calendar` (bool: mọi mã có cùng tập ngày),
  `n_nan`, `n_big_moves` (số dòng có |close/close_prev − 1| > 0.07).
- Produces: CLI `python -m nlp_module.trial_forecast --build-prices [--stock-dir data/stock] [--out data/prices.csv]`:
  lấy `symbols` = các `symbol` trong `data/companies.csv`, ghi file, in `check_prices` và danh sách mã của công ty
  chưa có giá. Khi có cờ này thì chỉ dựng giá rồi thoát.

- [ ] **Step 1: Viết test (thất bại)** — `prices_from_stock` trên 2 file CSV giả trong `tmp_path` (AAA, BBB cùng 3
  ngày; một file CCC không thuộc `symbols`) → đúng 6 dòng, chỉ AAA/BBB, cột `date,symbol,close`.
  `check_prices`: lịch cùng ngày → `same_calendar` True; xóa một ngày của BBB → False; đặt một close nhảy +20% →
  `n_big_moves == 1`.
- [ ] **Step 2: Chạy, xác nhận FAIL.**
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy toàn bộ test** — PASS.
- [ ] **Step 5: Kiểm tra trên dữ liệu thật** — chạy `--build-prices --out <scratch>/prices.csv` (KHÔNG ghi đè
  `data/prices.csv`), rồi so với `data/prices.csv` hiện có (10 mã). Expected: cùng tập mã và cùng giá trị close;
  danh sách mã thiếu là BID, CTG, HDB, NVL, VHM. Ghi kết quả so sánh vào báo cáo của task.

---

### Task 2: Contract v3: cường độ, cú sốc, suy giảm

**Files:**
- Modify: `nlp_module/contract.py`
- Test: `tests/test_snapshot.py`

**Interfaces:**
- Produces: `FEATURE_VERSION = "v3"`, comment `# v3 adds sentiment_abs_max, sentiment_shock_24h, sentiment_decay`.
  Ba khóa mới trong `Snapshot.features`, đều tính trên các bài không `uncertain` trong cửa sổ 72h (`win`):
  - `sentiment_abs_max = max(|p_pos − p_neg|)`, None nếu không có bài.
  - `sentiment_shock_24h = |sentiment_mean_24h|`, None nếu `sentiment_mean_24h` là None.
  - `sentiment_decay` = trung bình có trọng số của `p_pos − p_neg`, trọng số `0.5 ** (age_hours / 24)`; None nếu
    không có bài.
  Các khóa cũ không đổi giá trị.

- [ ] **Step 1: Viết test (thất bại)**

```python
from datetime import timedelta
import pytest
from nlp_module.contract import ArticleCompanyResult, build_snapshot, utc

def _r(i, hours_before, pos, neg, unc=False, cut=utc(2026, 1, 5, 8)):
    t = cut - timedelta(hours=hours_before) - timedelta(minutes=30)        # backtest availability = published + 30 min
    return ArticleCompanyResult(f"r{i}", "AAA", "s", t, t, "direct", pos, 1 - pos - neg, neg, uncertain=unc)

def test_v3_intensity_shock_decay():
    rs = [_r(1, 0, 0.9, 0.0), _r(2, 24, 0.0, 0.6)]                        # +0.9 now, -0.6 one half-life ago
    f = build_snapshot(rs, "AAA", utc(2026, 1, 5, 8), "backtest").features
    assert f["sentiment_abs_max"] == pytest.approx(0.9)
    assert f["sentiment_shock_24h"] == pytest.approx(abs(f["sentiment_mean_24h"]))
    assert f["sentiment_decay"] == pytest.approx((0.9 * 1 + (-0.6) * 0.5) / 1.5)

def test_v3_all_uncertain_window_is_none():
    f = build_snapshot([_r(1, 1, 0.9, 0.0, unc=True)], "AAA", utc(2026, 1, 5, 8), "backtest").features
    assert f["sentiment_abs_max"] is None and f["sentiment_decay"] is None and f["sentiment_shock_24h"] is None
```
- [ ] **Step 2: Chạy, xác nhận FAIL.**
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy toàn bộ test** — PASS. Test nào đang khẳng định FEATURE_VERSION là "v2" thì cập nhật theo.

---

### Task 3: Nhóm đặc trưng `--news3` trong forecaster

**Files:**
- Modify: `nlp_module/trial_forecast.py`, `nlp_module/walk_forward.py`
- Test: `tests/test_trial_forecast.py`

**Interfaces:**
- Consumes: ba khóa v3 từ Task 2.
- Produces: `NEWS3 = ["sentiment_std", "hours_since_last_news", "sentiment_abs_max", "sentiment_shock_24h", "sentiment_decay", "news_surprise"]`.
- `add_text` lấy thêm từ snapshot 5 khóa đầu của `NEWS3`. Chính sách điền (ghi rõ trong comment):
  `hours_since_last_news` None → 720, giá trị lớn hơn bị chặn ở 720; các khóa còn lại None → 0.0 (dựa vào
  `has_news` / `uncertain_ratio` như chính sách hiện có).
- Produces: `add_news_surprise(df) -> df` thêm
  `news_surprise = rank_pct(news_count_24h) * (1 - rank_pct(vol10))`, rank theo `groupby("date")` trên giá trị
  **thô** (gọi trước `rank_cs`). Dòng có `vol10` NaN thì `news_surprise` NaN; các dòng này đã bị `run`/`_split` bỏ.
- Produces: cờ `--news3` trong cả `trial_forecast.main` và `walk_forward.main`. Thêm `("news3", NEWS3)` vào dict
  `extras` hiện có (cạnh `fund` và `bctc`). Khi bật thì gọi `add_news_surprise` trước bước rank.

- [ ] **Step 1: Viết test (thất bại)** — (a) `add_news_surprise` trên 3 mã cùng một ngày: mã có `news_count_24h` cao
  nhất và `vol10` thấp nhất có giá trị lớn nhất, mã ngược lại có giá trị nhỏ nhất; hai ngày khác nhau không ảnh hưởng
  lẫn nhau. (b) `add_text` với một mã chưa từng có tin cho `hours_since_last_news == 720` và `sentiment_decay == 0.0`.
  Dùng một `ArticleCompanyResult` của mã khác làm `results`, theo kiểu test hiện có.
- [ ] **Step 2: Chạy, xác nhận FAIL.**
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy toàn bộ test** — PASS (kể cả `test_walk_forward.py`; `TEXT` không đổi).

---

### Task 4: Đo `--news3` và kết luận

**Files:**
- Modify: `docs/model_report.md`
- Data: `data/news3_report.txt`

**Tiêu chí kết luận (cố định trước khi chạy), cho từng target:** "news3 có giá trị gia tăng" khi trên walk-forward
(`walk_forward --skip-nlp --news3 --target <t>`) đạt cả hai điều kiện:
(i) validation chọn một tập có `news3` ở ít nhất 3 trong 6 ô fold × h;
(ii) trung bình test AUC của cấu hình được chọn ≥ trung bình test AUC price-only + 0.005.
Ngược lại thì kết luận "không có giá trị gia tăng". Single split (`trial_forecast --news3`) chỉ ghi để tham khảo.

- [ ] **Step 1: Chạy** `trial_forecast --news3 --target absmove` và `--target excess`, cùng
  `walk_forward --skip-nlp --news3 --target absmove` và `--target excess`. Lệnh dài thì chạy nền, có log. Ghi toàn bộ
  bảng vào `data/news3_report.txt`.
- [ ] **Step 2: Áp tiêu chí** cho từng target, ghi (i) và (ii) kèm số liệu, rồi kết luận.
- [ ] **Step 3: `docs/model_report.md`** — mục mới "News intensity/surprise features (news3, 2026-10-29)": định nghĩa
  6 đặc trưng, chính sách điền, bảng single split và walk-forward, tiêu chí, kết luận. Thêm AUC đơn biến trên test
  window của fold C cho từng đặc trưng news3 với nhãn T+5 `absmove`, để so với bảng 0.582/0.558 hiện có.
- [ ] **Step 4: Chạy toàn bộ test** — PASS.

---

### Task 5: Hồi quy Text→Return (TF-IDF + Ridge) và đo theo fold

**Files:**
- Create: `nlp_module/text_return.py`
- Test: `tests/test_text_return.py`
- Modify: `docs/model_report.md`
- Data: `data/text_return_report.txt`

**Interfaces:**
- Produces: `assign_sessions(published: pd.Series, dates: pd.DatetimeIndex) -> pd.Series`: với mỗi thời điểm (UTC),
  trả về phiên đầu tiên trong `dates` có `date + CLOSE_UTC ≥ published + BACKTEST_DELAY`; NaT nếu không có phiên nào.
- Produces: `article_labels(articles, prices, h, target) -> pd.DataFrame`. `articles` có cột
  `symbol, published_at, passage_text`; `prices` là đầu ra của `trial_forecast.load_prices`. Kết quả thêm `session`
  và `y`, trong đó `r = close[t+h]/close[t] − 1` của mã đó:
  - `target="excess"`: `y = r − mean_r` của các mã trong ngày;
  - `target="absmove"`: `y = |r| − median|r|` của các mã trong ngày.
  Bỏ bài không có phiên hoặc không có giá t+h.
- Produces: `fit_ridge(texts, y)`: `make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), Ridge(alpha=10.0))`,
  đầu vào `f"[{symbol}] {passage_text}"` (Version 2 của bài báo).
- Produces: `daily_feature(articles_with_pred, df) -> df` thêm `ridge_24h` (trung bình dự báo của các bài có
  `published_at + BACKTEST_DELAY` thuộc khoảng (cutoff − 24h, cutoff], với cutoff = date + CLOSE_UTC) và `has_ridge`
  (0/1). Ô không có bài thì `ridge_24h = 0`.
- Produces: `main(argv)` (`python -m nlp_module.text_return --target absmove|excess`):
  1. Đọc bài: `article_company_result` của `baseline-tfidf-lr-v1` với `relation='direct'`, kèm
     `document_version.is_representative=1` và `published_at` khác NULL. Chỉ đọc (read-only), giới hạn ở các mã có
     trong `data/prices.csv`.
  2. Với mỗi fold trong `FOLDS` và h ∈ (2, 5): fit Ridge trên các bài có `session` trong `[train_start, val_start)`
     **và** chỉ số phiên `session + h < val_start`; dự báo cho mọi bài; dựng `ridge_24h`; tách bằng
     `walk_forward._split`.
  3. Forecaster `StandardScaler + LogisticRegression(C=1.0, max_iter=1000)` **huấn luyện trên validation**, chấm
     trên test. Hai bộ đặc trưng: `PRICE` và `PRICE + ["ridge_24h", "has_ridge"]`. Ghi thêm AUC đơn biến của
     `ridge_24h` trên các dòng test có `has_ridge = 1`.
  4. Ghi bảng (fold, h, n_train_articles, test AUC price, test AUC price+ridge, diff, AUC đơn biến) vào
     `data/text_return_report.txt`.
- **Tiêu chí kết luận (cố định trước khi chạy), cho từng target:** "Text→Return có giá trị gia tăng" khi diff > 0 ở
  ít nhất 2 trong 3 fold ở **cả** h=2 lẫn h=5, **và** trung bình diff qua 6 ô > 0.005.

- [ ] **Step 1: Viết test (thất bại)**

```python
import pandas as pd
from nlp_module.text_return import assign_sessions

DATES = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-09"])     # Mon, Tue, Fri (Wed-Thu holiday)

def test_session_respects_close_and_holidays():
    pub = pd.Series(pd.to_datetime(["2026-01-05 07:00", "2026-01-05 07:45", "2026-01-06 09:00", "2026-01-03 10:00"]))
    got = assign_sessions(pub, DATES)
    assert list(got) == list(pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-09", "2026-01-05"]))
    # 07:00 + 30 min <= 08:00 close -> same day; 07:45 + 30 min > close -> next session; after Tue close -> Fri; Saturday -> Monday
```
  Cộng các test: `article_labels` với 2 mã × 4 ngày giá tay → `y` excess bằng r trừ trung bình; `y` absmove bằng
  |r| trừ trung vị; bài ở phiên cuối (thiếu t+h) bị bỏ. Một test purge: với `val_start` cho trước, hàm chọn tập train
  (tách ra thành `train_articles(labeled, dates, val_start, h)`) loại bài có `session + h ≥ val_start`.
  `daily_feature`: bài lúc 07:00 UTC ngày D vào ô D; bài 09:00 UTC ngày D vào ô D+1; ô không có bài thì
  `has_ridge = 0, ridge_24h = 0`.
- [ ] **Step 2: Chạy, xác nhận FAIL.**
- [ ] **Step 3: Cài đặt** theo Interfaces.
- [ ] **Step 4: Chạy toàn bộ test** — PASS.
- [ ] **Step 5: Chạy** `python -m nlp_module.text_return --target absmove` và `--target excess` (chạy nền nếu dài).
  Áp tiêu chí, ghi kết luận vào `data/text_return_report.txt`.
- [ ] **Step 6: `docs/model_report.md`** — mục mới "Text→Return regression (Ridge, 2026-10-29)": thiết kế (đầu vào,
  nhãn, purge, forecaster học trên validation), bảng theo fold, tiêu chí, kết luận, cùng một dòng nhắc rằng chỉ có
  10 mã có giá (Task 1, việc của người dùng).
