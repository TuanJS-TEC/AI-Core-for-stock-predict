# Phương án cải thiện tổng thể NLP_Module (đề xuất, 2026-10-29)

Nguồn: đề xuất do người dùng dán vào phiên làm việc ngày 2026-10-29. Bản dưới đây giữ nguyên nội dung, chỉ sửa định
dạng. Phần 1 có kế hoạch triển khai tại `docs/superpowers/plans/2026-10-29-volcast-v3.md`. Phần 2 và Phần 3 là hai hệ
thống con độc lập, mỗi phần sẽ có kế hoạch riêng.

Ràng buộc chung của đề xuất: không thêm thư viện ngoài `sklearn/pandas/numpy`, không rò rỉ dữ liệu qua mốc 15:00,
không chạm vào tập `eval` đóng băng khi chọn tham số.

## PHẦN 1: Nâng cấp `nlp_module/volcast.py` (đưa Test Accuracy tiệm cận/vượt mốc 0.65 ổn định)

Mục tiêu của `volcast.py` là dự đoán nhãn `absmove`: |return t→t+h| > median ngày t. Hiện tại Test Accuracy trung bình
đạt `0.642` (Fold C đạt `0.625` ở h=2 và `0.604` ở h=5).

### 1.1. Bổ sung đặc trưng xếp hạng chéo trong ngày (cross-sectional ranks)
- Vấn đề: nhãn `absmove` so sánh tương đối giữa 15 mã trong cùng một ngày, và luật cơ sở `_rule_score` cũng xếp hạng
  `groupby("date").vol10.rank()`. Nhưng `logit`/`gbm` chỉ nhận giá trị tuyệt đối (`vol10`, `vol20`, `range`, `ret1`).
  Khi thị trường biến động cao, mọi mã đều có `vol10` cao, làm mô hình dự đoán lệch.
- Giải pháp: thêm trong `add_features(df)` (chỉ dùng dữ liệu tại close ngày t):
  - `cs_vol10`, `cs_vol20`, `cs_range`, `cs_abs_ret1`: thứ hạng phần trăm trong ngày t của `vol10`, `vol20`,
    `(high - low) / close`, `|ret1|`.
  - `vol_ratio = vol10 / (vol20 + 1e-8)`.

```python
df["abs_ret1"] = df["ret1"].abs()
df["vol_ratio"] = df["vol10"] / df["vol20"].replace(0, np.nan)
for c in ("vol10", "vol20", "range", "abs_ret1"):
    g_cs = df.groupby("date")[c]
    df[f"cs_{c}"] = (g_cs.rank(method="average") - 0.5) / g_cs.transform("count")
```
Thêm `"vol_ratio", "cs_vol10", "cs_vol20", "cs_range", "cs_abs_ret1"` vào `FEATS`.

### 1.2. Sửa tiêu chí chọn model (`logit` vs `gbm`) và đệm an toàn ngưỡng trong `walk()`
- Vấn đề 1: `pick_threshold` tìm độ phủ lớn nhất với val accuracy ≥ 0.65, nên `okv[cv >= t].mean()` của mọi model đều
  ≈ 0.650x. So sánh `okv[cv >= t].mean() > best[1]` thực chất chọn theo chữ số thập phân thứ 4, thay vì chọn model có
  độ phủ validation (`va_cov`) lớn nhất tại mốc 0.65.
- Vấn đề 2: ngưỡng vừa chạm 0.650 trên validation sẽ tụt về 0.61–0.63 trên test (regression to the mean).
- Giải pháp: (1) chọn model theo `va_cov` lớn nhất, bằng nhau mới xét `va_acc`; (2) nâng `min_acc` chọn ngưỡng lên
  `0.665` (hoặc lấy ngưỡng theo phân vị của `cv`) để có biên an toàn ~1.5%.

### 1.3. Chống lệch thang xác suất khi refit trên train + val
- Vấn đề: `gbm` refit trên `trva` với `min_samples_leaf=100` cố định cho xác suất `pt` sắc hơn `pv`.
- Giải pháp: lưu tỷ lệ độ phủ `va_cov` thay vì ngưỡng tuyệt đối, hoặc `min_samples_leaf = max(50, int(0.015 * len(X)))`,
  hoặc bọc `CalibratedClassifierCV(MODELS[m](), cv=3, method="sigmoid")`.

### 1.4. Vá 2 lỗi đã ghi chú `# ponytail` trong `volcast.py`
1. Purge validation ở biên test (`_split` trong `walk_forward.py`): `va` lấy `dates[iv:it]`, nên h phiên cuối của
   validation nhìn thấy close của h phiên đầu test. Sửa thành `d[d.date.isin(dates[iv:max(it - h, iv)])]`.
2. Khe hở cổ tức trong `bars()`: nếu `abs(ret1) > 0.072` (vượt biên độ ±7% của HOSE) và có `ref_price` từ SSI, tính
   `ret1 = (matched_price - ref_price) / ref_price`.

## PHẦN 2: Nâng cấp `baseline.py`, `entities.py`, `passages.py` (tăng Recall lớp positive từ 0.50, cải thiện Macro-F1)

### 2.1. TF-IDF + Logistic Regression trong `nlp_module/sentiment/baseline.py`
- Hiện tại: `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)` +
  `LogisticRegression(max_iter=2000, class_weight="balanced")`, 1.324 mẫu, `neutral` 892 mẫu (67.4%), `negative` 68 mẫu.
- Đề xuất: `min_df=2, max_df=0.90`; `FeatureUnion` word n-gram (1, 3) (`min_df=2`) + char-wb n-gram (3, 5)
  (`min_df=5`); trọng số lớp (N / n_c)^0.5 thay cho `balanced`, chọn `C` trên 80/20 của `dev`; nhân hệ số
  α = (α_pos, 1.0, α_neg) lên xác suất rồi chuẩn hóa lại, chọn α tối ưu `macro_f1` trên calibration twin (20% cuối
  `dev`). Kỳ vọng Recall positive từ 0.50 lên ~0.68–0.72.

### 2.2. Linker bắt nhầm địa danh và tài khoản ngân hàng (`nlp_module/entities.py`)
- Lỗi 1: "BIDV thu giữ biệt thự tại Vinhomes Riverside" → `direct` (1.0) cho VHM.
- Lỗi 2: tin cảnh báo lừa đảo chuyển khoản qua tài khoản VCB → `direct` (0.8) cho VCB.
- Đề xuất:
```python
_LANDMARK_SFX = re.compile(
    r"^\s+(riverside|ocean\s+park|smart\s+city|grand\s+park|central\s+park|"
    r"golden\s+river|metropolis|times\s+city|royal\s+city|imperia|green\s+bay|"
    r"skylake|west\s+point|symphony|wonder\s+park|vũng\s+áng)\b", re.I
)
_LOC_PREP = re.compile(r"(tại|ở|khu\s+đô\s+thị|kđt|dự\s+án|biệt\s+thự|căn\s+hộ)\s+$", re.I)
_FRAUD_CTX = re.compile(r"lừa\s+đảo|chiếm\s+đoạt|giả\s+mạo|công\s+an|tài\s+khoản\s+ngân\s+hàng", re.I)
```
  - `Linker.find()`: alias "Vinhomes" có `_LOC_PREP` phía trước và `_LANDMARK_SFX` phía sau, câu không có từ chủ thể
    doanh nghiệp ("CTCP", "cổ phiếu VHM", "chủ đầu tư") → bỏ mention.
  - `Linker.link()`: bài khớp `_FRAUD_CTX` và ngân hàng chỉ xuất hiện dạng tên thường (không có "cổ phiếu", "mã",
    "HOSE") → hạ `relation` từ `direct` xuống `ambiguous`.

### 2.3. Bài báo "chỉ khớp tiêu đề" (`nlp_module/passages.py`)
- Vấn đề: bài VIC ngày 26/10/2026, tiêu đề nhắc "Vingroup" nhưng thân bài gọi "Tập đoàn"; `select_passage()` chỉ trả
  câu tiêu đề 97 ký tự, `difficult=True`.
- Đề xuất: khi `link.in_title` và không có `ids` trong thân bài, lấy câu sapo:
```python
if not ids and link.in_title and sents:
    ids = [0]   # Lấy câu mở đầu (Sapo) khi tiêu đề đã nhắc trực tiếp công ty
```

## PHẦN 3: Khai thác tín hiệu NLP cho dự báo (`contract.py`, `trial_forecast.py`) và ý tưởng bài báo arXiv

`docs/model_report.md` (mục Magnitude target): `news_count_72h` có AUC đơn biến 0.558 trên `absmove`,
`sentiment_mean_24h` 0.534, nhưng ghép với `vol10` (0.582) trong Logistic Regression thì không tăng AUC.

### 3.1. Biểu diễn đặc trưng cảm xúc cho bài toán biến động
- `contract.py` (`build_snapshot`): trọng số suy giảm theo thời gian (half-life 24h) hoặc ưu tiên cửa sổ 24h; thêm
  `sentiment_abs_max` = max |p_pos − p_neg| trong 72h và `sentiment_shock_24h` = |sentiment_mean_24h|.
- `trial_forecast.py`: đưa `sentiment_std` và `hours_since_last_news` vào `TEXT`; thêm
  `news_surprise = rank_cs(news_count_24h) × (1 − rank_cs(vol10))`.

### 3.2. Hồi quy trực tiếp Text → Return / AbsReturn bằng MSE (Taylor & Ng, arXiv 2401.01487v1)
- Mỗi `passage_text` có `relation == "direct"`, trước 15:00 ngày t; đầu vào `f"[{symbol}] {passage_text}"`.
- Nhãn: lợi suất vượt trội T+2 (r_i − r̄) cho hướng giá; biên độ vượt trội T+2 (|r_i| − median |r|) cho volcast.
- `make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), Ridge(alpha=10.0))`, train trên tập
  Train của từng fold A, B, C (purge h phiên); dự báo ŷ cho từng bài, đưa vào `walk_forward` so với price-only.

### 3.3. Bổ sung 5 mã thiếu giá (BID, CTG, HDB, VHM, NVL)
- `data/prices.csv` chỉ có 10/15 mã của NLP; thêm 5 file CSV lịch sử giá vào `data/stock/` và cập nhật
  `data/prices.csv` đủ 15 mã.

## Thứ tự ưu tiên
1. Phần 1 (volcast, walk_forward): thêm `cs_*`/`vol_ratio`, sửa tiêu chí chọn model, `min_acc` 0.665, purge
   validation; chạy `python -m nlp_module.volcast eval` so với `volcast-v2`.
2. Phần 2: sửa `baseline.py`, `entities.py`, `passages.py`; huấn luyện lại `baseline-cal.joblib`, kiểm tra trên
   validation của `dev` trước khi chấm một lần trên `eval`.
3. Phần 3: `news_surprise`, `sentiment_abs_max`, nhánh Ridge (MSE) trên `walk_forward`.
