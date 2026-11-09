# Pipeline dự đoán xu hướng giá 15 cổ phiếu

Tài liệu tổng hợp toàn bộ pipeline. Bài toán gốc là dự báo chuỗi thời gian dài hạn (LTSF) giá đóng cửa của 15 mã
(FPT, VNM, STB, ACB, SSI, VIC, HPG, VCB, MSN, MBB, GAS, MWG, VPB, VRE, TCB). Pipeline hiện tại dùng ba mô hình LTSF
tuyến tính (Linear, NLinear, DLinear) cho hai nhiệm vụ cùng lúc:

- **Hồi quy:** dự báo T giá trị log giá đóng cửa tiếp theo (đã chuẩn hóa).
- **Phân loại 3 nhãn:** giá đóng cửa sau T phiên sẽ **Giảm**, **Đi ngang** hay **Tăng** so với phiên cuối của cửa
  sổ, kèm xác suất của từng nhãn. Đây là đầu ra chính để chọn mô hình và để dự đoán.

Cấu hình duy nhất đang dùng: **huấn luyện gộp 15 mã**, đầu vào chỉ là **log giá đóng cửa**, đầu phân loại có
**embedding mã**, ngưỡng nhãn **K_THETA = 0.43**. Lưới thí nghiệm: T ∈ {2, 5} × L ∈ {30, 90} × 3 mô hình × 2
weight decay × 5 seed = **120 lượt huấn luyện**.

---

## 1. Ba mô hình thuật toán

Cả ba mô hình đến từ bài báo *"Are Transformers Effective for Time Series Forecasting?"* (Zeng và cộng sự, 2023).
Ý tưởng chung: một tầng tuyến tính duy nhất ánh xạ trực tiếp L điểm quá khứ sang T điểm tương lai (dự báo trực tiếp,
không tự hồi quy từng bước). Các mô hình này cho kết quả ngang hoặc tốt hơn Transformer trên nhiều bộ LTSF trong khi
rất nhỏ và nhanh.

Ký hiệu: x = (x₁, …, x_L) là cửa sổ đầu vào (log giá đã chuẩn hóa), ŷ ∈ ℝᵀ là dự báo.

### 1.1 Linear

```
ŷ = W · x + b          W ∈ ℝ^{T×L}, b ∈ ℝ^T
```

Mỗi bước tương lai là tổng có trọng số của L điểm quá khứ. Đây là mô hình tham chiếu đơn giản nhất.

### 1.2 NLinear (Normalization-Linear)

```
x̃ = x − x_L                (trừ giá trị cuối của cửa sổ)
ŷ = W · x̃ + b + x_L        (dự báo độ lệch rồi cộng lại giá trị cuối)
```

Mô hình chỉ học **độ lệch so với mức giá hiện tại**, không học mức giá tuyệt đối. Giá cổ phiếu không dừng (mức
giá ở val/test thường nằm ngoài khoảng của train); việc trừ rồi cộng lại x_L giúp mô hình không bị lệch khi mức giá
thay đổi (distribution shift). Khi W = 0, NLinear chính là baseline *repeat*.

### 1.3 DLinear (Decomposition-Linear)

```
trend  = MovingAvg_25(x)          (trung bình trượt kernel 25, padding hai đầu bằng giá trị biên)
season = x − trend
ŷ = W_t · trend + W_s · season + b
```

Chuỗi được tách thành thành phần xu hướng (trơn) và phần dư (dao động ngắn hạn), mỗi phần có một tầng tuyến tính
riêng. Padding hai đầu, lặp (25 − 1)/2 = 12 lần giá trị đầu và cuối, giữ độ dài chuỗi bằng L. Cách tách này giúp
tầng tuyến tính học xu hướng và dao động tách biệt, phù hợp chuỗi giá có xu hướng rõ.

### 1.4 Giống và khác nhau

| | Linear | NLinear | DLinear |
|---|---|---|---|
| Lõi | 1 tầng tuyến tính L → T | 1 tầng tuyến tính L → T | 2 tầng tuyến tính L → T |
| Tiền xử lý trong mô hình | không | trừ giá trị cuối, cộng lại sau | tách trend (MA 25) / season |
| Xử lý mức giá thay đổi | kém | tốt (học độ lệch) | trung bình (trend vẫn mang mức giá) |
| Số tham số (hồi quy) | L·T + T | L·T + T | 2·(L·T + T) |

Giống nhau: đều tuyến tính, dự báo trực tiếp T bước, không có tầng ẩn, không có attention. Khác nhau ở cách biến
đổi đầu vào trước tầng tuyến tính.

### 1.5 Cách ba mô hình được dùng trong pipeline (mô hình đa nhiệm gộp)

Mỗi lượt huấn luyện dùng một mô hình `PooledMultiTask` gồm hai đầu cùng họ kiến trúc (cùng Linear, cùng NLinear
hoặc cùng DLinear):

1. **Đầu hồi quy:** đúng mô hình Linear/NLinear/DLinear ở trên, L → T, dùng chung cho 15 mã.
2. **Đầu phân loại có embedding mã (`EmbedClsHead`):**
   - Tầng 1, chiếu thời gian L → d = 8 theo đúng họ kiến trúc:
     - Linear: `h = P·x`.
     - NLinear: `h = [P·(x − x_L), x_L]`, giữ thêm giá trị cuối để không mất mức hiện tại.
     - DLinear: `h = P_t·trend + P_s·season`.
   - Tầng 2: nối h với vector embedding 4 chiều của mã, rồi Linear → 3 logit (Giảm / Đi ngang / Tăng).

`forward(x, ticker_id)` trả về `(dự báo hồi quy (B, T), logits (B, 3))`.

### 1.6 Lý do áp dụng vào bài toán

- **Dữ liệu ít:** mỗi mã chỉ có khoảng 1.000 phiên train. Mô hình sâu dễ quá khớp; mô hình tuyến tính ít tham số
  hơn nhiều và ổn định hơn.
- **Chuỗi giá gần với bước ngẫu nhiên:** phần dự báo được thường rất nhỏ. Mô hình đơn giản kèm baseline *repeat* cho
  phép đo trung thực mô hình có vượt được "giá ngày mai bằng giá hôm nay" hay không.
- **Ba cách xử lý đầu vào khác nhau** (thô, bỏ mức giá, tách xu hướng) cho phép so sánh cách nào giữ tín hiệu tốt
  nhất. Thực tế trên val, DLinear/wd = 0 được chọn ở cả 4 cặp (T, L).
- **Gộp 15 mã:** gấp 15 lần số mẫu huấn luyện cho phần chiếu thời gian (học chung). Embedding mã cho phép mỗi mã
  có độ dịch riêng ở tầng phân loại mà không cần một mô hình riêng cho từng mã.
- **Đa nhiệm:** đầu hồi quy giữ đúng yêu cầu LTSF gốc (dự báo giá); đầu phân loại trả lời trực tiếp câu hỏi thực
  tế (giá sẽ tăng, giảm hay đi ngang) kèm xác suất.

---

## 2. Cấu trúc file

```
requirements.txt            thư viện cần cài
sync_stock_parquet.py       đồng bộ dữ liệu giá (chạy riêng, trước pipeline)
demo.py                     giao diện Streamlit: nhãn dự đoán và đồ thị giá dự báo / thực tế của một mã
src/
├── __init__.py             đánh dấu package (rỗng)
├── config.py               mọi tham số, cấu hình, đường dẫn
├── data.py                 đọc dữ liệu, chia tập, cắt cửa sổ, chuẩn hóa, gán nhãn
├── models.py               Linear, NLinear, DLinear, đầu phân loại, mô hình gộp
├── metrics.py              chỉ số đánh giá và hiệu chỉnh xác suất
├── pipeline.py             huấn luyện lưới, chọn cấu hình, đánh giá test
├── report.py               bảng kết quả cuối, baseline, khoảng tin cậy, ma trận nhầm lẫn, đồ thị dự đoán của kho model
└── predict.py              kho model tốt nhất từng mã và dự đoán phiên tới
outputs/
├── results/                bảng kết quả (CSV/JSON) và trạng thái huấn luyện
│   ├── pooled_embed_k*/         kết quả lần chạy đầy đủ
│   ├── pooled_embed_quick_k*/   kết quả lần chạy thử --quick (nếu có)
│   ├── predictions/                dự đoán phiên tới
│   └── models_regression.csv       MSE/MAE của từng model trong kho
├── figures/                hình ma trận nhầm lẫn và đồ thị dự đoán
│   ├── pooled_embed_k0.43/
│   └── models/                     60 đồ thị dự đoán/thực tế, mỗi model một hình
└── models/                 60 model tốt nhất (15 mã × 2 T × 2 L) và sổ đăng ký
```

### 2.1 Chức năng từng file

| File | Chức năng |
|---|---|
| `requirements.txt` | torch 2.14.1, scikit-learn 1.9.1, pandas 3.0.6, pyarrow 26.0.0, numpy 2.5.4, matplotlib 3.11.2, pytest 9.1.1, streamlit 1.66.1, plotly 7.1.0 (hai thư viện cuối chỉ dùng cho `demo.py`). Cài bằng `pip install -r requirements.txt`. |
| `sync_stock_parquet.py` | Đọc `{mã}_stock.csv`, lấy các phiên mới từ vnstock (khung 1D, thử lại tối đa 3 lần), chuẩn hóa thời gian mỗi phiên về ngày + 07:00 UTC, lọc trùng theo ngày (giữ dòng mới nhất), ép kiểu cột, ghi `{mã}_stock.parquet` và cập nhật CSV. Cảnh báo nếu các mã lệch ngày. Cờ: `--no-fetch` (chỉ làm sạch, không gọi API), `--symbols ACB FPT` (chỉ một số mã), `--delay` (giây nghỉ giữa các lần gọi API), `--stock-dir`. **Không** được pipeline gọi tự động. |
| `demo.py` | Giao diện web Streamlit, chạy bằng `streamlit run demo.py`. Chọn mã trước; phần trên hiện nhãn dự đoán kèm xác suất cho T+2 và T+5 (mỗi T hai model L = 30, L = 90), phần dưới vẽ đồ thị tương tác (Plotly) giá dự báo của đầu hồi quy so với giá thực tế và baseline repeat từ `DEMO_CHART_START` đến hôm nay. Chỉ đọc dữ liệu và kho model, không ghi file. Xem mục 4.3. |
| `src/config.py` | Nguồn duy nhất cho SEED, danh sách mã, tỷ lệ chia, K_THETA, lưới (T, L, mô hình, weight decay), siêu tham số huấn luyện, lưới hiệu chỉnh, tên file đầu ra, tham số báo cáo và đồng bộ. Module khác chỉ đọc qua `from . import config as C`. |
| `src/data.py` | `build_panel`: đọc parquet 15 mã, kiểm tra cùng dãy ngày, không trùng ngày, lấy log close. `split_sizes`: chia 70:15:15 theo thời gian. `build_pooled`: với mỗi mã fit scaler trên train, cắt cửa sổ, tính log-return r, θ, nhãn và prior, rồi nối mẫu của 15 mã. `make_loader`: DataLoader (chỉ train được shuffle). `raw_windows`: cắt cửa sổ thô dùng khi dự đoán. |
| `src/models.py` | `Linear`, `NLinear`, `MovingAvg`, `DLinear` (đầu hồi quy); `FeatureProjection`, `EmbedClsHead` (đầu phân loại có embedding mã); `PooledMultiTask` (ghép hai đầu); `arch_spec` (chữ ký kiến trúc để phát hiện thay đổi). |
| `src/metrics.py` | MSE/MAE, confusion matrix, F1, Macro-F1, Brier, Cohen's kappa, hit rate, F1 hoán vị, block bootstrap; hiệu chỉnh xác suất (nhiệt độ, alpha, co về prior); `ticker_metrics` hiệu chỉnh trên val của một mã rồi tính chỉ số val và test. |
| `src/pipeline.py` | Điểm vào `python -m src.pipeline`. Huấn luyện 120 lượt (tiếp tục được nếu bị ngắt), đánh giá từng mã, tổng hợp val, chọn (mô hình, wd) cho mỗi (T, L) bằng val, ghi kết quả test, gọi `report.py`, cập nhật kho model, rồi vẽ đồ thị dự đoán của kho model. Cờ `--quick`, `--force`. |
| `src/report.py` | Từ dự đoán test của cấu hình đã chọn: tính bảng cuối cho model, ensemble 5 seed, baseline repeat và tần suất lớp; F1 hoán vị; khoảng tin cậy bootstrap; Brier skill; bảng theo mã; ma trận nhầm lẫn (CSV và hình). `plot_models`: với mỗi model trong `outputs/models/`, vẽ đường dự đoán bước T của đầu hồi quy và đường giá thực tế trên val và test (không vẽ train), tính MSE/MAE của model và baseline repeat. Điểm vào riêng `python -m src.report` (chỉ vẽ lại đồ thị kho model, không huấn luyện). |
| `src/predict.py` | `update_models`: chọn và cập nhật model tốt nhất cho từng (mã, T, L) vào `outputs/models/`. Điểm vào `python -m src.predict [mã ...]`: dự đoán 3 nhãn kèm xác suất cho T phiên sau phiên cuối của dữ liệu, ghi phiên mục tiêu theo lịch giao dịch (bỏ cuối tuần, ngày nghỉ) và cảnh báo khi dữ liệu chưa sync. `forward`: chạy model đã lưu cho cả hai đầu (hồi quy và phân loại), dùng chung cho dự đoán và vẽ đồ thị. |

### 2.2 Chức năng các thư mục trong `outputs/`

| Thư mục | Nội dung |
|---|---|
| `outputs/results/pooled_embed_k0.43/` | Toàn bộ kết quả của lần chạy đầy đủ: cấu hình, chỉ số từng lượt, trạng thái mô hình, bảng chọn cấu hình, bảng test cuối, ma trận nhầm lẫn. Tên thư mục chứa K_THETA nên đổi K sẽ tạo thư mục mới. |
| `outputs/results/pooled_embed_quick_k0.43/` | Cùng cấu trúc nhưng của lần chạy thử `--quick`; chỉ để kiểm tra luồng, không dùng để kết luận. |
| `outputs/results/predictions/` | Mỗi lần chạy `python -m src.predict` ghi một file `predictions_{ngày cuối dữ liệu}.csv`. |
| `outputs/figures/pooled_embed_k0.43/` | `confusion_T2.png`, `confusion_T5.png`. |
| `outputs/figures/models/` | 60 file `{mã}_T{T}_L{L}.png`: đường dự đoán và đường thực tế của từng model trong kho, chỉ trên val và test. |
| `outputs/results/models_regression.csv` | MSE/MAE (thang chuẩn hóa và thang giá gốc) của từng model trong kho và baseline repeat, trên val và test. |
| `outputs/models/` | 60 file `{mã}_T{T}_L{L}.pt`, `registry.csv` (model đang dùng), `registry_history.csv` (lịch sử cập nhật). |

---

## 3. Tham số, cấu hình, đặc trưng và lựa chọn

Tất cả nằm trong `src/config.py`.

### 3.1 Tái lập kết quả

| Tham số | Giá trị | Cách áp dụng và lý do |
|---|---|---|
| `SEED` | 59 | Seed gốc duy nhất, khai báo một lần. Mọi seed khác suy ra từ đây để toàn bộ kết quả tái lập được. |
| `SEEDS` | 59, 60, 61, 62, 63 | Mỗi cấu hình huấn luyện 5 lần với 5 seed. Báo cáo mean ± std để thấy mức dao động do khởi tạo ngẫu nhiên, tránh kết luận từ một lần chạy may mắn. |
| `DETERMINISTIC` | True | `torch.use_deterministic_algorithms(True)`, huấn luyện trên CPU: cùng seed cho cùng kết quả đến từng chữ số. |

### 3.2 Dữ liệu

| Tham số / lựa chọn | Giá trị | Cách áp dụng và lý do |
|---|---|---|
| `TICKERS` | 15 mã | Các mã vốn hóa lớn, đủ thanh khoản, có cùng dãy ngày giao dịch. |
| Đặc trưng đầu vào | **log giá đóng cửa** (`TARGET_COL = "close"`, một kênh) | Log biến tỷ lệ tăng/giảm thành hiệu số, nên θ và nhãn so sánh được giữa các mức giá. Các kênh khác (open/high/low/volume) đã thử và không cải thiện, nên bị loại để mô hình gọn. |
| `TRAIN_RATIO`, `VAL_RATIO` | 0.70, 0.15 (test là phần còn lại) | Chia **theo thời gian, không xáo trộn**, trên trục ngày chung của 15 mã. Hiện có 1446 dòng (2021-01-04 đến 2026-10-23) → 1012 / 216 / 218. Mục tiêu của mọi cửa sổ train nằm trước ngày đầu của val nên không rò rỉ tương lai. Đầu vào của cửa sổ val/test được lấy L điểm liền trước (có thể nằm trong train, nhưng mục tiêu thì không). |
| Chuẩn hóa | `StandardScaler` **riêng từng mã**, fit chỉ trên dòng train | Các mã có mức giá rất khác nhau; chuẩn hóa từng mã đưa chúng về cùng thang để học chung. Fit chỉ trên train để không dùng thông tin val/test. |
| Cửa sổ | cắt trong chuỗi của từng mã rồi mới nối mẫu | "Gộp mẫu, không gộp chuỗi": một cửa sổ không bao giờ chứa dữ liệu của hai mã. Mỗi mẫu mang `ticker_id`, `end_idx` và ngày mục tiêu. |

### 3.3 Nhãn phân loại

| Tham số | Giá trị | Cách áp dụng và lý do |
|---|---|---|
| r | `log(close_{t+T}) − log(close_t)` | Log-return thật (trên giá gốc, không phụ thuộc chuẩn hóa) giữa phiên cuối cửa sổ t và phiên t + T. |
| θ | `K_THETA · std(r_train)`, riêng từng mã và từng T | Mã biến động mạnh có ngưỡng rộng hơn, nên "Đi ngang" mang cùng ý nghĩa tương đối ở mọi mã. Chỉ tính từ train. |
| Nhãn | Giảm (0) nếu r < −θ; Đi ngang (1) nếu \|r\| ≤ θ; Tăng (2) nếu r > θ | Ba lớp phản ánh quyết định thực tế (bán / giữ / mua) và tránh coi biến động nhỏ là tín hiệu. |
| `K_THETA` | **0.43** | Nếu r xấp xỉ chuẩn thì P(\|r\| ≤ 0.43σ) ≈ 1/3, nên ba lớp gần cân bằng. Khi quét K, 0.43 là mức duy nhất cho Brier skill test dương ở cả 4 cặp (T, L). Ví dụ θ: VNM T = 2 khoảng ±0.78 %, FPT T = 5 khoảng ±1.5 %. |
| prior | tần suất 3 lớp ở train của từng mã | Dùng làm baseline xác suất, đích co xác suất và mẫu số của quy tắc chọn nhãn. |

### 3.4 Mô hình

| Tham số | Giá trị | Cách áp dụng và lý do |
|---|---|---|
| `MODELS` | Linear, NLinear, DLinear | So sánh ba cách biến đổi đầu vào (mục 1). |
| `LOOKBACKS` (L) | 30, 90 | Khoảng 1.5 tháng và 4.5 tháng giao dịch: nhìn ngắn hạn và trung hạn. |
| `HORIZONS` (T) | 2, 5 | Dự báo 2 phiên và 1 tuần giao dịch. |
| `MA_KERNEL` | 25 | Kernel trung bình trượt của DLinear, theo bài báo gốc, padding hai đầu để giữ độ dài L. |
| `CLS_TEMPORAL_DIM` (d) | 8 | Đầu phân loại nén L điểm còn 8 đặc trưng thời gian trước khi phân loại. Nhỏ để hạn chế quá khớp. |
| `TICKER_EMB_DIM` | 4 | Mỗi mã có một vector 4 chiều học được, cho phép đầu phân loại dịch quyết định theo đặc tính từng mã. Đây là phương án tốt nhất khi so với không embedding và với các đặc trưng thêm. |

### 3.5 Huấn luyện

| Tham số | Giá trị | Cách áp dụng và lý do |
|---|---|---|
| Hàm mất mát | `MSE(hồi quy) + CrossEntropy(phân loại)`, không trọng số lớp | Huấn luyện hai đầu cùng lúc; các lớp đã gần cân bằng nhờ K = 0.43 nên không cần trọng số lớp. |
| Optimizer | Adam, `LR = 1e-3` | Mặc định ổn định cho mô hình tuyến tính nhỏ. |
| `BATCH_SIZE` | 128 | Mẫu của 15 mã trộn chung (khoảng 13.800–14.700 mẫu train tùy L, T) nên dùng batch lớn hơn huấn luyện từng mã. |
| `MAX_EPOCHS`, `PATIENCE`, `MIN_DELTA` | 100, 10, 1e-9 | Dừng sớm **riêng cho từng đầu**: đầu hồi quy theo MSE val, đầu phân loại theo NLL val. Dừng khi cả hai cùng 10 epoch không cải thiện. Trạng thái cuối ghép trọng số tốt nhất của từng đầu (tham số `reg.*` từ epoch MSE val tốt nhất, `cls.*` từ epoch NLL val tốt nhất). |
| `WEIGHT_DECAYS` | 0, 1e-2 | Regularization L2. Thử có và không để val quyết định. |
| Shuffle | chỉ DataLoader train | Val/test giữ thứ tự thời gian. |
| `QUICK_GRID` | T = 2, L = 30, NLinear, 2 seed | Chạy thử toàn bộ luồng trong vài phút. |

### 3.6 Hiệu chỉnh xác suất và quy tắc chọn nhãn

Logit của mạng thường quá tự tin hoặc lệch lớp. Ba tham số được fit **trên val của từng mã** (không đọc test),
theo thứ tự:

| Bước | Tham số | Công thức và cách fit | Lý do |
|---|---|---|---|
| 1 | Nhiệt độ τ (`TEMP_BETA_RANGE = (0.01, 10)`, 80 vòng) | `p = softmax(z / τ)`; τ cực tiểu NLL val, tìm kiếm chia ba theo β = 1/τ (NLL lồi theo β) | Sửa độ tự tin của xác suất mà không đổi thứ hạng lớp. |
| 2 | α (`ALPHA_GRID = 0.0, 0.1, …, 1.0`) | nhãn = `argmax_c p_c / prior_c^α`; α cho Macro-F1 val cao nhất (hòa thì lấy α nhỏ nhất) | Mạng hay dồn về lớp phổ biến; chia cho prior^α bù lại để không bỏ quên lớp hiếm. α = 0 là argmax thông thường. |
| 3 | λ (co về prior) | `p' = λ·p + (1 − λ)·prior`; λ ∈ [0, 1] cực tiểu Brier val, có nghiệm đóng `λ = ⟨onehot − prior, p − prior⟩ / ‖p − prior‖²` | Khi tín hiệu yếu, kéo xác suất về tần suất lớp để không bao giờ tệ hơn baseline tần suất (về Brier). |

Xác suất báo cáo (`p_down`, `p_flat`, `p_up`) là p' (đã co). Nhãn dự đoán dùng điểm quyết định
`s = p / prior^α` (cột `s_*`). Vì vậy nhãn có thể khác lớp có xác suất lớn nhất khi α > 0.
`PRIOR_EPS = 1e-6` là sàn của prior để tránh chia cho 0.

### 3.7 Chọn cấu hình và chọn model từng mã

| Lựa chọn | Quy tắc | Lý do |
|---|---|---|
| Cấu hình chung cho mỗi (T, L) | (mô hình, wd) có Macro-F1 val (trung bình 15 mã, rồi trung bình 5 seed) cao nhất | Chỉ dùng val để chọn; test chỉ dùng một lần để báo cáo nên kết quả test không bị thổi phồng. |
| Model cho từng (mã, T, L) (`MODEL_SELECT_METRIC = "val_f1_macro"`) | trong 30 lượt (3 mô hình × 2 wd × 5 seed), lấy lượt có Macro-F1 val **của chính mã đó** cao nhất | Mỗi mã có thể hợp một kiến trúc khác. Hiện tại: DLinear 27, NLinear 21, Linear 12 model. |
| Cập nhật khi huấn luyện lại (`MODEL_MIN_GAIN = 1e-9`) | model đang lưu được chấm lại trên val hiện tại (hiệu chỉnh lại như ứng viên); ứng viên cao hơn quá 1e-9 thì thay, không thì giữ. Khác K_THETA hoặc kiến trúc thì luôn thay | So hai model trên cùng một tập và cùng một quy trình. 1e-9 bỏ qua sai số làm tròn khi đọc CSV. |

### 3.8 Báo cáo

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `STD_DDOF` | 0 | std qua 5 seed. |
| `PERM_N` | 200 | Số lần hoán vị nhãn dự đoán khi tính F1 hoán vị. |
| `BOOTSTRAP_BLOCK`, `BOOTSTRAP_N`, `CI_LEVEL` | 20 phiên, 500 lần, 95 % | Khoảng tin cậy block bootstrap theo ngày cho F1 và kappa trên test. |
| `SCALE_NOTE` | "nhãn từ log-return thật, không phụ thuộc thang" | Ghi rõ thang của chỉ số phân loại trong bảng kết quả. |
| `FIG_*`, `CONFUSION_*`, `DISPLAY_*` | | Định dạng hình và in bảng. |
| `MODEL_FIGURES_DIR`, `MODEL_FIG_FILE` | `outputs/figures/models/`, `{mã}_T{T}_L{L}.png` | Nơi lưu đồ thị dự đoán của từng model trong kho (cùng tên với file `.pt`). |
| `MODEL_REGRESSION_FILE` | `outputs/results/models_regression.csv` | Bảng MSE/MAE của các model trong kho và baseline repeat. |
| `MODEL_FIGSIZE`, `MODEL_ACTUAL_COLOR`, `MODEL_PRED_COLOR`, `MODEL_TEST_SHADE`, `MODEL_LINE_WIDTH` | (11, 4.8), đen, đỏ, nền xanh nhạt, 1.1 | Kích thước hình, màu đường thực tế/dự đoán, màu nền vùng test, độ dày đường. |
| `MARKET_HOLIDAYS` | `("2027-01-01",)` | Ngày nghỉ **sắp tới** của sàn, dùng để tính ngày mục tiêu của dự đoán. Ngày nghỉ đã qua không cần khai báo: suy ra từ dữ liệu (ngày thường không có phiên). Cần bổ sung khi sở giao dịch công bố lịch nghỉ (Tết, 30/4, 1/5, 2/9, ngày nghỉ bù). |
| `MARKET_CALENDAR_UNTIL` | "2027-01-31" | Đã khai báo đủ ngày nghỉ đến ngày này. Ngày mục tiêu sau mốc này được ghi `dự kiến` vì có thể lùi khi có ngày nghỉ chưa khai báo. Nhớ dời mốc khi bổ sung `MARKET_HOLIDAYS`. |
| `SESSION_DATA_READY` | "15:00" | Giờ máy mà từ đó phiên của hôm nay được coi là đã có dữ liệu. Dùng để cảnh báo dữ liệu chưa sync và đánh dấu phiên mục tiêu `đã qua`. |
| `DEMO_CHART_START` | "2026-10-01" | Phiên mục tiêu đầu tiên trên đồ thị của `demo.py` (đồ thị chạy đến hôm nay). |
| `DEMO_LABEL_COLORS`, `DEMO_ACTUAL_COLOR`, `DEMO_PRED_COLOR`, `DEMO_REPEAT_COLOR` | | Màu nhãn (Giảm đỏ, Đi ngang xám, Tăng xanh) và màu ba đường trong `demo.py`. |

### 3.9 Đồng bộ dữ liệu

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `SYNC_INTERVAL` | "1D" | Dữ liệu theo phiên. |
| `SYNC_MAX_RETRIES`, `SYNC_RETRY_BACKOFF` | 3, 2.0 | Thử lại khi API lỗi, chờ 2 · lần_thử giây. |
| `SYNC_API_DELAY` | 1.0 giây | Nghỉ giữa hai lần gọi API để tránh bị chặn. |
| `SESSION_HOUR_UTC` | 7 | Mỗi phiên ghi đúng một mốc ngày + 07:00 UTC, nên lọc trùng theo ngày là chính xác. |

### 3.10 Manifest, dataset_id, snapshot và các module nhãn thích ứng

`sync_stock_parquet.py` ghi `stock/manifest.json` (số dòng, phiên cuối, băm nội dung từng mã) và `dataset_id` suy ra từ đó; `build_panel` chỉ nhận `dataset_id` khi manifest khớp dữ liệu vừa đọc. `python -m src.snapshot freeze` sao parquet và manifest vào `stock/snapshots/<dataset_id[:16]>` để chạy lại đúng dữ liệu đã dùng.

- `pipeline`: `experiment.json` có thêm khóa `dataset_id`; đổi `dataset_id` được coi như dữ liệu đổi nên chạy lại từ đầu.
- `predict`: model lưu `dataset_id` trong `trained_on`/`evaluated_on`. Bảng dự đoán có thêm cột `status` (`OK` hoặc `DATA_INVALID`) và `reason`. Hàng `DATA_INVALID` (lý do `MISSING_SESSION`: thiếu phiên lẽ ra đã có, hoặc có ngày thường không giải thích được trong L phiên cuối) có xác suất NaN và `label = DATA_INVALID`; `target_date`, `target_note` vẫn được điền.

Các module mới `validation`, `snapshot`, `price_features`, `labels`, `splits`, `dataset` (kiểm tra dữ liệu, manifest, đặc trưng giá/khối lượng, nhãn thích ứng theo biến động, fold theo thời gian, bộ dữ liệu huấn luyện/suy luận) chưa được các model đang lưu trong `outputs/models/` sử dụng: các model này vẫn huấn luyện bằng nhãn K_THETA ở trên. Riêng kiểm tra dữ liệu (`validation`, `Panel.gaps`) đã được `build_panel` và `predict` dùng. `features_ok` cô lập lỗi theo từng mã chỉ khi Panel được dựng tay (như trong `tests/test_dataset.py`); qua `build_panel`, một mã có tập ngày khác các mã còn lại làm `check_aligned` ném `ValueError` cho cả lần dựng, nên không tới được bước đó.

---

## 4. Cách chạy và luồng hoạt động

### 4.1 Các lệnh theo thứ tự

```bash
pip install -r requirements.txt     # 1. cài thư viện (một lần)
python sync_stock_parquet.py        # 2. lấy phiên mới nhất từ vnstock (vài chục giây)
python -m src.pipeline              # 3. huấn luyện, đánh giá, cập nhật kho model, vẽ đồ thị (~25–35 phút nếu phải huấn luyện lại)
python -m src.predict               # 4. dự đoán phiên tới cho cả 15 mã (vài giây)
python -m src.predict VNM FPT       #    hoặc chỉ một vài mã
streamlit run demo.py               # 5. (tùy chọn) mở giao diện demo trên trình duyệt
```

Lệnh phụ:

- `python sync_stock_parquet.py --no-fetch`: chỉ làm sạch, lọc trùng dữ liệu đang có.
- `python -m src.pipeline --quick`: chạy thử lưới nhỏ trong vài phút.
- `python -m src.pipeline --force`: bỏ kết quả cũ, huấn luyện lại từ đầu.
- `python -m src.report`: chỉ vẽ lại đồ thị dự đoán và bảng MSE/MAE cho các model đang có trong kho (khoảng 20 giây,
  không huấn luyện). Dùng sau khi sync dữ liệu mới để xem model hiện tại bám giá thế nào trên dữ liệu mới nhất.

### 4.2 Luồng hoạt động

**Bước 1: Đồng bộ dữ liệu (`sync_stock_parquet.py`).** Script đọc file CSV của từng mã, hỏi vnstock các phiên
mới hơn phiên cuối đang có, ghép vào, đưa mọi mốc thời gian về "ngày + 07:00 UTC" rồi bỏ các ngày bị lặp. Kết quả
ghi ra `{mã}_stock.parquet`. Sau bước này mỗi phiên có đúng một dòng và 15 mã có cùng dãy ngày.

**Bước 2: Đọc và kiểm tra dữ liệu (`pipeline.main` → `data.build_panel`).** Pipeline đọc 15 file parquet, kiểm tra
15 mã có đúng cùng dãy ngày và không có ngày lặp (nếu có lỗi thì dừng và nhắc chạy lại bước 1), rồi lấy log giá
đóng cửa. Nó in số dòng và mốc chia train/val/test.

**Bước 3: Kiểm tra có cần huấn luyện lại (`Store`).** Pipeline so chữ ký hiện tại (kiến trúc, K_THETA, danh sách mã,
số dòng, ngày cuối) với `experiment.json`. Nếu giống, nó dùng lại các lượt đã chạy và chỉ chạy những lượt còn
thiếu (nên có thể dừng giữa chừng rồi chạy tiếp). Nếu khác (ví dụ vừa sync thêm phiên mới), nó huấn luyện lại từ đầu.

**Bước 4: Huấn luyện lưới 120 lượt (`run_experiment`, `fit`).** Với mỗi cặp (T, L), dữ liệu gộp được xây một lần:
chuẩn hóa từng mã theo train, cắt cửa sổ, tính θ và nhãn. Sau đó, với mỗi mô hình, mỗi weight decay và mỗi seed,
một mô hình đa nhiệm được huấn luyện trên mẫu trộn của 15 mã cho đến khi dừng sớm. Mỗi lượt mất khoảng 8–15 giây và
in ra Macro-F1, kappa và thời gian còn lại.

**Bước 5: Đánh giá từng mã (`evaluate_run`, `metrics.ticker_metrics`).** Với từng mã, pipeline lấy logit val, fit
nhiệt độ, α và λ trên val của mã đó, rồi áp dụng nguyên các tham số ấy lên test. Nó ghi chỉ số val, chỉ số test, ma
trận nhầm lẫn và xác suất test của mọi lượt.

**Bước 6: Chọn cấu hình bằng val (`val_summary`, `select_configs`).** Pipeline lấy trung bình chỉ số val qua 15 mã
rồi qua 5 seed cho từng (T, L, mô hình, wd), và chọn (mô hình, wd) có Macro-F1 val cao nhất cho mỗi (T, L).

**Bước 7: Báo cáo test (`finalize`, `report.main`).** Chỉ dự đoán test của cấu hình đã chọn được dùng. Bảng cuối so
mô hình với ensemble 5 seed, baseline repeat và baseline tần suất lớp, kèm F1 hoán vị, khoảng tin cậy bootstrap và
Brier skill. Bảng theo mã và ma trận nhầm lẫn (CSV và hình) cũng được ghi ra.

**Bước 8: Cập nhật kho model (`predict.update_models`).** Với mỗi (mã, T, L), lượt có Macro-F1 val của mã đó cao nhất
là ứng viên. Model đang lưu (nếu có) được chấm lại trên val hiện tại; model nào cao hơn thì được giữ trong
`outputs/models/`. Mọi quyết định được ghi vào `registry_history.csv`. Bước này cũng chạy sau `--quick`.

**Bước 9: Vẽ đồ thị kho model (`report.plot_models`).** Với từng model trong `outputs/models/`, pipeline dựng lại
các cửa sổ val và test của mã đó từ dữ liệu hiện tại, chuẩn hóa bằng scaler lưu cùng model và chạy đầu hồi quy để
lấy dự báo T bước. Đồ thị vẽ đường giá đóng cửa thực tế từ ngày đầu của val đến phiên cuối (bỏ phần train để nhìn
rõ phần dự đoán) và đường dự đoán ở bước T (giá dự đoán cho ngày t + T, đặt tại ngày t + T), trên thang giá gốc;
vùng test được tô nền. Tiêu đề ghi MSE/MAE của model và của baseline repeat trên val và test. Mọi chỉ số được ghi vào
`models_regression.csv` và trung bình qua mã được in ra terminal. Bước này cũng chạy sau `--quick`, và chạy riêng
được bằng `python -m src.report`.

**Bước 10: Dự đoán (`python -m src.predict`).** Với mỗi mã được yêu cầu và mỗi (T, L), lệnh nạp model đã lưu, lấy L
phiên mới nhất, chuẩn hóa bằng scaler lưu cùng model, tính logit, áp dụng τ, α, λ đã lưu, rồi in xác suất 3 nhãn,
nhãn dự đoán và ngưỡng % tương ứng. Kết quả cũng được ghi ra CSV.

Ngày của dự đoán được tính như sau:
- **Gốc** là phiên cuối của dữ liệu (`last_date`), không phải hôm nay. Chạy vào thứ Bảy, Chủ Nhật hay ngày nghỉ cho
  cùng kết quả với phiên giao dịch gần nhất trước đó, vì không có dữ liệu mới.
- **Phiên mục tiêu** (`target_date`) là phiên giao dịch thứ T sau gốc: đếm thứ Hai đến thứ Sáu, bỏ ngày nghỉ. Ngày nghỉ
  đã qua lấy từ chính dữ liệu, ngày nghỉ sắp tới lấy từ `MARKET_HOLIDAYS`. Ví dụ gốc thứ Sáu 2026-10-23 cho T = 2 là
  thứ Ba 2026-10-27 và T = 5 là thứ Sáu 2026-10-30; gốc 2026-04-29 cho T = 2 là 2026-05-05 (nghỉ 30/4 và 1/5).
- **Thời điểm dự đoán** (`predicted_at`) là lúc chạy lệnh.
- **Kiểm tra dữ liệu cũ:** phiên mới nhất lẽ ra đã có là hôm nay (nếu là ngày giao dịch và đã qua `SESSION_DATA_READY`)
  hoặc phiên giao dịch trước đó. Nếu dữ liệu dừng trước phiên này, lệnh in cảnh báo kèm danh sách phiên thiếu và
  nhắc chạy `python sync_stock_parquet.py`; nếu phiên thiếu thực ra là ngày nghỉ chưa khai báo thì thêm vào
  `MARKET_HOLIDAYS`. Dự đoán vẫn được tính từ phiên cuối hiện có.
- Màn hình in thêm phiên mục tiêu của từng T kèm thứ trong tuần.

### 4.3 Demo Streamlit (`demo.py`)

```bash
pip install -r requirements.txt     # cần streamlit và plotly (một lần)
streamlit run demo.py               # mở http://localhost:8501; dừng bằng Ctrl+C
```

Chạy từ thư mục gốc của project, sau khi đã có kho model (`python -m src.pipeline`). Nên chạy
`python sync_stock_parquet.py` trước để có phiên mới nhất; demo tự đọc lại dữ liệu và model khi file thay đổi (chỉ cần
tải lại trang), không cần khởi động lại.

1. **Chọn mã.** Trang chỉ hiện hộp chọn mã; chưa chọn thì chưa hiện gì khác.
2. **Phần trên: nhãn dự đoán.** Hai cột T+2 và T+5, tiêu đề là phiên mục tiêu kèm thứ trong tuần (tính như Bước 10).
   Mỗi cột có hai thẻ, một cho model L = 30 và một cho L = 90: nhãn (▼ Giảm đỏ, ▶ Đi ngang xám, ▲ Tăng xanh), xác suất
   của nhãn đó, ba xác suất đầy đủ, ngưỡng % của Tăng / Giảm và Macro-F1 val của model. Kết quả giống
   `python -m src.predict` nhưng không ghi file CSV. Nếu dữ liệu thiếu phiên lẽ ra đã có, trang hiện cảnh báo như ở
   Bước 10. Nhãn chọn theo argmax(p / prior^alpha) nên có thể khác lớp có xác suất lớn nhất.
3. **Phần dưới: đồ thị giá dự báo và giá thực tế.** Thanh chọn tầm dự báo (T+2 / T+5), độ dài cửa sổ (L = 30 / 90) và
   bật/tắt baseline repeat. Với mỗi phiên gốc d, model của (mã, T, L) dự báo giá đóng cửa tại phiên thứ T sau d (bước
   cuối của đầu hồi quy, đổi về thang giá gốc bằng scaler lưu cùng model) từ L phiên kết thúc tại d; điểm dự báo đặt tại
   phiên mục tiêu, viền điểm mang màu nhãn dự đoán của đầu phân loại. Đồ thị gồm:
   - giá thực tế từ `DEMO_CHART_START` đến hôm nay;
   - giá dự báo (đường gạch);
   - dự báo có phiên mục tiêu chưa có giá thật (điểm rỗng, đường chấm), có thể nằm sau hôm nay;
   - baseline repeat (giá đóng cửa của phiên gốc);
   - đường chấm dọc tại phiên cuối của dữ liệu.

   Đồ thị Plotly tương tác: rê chuột xem giá, phiên gốc, nhãn và xác suất; kéo để phóng to, nhấp đúp để về ban đầu;
   nhấp chú thích để ẩn/hiện từng đường; trục ngày bỏ thứ Bảy, Chủ Nhật. Dưới đồ thị là bảng MAE/MSE ở **thang giá gốc**
   của model và baseline repeat trên các phiên trong khoảng đã có giá thật, và một bảng chi tiết từng điểm (mở rộng).
   Giai đoạn từ 2026-10-01 nằm trong tập test hoặc sau dữ liệu huấn luyện, nên đây là dự báo ngoài mẫu. Model trong kho
   được chọn theo Macro-F1 của đầu phân loại, nên đường dự báo hồi quy thường bám sát repeat.

Đổi mốc bắt đầu của đồ thị bằng `DEMO_CHART_START` trong `src/config.py`.

---

## 5. Các file kết quả

### 5.1 `outputs/results/pooled_embed_k0.43/`

| File | Biểu diễn cho |
|---|---|
| `experiment.json` | Chữ ký của lần chạy: `input`, `cls_head`, `cls_temporal_dim`, `ticker_emb_dim`, `k_theta`, `tickers`, `n_rows`, `last_date`. Khác chữ ký hiện tại thì pipeline huấn luyện lại. |
| `runs.csv` | Một dòng cho mỗi (mã, T, L, mô hình, wd, seed): 15 × 120 = 1.800 dòng. Gồm θ, kích thước tập, tham số hiệu chỉnh (`temperature`, `alpha`, `shrink`), chỉ số val (`val_f1_macro`, `val_kappa`, `val_brier_skill`, `val_nll_raw`, …), chỉ số hồi quy test (`mse_scaled`, `mae_scaled`), chỉ số phân loại test (`accuracy`, `f1_macro`, `f1_weighted`, F1 từng lớp, `kappa`, `brier`) và ma trận nhầm lẫn test `cm_ij` (hàng i là nhãn thật, cột j là nhãn dự đoán). Đây là dữ liệu gốc của mọi bảng khác. |
| `preds_all.csv` | Xác suất và điểm quyết định trên test của mọi lượt (mọi mẫu). |
| `states.pt` | Trọng số của mọi lượt; dùng để tiếp tục khi bị ngắt và để xuất model vào kho. |
| `val_summary.csv` | 24 dòng (4 cặp T, L × 3 mô hình × 2 wd): mean và std qua 5 seed của các chỉ số val (đã trung bình qua 15 mã). Là căn cứ để chọn cấu hình. |
| `final_config.csv` | 4 dòng: cấu hình được chọn cho mỗi (T, L) và điểm val của nó. Hiện tại cả 4 đều là DLinear/wd = 0. |
| `test_predictions.csv` | Dự đoán test của cấu hình đã chọn: `ticker, T, L, model, wd, seed, date, y_true, p_down, p_flat, p_up, s_down, s_flat, s_up, y_pred`. |
| `train_class_freq.csv` | prior (tần suất 3 lớp ở train) của từng (mã, T, L). |
| `final_table.csv` | **Bảng kết quả chính**, 16 dòng: mỗi (T, L) có 4 phương pháp `model`, `ensemble`, `repeat`, `class_freq`. Các cột: khoảng ngày test, số mẫu, thang, mean ± std qua seed của `f1_macro`, `brier`, `hit_rate`, `directional_share`, `kappa`, `f1_perm`, khoảng tin cậy 95 % của F1 và kappa, `brier_skill`, `f1_minus_class_freq`, `f1_minus_perm`. `headline = True` đánh dấu L có val tốt nhất cho mỗi T. |
| `per_ticker.csv` | Theo từng (T, L, mã), trung bình 5 seed: `f1_macro`, `kappa`, `f1_perm`, `brier`, `f1_minus_perm`. Cho thấy mã nào có tín hiệu, mã nào không. |
| `confusion_T{T}_L{L}.csv` | Ma trận nhầm lẫn test 3 × 3 (cộng 15 mã, trung bình 5 seed), số mẫu. |
| `confusion_T{T}_L{L}_norm.csv` | Cùng ma trận, chuẩn hóa theo hàng: P(dự đoán \| nhãn thật). |

Kết quả hiện tại trên test (thang: nhãn từ log-return thật, mean qua 5 seed):

Mô hình vượt cả hai baseline ở mọi (T, L) nhưng tín hiệu nhỏ (kappa 0.01–0.04, F1 hơn F1 hoán vị 0.007–0.031).

### 5.2 `outputs/figures/pooled_embed_k0.43/`

| File | Biểu diễn cho |
|---|---|
| `confusion_T2.png`, `confusion_T5.png` | Ma trận nhầm lẫn chuẩn hóa theo hàng của cấu hình đã chọn, mỗi ô con là một L. Mỗi ô ghi % và số mẫu, cho thấy mô hình hay nhầm lớp nào sang lớp nào. |

### 5.3 `outputs/models/`

| File | Biểu diễn cho |
|---|---|
| `{mã}_T{T}_L{L}.pt` (60 file) | Một model hoàn chỉnh để dự đoán: trọng số, chỉ số mã trong embedding, scaler của mã, τ, α, λ, prior, θ, K_THETA, chữ ký kiến trúc, chỉ số val/test và ngày dữ liệu đã huấn luyện/đánh giá. |
| `registry.csv` | 60 dòng, model đang dùng: `ticker, T, L, model, wd, seed, val_f1_macro, val_kappa, val_brier_skill, test_f1_macro, test_kappa, trained_last_date, evaluated_last_date, trained_at`. |
| `registry_history.csv` | Mỗi lần cập nhật thêm 60 dòng: `updated_at, ticker, T, L, action` (`new` / `kept` / `replaced`), điểm val của ứng viên và của model cũ, ứng viên là gì, model nào được giữ. |

### 5.4 Đồ thị và chỉ số hồi quy của kho model

| File | Biểu diễn cho |
|---|---|
| `outputs/figures/models/{mã}_T{T}_L{L}.png` (60 file) | Mỗi hình là một model trong kho. Trục ngang là ngày, chỉ từ đầu val đến phiên cuối (không vẽ train). Đường đen là giá đóng cửa thực tế; đường đỏ là dự đoán bước T của đầu hồi quy, đặt tại ngày mục tiêu t + T (đổi về thang giá gốc). Vùng nền xanh là test, vùng trắng là val. Tiêu đề: mã, T, L, cấu hình (mô hình/wd/seed), ngày cuối của dữ liệu; MSE / MAE thang chuẩn hóa (trung bình T bước) của model và repeat trên val và test; MAE thang giá gốc. Đường dự đoán gần như là đường thực tế dịch sang phải T phiên nghĩa là mô hình chủ yếu lặp giá cuối, tức gần với repeat. |
| `outputs/results/models_regression.csv` | 4 dòng cho mỗi model (val/test × model/repeat), tổng 240 dòng: `ticker, T, L, config, split, method, date_from, date_to, n, mse_scaled, mae_scaled, mse_price, mae_price`. `method = repeat` là baseline lặp giá trị cuối của cửa sổ cho cả T bước. `date_from`/`date_to` là ngày mục tiêu (bước T) đầu và cuối của tập. Với model mới huấn luyện, `mse_scaled`/`mae_scaled` trên test trùng với `runs.csv` của lượt đó. |

Đọc kết quả: model nào có `mse_scaled` thấp hơn repeat trên test là đầu hồi quy có ích hơn "giá không đổi". Ở dữ
liệu hiện tại (đến 2026-10-23), trung bình qua 15 mã thì đầu hồi quy chưa vượt repeat ở cả 4 (T, L) (test T = 2:
0.0290–0.0292 so với 0.0288; T = 5: 0.0600–0.0628 so với 0.0589). Điều này phù hợp với giá gần bước ngẫu nhiên.
Lưu ý model trong kho được chọn theo Macro-F1 val của đầu phân loại, không theo MSE.

### 5.5 `outputs/results/predictions/predictions_{ngày}.csv`

Mỗi dòng là một (mã, T, L): `p_down, p_flat, p_up` (xác suất đã hiệu chỉnh, tổng bằng 1), `label` (nhãn dự đoán),
`up_above_pct = 100·(e^θ − 1)` và `down_below_pct = 100·(e^{−θ} − 1)` (ngưỡng % để được tính là Tăng / Giảm),
model dùng (`model, wd, seed`), `val_f1_macro` của model đó và `trained_last_date`. Các cột ngày:

| Cột | Ý nghĩa |
|---|---|
| `last_date` | Phiên cuối của dữ liệu, là gốc của dự đoán (cửa sổ L phiên kết thúc tại đây). |
| `target_date` | Phiên giao dịch thứ T sau `last_date` (đã bỏ cuối tuần và ngày nghỉ): nhãn nói về giá đóng cửa phiên này so với `last_date`. |
| `predicted_at` | Thời điểm chạy dự đoán. |
| `target_note` | Trống nếu bình thường; `đã qua` nếu phiên mục tiêu đã diễn ra lúc chạy (dữ liệu chưa sync); `dự kiến` nếu phiên mục tiêu nằm sau `MARKET_CALENDAR_UNTIL`, tức có thể lùi khi có ngày nghỉ chưa khai báo. |

`{ngày}` là phiên cuối của dữ liệu, nên chạy lại khi chưa có phiên mới sẽ ghi đè cùng một file. Chạy cho một vài mã
(`python -m src.predict VNM`) cũng ghi vào file này và chỉ còn các mã đó.

Cách đọc: nên xem `val_f1_macro`. Model có giá trị thấp (ví dụ < 0.25) hoặc ba xác suất đều quanh 33 % gần như không
có tín hiệu cho mã đó.

---

## 6. Chỉ số đánh giá

Ký hiệu: N mẫu, 3 lớp c ∈ {Giảm, Đi ngang, Tăng}, y là nhãn thật, ŷ là nhãn dự đoán, p_c là xác suất dự đoán của
lớp c. Ma trận nhầm lẫn CM[i, j] = số mẫu có nhãn thật i và nhãn dự đoán j.

### 6.1 Chỉ số hồi quy (thang chuẩn hóa)

| Chỉ số | Công thức | Lý do |
|---|---|---|
| MSE (`mse_scaled`) | (1 / N·T) Σ (ŷ − y)² | Hàm mất mát và tiêu chí dừng sớm của đầu hồi quy. |
| MAE (`mae_scaled`) | (1 / N·T) Σ \|ŷ − y\| | Ít nhạy với ngoại lai hơn MSE. |

Cả hai tính trên log giá **sau StandardScaler** của từng mã, nên so được giữa các mã nhưng không phải đơn vị đồng.

Trong `models_regression.csv` và đồ thị kho model có thêm hai chỉ số **thang giá gốc**:

| Chỉ số | Công thức | Lý do |
|---|---|---|
| MSE giá gốc (`mse_price`) | (1 / N·T) Σ (e^{ŷ_log} − e^{y_log})², với y_log = y·σ + μ (đảo scaler) | Sai số theo đơn vị giá trong dữ liệu, dễ hình dung hơn thang chuẩn hóa. |
| MAE giá gốc (`mae_price`) | (1 / N·T) Σ \|e^{ŷ_log} − e^{y_log}\| | Sai lệch giá trung bình mỗi bước dự báo. Không so được giữa các mã vì mức giá khác nhau. |

Mọi chỉ số hồi quy đều được báo cáo cùng baseline **repeat** (ŷ = giá trị cuối của cửa sổ cho cả T bước).

### 6.2 Chỉ số phân loại

| Chỉ số | Công thức | Lý do dùng |
|---|---|---|
| Accuracy | Σ_c CM[c, c] / N | Dễ hiểu nhưng có thể cao khi chỉ đoán lớp phổ biến; chỉ dùng tham khảo. |
| Precision, Recall, F1 lớp c | P_c = CM[c,c] / Σ_i CM[i,c]; R_c = CM[c,c] / Σ_j CM[c,j]; F1_c = 2·P_c·R_c / (P_c + R_c) | Chất lượng từng lớp (`f1_down`, `f1_flat`, `f1_up`). |
| **Macro-F1** (`f1_macro`) | (1/3) Σ_c F1_c | **Chỉ số chính** để chọn cấu hình, alpha và model. Ba lớp có trọng số ngang nhau, nên mô hình chỉ đoán "Đi ngang" bị phạt (F1 ≈ 0.19), buộc phải nhận ra cả Tăng và Giảm. |
| Weighted-F1 | Σ_c F1_c · n_c / N | Tham khảo: F1 có trọng số theo số mẫu của lớp. |
| **Cohen's kappa** (`kappa`) | κ = (p_o − p_e) / (1 − p_e), p_o = accuracy, p_e = Σ_c (Σ_j CM[c,j] / N)·(Σ_i CM[i,c] / N) | Mức khớp **vượt quá ngẫu nhiên** khi giữ nguyên phân bố nhãn thật và nhãn dự đoán. κ = 0 là không có tín hiệu, < 0 là tệ hơn ngẫu nhiên. Với chuỗi giá gần bước ngẫu nhiên, đây là cách rõ nhất để biết mô hình có tín hiệu thật hay không. |
| **Brier đa lớp** (`brier`) | (1/N) Σ_n Σ_c (p_{n,c} − 1[y_n = c])² (khoảng 0 đến 2, thấp hơn là tốt) | Đánh giá **chất lượng xác suất**, không chỉ nhãn. Cần vì đầu ra của bài toán là xác suất của từng nhãn. |
| **Brier skill** (`brier_skill`, `val_brier_skill`) | 1 − Brier_model / Brier_class_freq | > 0 nghĩa là xác suất của mô hình tốt hơn việc luôn đưa ra tần suất lớp train. Là tiêu chí chọn K_THETA = 0.43. |
| NLL (`val_nll_raw`) | −(1/N) Σ_n log p_{n, y_n} | Hàm mất mát của đầu phân loại, tiêu chí dừng sớm và tiêu chí fit nhiệt độ τ. |
| Hit rate (`hit_rate`) | (CM[Giảm,Giảm] + CM[Tăng,Tăng]) / (số mẫu dự đoán Giảm hoặc Tăng) | Khi mô hình "ra tín hiệu" (mua/bán), tỷ lệ đúng là bao nhiêu. NaN nếu không bao giờ dự đoán Giảm/Tăng (repeat, class_freq). |
| Directional share (`directional_share`) | tỷ lệ mẫu có ŷ ≠ Đi ngang | Mô hình ra tín hiệu thường đến mức nào; đọc cùng hit rate. |

### 6.3 Baseline và kiểm định

| Chỉ số / baseline | Cách tính | Lý do |
|---|---|---|
| **repeat** | Lặp giá trị cuối của cửa sổ ⇒ r = 0 ⇒ luôn đoán Đi ngang, xác suất one-hot | Baseline bắt buộc của LTSF ("giá không đổi"). Có mặt trong mọi bảng. |
| **class_freq** | Xác suất luôn bằng prior train, nhãn là lớp phổ biến nhất | Mốc của Brier skill (bằng 0) và mốc "không dùng đầu vào". |
| **ensemble** | Trung bình xác suất và điểm quyết định của 5 seed cho mỗi (mã, ngày), rồi argmax | Kiểm tra việc gộp seed có làm tín hiệu ổn định hơn không. |
| F1 hoán vị (`f1_perm`) | Trung bình Macro-F1 sau 200 lần hoán vị ngẫu nhiên nhãn dự đoán (giữ tỷ lệ từng nhãn, cắt liên hệ với nhãn thật) | Mức F1 một bộ dự đoán **không có tín hiệu** với cùng tỷ lệ nhãn đạt được. Macro-F1 có thể tăng chỉ vì đoán đều ba lớp; `f1_minus_perm = f1_macro − f1_perm` tách phần tín hiệu thật. |
| `f1_minus_class_freq` | f1_macro − f1_macro(class_freq) | Mức vượt baseline tần suất lớp. |
| Khoảng tin cậy 95 % (`f1_ci_lo/hi`, `kappa_ci_lo/hi`) | Moving block bootstrap trên trục ngày test: khối 20 phiên liền nhau, 500 lần, phân vị 2.5 % và 97.5 %. Mọi mã và seed dùng chung chỉ số ngày | Test chỉ khoảng 215 phiên và các phiên liền nhau tương quan, các mã cùng ngày cũng tương quan. Bootstrap theo khối ngày giữ được các tương quan này, nên khoảng tin cậy trung thực hơn bootstrap từng mẫu. Khoảng của kappa chứa 0 nghĩa là chưa chắc có tín hiệu. |
| mean ± std qua 5 seed | Trung bình qua 15 mã cho từng seed, rồi mean và std (ddof = 0) qua 5 seed | Đo độ ổn định theo khởi tạo, theo ràng buộc báo cáo của project. |

### 6.4 Vì sao chọn bằng Macro-F1 val và báo cáo thêm kappa, Brier

- **Macro-F1** phù hợp mục tiêu "nhận ra cả ba hướng": không thể đạt điểm cao bằng cách chỉ đoán một lớp.
- **Kappa** và **F1 hoán vị** chống lại việc Macro-F1 bị thổi phồng do phân bố nhãn dự đoán.
- **Brier/Brier skill** đánh giá xác suất, là đầu ra mà `predict` đưa cho người dùng.
- Mọi lựa chọn (cấu hình, α, τ, λ, model từng mã) chỉ dùng **val**; **test chỉ để báo cáo**. Riêng điểm val của
  model chọn theo từng mã bị lạc quan (chọn 1 trong 30 lượt trên chính val): kappa val trung bình 0.09–0.17 trong khi
  trên test chỉ 0.02–0.06. Vì vậy cần đọc chỉ số test trong `registry.csv` khi đánh giá độ tin cậy của một model.
