# Nguồn dữ liệu ngoài tin cho nhãn độ lớn "mag": baseline HAR và lịch sự kiện doanh nghiệp, thiết kế

Ngày: 2026-11-12. Nguồn ý tưởng: đề xuất của chủ dự án cùng ngày ("Nguồn dữ liệu ngoài tin tức cho fusion model dự báo độ lớn biến động").
Bối cảnh: `docs/reports/mmf_mag_report.md` (R7, R8), phụ lục 2026-11-12 của `docs/reports/mmf_preregistration.md`.

## Mục đích và lý do

R7/R8 cho thấy tín hiệu độ lớn của nhãn `mag-median-v1` chủ yếu là **tính bền của biến động**: mã vốn biến động mạnh và gần đây biến động mạnh thì sắp tới cũng mạnh. `prior_adaptive` gần ngang mạng sâu ở vài ô, và trên fold lùi (R8a) fusion không hơn `prior_fixed`. Tin tức không thêm gì đo được (R3, R6, R7c, R8c), phù hợp với nhận định "tin thường đến sau giá".

Nghiên cứu này hỏi đúng một câu: **có nguồn dữ liệu biết trước (ex-ante) và khác nhau giữa các mã nào mang thông tin chưa nằm trong lịch sử biến động không?** Để trả lời công bằng, phải có một baseline biến động mạnh (dạng HAR, Corsi 2009) được đăng ký trước, rồi mới đo phần đóng góp thêm của nguồn mới.

## Đã kiểm tra trước khi viết (2026-11-12)

| Nguồn | Kết quả kiểm tra | Hệ quả |
|---|---|---|
| Lịch sự kiện, vnstock `Company(source="vci")._fetch_events` | Có `public_date` (ngày công bố), `record_date`, `exright_date`, `display_date1`, mã sự kiện `DIV, ISS, AGME, AGMR, EGME, …`; lịch sử từ 2020 (VNM 29 sự kiện, HPG 17, VCB 23 cho 5 loại trên). `source="kbs"` trả bảng rỗng. | Dùng được, point-in-time theo `public_date`. Thưa: khoảng 3–6 sự kiện/mã/năm. |
| Tin trong `nlp.db` về chốt quyền | Khoảng 10–20 tiêu đề/năm cho cả thị trường | Không đủ làm lịch sự kiện; chỉ dùng để đối chiếu ngày công bố. |
| Dòng tiền khối ngoại, tự doanh | vnstock 4 (`Trading`) không có lịch sử; Finhay chỉ từ 2026-07-28 (cron 15:40 đang tích lũy); CafeF khoảng 248 phiên | **Không vào nghiên cứu này.** Chỉ thu thập tiến (đã chạy). |
| Sổ lệnh | `market_quote` chỉ có giá khớp, cao/thấp, khối lượng; không có bid/ask, độ sâu | **Không vào nghiên cứu này.** Không xây collector mới bây giờ. |
| OHLCV | Snapshot `24251acc1a5695bd`, 15 mã; `close` liên tục qua ngày GDKHQ (phụ lục mag, mục "Giá dùng cho nhãn") | Ngày GDKHQ không làm nhiễu nhãn; cờ GDKHQ trong cửa sổ đo biến động thật, không đo bước nhảy kỹ thuật. |
| `FEATURES` hiện có | Đã có `std_10/20/60`, `h_l`, `o_pc`, nhưng ở mức tuyệt đối, không trừ mức chung của phiên | HAR "tương đối trong phiên" là đặc trưng mới thật, không trùng `logit_price`. |

## Mục tiêu

1. Đăng ký và đo một baseline HAR tương đối trên cùng nhãn `mag`, cùng hàng, cùng fold với R7/R8.
2. Đo phần đóng góp thêm của lịch sự kiện doanh nghiệp so với baseline đó (ablation lồng nhau).
3. Bắt đầu lưu ảnh chụp lịch sự kiện mỗi ngày, để mọi kiểm chứng tiến sau này là point-in-time thật (không phụ thuộc dữ liệu nhà cung cấp đã bị sửa lùi).

## Không làm (non-goals)

- Dòng tiền khối ngoại/tự doanh, sổ lệnh, phái sinh VN30, vĩ mô, beta × chế độ thị trường: chưa đủ lịch sử hoặc chỉ ở mức thị trường (cùng giá trị cho 15 mã, không giúp xếp hạng trong nhãn tương đối). Đặc trưng bối cảnh của price-v2 đã làm log-loss xấu đi.
- Lịch công bố BCTC quý: hạn theo quy định gần như giống nhau giữa các mã (mức thị trường); ngày công bố thật của từng mã trong quá khứ không có ở dạng point-in-time.
- Mạng sâu mới hay fusion head học được. Câu hỏi là "nguồn có mang thông tin không", trả lời gọn nhất bằng hai mô hình tuyến tính lồng nhau.
- Không đổi nhãn, fold, `FEATURES`, `config.ARMS`, `config.BASELINES`, `mmf_lock.json`; không chạy holdout; không đọc vùng sau 2026-10-30 để phát triển.

## Điều không được xảy ra (anti-goals)

- Rò rỉ thời gian qua lịch sự kiện: dùng sự kiện chưa công bố tại thời điểm dự báo, hoặc dùng ngày đã bị nhà cung cấp sửa lùi mà không ghi rõ.
- Thử thêm biến thể đặc trưng/cửa sổ sự kiện sau khi xem kết quả để "cứu" R9a.
- Trình bày kết quả trên fold A–Fb như xác nhận ngoài mẫu. Cả sáu fold đã bị nhìn bởi R7/R8.

## Thiết kế

### Kiến trúc (quyết định của trợ lý)

```text
OHLCV ──→ 7 đặc trưng HAR, trừ trung bình 15 mã cùng phiên ──┐
                                                             ├──→ hồi quy logistic L2 (C=0.1) → P(Mạnh)
Lịch sự kiện VCI (point-in-time) ──→ 4 đặc trưng sự kiện ────┘
```

"Fusion" ở đây là hồi quy logistic trên các nhánh ghép lại. Hai arm lồng nhau: `logit_har` (chỉ HAR) và `logit_har_ev` (HAR + sự kiện). Khi nhánh sự kiện không mang thông tin, hệ số của nó co về 0 và mô hình quay về baseline, đúng ý "giữ nhánh biến động làm nền". Không có tham số nào được dò.

Đường chạy: dùng lại `experiment.fold_bundle(panel, h, fold, label="mag")` để có **đúng các hàng** của R7/R8 (cùng split, purge, cửa sổ đủ dài), dùng lại `baselines._model("logit")`, `E.fit_alpha` trên calib và `experiment._write_preds` như mọi baseline khác. Chấm bằng `mag.two_class` và `compare.compare`. Mã mới nằm trong:
- `ai_core/prices/events.py`: tải và chuẩn hóa lịch sự kiện VCI, lưu ảnh chụp theo ngày (lớp dữ liệu, cạnh `sync.py`).
- `ai_core/models/mmf/har.py`: đặc trưng HAR, đặc trưng sự kiện, chạy hai arm, chấm R9.
- `ai_core/paths.py`: thêm `EVENTS = PRICES / "events"`.

### Đặc trưng HAR (7, cố định)

Với mỗi mã, phiên t, từ OHLCV đến hết phiên t: `r = ln(close_t / close_{t-1})`, `pk = ln(high/low)² / (4 ln 2)` (Parkinson).
- `har_rv_d = ln(r_t² + ε)`, `har_rv_w = ln(mean_5(r²) + ε)`, `har_rv_m = ln(mean_22(r²) + ε)`
- `har_pk_d`, `har_pk_w`, `har_pk_m`: như trên với `pk`
- `har_gap_w = ln(mean_5(o_pc²) + ε)`, `o_pc = ln(open_t / close_{t-1})`

`ε = 1e-8`. `min_periods` bằng độ dài cửa sổ. Sau đó **trừ trung bình cắt ngang** của 15 mã cùng phiên (nhãn là tương đối, nên chỉ thứ hạng trong phiên có ích). Chuẩn hóa: mean/std trên khối train, clip ±5, NaN → 0 (như `logit_price`).

### Đặc trưng sự kiện (4, cố định)

Một sự kiện được biết tại phiên t nếu `public_date < t` (chặt: thông báo ngày t có thể ra sau giờ đóng cửa). Loại dùng: `DIV, ISS, AGME, AGMR, EGME`. Ngày sự kiện: `exright_date` cho `DIV, ISS`; ngày họp cho `AGME, EGME` (dự kiến là `display_date1`, phải xác minh ở bước kiểm toán dữ liệu); `AGMR` chỉ tính là một thông báo.
- `ev_exr_in_h`: 1 nếu có ngày GDKHQ đã biết rơi vào phiên t+1..t+h.
- `ev_mtg_in_h`: 1 nếu có ngày họp đã biết rơi vào t+1..t+h.
- `ev_next`: số phiên giao dịch đến sự kiện đã biết gần nhất sau t (GDKHQ hoặc họp), cắt ở 20, chia 20; không có thì 1.
- `ev_new5`: số thông báo có `public_date` trong 5 phiên t−5..t−1.

Đếm phiên theo `panel.calendar`. Không trừ trung bình cắt ngang (đã khác nhau giữa các mã).

### Lưu dữ liệu sự kiện

- `python -m ai_core.prices.events` tải 5 loại cho 15 mã từ 2020-01-01 (phân trang, nghỉ 3.5 s giữa các lệnh vì giới hạn 20 lệnh/phút), ghi `data/prices/events/<YYYY-MM-DD>.parquet` (ngày tải). Cột: `symbol, event_code, public_date, record_date, exright_date, display_date1, event_id`.
- Lần chạy R9 dùng **một file cố định**, ghi tên file và sha256 vào báo cáo.
- Ảnh chụp hằng ngày (cron 15:50 ngày thường, chủ dự án thêm vào crontab) bắt đầu ngay, để có lịch sự kiện point-in-time thật cho kiểm chứng tiến.

## Phụ lục đăng ký trước (nháp, chủ dự án xác nhận rồi mới ghi vào cuối `mmf_preregistration.md` và mới chạy)

**Phụ lục 2026-11-1x (R9: baseline HAR và lịch sự kiện cho nhãn mag; soạn trước khi chạy bất kỳ arm nào dưới đây).**

- Dữ liệu: snapshot `24251acc1a5695bd`, nhãn `mag-median-v1`, fold A, B, C (`DATE_FOLDS`) và D, E, Fb (`BACK_FOLDS`): 6 fold, eval 2024-10-22 đến 2026-10-30. Hàng giống hệt R7/R8. Lịch sự kiện: file cố định ghi trong báo cáo.
- Arm mới: `logit_har`, `logit_har_ev`, ghi vào `data/mmf/exp/har/<dataset_id>`. Mốc `prior_fixed`, `prior_adaptive`, `logit_price`, `dl_price`, `fusion_concat` đọc từ run mag `data/mmf/exp/mag/24251acc1a5695bd` (không chạy lại); bộ chấm kiểm khóa (mã, phiên) của eval hai run trùng nhau ở từng fold, lệch thì dừng. Đặc trưng và mô hình như mục Thiết kế. Không thêm arm sau khi xem kết quả.
- Phép đo: chênh lệch log-loss ghép cặp trên eval, xác suất chuẩn hóa hai lớp, 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần, seed 0).
- Tiêu chí chính (theo từng h, đúng 4 kiểm định):
  - **R9a, đóng góp của lịch sự kiện:** `logit_har_ev − logit_har`.
  - **R9b, baseline HAR:** `logit_har − prior_adaptive`.
  - "Có bằng chứng": tốt hơn ở ít nhất 4/6 fold và cận trên CI gộp < 0. "Có tín hiệu, bằng chứng yếu": mean < 0 nhưng CI chứa 0, hoặc CI đạt nhưng dưới 4/6 fold. "Chưa chứng minh được": còn lại.
- Báo cáo kèm, không dùng để quyết định: `logit_har − prior_fixed`, `logit_har_ev − prior_fixed`; `logit_har` và `logit_har_ev` so với `dl_price`, `fusion_concat` của run mag (cùng hàng; chỉ các fold mà run đó có); kết quả từng fold; AUC, Brier; tỷ lệ hàng có `ev_exr_in_h` hoặc `ev_mtg_in_h` = 1; hệ số của 4 đặc trưng sự kiện theo fold; **phép giả dược**: 20 hoán vị ngẫu nhiên không điểm cố định của bảng sự kiện giữa các mã (giữ nguyên mùa vụ, phá liên kết mã), báo cáo R9a thật so với phân phối giả dược.
- Kiểm tra rò rỉ: unit test chứng minh sự kiện có `public_date ≥ t` không ảnh hưởng đặc trưng tại t; đối chiếu ít nhất 10 sự kiện (5 mã) giữa `public_date` của VCI và ngày bài đầu tiên trong `nlp.db` hoặc trang VSDC/HOSE, ghi kết quả vào báo cáo **trước khi** chạy arm. Nếu `public_date` của VCI sớm hơn ngày công bố thật ở bất kỳ ca nào, dừng và báo cáo. Ngưỡng AUC nhãn thật ≥ 0.75 của phụ lục mag giữ nguyên.
- Quy tắc dừng: chạy đúng một lần. Không thử đặc trưng, cửa sổ, loại sự kiện hay C khác sau khi xem kết quả. Nếu R9a là "Chưa chứng minh được" ở cả hai h, ghi kết luận: lịch sự kiện (VCI, 5 loại) không thêm thông tin đo được ngoài lịch sử biến động cho 15 mã. Nếu R9a "Có bằng chứng" ở ít nhất một h, việc khóa `logit_har_ev` cho kiểm chứng tiến là quyết định của chủ dự án (file khóa riêng `docs/reports/har_ev_lock.json`), chấm một lần sau ít nhất 40 phiên đã trưởng thành **sau ngày khóa**, dùng ảnh chụp sự kiện hằng ngày. `mmf_lock.json` và holdout mmf không đổi.
- Giới hạn biết trước: sáu fold đã bị nhìn bởi R7/R8 trên cùng nhãn, nên R9 là vùng phát triển; volcast đã dùng biên độ ngày (`range`) cho cùng mục tiêu. Lịch sự kiện là dữ liệu nhà cung cấp tải ngày 2026-11-1x: sự kiện bị hủy có thể đã biến mất, ngày có thể đã sửa (thiên lệch nhìn trước không kiểm soát hết được bằng `public_date`). Sự kiện thưa (ước đoán của trợ lý, chưa đo: khoảng 4% hàng ở T+2 và 9% ở T+5 có cờ trong cửa sổ), nên R9a có thể thiếu lực. Bốn kiểm định chính chưa hiệu chỉnh đa kiểm định; 15 mã biến động cùng nhau.

## Thời gian

Hạn đồ án 2026-11-15. Phần dữ liệu, code và lượt chạy R9 trên CPU vừa trong hạn (hồi quy logistic, vài phút). Kiểm chứng tiến chỉ có sau hạn (40 phiên sau ngày khóa, khoảng cuối tháng 1/2027), nên **mọi kết quả R9 trong đồ án là vùng phát triển** và phải ghi đúng như vậy.

## Các quyết định trợ lý tự đặt (chủ dự án kiểm lại)

1. Fusion bằng logistic lồng nhau thay vì fusion head mạng nơ-ron.
2. Bảy đặc trưng HAR, trừ trung bình cắt ngang; bốn đặc trưng sự kiện; `public_date < t` chặt.
3. Gộp 6 fold (A–Fb) với ngưỡng 4/6, thay vì tách R7/R8.
4. R9b so với `prior_adaptive` (không phải `prior_fixed`), theo gợi ý của đề xuất rằng `prior_adaptive` là mốc khó hơn.
5. Bỏ dòng tiền, sổ lệnh, phái sinh, vĩ mô, lịch BCTC khỏi nghiên cứu này (lý do ở bảng kiểm tra).

## Việc để người làm tự quyết

Tên hàm, cách phân trang VCI, định dạng bảng báo cáo, vị trí test; miễn giữ đúng định nghĩa đặc trưng, hàng, phép đo ở trên.
