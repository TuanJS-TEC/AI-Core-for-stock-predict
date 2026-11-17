# Báo cáo phát triển: nhãn trung hòa thị trường "xs" cho mmf (T+2, T+5)

Ngày chạy: 2026-11-11. Luật: phụ lục 2026-11-11 của `docs/reports/mmf_preregistration.md`, chủ dự án xác nhận trước khi chạy. Plan: `docs/superpowers/plans/2026-11-11-xs-label.md`.

**Đây là vùng phát triển, chưa phải kết quả cuối.** Khối eval của fold A, B, C đã bị nhìn nhiều lần. Holdout không được chạy, `mmf_lock.json` không đổi.

Dataset: snapshot `24251acc1a5695bd` (15 mã). Nhãn: `xs-k0.43-w60-v1`. Run: `data/mmf/exp/xs/24251acc1a5695bd`. Bảng số: `data/mmf/exp/xs/r5/*.csv`. Xác suất mạng sâu là trung bình 5 seed.

## Kết luận

**R5 là "Có tín hiệu, bằng chứng yếu" ở cả hai h.** Ứng viên xs có log-loss thấp hơn `prior_fixed` cùng nhãn (−0.0042 ở T+2, −0.0051 ở T+5) nhưng CI 95% chứa 0 ở cả hai h.

- Quy tắc dừng không kích hoạt: nó đòi "Chưa chứng minh được" ở cả hai h.
- Điều kiện để khóa xs cho vùng chấm cuối cũng không đạt: không h nào có "Có bằng chứng". Theo khuyến nghị đã đăng ký, không khóa xs.
- Giả thuyết "bỏ thành phần thị trường khỏi nhãn làm lộ tín hiệu riêng từng mã" chưa được xác nhận. AUC Tăng/Giảm của các ứng viên nằm trong 0.46–0.57 (trừ `fusion_concat` T+5 fold B, 0.61, ở ô mà `prior_fixed` đã là 0.60), tức là khả năng xếp hạng mã gần mức ngẫu nhiên.
- Fusion không hơn nhánh giá trên nhãn xs: `fusion_concat` trừ `dl_price` là +0.0004 (T+2) và +0.0013 (T+5), thắng 1/3 và 0/3 fold.

## Kiểm tra rò rỉ (đọc trước R5)

`dl_price`, một seed, encoder lstm, 20 phép hoán vị nhãn train giữa các ngày. File: `data/mmf/exp/xs_shuffle/leak_check.md`.

|   h | fold | AUC nhãn thật | AUC nhãn xáo trộn: trung bình | độ lệch chuẩn | min | max |
|----:|:-----|------:|------:|------:|------:|------:|
| 2 | A | 0.527 | 0.493 | 0.021 | 0.462 | 0.533 |
| 2 | B | 0.477 | 0.474 | 0.022 | 0.445 | 0.527 |
| 2 | C | 0.503 | 0.497 | 0.014 | 0.471 | 0.521 |
| 5 | A | 0.508 | 0.485 | 0.024 | 0.444 | 0.524 |
| 5 | B | 0.454 | 0.445 | 0.026 | 0.393 | 0.507 |
| 5 | C | 0.490 | 0.492 | 0.019 | 0.442 | 0.517 |

- Luật đăng ký (fold C, lệch khỏi 0.5 quá 2 độ lệch chuẩn): không cảnh báo ở cả hai h (lệch 0.003 và 0.008, ngưỡng 0.028 và 0.038).
- Ngoài luật: fold B ở T+5 lệch 0.055, hơn 2 độ lệch chuẩn (0.052), về phía dưới 0.5. Đây là xếp hạng ngược chứ không phải điểm cao bất thường, và AUC nhãn thật ở ô đó cũng dưới 0.5 (0.454). Tôi chưa tìm nguyên nhân.
- Cảnh báo cũ (Macro-F1 ≥ 0.50, AUC ≥ 0.65): không arm nào chạm ngưỡng.

## Ứng viên được chọn (log-loss calib thấp nhất trong 5 arm)

|   h | fold | ứng viên xs | baseline tốt nhất |
|----:|:-----|:------------|:------------------|
| 2 | A | dl_price | logit_price |
| 2 | B | dl_price | logit_price |
| 2 | C | logit_all | logit_all |
| 5 | A | dl_price | logit_price |
| 5 | B | logit_price | logit_price |
| 5 | C | dl_price | gbm_price |

## R5: ứng viên xs trừ `prior_fixed` (tiêu chí chính)

Chênh lệch log-loss ghép cặp trên eval, block bootstrap theo ngày (khối 20 phiên, 2000 lần). Âm là ứng viên tốt hơn.

|   h | mean | lo | hi | fold thắng | n | số phiên | kết luận |
|----:|------:|------:|------:|:----:|----:|----:|:----|
| 2 | −0.0042 | −0.0123 | +0.0039 | 2/3 | 2745 | 183 | Có tín hiệu, bằng chứng yếu |
| 5 | −0.0051 | −0.0131 | +0.0037 | 3/3 | 2610 | 174 | Có tín hiệu, bằng chứng yếu |

Theo từng fold (thăm dò, mỗi fold chỉ khoảng 3 khối bootstrap):

|   h | fold | ứng viên | mean | lo | hi |
|----:|:-----|:---------|------:|------:|------:|
| 2 | A | dl_price | −0.0094 | −0.0272 | +0.0122 |
| 2 | B | dl_price | −0.0078 | −0.0168 | +0.0017 |
| 2 | C | logit_all | +0.0045 | −0.0031 | +0.0112 |
| 5 | A | dl_price | −0.0033 | −0.0231 | +0.0179 |
| 5 | B | logit_price | −0.0060 | −0.0217 | +0.0099 |
| 5 | C | dl_price | −0.0061 | −0.0134 | +0.0014 |

Cả 6 ô đều có CI chứa 0.

## Báo cáo kèm, không dùng để quyết định

### R1, R2 và từng arm so với `prior_fixed`

R1 và R2 dùng `fusion_concat` (fusion duy nhất trong 6 ứng viên). R3 không tính được: các arm bỏ nhánh không nằm trong danh sách đã đăng ký.

| so sánh |   h | mean | lo | hi | fold thắng | kết luận |
|:--------|----:|------:|------:|------:|:----:|:----|
| R1: fusion_concat − baseline tốt nhất | 2 | −0.0010 | −0.0068 | +0.0047 | 2/3 | Có tín hiệu, bằng chứng yếu |
| R2: fusion_concat − dl_price | 2 | +0.0004 | −0.0022 | +0.0029 | 1/3 | Chưa chứng minh được |
| logit_price − prior_fixed | 2 | −0.0063 | −0.0150 | +0.0014 | 3/3 | Có tín hiệu, bằng chứng yếu |
| logit_all − prior_fixed | 2 | −0.0031 | −0.0118 | +0.0051 | 2/3 | Có tín hiệu, bằng chứng yếu |
| gbm_price − prior_fixed | 2 | −0.0025 | −0.0127 | +0.0076 | 2/3 | Có tín hiệu, bằng chứng yếu |
| dl_price − prior_fixed | 2 | −0.0057 | −0.0130 | +0.0020 | 2/3 | Có tín hiệu, bằng chứng yếu |
| fusion_concat − prior_fixed | 2 | −0.0053 | −0.0123 | +0.0016 | 2/3 | Có tín hiệu, bằng chứng yếu |
| R1: fusion_concat − baseline tốt nhất | 5 | +0.0021 | −0.0110 | +0.0158 | 2/3 | Chưa chứng minh được |
| R2: fusion_concat − dl_price | 5 | +0.0013 | −0.0030 | +0.0060 | 0/3 | Chưa chứng minh được |
| logit_price − prior_fixed | 5 | −0.0104 | −0.0254 | +0.0016 | 3/3 | Có tín hiệu, bằng chứng yếu |
| logit_all − prior_fixed | 5 | −0.0060 | −0.0190 | +0.0054 | 3/3 | Có tín hiệu, bằng chứng yếu |
| gbm_price − prior_fixed | 5 | +0.0042 | −0.0149 | +0.0272 | 0/3 | Chưa chứng minh được |
| dl_price − prior_fixed | 5 | −0.0068 | −0.0171 | +0.0040 | 3/3 | Có tín hiệu, bằng chứng yếu |
| fusion_concat − prior_fixed | 5 | −0.0055 | −0.0156 | +0.0055 | 3/3 | Có tín hiệu, bằng chứng yếu |

`logit_price` chạy cố định cho mọi fold có chênh lệch lớn hơn ứng viên được chọn theo calib (−0.0063 và −0.0104), nhưng CI vẫn chứa 0 và đây là so sánh thăm dò, không phải R5.

### Chỉ số trên eval, sau calibration

| h | fold | arm | log-loss | Brier | Macro-F1 | AUC Tăng/Giảm |
|--:|:-----|:----|------:|------:|------:|------:|
| 2 | A | prior_fixed | 1.0819 | 0.6553 | 0.2229 | 0.4861 |
| 2 | A | logit_price | 1.0717 | 0.6485 | 0.3661 | 0.5289 |
| 2 | A | logit_all | 1.0753 | 0.6508 | 0.3643 | 0.5114 |
| 2 | A | gbm_price | 1.0772 | 0.6521 | 0.3782 | 0.5235 |
| 2 | A | dl_price | 1.0725 | 0.6488 | 0.3679 | 0.5353 |
| 2 | A | fusion_concat | 1.0728 | 0.6490 | 0.3687 | 0.5134 |
| 2 | B | prior_fixed | 1.0741 | 0.6504 | 0.2361 | 0.5305 |
| 2 | B | logit_price | 1.0670 | 0.6452 | 0.3533 | 0.5378 |
| 2 | B | logit_all | 1.0670 | 0.6456 | 0.3427 | 0.5475 |
| 2 | B | gbm_price | 1.0695 | 0.6464 | 0.3308 | 0.5281 |
| 2 | B | dl_price | 1.0663 | 0.6449 | 0.3205 | 0.5404 |
| 2 | B | fusion_concat | 1.0658 | 0.6448 | 0.3460 | 0.5427 |
| 2 | C | prior_fixed | 1.0793 | 0.6533 | 0.2201 | 0.4865 |
| 2 | C | logit_price | 1.0776 | 0.6523 | 0.3527 | 0.5069 |
| 2 | C | logit_all | 1.0838 | 0.6559 | 0.3454 | 0.4727 |
| 2 | C | gbm_price | 1.0810 | 0.6547 | 0.3405 | 0.5120 |
| 2 | C | dl_price | 1.0794 | 0.6538 | 0.3119 | 0.5171 |
| 2 | C | fusion_concat | 1.0809 | 0.6545 | 0.3144 | 0.4698 |
| 5 | A | prior_fixed | 1.0619 | 0.6434 | 0.2564 | 0.4846 |
| 5 | A | logit_price | 1.0415 | 0.6290 | 0.3410 | 0.5657 |
| 5 | A | logit_all | 1.0477 | 0.6335 | 0.3322 | 0.5427 |
| 5 | A | gbm_price | 1.0674 | 0.6425 | 0.3571 | 0.5039 |
| 5 | A | dl_price | 1.0587 | 0.6397 | 0.3609 | 0.5363 |
| 5 | A | fusion_concat | 1.0603 | 0.6411 | 0.3707 | 0.5358 |
| 5 | B | prior_fixed | 1.0795 | 0.6549 | 0.2884 | 0.5960 |
| 5 | B | logit_price | 1.0736 | 0.6505 | 0.3292 | 0.5695 |
| 5 | B | logit_all | 1.0777 | 0.6546 | 0.3474 | 0.5765 |
| 5 | B | gbm_price | 1.0830 | 0.6548 | 0.3341 | 0.5441 |
| 5 | B | dl_price | 1.0685 | 0.6460 | 0.3490 | 0.5481 |
| 5 | B | fusion_concat | 1.0702 | 0.6494 | 0.3103 | 0.6135 |
| 5 | C | prior_fixed | 1.0864 | 0.6589 | 0.2623 | 0.4696 |
| 5 | C | logit_price | 1.0817 | 0.6554 | 0.3310 | 0.4860 |
| 5 | C | logit_all | 1.0845 | 0.6574 | 0.3333 | 0.4657 |
| 5 | C | gbm_price | 1.0902 | 0.6611 | 0.3277 | 0.4823 |
| 5 | C | dl_price | 1.0804 | 0.6550 | 0.3430 | 0.4846 |
| 5 | C | fusion_concat | 1.0809 | 0.6548 | 0.3728 | 0.4575 |

### Phân phối lớp thật trên eval và lớp dự đoán của ứng viên được chọn

| h | fold | thật: Giảm / Đi ngang / Tăng | ứng viên | dự đoán: Giảm / Đi ngang / Tăng |
|--:|:-----|:-----------------------------|:---------|:--------------------------------|
| 2 | A | 25.5% / 42.5% / 32.0% | dl_price | 40.9% / 40.0% / 19.1% |
| 2 | B | 25.0% / 43.3% / 31.7% | dl_price | 19.5% / 72.6% / 8.0% |
| 2 | C | 26.1% / 43.7% / 30.2% | logit_all | 29.7% / 43.3% / 27.0% |
| 5 | A | 20.6% / 46.6% / 32.9% | dl_price | 57.8% / 29.2% / 13.0% |
| 5 | B | 24.9% / 41.7% / 33.3% | logit_price | 27.4% / 52.0% / 20.7% |
| 5 | C | 24.5% / 39.4% / 36.1% | dl_price | 10.7% / 64.6% / 24.7% |

Lớp Tăng nhiều hơn lớp Giảm ở mọi ô (30–36% so với 21–26%), dù nhãn đã trừ median. Lý do có thể là median 15 mã bị vài mã lớn kéo, hoặc phân phối phần dư lệch phải; tôi chưa kiểm tra. Phân phối dự đoán lệch xa phân phối thật ở fold A (dự đoán Giảm 41–58% trong khi thật là 21–26%).

### Trước và sau calibration (mạng sâu)

Chênh lệch log-loss eval, sau trừ trước; âm là calibration giúp. Baseline không qua nhiệt độ nên không có ở đây.

| h | fold | arm | nhiệt độ T (trung bình 5 seed) | trước | sau | sau − trước | lo | hi |
|--:|:-----|:----|------:|------:|------:|------:|------:|------:|
| 2 | A | dl_price | 1.84 | 1.0674 | 1.0725 | +0.0051 | −0.0016 | +0.0124 |
| 2 | A | fusion_concat | 1.94 | 1.0680 | 1.0728 | +0.0048 | −0.0023 | +0.0126 |
| 2 | B | dl_price | 1.05 | 1.0669 | 1.0663 | −0.0005 | −0.0016 | +0.0005 |
| 2 | B | fusion_concat | 0.95 | 1.0655 | 1.0658 | +0.0003 | −0.0009 | +0.0014 |
| 2 | C | dl_price | 1.27 | 1.0807 | 1.0794 | −0.0013 | −0.0045 | +0.0010 |
| 2 | C | fusion_concat | 1.31 | 1.0836 | 1.0809 | −0.0027 | −0.0067 | +0.0004 |
| 5 | A | dl_price | 1.73 | 1.0433 | 1.0587 | +0.0154 | +0.0014 | +0.0298 |
| 5 | A | fusion_concat | 1.68 | 1.0464 | 1.0603 | +0.0139 | +0.0027 | +0.0266 |
| 5 | B | dl_price | 0.71 | 1.0651 | 1.0685 | +0.0034 | −0.0084 | +0.0166 |
| 5 | B | fusion_concat | 0.73 | 1.0656 | 1.0702 | +0.0047 | −0.0072 | +0.0190 |
| 5 | C | dl_price | 1.26 | 1.0837 | 1.0804 | −0.0033 | −0.0066 | +0.0005 |
| 5 | C | fusion_concat | 1.46 | 1.0861 | 1.0809 | −0.0052 | −0.0091 | −0.0010 |

Ở fold A, T+5, calibration làm log-loss eval xấu đi rõ (+0.014 đến +0.015, CI trên 0): nhiệt độ học trên calib không chuyển sang eval. Quy trình đã đăng ký không đổi vì số này.

## Giới hạn

- Fold A, B, C đã bị nhìn nhiều lần; R5 ở đây chỉ để quyết định có đi tiếp, không phải kết quả cuối.
- 15 mã, mỗi fold 58–61 phiên, CI rộng. Hai kiểm định R5 chưa hiệu chỉnh đa kiểm định; mọi bảng khác là thăm dò.
- Không so log-loss của nhãn xs với nhãn v1: phân phối lớp khác nhau nên không so được.
- Median 15 mã là proxy thị trường thô.
- Kiểm tra rò rỉ dùng một seed và encoder lstm cho mọi fold, không phải encoder được chọn hay ensemble 5 seed.
- Lượt chạy làm trên cây làm việc có thay đổi chưa commit (commit ghi trong file dự báo là `64ea661`, không phản ánh code nhãn xs).
