# Báo cáo: nhãn độ lớn biến động "mag" cho mmf (T+2, T+5), R7 và R8

Ngày chạy: 2026-11-12. Luật: hai phụ lục 2026-11-12 (và phần sửa đổi) của `docs/reports/mmf_preregistration.md`, chủ dự án duyệt trước khi chạy. Mỗi phép đo chạy đúng một lần.

**Không có số nào ở đây là kết quả chấm cuối.** R7 là vùng phát triển (eval của fold A, B, C, đã bị nhìn nhiều lần). R8 là xác nhận ngoài thời gian theo chiều lùi (eval của fold D, E, Fb, 2024-10-22 đến 2025-07-25), yếu hơn một holdout tiến vì các lý do ghi trong phụ lục. Vùng sau 2026-10-30 không được đọc; `mmf_lock.json` không đổi; holdout không được chạy.

Dataset: snapshot `24251acc1a5695bd` (15 mã). Nhãn: `mag-median-v1` (Mạnh nếu |lợi suất log h phiên| lớn hơn trung vị 15 mã cùng phiên; 7/15 mã Mạnh mỗi phiên, tỷ lệ nền 46.7%). Run: `data/mmf/exp/mag/24251acc1a5695bd`. Bảng đầy đủ: `data/mmf/exp/mag/r7.md`, `data/mmf/exp/mag/r8.md`. Chấm bằng `python -m ai_core.models.mmf.mag`. Xác suất mạng sâu là trung bình 5 seed, chuẩn hóa lại trên hai lớp trước khi chấm. Fold F của phụ lục R8 mang tên `Fb` trong code và file (tên `F` đã dùng cho fold cuối của v1); ngày tháng đúng như đã đăng ký.

## Kết luận

1. **Trên vùng phát triển (R7), mô hình có tín hiệu và mạng sâu hơn baseline cổ điển, ở cả hai h.** R7a và R7b đều "Có bằng chứng".
2. **Trên fold lùi (R8), mô hình fusion không được xác nhận là hơn tần suất lớp của từng mã.** R8a là "Có tín hiệu, bằng chứng yếu" ở cả hai h (CI chứa 0). Theo tiêu chí đã đăng ký, **mô hình fusion chưa đạt mức "dùng được cho dự báo độ lớn"** ở h nào.
3. **Fusion hơn hồi quy logistic trên giá ở cả hai vùng** (R8b "Có bằng chứng" ở cả hai h), nhưng trên fold lùi chính `logit_price` lại kém `prior_fixed`, nên R8b không nói rằng fusion có ích so với mốc đơn giản nhất.
4. **Tin không đóng góp đo được.** R7c "Chưa chứng minh được" ở cả hai h; R8c yếu ở T+2 và "Chưa chứng minh được" ở T+5. Chênh lệch nằm trong ±0.002 log-loss và đổi dấu giữa các fold. Khớp với R3 và R6.

Cách đọc gộp: tín hiệu độ lớn có thật nhưng phần lớn là thứ đơn giản và bền (mã nào vốn biến động nhiều, và biến động gần đây cao thì sắp tới cao). Mạng sâu khai thác nó tốt hơn logit/GBM trên dữ liệu 2025-2026, nhưng trên 2024-10 đến 2025-07 thì tần suất theo mã đã đủ, mạng sâu không thêm được gì đáng tin.

## Kiểm tra rò rỉ (đọc trước R7)

`dl_price`, một seed, encoder lstm, 20 phép hoán vị nhãn train giữa các ngày. File: `data/mmf/exp/mag_shuffle/leak_check.md`. Mốc so sánh theo luật là AUC của `prior_fixed` cùng (h, fold).

| h | fold | AUC nhãn thật | AUC xáo trộn: trung bình | max | AUC `prior_fixed` | trung bình − mốc | max − mốc |
|--:|:-----|------:|------:|------:|------:|------:|------:|
| 2 | A | 0.615 | 0.582 | 0.617 | 0.546 | 0.036 | 0.071 |
| 2 | B | 0.620 | 0.607 | 0.619 | 0.566 | 0.041 | 0.053 |
| 2 | C | 0.569 | 0.565 | 0.576 | 0.545 | 0.021 | 0.032 |
| 5 | A | 0.622 | 0.616 | 0.627 | 0.590 | 0.026 | 0.037 |
| 5 | B | 0.612 | 0.585 | 0.600 | 0.541 | 0.044 | 0.059 |
| 5 | C | 0.617 | 0.630 | 0.646 | 0.594 | 0.036 | 0.052 |

- Luật đăng ký (fold C; cảnh báo nếu trung bình vượt mốc quá 0.05 hoặc có hoán vị vượt quá 0.08): **không cảnh báo** ở cả hai h. Năm ô còn lại cũng dưới cả hai ngưỡng.
- Ngưỡng AUC nhãn thật ≥ 0.75: không arm nào chạm (cao nhất 0.662).
- Báo cáo kèm, không phải phép thử rò rỉ: AUC nhãn thật chỉ vượt max của 20 hoán vị ở 2/6 ô (T+2 B, T+5 B). Mạng huấn luyện trên nhãn xáo ngày vẫn học được "đặc trưng biến động cao thì Mạnh" từ khác biệt giữa các mã, và quy luật đó áp dụng được theo thời gian. Vì vậy phép xáo này không tách được kỹ năng dự báo theo thời điểm; nó chỉ cho thấy phần tín hiệu vượt lên trên quy luật đơn giản đó là nhỏ.

## R7: vùng phát triển, fold A, B, C

Chênh lệch log-loss ghép cặp trên eval, block bootstrap theo ngày (khối 20 phiên, 2000 lần). Âm là vế trái tốt hơn.

| h | so sánh | vế trái (theo fold) | vế phải | mean | lo | hi | fold thắng | kết luận |
|--:|:--|:--|:--|--:|--:|--:|:--:|:--|
| 2 | R7a ứng viên − prior_fixed | A=dl_price B=gbm_all C=no_text | prior_fixed | −0.0160 | −0.0306 | −0.0029 | 3/3 | Có bằng chứng |
| 2 | R7b mạng sâu − baseline | A=dl_price B=dl_price C=no_text | A=logit_price B=gbm_all C=gbm_price | −0.0071 | −0.0134 | −0.0020 | 2/3 | Có bằng chứng |
| 2 | R7c fusion_concat − no_text | fusion_concat | no_text | +0.0006 | −0.0010 | +0.0020 | 1/3 | Chưa chứng minh được |
| 5 | R7a ứng viên − prior_fixed | dl_price (cả 3 fold) | prior_fixed | −0.0196 | −0.0355 | −0.0053 | 3/3 | Có bằng chứng |
| 5 | R7b mạng sâu − baseline | dl_price (cả 3 fold) | A=gbm_price B=logit_all C=gbm_price | −0.0101 | −0.0165 | −0.0041 | 3/3 | Có bằng chứng |
| 5 | R7c fusion_concat − no_text | fusion_concat | no_text | +0.0002 | −0.0018 | +0.0026 | 2/3 | Chưa chứng minh được |

n = 2745 hàng, 183 phiên (T+2); 2610 hàng, 174 phiên (T+5). 2745 = 15 mã × 183 phiên.

Báo cáo kèm, không dùng để quyết định:
- `fusion_attention − no_text`: +0.0017 (T+2), +0.0012 (T+5), CI chứa 0, thắng 1/3 và 0/3 fold. Cơ chế attention không khai thác tin tốt hơn concat.
- `dl_text − prior_fixed`: −0.0057 (CI −0.0095 đến −0.0022, 3/3 fold) ở T+2; −0.0067 (CI chứa 0) ở T+5. Tin đứng một mình hơn tần suất theo mã một chút, nhưng kém xa nhánh giá (AUC 0.54–0.62 so với 0.58–0.65) và không thêm gì khi đã có giá.
- Ứng viên được chọn (log-loss calib thấp nhất) là `dl_price` ở 4/6 ô; mô hình fusion không được chọn ở ô nào.
- AUC Mạnh/Yếu trên eval: `dl_price` 0.58–0.65, `fusion_concat` 0.58–0.65, `logit_price` 0.57–0.62, `prior_fixed` 0.54–0.59.
- `prior_adaptive` (tần suất 120 nhãn gần nhất của mã, không là ứng viên) gần ngang mạng sâu ở một số ô (T+2 B: 0.6692 so với `dl_price` 0.6677; T+5 C: 0.6648 so với 0.6659). Điều này ủng hộ cách đọc "tín hiệu chủ yếu là tính bền của biến động".

## R8: xác nhận ngoài thời gian, fold D, E, Fb

Arm cố định trước, không chọn. Cùng phép đo.

| h | so sánh | mean | lo | hi | fold thắng | kết luận |
|--:|:--|--:|--:|--:|:--:|:--|
| 2 | R8a fusion_concat − prior_fixed | −0.0056 | −0.0124 | +0.0021 | 2/3 | Có tín hiệu, bằng chứng yếu |
| 2 | R8b fusion_concat − logit_price | −0.0087 | −0.0131 | −0.0045 | 3/3 | Có bằng chứng |
| 2 | R8c fusion_concat − no_text | −0.0005 | −0.0027 | +0.0016 | 2/3 | Có tín hiệu, bằng chứng yếu |
| 5 | R8a fusion_concat − prior_fixed | −0.0002 | −0.0089 | +0.0084 | 1/3 | Có tín hiệu, bằng chứng yếu |
| 5 | R8b fusion_concat − logit_price | −0.0094 | −0.0169 | −0.0007 | 3/3 | Có bằng chứng |
| 5 | R8c fusion_concat − no_text | +0.0011 | −0.0013 | +0.0037 | 1/3 | Chưa chứng minh được |

R8a theo từng fold (fusion_concat − prior_fixed):

| h | D | E | Fb |
|--:|--:|--:|--:|
| 2 | +0.0042 | −0.0080 | −0.0130 |
| 5 | +0.0076 | +0.0034 | −0.0117 |

- Ở fold D (eval 2024-10-22 đến 2025-01-17) `prior_fixed` thắng mọi mô hình ở cả hai h: AUC của nó là 0.620 (T+2) và 0.649 (T+5), cao hơn hẳn các fold khác. Trong giai đoạn đó thứ hạng biến động giữa các mã gần như đứng yên, nên tần suất theo mã đã là dự báo tốt nhất.
- Ở fold D và E các mô hình dự đoán Mạnh quá ít (tỷ lệ dự đoán Mạnh 6–33% so với thực tế 46.7%): mức biến động chung của khối train (2022–2024) khác khối eval, và xác suất bị lệch mức. AUC của `fusion_concat` ở fold D vẫn là 0.632 và 0.652, tức là thứ hạng còn tốt nhưng xác suất thì lệch. Đây là quan sát sau khi chạy, không phải phép đo đã đăng ký; không có phép sửa nào được thử.
- R8b đạt, nhưng vì `logit_price` kém `prior_fixed` ở D và E nên R8b chỉ nói fusion tốt hơn một mô hình tuyến tính trên cùng đặc trưng.

## Trả lời các câu hỏi của đồ án

| Câu hỏi | Trả lời theo luật đã đăng ký |
|:--|:--|
| Hướng giá T+2, T+5 có dự báo được không? | Không có bằng chứng (R1–R5, các báo cáo trước). |
| Độ lớn biến động có dự báo được không? | Có trên vùng phát triển (R7a). Chưa được xác nhận trên fold lùi so với tần suất theo mã (R8a). |
| Mạng sâu có hơn mô hình cổ điển không? | Có trên vùng phát triển (R7b) và so với logit trên fold lùi (R8b). |
| Hợp nhất đa phương thức có lợi không? | Không đo được: `fusion_concat` ngang `no_text` và `dl_price`. |
| Tin tức có đóng góp không? | Không đo được ở cả hướng lẫn độ lớn (R3, R6, R7c, R8c). |

## Giới hạn

- Fold A, B, C đã bị nhìn nhiều lần; volcast đã đo cùng mục tiêu trên cùng giai đoạn. R7 không phải phát hiện mù.
- R8 là chiều lùi: thiết kế mmf được hình thành khi nhìn dữ liệu sau các khối này, và volcast đã chấm một phần giai đoạn này.
- Mười hai kiểm định chính (6 của R7, 6 của R8) chưa hiệu chỉnh đa kiểm định. 15 mã biến động cùng nhau; mỗi fold 58–61 phiên.
- Nhãn mag là tương đối trong rổ 15 mã: nó nói mã nào biến động mạnh hơn các mã khác, không nói mức biến động tuyệt đối, và không nói hướng.
- Chưa có phép đo tiến nào. Một xác nhận thật cần cấu hình mag được khóa và ít nhất 40 phiên đã trưởng thành sau 2026-10-30.
- Theo quy tắc dừng đã đăng ký, không có biến thể nhãn, arm hay siêu tham số thứ hai nào được thử sau khi xem các kết quả này.
