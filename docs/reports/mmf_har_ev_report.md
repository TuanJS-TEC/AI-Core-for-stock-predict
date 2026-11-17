# Báo cáo R9: baseline HAR và lịch sự kiện doanh nghiệp cho nhãn mag (T+2, T+5)

Ngày chạy: 2026-11-12. Luật: phụ lục R9 và phần sửa đổi trong `docs/reports/mmf_preregistration.md`, cả hai ghi trước khi chạy. Chạy đúng một lần: 24 file arm, 240 file giả dược, một lượt chấm.

**Đây là kết quả vùng phát triển.** Cả sáu fold (A, B, C, D, E, Fb; eval từ 2024-10-22 đến 2026-10-30) đã được R7/R8 dùng trên cùng nhãn. Vùng sau 2026-10-30 không được đọc, `mmf_lock.json` không đổi, holdout không được chạy.

- Dataset: snapshot `24251acc1a5695bd`, nhãn `mag-median-v1`, hàng giống hệt R7/R8 (bộ chạy đối chiếu từng (h, fold) với run mag trước khi fit).
- Lịch sự kiện: `data/prices/events/2026-11-12.parquet`, sha256 `6f7a7f21…5863b`. Kiểm toán ở `docs/reports/mag_events_audit.md`.
- Run: `data/mmf/exp/har/24251acc1a5695bd`. Bảng đầy đủ: `data/mmf/exp/har/r9.md`. Code: `ai_core/models/mmf/har.py`, `ai_core/prices/events.py`.

## Kết luận

1. **Lịch sự kiện không thêm thông tin đo được ngoài HAR (R9a).** Kết quả là "Có tín hiệu, bằng chứng yếu" ở cả hai h: trung bình có lợi, nhưng rất nhỏ (−0.0004 và −0.0001 log-loss), CI chứa 0, thắng 4/6 fold. Theo luật, không khóa cấu hình nào cho kiểm chứng tiến và không thử biến thể nào.
2. **Baseline HAR tương đối là mốc mạnh nhất từ trước tới nay cho nhãn mag (R9b).** `logit_har` hơn `prior_adaptive` ("Có bằng chứng" ở cả hai h, 5/6 fold). Nó cũng hơn `prior_fixed` ở cả 6 fold (T+2) và 5/6 fold (T+5).
3. **Mô hình tuyến tính 7 đặc trưng ngang hoặc hơn mạng sâu (báo cáo kèm).** Log-loss của `logit_har` thấp hơn `dl_price` (−0.0034 ở T+2, −0.0060 ở T+5), và `logit_har_ev` thấp hơn `fusion_concat` (−0.0039, −0.0050). Trên fold lùi D, E, Fb, đúng chỗ R8a không xác nhận được fusion, HAR thắng `prior_fixed` ở 5/6 ô. Đây là so sánh thăm dò, không phải kiểm định chính.

Cách đọc gộp: tín hiệu độ lớn trong rổ 15 mã gần như hoàn toàn là **thứ hạng biến động gần đây giữa các mã trong cùng phiên**. Các bước của HAR là đúng hướng: trừ mức chung của phiên, dùng biên độ trong phiên, dùng ba thang 1, 5, 22 phiên. Mạng sâu, tin tức và lịch sự kiện đều không thêm được gì đáng tin trên phần đó.

## Kiểm tra trước khi đọc

- Ngưỡng AUC nhãn thật ≥ 0.75: không arm nào chạm (cao nhất 0.666).
- Cổng rò rỉ của lịch sự kiện: qua. VCI ghi ngày công bố muộn hơn tin, trung vị 4 ngày (`mag_events_audit.md`).
- Review code trước khi chạy (một reviewer độc lập) tìm ra 1 lỗi crash và 3 vấn đề, đều đã sửa trước lượt chạy. Một vấn đề làm đổi một câu của phụ lục (bỏ clip ±5 cho đặc trưng sự kiện) và đã được ghi thành sửa đổi trước khi chạy.

## R9: kiểm định chính

Chênh lệch log-loss ghép cặp trên eval, 6 fold, block bootstrap theo ngày (khối 20 phiên, 2000 lần). Âm là vế trái tốt hơn.

| h | so sánh | mean | lo | hi | fold thắng | kết luận |
|--:|:--|--:|--:|--:|:--:|:--|
| 2 | R9a `logit_har_ev − logit_har` | −0.0004 | −0.0013 | +0.0004 | 4/6 | Có tín hiệu, bằng chứng yếu |
| 2 | R9b `logit_har − prior_adaptive` | −0.0085 | −0.0157 | −0.0010 | 5/6 | Có bằng chứng |
| 5 | R9a `logit_har_ev − logit_har` | −0.0001 | −0.0006 | +0.0003 | 4/6 | Có tín hiệu, bằng chứng yếu |
| 5 | R9b `logit_har − prior_adaptive` | −0.0164 | −0.0273 | −0.0066 | 5/6 | Có bằng chứng |

n = 5490 hàng, 366 phiên (T+2); 5220 hàng, 348 phiên (T+5).

R9a theo từng fold:

| h | A | B | C | D | E | Fb |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | −0.0019 | −0.0003 | +0.0008 | +0.0000 | −0.0008 | −0.0003 |
| 5 | −0.0010 | −0.0003 | −0.0006 | +0.0003 | −0.0000 | +0.0008 |

R9b theo từng fold:

| h | A | B | C | D | E | Fb |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | −0.0107 | +0.0054 | −0.0002 | −0.0042 | −0.0144 | −0.0268 |
| 5 | −0.0169 | −0.0231 | +0.0005 | −0.0072 | −0.0149 | −0.0371 |

## Báo cáo kèm (không dùng để quyết định)

| h | so sánh | mean | lo | hi | fold thắng |
|--:|:--|--:|--:|--:|:--:|
| 2 | `logit_har − prior_fixed` | −0.0157 | −0.0262 | −0.0061 | 6/6 |
| 2 | `logit_har − dl_price` | −0.0034 | −0.0092 | +0.0023 | 3/6 |
| 2 | `logit_har_ev − fusion_concat` | −0.0039 | −0.0096 | +0.0017 | 3/6 |
| 5 | `logit_har − prior_fixed` | −0.0150 | −0.0282 | −0.0037 | 5/6 |
| 5 | `logit_har − dl_price` | −0.0060 | −0.0129 | +0.0002 | 5/6 |
| 5 | `logit_har_ev − dl_price` | −0.0062 | −0.0129 | −0.0000 | 5/6 |
| 5 | `logit_har_ev − fusion_concat` | −0.0050 | −0.0128 | +0.0025 | 5/6 |

- AUC Mạnh/Yếu trên eval: `logit_har` 0.58–0.66, `dl_price` 0.58–0.65, `prior_fixed` 0.54–0.65.
- Ở fold D (giai đoạn `prior_fixed` thắng mọi mô hình trong R8), `logit_har` vượt `prior_fixed` ở T+2 (0.6622 so với 0.6688) và ngang ở T+5 (0.6569 so với 0.6568). Ở fold E và Fb, HAR thắng ở cả hai h.
- **Giả dược.** Bảng sự kiện được trao cho mã khác (20 hoán vị không điểm cố định). Ở T+2, R9a thật (−0.0004) tốt hơn cả 20 giả dược (phân vị 5–95%: −0.0001 đến +0.0007). Ở T+5, 2/20 giả dược tốt bằng hoặc hơn R9a thật. Lịch sự kiện của đúng mã có mang một chút thông tin ở T+2, nhưng quá nhỏ để vượt nhiễu của phép đo chính.
- **Hệ số (lớp Mạnh, đặc trưng đã chuẩn hóa).** `ev_exr_in_h` dương ở 12/12 ô (0.023–0.049): có ngày GDKHQ trong cửa sổ thì nhiều khả năng biến động mạnh hơn. `ev_new5` âm ở 12/12 ô (−0.009 đến −0.029). `ev_mtg_in_h` dương ở T+2 (6/6) và lẫn dấu ở T+5. Dấu ổn định, nhưng độ lớn nhỏ vì chỉ 1.8–4.0% hàng có cờ.
- Mười kiểm định "kèm" chưa hiệu chỉnh đa kiểm định. Dòng `logit_har_ev − dl_price` ở T+5 đạt "Có bằng chứng" với cận trên CI làm tròn về −0.0000, sát biên.

## Trả lời câu hỏi của đề xuất "nguồn dữ liệu ngoài tin"

| Câu hỏi | Trả lời |
|:--|:--|
| Baseline đúng cho nhãn mag là gì? | HAR tương đối (7 đặc trưng, trừ trung bình phiên), không phải `prior_fixed`. Nó hơn cả `prior_adaptive` (R9b). |
| Lịch sự kiện (biết trước, khác nhau giữa các mã) có thêm thông tin không? | Không đo được ở mức đã đăng ký (R9a yếu). Có dấu hiệu nhỏ, dấu ổn định ở T+2 (giả dược, hệ số), nhưng sự kiện quá thưa. |
| Dòng tiền khối ngoại, sổ lệnh? | Chưa kiểm được: không có lịch sử (Finhay từ 2026-07-28, không có bid/ask). Chỉ thu thập tiến. |
| Mạng sâu có cần không? | Trên nhãn mag, mô hình tuyến tính HAR ngang hoặc hơn `dl_price`/`fusion_concat` (báo cáo kèm). |

## Giới hạn

- Sáu fold đã bị nhìn bởi R7/R8; volcast đã dùng biên độ ngày cho cùng mục tiêu. R9 là vùng phát triển, chưa có phép đo tiến.
- HAR được chọn theo tài liệu (Corsi 2009) và đề xuất, trước khi chạy, không dò trên các fold này. Tuy vậy, việc "biến động gần đây là tín hiệu chính" đã được biết từ R7.
- Lịch sự kiện là dữ liệu nhà cung cấp tải ngày 2026-11-12: sự kiện bị hủy có thể đã biến mất, ngày có thể đã sửa. `public_date` của VCI thường muộn hơn tin vài ngày, nên đặc trưng thấy thông tin muộn (thận trọng).
- Sự kiện thưa: 1.8% (T+2) và 4.0% (T+5) hàng có cờ trong cửa sổ. R9a thiếu lực; "bằng chứng yếu" không đồng nghĩa với "không có tác động".
- Bốn kiểm định chính và mười kiểm định kèm chưa hiệu chỉnh đa kiểm định; 15 mã biến động cùng nhau.
- Theo quy tắc dừng: không thử thêm đặc trưng, cửa sổ, loại sự kiện hay C nào sau khi xem kết quả.

## Việc tiếp theo có thể làm (quyết định của chủ dự án, ngoài R9)

- Nếu muốn một mô hình mag cho kiểm chứng tiến, ứng viên hợp lý nhất theo dữ liệu này là `logit_har`. Việc khóa nó cần một phụ lục riêng, và phải chấm sau ít nhất 40 phiên đã trưởng thành kể từ ngày khóa.
- Ảnh chụp lịch sự kiện hằng ngày (cron 15:50) giữ dữ liệu sự kiện point-in-time cho mọi kiểm chứng tiến sau này.
