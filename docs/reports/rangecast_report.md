# Báo cáo R11: dải giá 80% cho T+2, T+5 (rangecast)

Ngày chạy: 2026-11-13. Luật: phụ lục R11 trong `docs/reports/mmf_preregistration.md`, ghi trước khi chấm. Chạy đúng một lần, một cấu hình.

**Đây là kết quả vùng phát triển.** Sáu fold (A, B, C, D, E, Fb; eval từ 2024-10-22 đến 2026-10-30) đã được R7–R9 dùng cho nhãn độ lớn. Vùng sau 2026-10-30 không được đọc, `mmf_lock.json` không đổi.

- Dataset: `data/prices_2018/snapshots/a0674e0c56bf9013` (15 mã, train từ 2019). Code: `ai_core/models/rangecast/rangecast.py`. Bảng đầy đủ: `data/rangecast/eval.md`.
- Mô hình: hồi quy HAR gộp mã dự báo mức biến động h phiên; dải = hai phân vị 10% và 90% của lợi suất chuẩn hóa, lấy trên khối calib.

## Kết luận

**Theo luật đã ghi: "Dùng được".** Độ phủ gộp nằm trong 77–83% ở cả hai kỳ hạn, và pinball loss thấp hơn mốc `naive20` với khoảng tin cậy loại 0 ở cả hai kỳ hạn.

| h | Độ phủ thực tế (mục tiêu 80%) | Rơi dưới / rơi trên | Độ rộng dải trung bình | Pinball har − naive20 (CI 95%) | Fold thắng |
|--:|--:|:--|--:|:--|:--:|
| 2 | 79.9% | 9.9% / 10.3% | 6.2% giá | −0.0003 (−0.0007 đến −0.0001) | 6/6 |
| 5 | 78.9% | 9.9% / 11.2% | 9.6% giá | −0.0006 (−0.0012 đến −0.0002) | 5/6 |

n = 5490 hàng, 366 phiên (T+2); 5220 hàng, 348 phiên (T+5).

So với hai mốc:

| h | Mô hình | Độ phủ | Độ rộng dải | Pinball |
|--:|:--|--:|--:|--:|
| 2 | har | 79.9% | 6.18% | 0.0101 |
| 2 | naive20 | 80.0% | 6.62% | 0.0104 |
| 2 | hist | 82.9% | 6.38% | 0.0106 |
| 5 | har | 78.9% | 9.59% | 0.0161 |
| 5 | naive20 | 78.5% | 10.37% | 0.0167 |
| 5 | hist | 83.4% | 10.43% | 0.0165 |

HAR cho dải hẹp hơn `naive20` khoảng 7% mà độ phủ tương đương. So với `hist`, HAR hơn rõ ở T+2 (CI loại 0, 5/6 fold); ở T+5 chênh lệch −0.0004 có CI chứa 0 (3/6 fold).

## Những chỗ dải chưa tốt

- **Độ phủ dao động mạnh giữa các fold.** T+2: 69.8% (fold E) đến 88.8% (fold Fb). T+5: 65.6% (fold E) đến 86.8% (fold Fb). Mức 80% chỉ đúng khi lấy trung bình trên nhiều giai đoạn; trong một quý cụ thể, độ phủ thực tế có thể lệch hơn 10 điểm phần trăm. Hai mốc so sánh cũng dao động tương tự.
- **Dải "biến động thấp" hơi hẹp.** Độ phủ ở mức thấp là 76.1% (T+2) và 75.6% (T+5), so với 82% ở mức bình thường và cao. Khi mô hình nói thị trường yên, giá vượt dải thường xuyên hơn mức công bố.
- **Không đều giữa các mã.** T+2: từ 74.9% (VIC) đến 85.8% (VNM). T+5: từ 70.1% (VIC) đến 85.9% (VNM). Với VIC ở T+5, 22.7% số lần giá vượt cận trên.
- **Hai phía không đối xứng ở T+5:** giá vượt cận trên (11.2%) nhiều hơn rơi dưới cận dưới (9.9%).

## Mức rủi ro

| h | Mức | Tỷ lệ số phiên | Độ rộng dải | Độ phủ |
|--:|:--|--:|--:|--:|
| 2 | thấp | 39% | 4.4% | 76.1% |
| 2 | bình thường | 37% | 6.3% | 82.1% |
| 2 | cao | 24% | 9.0% | 82.5% |
| 5 | thấp | 41% | 7.1% | 75.6% |
| 5 | bình thường | 36% | 9.8% | 80.6% |
| 5 | cao | 23% | 13.6% | 82.2% |

Dải ở mức cao rộng gấp đôi mức thấp, đúng chiều mong đợi. Ba mức không chia đều (thấp 39–41%, cao 23–24%) vì giai đoạn đánh giá yên hơn 250 phiên trước nó ở nhiều thời điểm.

## Giới hạn

- Vùng phát triển, chưa có phép đo tiến. Phép đo sạch là so dải đã đẩy lên web với giá thực tế trong các tuần tới.
- Mức cải thiện so với `naive20` là thật nhưng nhỏ (khoảng 3–4% pinball). Giá trị chính của mô hình là dải được hiệu chỉnh đúng mức phủ, không phải độ chính xác vượt trội.
- Phân vị lấy gộp mọi mã, nên mã có đuôi dày hơn (VIC) bị phủ thiếu. Không sửa sau khi xem kết quả (quy tắc dừng).
- Dải không nói gì về chiều giá: tâm dải gần bằng giá hiện tại.
- Chính sách điều chỉnh giá của nguồn vẫn là "unknown"; cửa sổ có bước nhảy quá 7.2% bị loại khỏi train và eval.
- Theo quy tắc dừng: không thử thêm đặc trưng, cửa sổ, mức phủ hay cách lấy phân vị nào.

## Bản chạy hằng ngày khác bản đánh giá ở một điểm

Bundle đang chạy live (`data/rangecast/rangecast.joblib`) được fit trên kho giá hiện hành `data/prices` (từ 2021-01-04), còn lượt đánh giá ở trên fit từ 2019 trên snapshot lịch sử dài. Lý do: bước dự báo hằng ngày phải đọc kho giá được sync, và `data/prices_2018` không được sync. Cùng công thức, cùng 14 biến; hai phân vị của bundle live lấy trên 126 phiên gần nhất không dùng để fit. Con số ở báo cáo này vì vậy mô tả cấu hình đã đăng ký, không phải đúng bundle đang chạy.
