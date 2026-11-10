# Luật đặt trước: model đa phương thức T+2, T+5

Viết trước khi có kết quả thật. Mọi thay đổi sau này phải ghi thêm ở cuối file kèm ngày và lý do, không sửa phần trên.

## Vùng dữ liệu
- Vùng phát triển: khối eval của fold A, B, C trong `DATE_FOLDS`. Các khối này đã bị nhìn trong thí nghiệm trước, nên kết quả ở đây chỉ để phát triển.
- Vùng chấm cuối: phiên sau 2026-10-30, chấm một lần cho mỗi h sau khi khóa cấu hình.

## Quy trình chọn
Trong mỗi (h, fold): early stopping trên tune; nhiệt độ và alpha trên calib; chọn baseline tốt nhất, nhánh đơn tốt nhất và fusion tốt nhất bằng log-loss trên calib.

## Tiêu chí, theo từng h
Dùng chênh lệch log-loss ghép cặp, 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần).
- R1, thắng baseline: fusion tốt hơn baseline tốt nhất ở ít nhất 2/3 fold và cận trên CI gộp < 0.
- R2, lợi ích fusion: fusion tốt hơn nhánh đơn tốt nhất ở ít nhất 2/3 fold và cận trên CI gộp < 0.
- R3, đóng góp nhánh: với mỗi nhánh, so `fusion_concat` với arm bỏ nhánh đó bằng cùng phép đo.

Diễn giải cho mỗi so sánh:
- "Có bằng chứng": đạt điều kiện trên.
- "Có tín hiệu, bằng chứng yếu": chênh lệch trung bình có lợi nhưng CI chứa 0, hoặc CI đạt nhưng không đủ 2/3 fold.
- "Chưa chứng minh được": còn lại.

## Chỉ số báo cáo, không dùng để chọn
Accuracy, Precision, Recall, Macro-F1, AUC-ROC, Brier, kappa, theo từng h, fold và mã. Không có ngưỡng tuyệt đối cho Macro-F1.

## Cảnh báo rò rỉ
Macro-F1 toàn mẫu >= 0.50 hoặc AUC Tăng so với Giảm >= 0.65 thì coi là nghi rò rỉ, phải kiểm tra trước khi tin.

## Công khai
Mọi kết quả chọn lọc theo độ tin cậy phải báo kèm độ phủ, số mẫu, số ngày, và đi cùng kết quả toàn mẫu.

## Phụ lục 2026-11-10 (trước khi chạy vùng chấm cuối)
Vùng chấm cuối chỉ có một fold, khoảng 40 phiên. Block bootstrap vòng tròn khối 20 phiên trên vùng này chỉ có 2 khối, và CI co lại: ở 20 phiên mỗi lần lấy mẫu giữ nguyên mọi phiên nên CI có độ rộng bằng 0. Vì vậy các so sánh R1, R2, R3 trên vùng chấm cuối dùng khối 5 phiên (`HOLDOUT_BLOCK`), vẫn 2000 lần. Lần chạy sớm (`--early`, tối thiểu 20 phiên) không đưa ra kết luận: cột kết luận ghi "Chưa đủ phiên để kết luận (chạy sớm)", số vẫn in. Luật của vùng phát triển giữ nguyên. Vùng chấm cuối chưa được đọc khi viết phụ lục này.
Khoảng tin cậy của mô phỏng luật khuyến nghị trên vùng chấm cuối cũng dùng khối 5 phiên.
Snapshot dữ liệu của lần chạy phát triển (24251acc1a5695bd) chứa 6 phiên sau 2026-10-30; các phiên này được nạp nhưng không dùng trong bất kỳ lần fit, chọn lựa hay chỉ số nào (không có hàng dự báo nào sau 2026-10-28).

## Phụ lục 2026-11-10 (v2 nhánh giá, viết trước khi chạy bất kỳ ứng viên v2 nào)
Ứng viên v2 là 12 cấu hình cố định: `hurdle_logit_price@base`, `hurdle_gbm_price@base`, và với mỗi variant `ext`, `ext_ctx`: `logit_price`, `gbm_price`, `hurdle_logit_price`, `hurdle_gbm_price`, `dl_price`. `ext` huấn luyện thêm trên các mã VN100 ngoài 15 mã; tune, calib, eval vẫn chỉ 15 mã. `ext_ctx` thêm 19 đặc trưng bối cảnh (VNINDEX, độ rộng thị trường, S&P 500, VIX, Brent, USD/VND, hạng cắt ngang). `hurdle_*` là mô hình hai tầng: có vượt ngưỡng hay không, rồi tăng hay giảm.
Chọn: trong mỗi (h, fold), ứng viên v2 có log-loss calib thấp nhất. Khóa: mỗi h lấy ứng viên được chọn ở nhiều fold nhất (hòa thì log-loss calib trung bình thấp hơn), ghi vào `mmf_lock_v2.json`.
R4, theo từng h, cùng phép đo và thang diễn giải như R1: v2 so với baseline tốt nhất của v1, và v2 so với `dl_price` của v1.
Trên vùng chấm cuối, v2 được chấm trong cùng lượt chạy với v1 (khối 5 phiên, 2000 lần), một lần cho mỗi h. Kết quả R1–R3 của v1 không đổi cách tính.
Giới hạn biết trước: fold A, B, C đã bị nhìn nhiều lần, nên R4 trên vùng phát triển chỉ để chọn; danh sách VN100 lấy ngày 2026-11-10 (thiên lệch sống sót ở tập train của mã ngoài); chuỗi toàn cầu lấy từ Yahoo chart API không chính thức. Vùng chấm cuối chưa được đọc khi viết phụ lục này.
Phụ lục này do trợ lý soạn theo plan `docs/superpowers/plans/2026-11-10-price-forecast-v2.md`; chủ dự án xác nhận trước khi nhánh `price-v2` được merge.
