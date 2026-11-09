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
