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

## Phụ lục 2026-11-11 (nhãn trung hòa thị trường "xs", viết trước khi chạy bất kỳ ứng viên xs nào)
Động cơ: trên vùng phát triển, baseline tốt nhất chỉ hơn `prior_fixed` khoảng 0.006 (T+2) và 0.011 (T+5) log-loss trung bình, fusion không hơn baseline, và AUC cao ở fold A đến từ hướng chung của thị trường (`mmf_v2_dev_report.md`). Giả thuyết cần kiểm: bỏ thành phần chung của thị trường khỏi nhãn làm lộ phần tín hiệu riêng của từng mã. Chưa có bằng chứng rằng giả thuyết đúng. Không có ứng viên xs nào đã được chạy khi viết phụ lục này, và vùng chấm cuối chưa được đọc.

**Nhãn xs (một cấu hình duy nhất, không dò tham số).** Với mã i, phiên t, h ∈ {2, 5}, trên 15 mã `TICKERS`:
- Lợi suất 1 phiên vượt trội `e[i,t] = ret[i,t] − median_j ret[j,t]`, trong đó `ret = ln(close[t]/close[t−1])` và median lấy qua 15 mã cùng phiên t.
- `σ_e[i,t]` = độ lệch chuẩn (ddof=1) của `e[i,·]` trong 60 phiên kết thúc tại t, cùng `min_periods` như v1.
- Lợi suất h phiên vượt trội `R[i,t] = Σ_{k=1..h} e[i,t+k]`.
- `θ = 0.43 · σ_e[i,t] · √h` (cùng K=0.43 với v1, không chỉnh). Tăng nếu `R > θ`, Giảm nếu `R < −θ`, còn lại Đi ngang.
- Trạng thái, `label_end_date = t+h` và purge giữ nguyên như v1. Một hàng chỉ `ok` khi cả 15 mã có `close` ở mọi phiên trong [t−60, t+h].
- Đặc trưng, folds, 15 mã, seeds, ngưỡng rò rỉ và quy trình chọn giữ nguyên. Không dùng mã ngoài (`ext`) hay đặc trưng bối cảnh.

**Ứng viên cố định trước (6 arm, mỗi (h, fold)):** `prior_fixed`, `logit_price`, `logit_all`, `gbm_price`, `dl_price`, `fusion_concat`. Không thêm arm sau khi xem kết quả.

**Chọn.** Trong mỗi (h, fold): tune, calib và chọn theo log-loss calib như phụ lục chính. Ứng viên xs của (h, fold) là arm có log-loss calib thấp nhất trong 5 arm không phải `prior_fixed`.

**Tiêu chí chính R5 (theo từng h, đúng 2 kiểm định).** Chênh lệch log-loss ghép cặp `ứng viên xs − prior_fixed` (cùng nhãn xs), 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần).
- "Có bằng chứng": ứng viên tốt hơn `prior_fixed` ở ít nhất 2/3 fold và cận trên CI gộp < 0.
- "Có tín hiệu, bằng chứng yếu": mean < 0 nhưng CI chứa 0, hoặc CI đạt nhưng không đủ 2/3 fold.
- "Chưa chứng minh được": còn lại.

**Báo cáo kèm, không dùng để quyết định:** R1–R3 trên nhãn xs (cùng thang), log-loss trước và sau calibration, Brier, Macro-F1, AUC Tăng/Giảm, phân phối lớp thật và dự đoán, CI theo từng fold. Không so log-loss của nhãn xs với nhãn v1: hai nhãn có phân phối lớp khác nhau nên log-loss không so được. Chỉ so mỗi mô hình với `prior_fixed` cùng nhãn.

**Kiểm tra rò rỉ (phải chạy và đọc trước khi đọc kết quả R5).** Nhãn train xáo trộn giữa các ngày (như `mmf_leak_check.md`), `dl_price`, fold C, 20 phép hoán vị, cho cả hai h. Cảnh báo nếu AUC Tăng/Giảm trung bình của nhãn xáo trộn lệch khỏi 0.5 quá 2 độ lệch chuẩn của chính nó. Cảnh báo cũ (Macro-F1 ≥ 0.50, AUC ≥ 0.65) giữ nguyên.

**Quy tắc dừng.** Nếu R5 là "Chưa chứng minh được" ở cả hai h, dừng thử nhãn mới cho nhánh giá và ghi kết luận: với 15 mã và dữ liệu này, đặc trưng giá không cho tín hiệu có thể đo được ở cả nhãn hướng lẫn nhãn trung hòa thị trường. Không thử biến thể nhãn thứ hai chỉ vì kết quả nhìn không đẹp. Nếu R5 đạt "Có bằng chứng" ở ít nhất một h, việc đưa xs vào vùng chấm cuối là quyết định riêng của chủ dự án và cần `mmf_lock_xs.json` khóa trước lượt holdout. Khuyến nghị của trợ lý: không khóa trừ khi cả hai h đạt "Có bằng chứng".

**Giới hạn biết trước.**
- Fold A, B, C đã bị nhìn nhiều lần (cả bởi thí nghiệm v1 và v2), nên R5 trên vùng phát triển chỉ để quyết định có nên đi tiếp, không phải kết quả cuối.
- 15 mã biến động cùng nhau và mỗi fold chỉ 58–61 phiên, nên CI rộng. Đây là lý do cần tăng số fold phát triển, một việc riêng chưa đăng ký ở đây.
- Median 15 mã là proxy thị trường rất thô; một mã lớn (VIC, VCB) kéo lệch được nó.
- Hai kiểm định chính (R5 ở T+2 và T+5) chưa hiệu chỉnh đa kiểm định; mọi so sánh khác là thăm dò.

Phụ lục này do trợ lý soạn theo plan `docs/superpowers/plans/2026-11-11-xs-label.md`; chủ dự án xác nhận trước khi chạy bất kỳ ứng viên xs nào.

## Phụ lục 2026-11-12 (nhãn độ lớn biến động "mag", soạn lúc 2026-11-12T14:23+07:00, trước khi chạy bất kỳ ứng viên mag nào)
Do trợ lý soạn; **chủ dự án xác nhận trước khi chạy trên dữ liệu thật.** Chủ dự án chọn hướng này ngày 2026-11-12, với hạn nộp 2026-11-15.

Động cơ: năm so sánh đã đăng ký trên nhãn hướng (R1–R5) không cho bằng chứng nào, còn mục tiêu độ lớn (`absmove` của volcast) có tín hiệu đo được (AUC test 0.57–0.65 với giá + BCTC, `data/news/reports/volcast_report.json`). Giả thuyết cần kiểm: kiến trúc đa phương thức của mmf dự báo được độ lớn biến động tốt hơn tần suất lớp của từng mã. Phụ lục này đổi mục tiêu dự báo, không đổi kiến trúc, đặc trưng hay fold.

**Nhãn mag (một cấu hình duy nhất, không dò tham số).** Với mã i, phiên t, h ∈ {2, 5}, trên 15 mã `TICKERS`:
- `r[i,t] = ln(close[t+h] / close[t])`; `m[t]` = trung vị của `|r[j,t]|` qua 15 mã cùng phiên t.
- Mạnh (lớp 1) nếu `|r[i,t]| > m[t]`, còn lại Yếu (lớp 0). Mỗi phiên có 7 mã Mạnh trên 15 khi không hòa (tỷ lệ nền 46.7%).
- Một hàng chỉ `ok` khi cả 15 mã có `close` ở mọi phiên trong [t, t+h]; `not_matured` như v1; không có `bad_sigma`. `label_end_date = t+h`, purge như v1. `theta_used = m[t]`. `label_version = "mag-median-v1"`.
- Đặc trưng, folds, 15 mã, seeds (5), scaler, early stopping, calibration giữ nguyên như v1. Không dùng mã ngoài hay đặc trưng bối cảnh.
- Mạng và baseline giữ đầu ra 3 lớp của code hiện có; lớp 2 không bao giờ xuất hiện. **Trước mọi phép chấm và chọn, xác suất của mọi arm (kể cả `prior_fixed`) được chuẩn hóa lại trên hai lớp 0 và 1**, để phần xác suất rơi vào lớp rỗng không làm lệch log-loss của arm nào.

**Ứng viên cố định trước (mỗi (h, fold)).** Baseline: `prior_fixed`, `logit_price`, `gbm_price`, `logit_all`, `gbm_all`. Mạng sâu: `dl_price`, `no_text`, `fusion_concat`, `fusion_attention` (xác suất trung bình 5 seed). Bộ chạy luôn sinh thêm `prior_adaptive`, `logit_text`, `gbm_text`; ba arm này chỉ báo cáo, không là ứng viên. Không thêm arm sau khi xem kết quả.

**Chọn.** Trong mỗi (h, fold), theo log-loss calib: ứng viên mag = thấp nhất trong 8 arm không phải `prior_fixed`; baseline tốt nhất = thấp nhất trong 4 baseline không phải prior; mạng sâu tốt nhất = thấp nhất trong 4 arm mạng sâu.

**Tiêu chí (theo từng h, đúng 6 kiểm định).** Chênh lệch log-loss ghép cặp trên eval, 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần, seed 0), thang diễn giải như R1.
- R7a, có tín hiệu: ứng viên mag − `prior_fixed`.
- R7b, lợi ích mạng sâu: mạng sâu tốt nhất − baseline tốt nhất.
- R7c, đóng góp của tin: `fusion_concat` − `no_text`.

**Báo cáo kèm, không dùng để quyết định:** AUC Mạnh/Yếu, Brier, accuracy, độ phủ lớp dự đoán, log-loss mọi arm theo (h, fold), CI theo từng fold. Không so log-loss của nhãn mag với nhãn v1 hay xs.

**Kiểm tra rò rỉ (chạy và đọc trước khi đọc R7).** Nhãn train xáo trộn giữa các ngày, `dl_price`, fold C, 20 phép hoán vị, cả hai h. Cảnh báo nếu AUC Mạnh/Yếu trung bình của nhãn xáo trộn lệch khỏi 0.5 quá 2 độ lệch chuẩn của chính nó. Ngưỡng cũ (Macro-F1 ≥ 0.50, AUC ≥ 0.65) không áp dụng cho nhãn này: biến động có tính bền (volatility clustering) và volcast đã đạt AUC tới 0.655 trên cùng mục tiêu. Ngưỡng thay thế: AUC Mạnh/Yếu ≥ 0.75 trên toàn mẫu eval của một (h, fold, arm) thì phải kiểm tra trước khi tin.

**Quy tắc dừng.** Chạy đúng một lần. Không thử biến thể nhãn, arm hay siêu tham số thứ hai vì kết quả nhìn không đẹp. Dù kết quả thế nào, `mmf_lock.json` và lượt holdout của v1 không đổi. Nếu R7a đạt "Có bằng chứng", việc khóa một cấu hình mag cho vùng chấm cuối là quyết định riêng của chủ dự án, cần file khóa riêng trước lượt holdout và ít nhất 40 phiên đã trưởng thành sau 2026-10-30.

**Giới hạn biết trước.**
- Không mù: volcast đã đo cùng mục tiêu trên cùng giai đoạn (fold khác, logit/GBM, giá + BCTC), nên việc baseline thắng prior là kết quả được kỳ vọng, không phải phát hiện mới. R6 (`docs/news/model_report.md`) đã cho thấy tin không thêm gì đo được cho mục tiêu này, nên R7c được kỳ vọng không đạt.
- Fold A, B, C đã bị nhìn nhiều lần. Đến hạn 2026-11-15 vùng chấm cuối mới có khoảng 10 phiên, nên **mọi kết quả mag trong báo cáo là kết quả vùng phát triển, chưa được xác nhận ngoài mẫu**, và phải được ghi đúng như vậy.
- 15 mã biến động cùng nhau, mỗi fold 58–61 phiên, CI rộng. Sáu kiểm định chưa hiệu chỉnh đa kiểm định.
- Trung vị 15 mã là ngưỡng tương đối: nhãn nói mã nào biến động mạnh hơn các mã khác trong rổ, không nói mức biến động tuyệt đối.
- Chạy trên CPU (driver GPU không hoạt động ngày 2026-11-12); không ảnh hưởng phép đo, chỉ ảnh hưởng thời gian.

### Sửa đổi phụ lục mag (2026-11-12T14:26+07:00, sau nhận xét của chủ dự án, vẫn trước khi chạy bất kỳ ứng viên mag nào)
Các điểm dưới đây thay cho điểm tương ứng ở phụ lục 2026-11-12; phần còn lại giữ nguyên.

**Kiểm tra rò rỉ (thay toàn bộ đoạn cũ).** Nhãn train xáo trộn giữa các ngày, `dl_price`, fold C, 20 phép hoán vị, cả hai h; đo AUC Mạnh/Yếu trên eval.
- Mốc so sánh là AUC eval của `prior_fixed` cùng (h, fold), không phải 0.5. Lý do: phép xáo dùng một hoán vị ngày chung cho mọi mã, nên giữ nguyên tần suất Mạnh của từng mã; mạng có embedding mã và đặc trưng biến động bền theo mã, nên học lại được tần suất đó từ nhãn xáo trộn. Với nhãn hướng, tần suất theo mã gần như không mang thông tin nên mốc 0.5 là đúng; với nhãn mag thì không.
- Cảnh báo rò rỉ nếu: AUC trung bình của 20 hoán vị vượt AUC của `prior_fixed` quá 0.05, hoặc có hoán vị nào vượt quá 0.08. (Chủ dự án đề xuất 0.55 và 0.58 trên mốc 0.5; giữ nguyên hai biên độ đó, chỉ đổi mốc.)
- Báo cáo kèm: AUC nhãn thật so với max của 20 hoán vị. Đây là phép thử tín hiệu, không phải phép thử rò rỉ, nên không là điều kiện qua kiểm tra.
- Ngưỡng 0.75 ở phụ lục gốc áp dụng cho AUC **nhãn thật** trên eval (thay ngưỡng 0.65 cũ của nhãn hướng), không áp dụng cho nhãn xáo trộn. Giữ nguyên.

**Giá dùng cho nhãn (bổ sung).** Kiểm ngày 2026-11-12 trên snapshot `24251acc1a5695bd`, 15 mã, từ 2021-10-01: lợi suất log 1 phiên có trị tuyệt đối lớn nhất là 0.073, tức đúng biên độ sàn −7% của HOSE; có 84 phiên chạm sàn và không có bước nhảy nào vượt biên độ. Chuỗi `close` của snapshot vì vậy liên tục qua các ngày chia tách và cổ tức, tương đương chuỗi `adj` của volcast. Thêm chốt chặn: hàng (i, t) có trạng thái `seam` (không `ok`) nếu mã i có phiên nào trong t+1..t+h với `|close/close_trước − 1| > 0.072` (`volcast.RET_BAND`), và `|r[i,t]|` của hàng đó không tham gia trung vị `m[t]`. Trên snapshot hiện tại chốt này không loại hàng nào.

**Ứng viên (bổ sung).** Thêm `dl_text` vào nhóm mạng sâu: 5 arm mạng sâu, 9 arm không phải prior.

**R7c (bổ sung).** Kiểm định chính vẫn là `fusion_concat − no_text` (2 kiểm định, một cho mỗi h): đây là cặp duy nhất chỉ khác nhau ở nhánh tin. Báo cáo kèm, không dùng để quyết định: `fusion_attention − no_text` (khác cả nhánh tin lẫn cơ chế hợp nhất, vì `ARMS` không có arm attention bỏ tin) và `dl_text − prior_fixed` (tin đứng một mình).

**R7b (không đổi, ghi rõ).** Việc chọn "tốt nhất" làm trên calib, phép đo làm trên eval, nên không có thiên lệch chọn lọc trong cùng một khối; thiên lệch còn lại là fold A, B, C đã bị nhìn nhiều lần, đã nêu ở mục giới hạn.

**GPU.** Card có trong máy (RTX 5060 Ti) và torch của `.venv_mmf` là bản CUDA (2.14.1+cu130). Nguyên nhân: kernel đang chạy 7.0.0-43 không có module nvidia (gói `linux-modules-nvidia-595-open` mới cài tới 7.0.0-42). Sửa cần quyền sudo nên do chủ dự án làm; nếu không sửa thì chạy CPU.

## Phụ lục 2026-11-12 (R8: xác nhận ngoài thời gian trên fold lùi D, E, F cho nhãn mag; soạn lúc 2026-11-12T14:44+07:00, trước khi chạy R7 và R8)
Do trợ lý soạn; **chủ dự án xác nhận trước khi chạy.** Mục đích: có một phép đo ngoài các khối eval A, B, C trước hạn 2026-11-15 mà không chạm vùng chấm cuối. Vùng sau 2026-10-30 không được đọc; `mmf_lock.json` và lượt holdout không đổi.

**Fold.** `make_date_folds(calendar, "2025-07-25", n=3)` trên snapshot `24251acc1a5695bd`, khối 63 phiên (train_end, tune_end, calib_end, eval_end):
- D: 2024-04-19, 2024-07-22, 2024-10-21, 2025-01-17
- E: 2024-07-22, 2024-10-21, 2025-01-17, 2025-04-24
- F: 2024-10-21, 2025-01-17, 2025-04-24, 2025-07-25

Ba khối eval phủ 2024-10-22 đến 2025-07-25 (189 phiên). Trong mọi lượt chạy mmf trước đây các phiên này chỉ nằm trong khối train của A, B, C; chưa từng là tune, calib hay eval. Train bắt đầu từ `START` = 2022-01-01 như v1. Cảm xúc tin: `phobert-foldA` cho cả ba fold (huấn luyện trên nhãn trước 2024-01-01).

**Arm cố định, không chọn giữa các arm.** `prior_fixed`, `logit_price`, `dl_price`, `no_text`, `fusion_concat`. Nhãn mag, đặc trưng, siêu tham số, 5 seed, chọn encoder giá trên tune, calibration: đúng như R7 và v1, không đổi gì sau khi xem R7. Danh sách arm này cố định ở đây, không phụ thuộc kết quả R7.

**Tiêu chí (theo từng h, đúng 6 kiểm định).** Chênh lệch log-loss ghép cặp trên eval của D, E, F, 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần, seed 0), xác suất chuẩn hóa hai lớp, thang diễn giải như R1.
- R8a, mô hình fusion có tín hiệu: `fusion_concat − prior_fixed`. Đây là tiêu chí để gọi mô hình fusion là "dùng được cho dự báo độ lớn ở h đó": phải đạt "Có bằng chứng".
- R8b, fusion so với mô hình đơn giản: `fusion_concat − logit_price`.
- R8c, đóng góp của tin: `fusion_concat − no_text`.

**Thứ tự và quy tắc dừng.** Chạy R7 trước (gồm kiểm tra rò rỉ). Nếu kiểm tra rò rỉ của R7 cảnh báo thì không chạy R8 cho tới khi tìm ra nguyên nhân. Ngoài ra R8 chạy đúng một lần, bất kể R7 ra sao. Không thêm fold, arm hay biến thể sau khi xem kết quả. Nếu R8a đạt nhưng R8b không đạt, báo cáo phải ghi: mô hình fusion có tín hiệu nhưng chưa chứng minh được là hơn hồi quy logistic trên giá.

**Giới hạn biết trước (R8 yếu hơn một holdout tiến).**
- Không mù với mục tiêu này: volcast (logit/GBM, giá + BCTC, cùng mục tiêu) đã chấm test trên 2024-07 đến 2025-01 và 2025-07 đến 2026-01, và dùng 2025-01 đến 2025-07 làm validation. Quyết định chuyển sang nhãn mag dựa trên các kết quả đó. Kiến trúc mmf thì chưa từng được chấm trên các khối này.
- Thiết kế của mmf (đặc trưng, kiến trúc, siêu tham số) được hình thành khi nhìn eval của A, B, C, tức là dữ liệu *sau* các khối này. R8 vì vậy là xác nhận ngoài thời gian theo chiều lùi, không phải dự báo tiến; không được gọi là "kết quả chấm cuối".
- Train ngắn hơn fold A, B, C (bắt đầu 2022-01, kết thúc 2024-04 đến 2024-10).
- Sáu kiểm định chưa hiệu chỉnh đa kiểm định; 15 mã biến động cùng nhau.
- Hai phương án khác được xem xét và không dùng: nhìn sớm vùng chấm cuối ở khoảng 10 phiên (kiểm định đổi dấu theo ngày giả định các ngày trao đổi được, trong khi nhãn h=5 chồng lấn giữa các ngày liên tiếp; 10 phiên không đủ lực để phát hiện chênh lệch cỡ 0.005 log-loss; và tiêu một phần lượt holdout); và vùng đệm sau fold C (không tồn tại: eval của C kết thúc đúng 2026-10-30).

## Phụ lục 2026-11-12 (R9: baseline HAR và lịch sự kiện cho nhãn mag; soạn lúc 2026-11-12T21:42+07:00, trước khi chạy bất kỳ arm nào dưới đây)
Do trợ lý soạn theo spec `docs/superpowers/specs/2026-11-12-mag-har-events-design.md` và plan `docs/superpowers/plans/2026-11-12-mag-har-events.md`. Chủ dự án được hỏi xác nhận phụ lục nháp trong spec và trả lời "ok làm ngay cho tôi" (2026-11-12). Kiểm toán dữ liệu (không đọc nhãn): `docs/reports/mag_events_audit.md`.

Động cơ: R7/R8 cho thấy tín hiệu của nhãn mag chủ yếu là tính bền của biến động, và tin không thêm gì đo được. Câu hỏi: một nguồn biết trước và khác nhau giữa các mã (lịch sự kiện doanh nghiệp) có thêm thông tin ngoài một baseline biến động dạng HAR không.

- **Dữ liệu.** Snapshot `24251acc1a5695bd`, nhãn `mag-median-v1`, fold A, B, C (`DATE_FOLDS`) và D, E, Fb (`BACK_FOLDS`): 6 fold, eval 2024-10-22 đến 2026-10-30. Hàng giống hệt R7/R8. Lịch sự kiện: `data/prices/events/2026-11-12.parquet` (sha256 `6f7a7f21ce1e4b2feb81741c69069642e3a7f2c17bfeb8132e7ecf92cdc5863b`, 330 sự kiện VCI loại DIV, ISS, AGME, EGME; nguồn không trả AGMR).
- **Đặc trưng HAR (7).** `har_rv_d/w/m` = ln(trung bình 1/5/22 phiên của r² + 1e-8), `har_pk_d/w/m` cùng dạng với Parkinson ln(high/low)²/(4 ln 2), `har_gap_w` = ln(trung bình 5 phiên của ln(open/close_trước)² + 1e-8); trừ trung bình của các mã cùng phiên.
- **Đặc trưng sự kiện (4).** Biết tại t nếu `public_date < t`. Ngày sự kiện: `exright_date` cho DIV, ISS; `display_date1` (ngày họp) cho AGME, EGME. `ev_exr_in_h`, `ev_mtg_in_h` = có sự kiện trong t+1..t+h; `ev_next` = số phiên đến sự kiện đã biết gần nhất, cắt ở 20, chia 20 (không có thì 1); `ev_new5` = số thông báo trong t−5..t−1. Ngày không phải phiên → phiên kế tiếp.
- **Arm.** `logit_har` (HAR) và `logit_har_ev` (HAR + sự kiện): hồi quy logistic, cài đặt như `logit_price` (C=0.1), chuẩn hóa trên train, clip ±5, NaN → 0, alpha trên calib như mọi baseline. Ghi vào `data/mmf/exp/har/24251acc1a5695bd`. Mốc `prior_fixed`, `prior_adaptive`, `dl_price`, `fusion_concat` đọc từ run mag `data/mmf/exp/mag/24251acc1a5695bd` (không chạy lại); bộ chấm dừng nếu hàng eval của hai run không trùng. Không thêm arm sau khi xem kết quả.
- **Phép đo.** Chênh lệch log-loss ghép cặp trên eval, xác suất chuẩn hóa hai lớp, 95% block bootstrap theo ngày (khối 20 phiên, 2000 lần, seed 0).
- **Tiêu chí chính (theo từng h, đúng 4 kiểm định).** R9a, đóng góp của lịch sự kiện: `logit_har_ev − logit_har`. R9b, baseline HAR: `logit_har − prior_adaptive`. "Có bằng chứng": tốt hơn ở ít nhất 4/6 fold và cận trên CI gộp < 0; "Có tín hiệu, bằng chứng yếu": mean < 0 nhưng CI chứa 0, hoặc CI đạt nhưng dưới 4/6 fold; "Chưa chứng minh được": còn lại.
- **Báo cáo kèm, không dùng để quyết định.** `logit_har − prior_fixed`, `logit_har_ev − prior_fixed`, `logit_har − dl_price`, `logit_har_ev − dl_price`, `logit_har_ev − fusion_concat`; từng fold; AUC, Brier; hệ số 4 đặc trưng sự kiện theo fold; giả dược: 20 hoán vị không điểm cố định của bảng sự kiện giữa các mã (seed 0..19), R9a gộp thật so với phân phối giả dược.
- **Rò rỉ.** Unit test chứng minh sự kiện công bố tại t không ảnh hưởng đặc trưng tại t. Đối chiếu ngày công bố VCI với tin (`mag_events_audit.md`): VCI ghi muộn hơn tin, trung vị 4 ngày, không có ca nào sớm hơn; cổng qua. Ngưỡng AUC nhãn thật ≥ 0.75 của phụ lục mag giữ nguyên.
- **Quy tắc dừng.** Chạy đúng một lần. Không thử đặc trưng, cửa sổ, loại sự kiện hay C khác sau khi xem kết quả. Nếu R9a là "Chưa chứng minh được" ở cả hai h: ghi kết luận "lịch sự kiện (VCI, 4 loại) không thêm thông tin đo được ngoài lịch sử biến động cho 15 mã". Nếu R9a "Có bằng chứng" ở ít nhất một h: khóa `logit_har_ev` cho kiểm chứng tiến là quyết định của chủ dự án (`docs/reports/har_ev_lock.json`), chấm một lần sau ít nhất 40 phiên đã trưởng thành sau ngày khóa, dùng ảnh chụp sự kiện hằng ngày. `mmf_lock.json` và holdout mmf không đổi.
- **Giới hạn biết trước.** Sáu fold đã bị nhìn bởi R7/R8 trên cùng nhãn, nên R9 là vùng phát triển. Volcast đã dùng biên độ ngày cho cùng mục tiêu. Lịch sự kiện là dữ liệu nhà cung cấp tải ngày 2026-11-12: sự kiện bị hủy có thể đã biến mất, ngày có thể đã bị sửa. Sự kiện rất thưa (đo được: 1.8% hàng ở T+2 và 4.0% ở T+5 có cờ trong cửa sổ), nên R9a có thể thiếu lực. Bốn kiểm định chính chưa hiệu chỉnh đa kiểm định; 15 mã biến động cùng nhau.

### Sửa đổi phụ lục R9 (2026-11-12T21:47+07:00, sau review code, vẫn trước khi chạy bất kỳ arm R9 nào)
- **Clip ±5 chỉ áp cho 7 đặc trưng HAR**, không áp cho 4 đặc trưng sự kiện (spec chỉ ghi clip ở mục HAR). Lý do: một cờ hiếm có tỷ lệ p, sau chuẩn hóa, bằng √((1−p)/p) khi bật, vượt 5 khi p < 3.85%. Độ phủ đo được là 0.9–2.3%, nên clip sẽ ép mọi cờ về 5 và gộp `ev_new5` ≥ 2 về cùng một giá trị. Đặc trưng sự kiện vẫn được chuẩn hóa bằng mean/std của train.
- Bộ chạy kiểm hàng eval của mỗi (h, fold) với `prior_fixed` của run mag **trước khi fit**, và dừng nếu HAR thiếu trên quá 5% hàng train (dấu hiệu lệch khóa).
- Hệ số được báo cáo là hệ số của lớp Mạnh trên đặc trưng đã chuẩn hóa (không clip với đặc trưng sự kiện).

## Phụ lục 2026-11-13 (R10: luật v3 cho nhãn hướng 3 lớp và hai thử nghiệm nhánh giá; soạn lúc 2026-11-13T13:13+07:00, trước khi chạy bất kỳ arm nào dưới đây)

Chủ dự án yêu cầu ngày 2026-11-13 ("ok vậy thử đi") sau khi xem bản kiểm tra dữ liệu và luật. Chạy đúng một lần, một cấu hình, trên fold A, B, C của snapshot `24251acc1a5695bd`, train từ 2022, nhãn `adaptive-k0.43-w60-v1` không đổi. **Vùng phát triển:** ba khối eval này đã bị nhìn rất nhiều lần, kể cả để nghĩ ra chính các thay đổi dưới đây; kết quả không phải kiểm chứng ngoài mẫu. Vùng sau 2026-10-30 không được đọc, `mmf_lock.json` không đổi, holdout không chạy. Phần luật cũ (R1–R3 theo log-loss) giữ nguyên hiệu lực cho cấu hình đã khóa.

Thay đổi so với v1 (áp cho mọi arm mạng sâu của R10):
1. Hàm mất mát cross-entropy có trọng số lớp w_c = N / (3 n_c) tính trên khối train; dừng sớm theo cùng hàm đó trên tune.
2. Nhiệt độ và alpha fit trên tune + calib gộp (126 phiên) thay cho riêng calib. Tune đã dùng để dừng sớm nên hơi lạc quan cho hai tham số này; chấp nhận, ghi rõ.
3. Chọn trong mỗi (h, fold) bằng Macro-F1 của quyết định chính thức (argmax p / prior^alpha) trên tune + calib: fusion trong {fusion_concat, fusion_attention}, baseline trong {logit_price, gbm_price, logit_all, gbm_all} (baseline không đổi cách fit; alpha của chúng cũng fit trên tune + calib).
4. Bộ mã hóa giá của từng (h, fold) lấy từ `select.json` của lượt chạy đã khóa, không chọn lại. 5 seed, xác suất là trung bình seed.

Arm: `dl_price`, `no_text`, `fusion_concat`, `fusion_attention` (kiến trúc v1 với các thay đổi trên), cộng hai thử nghiệm:
- E1 `short_price`: nhánh giá nhận chuỗi 15 phiên cuối của 6 đặc trưng bước (ret_1, c_o, h_l, o_pc, close_loc, v_ma20) qua GRU 32 chiều, nối với 25 đặc trưng của phiên t qua một lớp tuyến tính 32 chiều + ReLU + dropout.
- E2 `dl_price_m`, `fusion_concat_m`: như `dl_price`, `fusion_concat` nhưng trên train bỏ (trọng số 0) các mẫu có |r| trong khoảng 0.85–1.15 lần ngưỡng θ của chính nó. Tune, calib, eval không đổi.

Thước đo trên eval gộp ba fold, từng h; khoảng tin cậy 95% bằng block bootstrap theo ngày (khối 20 phiên, 2000 lần):
- R0, hơn ngẫu nhiên: Macro-F1 của fusion được chọn có cận dưới > 1/3 **và** AUC một-với-phần-còn-lại có cận dưới > 0.5 → "Có bằng chứng hơn ngẫu nhiên"; chỉ một trong hai → "bằng chứng yếu"; không cái nào → "Chưa chứng minh được".
- R1' (fusion được chọn trừ baseline được chọn) và R2' (fusion được chọn trừ `dl_price`): chênh lệch Macro-F1 và chênh lệch AUC, ghép cặp. "Có bằng chứng" khi cận dưới của chênh lệch Macro-F1 > 0 và thắng ít nhất 2/3 fold; "bằng chứng yếu" khi trung bình > 0; còn lại "Chưa chứng minh được". Log-loss chỉ báo cáo (mạng có trọng số lớp cho xác suất đã cân lại, không so log-loss với baseline được).
- Báo cáo kèm, cùng phép đo: tác động của luật v3 (`dl_price` và `fusion_concat` của R10 trừ cùng arm của lượt đã khóa với quyết định chính thức của nó), E1 (`short_price` trừ `dl_price`), E2 (`dl_price_m` trừ `dl_price`, `fusion_concat_m` trừ `fusion_concat`).
- Mọi arm: Accuracy, Precision và Recall từng lớp, Macro-F1, AUC một-với-phần-còn-lại, AUC Tăng so với Giảm, kèm khoảng tin cậy cho Accuracy, Macro-F1, AUC.

Quy tắc dừng: không thử thêm trọng số, dải bỏ mẫu, độ dài chuỗi, tập đặc trưng bước hay cách chọn nào sau khi xem kết quả. Không cấu hình nào của R10 được khóa cho vùng chấm cuối nếu không có phụ lục riêng. Cảnh báo rò rỉ giữ nguyên (Macro-F1 >= 0.50 hoặc AUC Tăng so với Giảm >= 0.65). Run: `data/mmf/exp/v3/24251acc1a5695bd`, code `ai_core/models/mmf/v3.py`.

## Phụ lục 2026-11-13 (R11: dải giá 80% "rangecast" cho T+2, T+5; soạn lúc 2026-11-13T15:15+07:00, trước khi chạy lượt chấm)

Chủ dự án duyệt thiết kế `docs/superpowers/specs/2026-11-13-price-range-forecast-design.md` ngày 2026-11-13 ("ok duyệt bản thiết kế, bắt đầu dựng đi"). Đây là **đích mới** (khoảng giá, không phải nhãn phân lớp), không phải biến thể của R7–R10; các quy tắc dừng của chúng giữ nguyên. Một cấu hình, chạy một lần. **Vùng phát triển:** sáu fold dưới đây đã bị R7–R9 dùng cho nhãn độ lớn, và việc "biến động gần đây dự báo được độ lớn" đã biết từ đó. Vùng sau 2026-10-30 không được đọc.

Mô hình (`ai_core/models/rangecast/rangecast.py`):
- Đích r = ln(close[t+h]/close[t]); hàng hợp lệ theo `status == ok` của nhãn mag (bỏ phiên thiếu, bỏ cửa sổ có bước nhảy > 7.2%).
- σ̂ = sqrt(exp(hồi quy tuyến tính bình phương tối thiểu)), gộp mọi mã, riêng từng h, đích ln(Σ lợi suất ngày² trong t+1..t+h + 1e-8), đầu vào 14 biến: 7 đại lượng HAR thô (như R9 nhưng không trừ trung bình phiên) và trung bình cắt ngang của từng đại lượng trong phiên.
- Dải: [z10·σ̂, z90·σ̂] theo log-lợi suất, z10 và z90 là phân vị 10% và 90% thực nghiệm của r/σ̂ trên khối calib (gộp mã), cắt trong biên độ ±7% mỗi phiên cộng dồn h phiên.
- Mức rủi ro: tỷ lệ các σ̂ trong tối đa 249 phiên trước của chính mã nhỏ hơn σ̂ hôm nay; ≥ 70% là cao, ≤ 30% là thấp, còn lại bình thường (dưới 59 giá trị: bình thường).

Dữ liệu và fold: snapshot lịch sử dài `data/prices_2018/snapshots/a0674e0c56bf9013` (15 mã, từ 2018-11-15, phần từ 2021 trùng snapshot dev), sáu fold `DATE_FOLDS` + `BACK_FOLDS`. Fit trên train + tune, phân vị trên calib, chấm eval.

Mốc so sánh: `naive20` (σ = độ lệch chuẩn lợi suất 20 phiên × √h, cùng thủ tục phân vị trên calib) và `hist` (phân vị 10/90 của r trong train + tune của từng mã).

Chỉ số trên eval gộp sáu fold, từng h: độ phủ thực tế, tỷ lệ rơi dưới và rơi trên, độ rộng dải trung bình, pinball loss (tổng hai phân vị), chênh lệch pinball ghép cặp har trừ từng mốc (block bootstrap 20 phiên, 2000 lần), độ phủ theo mã và theo mức rủi ro.

Cách đọc:
- "Dùng được": độ phủ gộp của cả hai h trong 77–83% **và** cận trên CI của pinball(har) − pinball(naive20) < 0 ở cả hai h.
- "Dùng được, có lưu ý": độ phủ cả hai h trong 75–85% nhưng không đạt điều kiện trên. Web vẫn hiện dải; báo cáo ghi rõ HAR không hơn mốc đơn giản.
- "Không đạt": độ phủ gộp của một h nằm ngoài 75–85%. Không đẩy dải sang web; web dùng nhãn mạnh/yếu của volcast.

Quy tắc dừng: không thử đặc trưng, cửa sổ, mức phủ, cách lấy phân vị hay tập train khác sau khi xem kết quả. Báo cáo: `docs/reports/rangecast_report.md`, bảng: `data/rangecast/eval.md`.

## Phụ lục 2026-11-14 (R12: nhánh tài chính theo quý với ngày công bố thật; soạn lúc 2026-11-14T16:55+07:00, trước khi chạy bất kỳ arm nào dưới đây)

Chủ dự án yêu cầu ngày 2026-11-14 ("làm ngay đi") sau khi xem phép đo tín hiệu từng chỉ số. Lý do: đề tài yêu cầu chỉ số từ BCTC quý/năm, lượt đã khóa chỉ dùng BCTC năm với ngày công bố giả định (01/04) và SSI không có BCTC. Đây là **dữ liệu mới** cho nhánh tài chính, không phải biến thể trọng số/cửa sổ/nhãn của R10; quy tắc dừng của R10 giữ nguyên. Chạy đúng một lần, một cấu hình, fold A, B, C của snapshot `24251acc1a5695bd`, nhãn `adaptive-k0.43-w60-v1`, train từ 2022. `mmf_lock.json` không đổi, holdout không chạy, vùng sau 2026-10-30 không được đọc.

**Đã nhìn trước (công khai):** ba khối eval này đã bị nhìn rất nhiều lần. Riêng với dữ liệu quý, trước khi soạn phụ lục đã đo tương quan hạng theo phiên giữa 9 chỉ số quý và lợi suất T+2/T+5 trên cả hai vùng: vùng train 2022-01..2025-07 có tín hiệu (E/P 0.039/0.062, gia tốc tăng trưởng lợi nhuận 0.037/0.051, giá đi theo tăng trưởng lợi nhuận sau công bố 0.18/0.27 trên 210 lần công bố, khoảng tin cậy loại trừ 0); vùng 2025-07..2026-10 (trùng các khối tune/calib/eval) thì gần 0 hoặc âm, khoảng tin cậy chứa 0. Kỳ vọng hợp lý vì vậy là **không cải thiện**; kết quả dương trên eval phải đọc là chưa kiểm chứng ngoài mẫu.

Dữ liệu: báo cáo quý thô của VCI cho 15 mã, 35 quý (2018-Q1..2026-Q3), có `publicDate`; tải 2026-11-14, `data/reference/bctc_quarter_vci_income_statement.parquet` (sha256 56beb6e7e0b5bf9b…) và `…_balance_sheet.parquet` (f85bf913d12f4164…). Giới hạn đã biết: (1) số liệu là bản hiện hành của nguồn, 19% dòng được cập nhật hơn 30 ngày sau ngày công bố nên có thể là số đã điều chỉnh; (2) `publicDate` với một số quý là ngày của bản soát xét/kiểm toán (muộn hơn bản đầu), tức thận trọng về rò rỉ nhưng làm đặc trưng "vừa công bố" kém chính xác; (3) E/P chia cho giá đã điều chỉnh nhân số cổ phần theo vốn góp của quý đó, như lượt cũ, không phải vốn hóa thật tại thời điểm đó; (4) ngân hàng không có biên gộp, tăng trưởng doanh thu, thanh khoản hiện hành (đặt 0).

Đặc trưng nhánh tài chính (`ai_core/models/mmf/fund_quarter.py`, 13 biến): E/P theo lợi nhuận 4 quý gần nhất, tăng trưởng lợi nhuận quý so với cùng kỳ, gia tốc của nó so với quý trước, ROE, ROA (lợi nhuận 4 quý), D/E, biên gộp, tăng trưởng doanh thu cùng kỳ, thanh khoản hiện hành, cờ "vừa công bố" (5 phiên đầu báo cáo dùng được), tăng trưởng lợi nhuận nhân cờ đó, cờ có BCTC, cờ ngân hàng. Một quý chỉ dùng được từ phiên **sau** ngày công bố, tối đa 150 ngày; thiếu thì cờ có BCTC = 0 và các tỷ số = 0 (như lượt cũ). Chuẩn hóa fit trên train.

Thủ tục: y hệt lượt đã khóa (hàm mất mát không trọng số, dừng sớm trên tune, nhiệt độ và alpha trên calib, 5 seed, xác suất trung bình seed), bộ mã hóa giá của từng (h, fold) lấy từ `select.json` của lượt đã khóa, không chọn lại. Đã kiểm tra trước khi chạy: mẫu, nhãn và cửa sổ giá của mọi khối trùng khít lượt đã khóa. Arm chạy lại với nhánh tài chính quý: `dl_fund`, `no_text`, `fusion_concat`, `fusion_attention`, cộng baseline `logit_all`, `gbm_all`. Arm không có nhánh tài chính (`prior_fixed`, `logit_price`, `dl_price`) lấy từ lượt đã khóa.

Tiêu chí, từng h, chênh lệch log-loss ghép cặp trên eval gộp ba fold (block bootstrap 20 phiên, 2000 lần), thang diễn giải của bản đăng ký gốc ("Có bằng chứng": cận trên CI < 0 và thắng ≥ 2/3 fold; "bằng chứng yếu": trung bình < 0; còn lại "Chưa chứng minh được"):
- R12a, quý so với năm: `fusion_concat` (quý) trừ `fusion_concat` của lượt đã khóa.
- R12b, fusion so với nhánh giá: `fusion_concat` (quý) trừ `dl_price`.
- R12c, nhánh tài chính đứng riêng: `dl_fund` (quý) trừ `prior_fixed`.
- Báo cáo kèm, cùng phép đo: từng arm quý trừ arm cùng tên của lượt đã khóa (`dl_fund`, `no_text`, `fusion_attention`, `logit_all`, `gbm_all`).
- Mọi arm: Accuracy, Precision và Recall từng lớp, Macro-F1, AUC một-với-phần-còn-lại, AUC Tăng so với Giảm theo quyết định chính thức (argmax p / prior^alpha), kèm khoảng tin cậy; chênh lệch Macro-F1/AUC của R12a–c chỉ báo cáo, không quyết định.

Cách đọc: mô hình hướng giá chỉ được gọi là "có cải thiện nhờ BCTC quý" khi R12a là "Có bằng chứng" ở ít nhất một h và không "Chưa chứng minh được" ở h còn lại. Dù kết quả nào, BCTC quý vẫn được ghi nhận là đã đáp ứng yêu cầu dữ liệu của đề tài; không cấu hình nào của R12 được khóa cho vùng chấm cuối nếu không có phụ lục riêng.

Quy tắc dừng: không thử thêm tập chỉ số, độ trễ công bố, độ dài cờ "vừa công bố", tuổi tối đa hay cách điền thiếu nào sau khi xem kết quả. Cảnh báo rò rỉ giữ nguyên (Macro-F1 >= 0.50 hoặc AUC Tăng so với Giảm >= 0.65). Run: `data/mmf/exp/r12/24251acc1a5695bd`, code `ai_core/models/mmf/r12.py`, bảng: `data/mmf/exp/r12/r12.md`.
