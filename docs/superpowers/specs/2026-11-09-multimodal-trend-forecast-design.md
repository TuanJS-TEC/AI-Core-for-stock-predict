# Model đa phương thức dự báo xu hướng giá T+2, T+5 — thiết kế

Ngày: 2026-11-09. Tóm tắt phần đã thống nhất trong hội thoại với chủ dự án. **[chủ dự án]** là điều họ đã nói hoặc đã duyệt, **[Claude]** là lựa chọn của người viết, chủ dự án có thể bác.

## 1. Mục đích

Đồ án đã được duyệt, giảng viên hướng dẫn TS. Lê Văn Vịnh, đề tài "Dự đoán Xu hướng Giá Cổ phiếu trên Thị trường Chứng khoán Việt Nam bằng Mô hình Học sâu Đa phương thức: Tích hợp Chỉ số Tài chính và Phân tích Cảm xúc Tin tức". **[chủ dự án]**

Cần một model giá T+2, T+5 cho 15 mã, kết hợp với model NLP đang có thành model fusion, phục vụ khuyến nghị đầu tư. **[chủ dự án]**

Câu hỏi phải trả lời trung thực: fusion có tốt hơn từng nhánh không, và mỗi nhánh đóng góp gì. Mục tiêu không phải chứng minh hệ thống có lãi. **[chủ dự án đã duyệt]**

## 2. Mục tiêu

1. Nhánh giá và BCTC dùng LSTM, GRU hoặc Transformer. Nhánh văn bản dùng PhoBERT hiện có. Hợp nhất bằng attention hoặc concatenation trước lớp dự đoán, theo đề cương. Có logistic và GBM làm đối chứng. **[chủ dự án đã duyệt]**
2. Dự báo 3 lớp Giảm, Đi ngang, Tăng, xuất xác suất, cho T+2 và T+5.
3. Báo cáo Accuracy, Precision, Recall, F1, AUC-ROC theo đề cương, cộng log-loss và Brier.
4. Ablation từng nhánh, và SHAP, LIME cho tài chính so với cảm xúc.
5. Khuyến nghị rút ra từ dự báo và phân tích SHAP. Mô phỏng giao dịch chỉ là phần phụ. **[chủ dự án đã duyệt]**

## 3. Tiêu chí thành công

**[chủ dự án đã duyệt]**
- Không rò rỉ dữ liệu.
- Thắng baseline về log-loss.
- Ablation cho thấy rõ từng nhánh có đóng góp hay không.
- Kết quả cuối được chấm một lần trên dữ liệu sau 30/10/2026.
- Macro-F1 được báo cáo, không đặt ngưỡng tuyệt đối. Con số 0.56–0.58 do chủ dự án tự đặt trước đây và đã bỏ khỏi nghiệm thu.

## 4. Không-mục-tiêu

- Báo cáo công ty chứng khoán, diễn đàn F247, Facebook: ghi là phạm vi chưa làm. **[chủ dự án đã duyệt]**
- BCTC quý, cảm xúc theo khía cạnh, FinBERT, LLM: không có dữ liệu hoặc nhãn. Ghi là giới hạn. **[Claude]**
- Không huấn luyện lại PhoBERT, không sửa `NLP_Module`, không thay đổi bảng quyết định live `decision-v0-nogo`. **[Claude]**
- Không làm giao diện demo trong đợt này. **[Claude]**

## 5. Phản-mục-tiêu

Đồ án coi là thất bại dù code chạy được nếu: **[chủ dự án đã duyệt]**
- Chọn model hoặc ngưỡng dựa trên kết quả test.
- Đạt điểm đẹp nhờ đổi nhãn hoặc giảm độ phủ mà không công khai.

## 6. Dữ liệu và giới hạn đã đo

- Giá: `Pipeline_V1.2`, 15 mã, từ 2021-01-04. Nhãn thích ứng `labels.py`, 25 đặc trưng `price_features.py`, fold `config.DATE_FOLDS`.
- Tin: 17.524 tài liệu, từ 2021-10-21. Điểm cảm xúc trong `article_company_result` theo từng phiên bản model. Mã GAS có rất ít tin.
- BCTC: `NLP_Module/data/bctc.parquet`, theo năm 2019–2025, 14/15 mã (thiếu SSI), không có ngày công bố. Năm Y được coi là biết từ 01/04 năm Y+1.
- 6/15 mã là ngân hàng (ACB, MBB, STB, TCB, VCB, VPB), không có biên gộp và thanh toán hiện hành.
- Cửa sổ eval của 3 fold đã bị nhìn trong các thí nghiệm trước, nên chỉ là vùng phát triển. Vùng chấm cuối là dữ liệu sau 2026-10-30.
- Bằng chứng trước đó: AUC hướng giá khoảng 0.52–0.54, nên kết quả khiêm tốn là kịch bản được kỳ vọng.

## 7. Các lựa chọn của người viết [Claude]

- Package mới `mmf/` ở gốc AI_Core, chạy bằng venv riêng `.venv_mmf` kế thừa gói của `NLP_Module/.venv`, để cài `shap`, `lime` mà không đụng môi trường cron.
- Mọi nhánh dùng cùng tập mẫu từ 2022-01-01, để ablation so được trên cùng hàng.
- Fold A, B dùng điểm của `phobert-foldB`, fold C dùng `phobert-foldC`, vùng chấm cuối dùng `phobert-phobert-v1`, để điểm cảm xúc không học từ nhãn tương lai.
- Bỏ qua cờ `uncertain` khi dựng đặc trưng văn bản, vì model theo fold và model live được chấm với ngưỡng khác nhau.
- Dùng E/P thay cho P/E vì EPS có thể âm hoặc gần 0; EPS đi vào qua E/P và tăng trưởng EPS.
- Vùng chấm cuối cần ít nhất 40 phiên đã có nhãn, hoặc 20 phiên với cờ `--early`, và chỉ chạy một lần cho mỗi tầm dự báo.
