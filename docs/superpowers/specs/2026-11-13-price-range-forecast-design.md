# Thiết kế: dự báo dải giá T+2, T+5 (rangecast) và hiển thị trên StockSenseVN

Ngày: 2026-11-13. Hạn nộp đồ án: 2026-11-15. Trạng thái: chủ dự án đã duyệt ngày 2026-11-13 ("ok duyệt bản thiết kế, bắt đầu dựng đi").

## 1. Ý định và lý do

Mọi thử nghiệm dự báo **hướng** giá (tăng/giảm) đều không có bằng chứng, kể cả khi train thêm dữ liệu 2019–2021
(`data/mmf/exp/hist2018/report.md`). Thứ dữ liệu hiện có dự báo được là **độ lớn** biến động: mô hình HAR tuyến tính là
mốc mạnh nhất (`docs/reports/mmf_har_ev_report.md`). Trong khi đó web StockSenseVN đang hiển thị nổi bật nhất chính phần
không có bằng chứng (nhãn tăng/giảm của `decision-v1-trend`).

Hướng mới: thay câu hỏi "giá đi đâu" bằng "giá có thể đi xa bao nhiêu". Với mỗi mã, mỗi phiên, mô hình đưa ra **dải giá
80%** cho T+2 và T+5 cùng **mức rủi ro** so với thường lệ của chính mã đó. Người dùng web dùng dải này để đặt điểm cắt
lỗ và cỡ vị thế trong Trade Plan, và tự kiểm chứng được qua từng phiên (nói 80% thì phải trúng xấp xỉ 80%).

Chủ dự án đã chọn "Cách 1": chỉ mô hình HAR, **không** train mạng sâu mới.

## 2. Mục tiêu, ngoài phạm vi, điều phải tránh

Mục tiêu
- Một mô hình dải giá chạy được hằng ngày cho 15 mã của `ai_core.prices.config.TICKERS`, h = 2 và 5.
- Một báo cáo đánh giá trên 6 fold sẵn có (A, B, C, D, E, Fb): độ phủ, độ rộng dải, so với hai mốc đơn giản.
- Web StockSenseVN **thôi hiển thị Tăng / Đi ngang / Giảm** (thứ mô hình chưa làm được) và thay bằng thứ mô hình làm
  được: mức biến động dự kiến và dải giá, ở mọi chỗ web đang hiện dự báo; cộng gợi ý điểm cắt lỗ trong Trade Plan.
  (Chủ dự án yêu cầu ngày 2026-11-13, thay cho quyết định ban đầu là giữ nhãn cũ.)

Ngoài phạm vi
- Mạng sâu, nhánh Fusion train trên đích mới, tin tức làm đầu vào của phép tính (tin vẫn hiển thị bên cạnh như hiện nay).
- Ứng dụng mobile và báo cáo PDF của StockSenseVN: không sửa trong đợt này. Chúng đọc cùng API, nên API giữ các trường
  cũ (`label`, `probabilities`, `confidence`) với giá trị null thay vì xóa; hai phần đó sẽ hiện "chưa có dự báo" cho tới
  khi nhóm web cập nhật. Rủi ro này phải báo cho nhóm web.
- Mô hình hướng giá và bảng `fusion_decision`: vẫn chạy nội bộ (bảng Streamlit, shadow), chỉ không còn được đẩy sang web.
- Holdout của mmf, `docs/reports/mmf_lock.json`, bundle volcast đang chạy live.

Điều phải tránh (làm được mà vẫn hỏng)
- Trình bày dải giá như giá mục tiêu hay khuyến nghị mua bán.
- Gọi kết quả là đã kiểm chứng ngoài mẫu: cả 6 fold là vùng phát triển đã bị nhìn nhiều lần.
- Thử nhiều biến thể rồi chọn cái đẹp nhất. Cấu hình ở mục 4 chạy đúng một lần.

## 3. Ràng buộc

- Hai ngày. Thứ tự làm: mô hình và đánh giá, đường chạy live, cầu nối sang web, giao diện, báo cáo. Hết giờ ở đâu thì
  dừng ở đó; mỗi bước xong là dùng được.
- Kiến trúc một chiều của `ai_core` (`tests/test_architecture.py`): `models` được import `prices`; `fusion` nằm trên cùng.
  Cron dùng `.venv` (không có torch); mô hình này chỉ cần numpy, pandas, scikit-learn.
- StockSenseVN là repo của nhóm khác (chủ dự án cho phép sửa): thêm bảng mới và trường mới, **không đổi nghĩa** cột hay
  trường cũ. Nhãn up/hold/down không bị tái sử dụng để chở ý nghĩa khác; chúng chỉ thôi được ghi và thôi được hiển thị.
  Dự báo đã ghi là bất biến (SDD 6.3 của họ); các batch `decision-v1-trend` cũ giữ nguyên trong cơ sở dữ liệu.
- Giá của AI_Core tính bằng nghìn đồng và đã điều chỉnh theo nhà cung cấp; giá của web là của SSI. Vì vậy cầu nối chỉ gửi
  **tỉ lệ so với giá đóng cửa** (ví dụ −3.1% / +3.4%), web tự nhân với giá đóng cửa của chính nó.
- Không commit ở cả hai repo cho đến khi chủ dự án bảo (yêu cầu đang có hiệu lực).

## 4. Mô hình (một cấu hình, định trước khi chạy)

Đích: r = ln(close[t+h] / close[t]) của từng mã. Hàng hợp lệ dùng đúng luật `status == "ok"` của nhãn mag (bỏ phiên thiếu,
bỏ cửa sổ có bước nhảy > 7.2% nghi đổi chuẩn điều chỉnh giá).

Bước 1, mức biến động dự kiến σ̂:
- Đặc trưng: 7 đại lượng HAR của `ai_core.models.mmf.har._raw_har` (bình phương lợi suất, biên độ Parkinson ở thang 1, 5,
  22 phiên, và khoảng trống mở cửa 5 phiên), dạng **thô**, không trừ trung bình phiên (đích ở đây là mức tuyệt đối, không
  phải thứ hạng giữa các mã), cộng trung bình cắt ngang của 7 đại lượng đó trong cùng phiên (mức chung của thị trường).
- Hồi quy tuyến tính bình phương tối thiểu, gộp mọi mã, riêng cho từng h, với đích ln(Σ lợi suất ngày² trong t+1..t+h + ε).
  σ̂ = sqrt(exp(dự báo)).

Bước 2, dải giá:
- Trên khối calib, tính z = r / σ̂ và lấy phân vị 10% và 90% thực nghiệm của z (gộp mọi mã). Cách này không giả định phân
  phối chuẩn và giữ được đuôi lệch.
- Dải của phiên t: [close·exp(z10·σ̂), close·exp(z90·σ̂)], rồi cắt trong biên độ sàn/trần ±7% mỗi phiên cộng dồn h phiên.

Mức rủi ro: vị trí của σ̂ hôm nay trong 250 giá trị σ̂ gần nhất của chính mã đó: từ phân vị 70 trở lên là "cao", từ 30
trở xuống là "thấp", còn lại "bình thường".

Dữ liệu và fold:
- Snapshot lịch sử dài `data/prices_2018/snapshots/a0674e0c56bf9013` (train từ 2019). Lý do: dải giá phải học được các
  giai đoạn đuôi (3/2020, 2022), và mô hình chỉ cần giá.
- 6 fold của `C.DATE_FOLDS` và `C.BACK_FOLDS`. Fit trên train + tune, phân vị trên calib, chấm trên eval, mỗi fold một lần.

## 5. Đánh giá (luật ghi trước khi chạy)

Hai mốc so sánh, cùng thủ tục phân vị trên calib:
- `naive20`: σ = độ lệch chuẩn lợi suất 20 phiên × √h.
- `hist`: dải cố định theo phân vị 10/90 của r trong train của từng mã (không có σ̂).

Chỉ số trên eval, gộp 6 fold và theo từng fold, từng h:
- Độ phủ thực tế của dải 80%, và tỉ lệ rơi dưới / rơi trên dải.
- Độ rộng dải trung bình (% giá).
- Pinball loss ở hai phân vị 10% và 90% (thước đo đúng cho dự báo phân vị); chênh lệch ghép cặp so với từng mốc với
  block bootstrap 20 phiên của `evaluate.paired_diff`.
- Độ phủ theo từng mã và theo ba mức rủi ro (dải "cao" phải rộng hơn và vẫn phủ ~80%).

Cách đọc, định trước:
- "Dùng được": độ phủ gộp nằm trong 77–83% ở cả hai h **và** pinball thấp hơn `naive20` với CI loại 0.
- "Dùng được, có lưu ý": độ phủ đạt nhưng không hơn `naive20` rõ ràng. Web vẫn hiển thị dải, báo cáo ghi rõ HAR không
  hơn mốc đơn giản.
- "Không đạt": độ phủ gộp ngoài 75–85%. Khi đó không đẩy dải sang web; báo cáo ghi kết quả âm.

Luật này được chép thành phụ lục mới ở cuối `docs/reports/mmf_preregistration.md` trước khi chạy bước chấm. Đây là đích
mới (dải giá), không phải biến thể của R7/R8/R9; các quy tắc dừng của chúng không bị đụng tới.

Báo cáo: `docs/reports/rangecast_report.md`.

## 6. Đường chạy live và web

AI_Core
- Module mới dưới `ai_core/models/` (tên gợi ý `rangecast`): `eval` (mục 5), `train` (fit trên mọi hàng đã có nhãn của kho
  giá hiện hành, lưu bundle joblib dưới `data/`), `predict` (ghi bảng SQLite `range_forecast` trong `nlp.db`: symbol,
  base_date, h, close, lo_pct, hi_pct, sigma, risk_level, model_version, created_at).
- Bundle live không lấy phân vị từ chính dữ liệu đã fit: hồi quy fit trên mọi hàng đã có nhãn **trừ 126 phiên gần nhất**,
  hai phân vị của z lấy trên 126 phiên đó (cùng độ dài calib + eval của một fold).
- `predict` đọc kho giá `data/prices` (parquet, có đủ open/high/low mà HAR cần), tức chạy sau lượt sync 15:20. Nếu phiên
  hôm nay chưa có, nó dự báo cho phiên mới nhất đã lưu và `base_date` ghi đúng phiên đó; không dựng bar từ quote SSI.
- `jobs/live_update.sh`: thêm `rangecast predict` sau `volcast predict`, cùng cửa sổ 15:00–16:00, lỗi thì chạy tiếp.
- `ai_core/fusion/stocksense.py`: đẩy `range_forecast` thay cho `fusion_decision`. Mỗi phiên một dòng `market_batches`
  (model_version của rangecast, state `experimental`) và các dòng `forecast_ranges`; cùng quy ước bất biến
  (`ON CONFLICT DO NOTHING`), chỉ các mã `analysis_enabled`, batch chỉ được tạo khi có ít nhất một dòng. Không còn ghi
  vào `forecasts` và `market_summary.labels_*`.

StockSenseVN
- Migration 025: bảng `forecast_ranges` (batch_id, symbol, horizon 't2'|'t5', low_pct, high_pct, coverage, risk_level
  'low'|'normal'|'high', created_at; duy nhất theo batch_id, symbol, horizon), và ba cột đếm mức biến động trong
  `market_summary`.
- API (`apps/api/src/market/repository.js`): `forecasts.t2` / `forecasts.t5` lấy từ `forecast_ranges` của batch mới nhất
  (giữ luật "dự báo mới nhất trong 7 ngày, gắn cờ stale"), trả `riskLevel`, `range {lowPct, highPct, coverage}`,
  `dataDate`, `stale`; các trường cũ trả null. Cập nhật `openapi.yaml`.
- Web, mọi chỗ hiện dự báo (`packages/core/src/labels/forecast.js`, `components/forecast.jsx`, `ForecastBadge.jsx`,
  trang mã, tổng quan thị trường, `AppShell`, trang Chat): huy hiệu Tăng/Đi ngang/Giảm và thanh ba xác suất được thay bằng
  huy hiệu **Biến động cao / bình thường / thấp** và dòng dải giá. Tổng quan thị trường đếm số mã theo ba mức biến động.
  Ghi chú thử nghiệm đổi thành: "Mô hình thử nghiệm: dự báo mức biến động và dải giá, không dự báo chiều tăng giảm".
- Trợ lý Chat (`apps/api/src/chat/assistant.js`): câu trả lời về dự báo dùng mức biến động và dải giá, không nói tăng giảm.
- Web, Trade Plan (`PlansPage.jsx`): khi đã chọn mã, dưới ô "Cắt lỗ" hiện dải T+5 và một nút điền nhanh cận dưới (lệnh
  mua) hoặc cận trên (lệnh bán) vào ô cắt lỗ. Người dùng vẫn sửa tay được.
- Tin tức và cảm xúc tin (phần NLP đang chạy tốt) giữ nguyên.
- Dòng bối cảnh tin trên mỗi thẻ dự báo (chủ dự án duyệt 2026-11-13): `range_forecast` và `forecast_ranges` mang thêm
  `has_news`, `sentiment_24h` (null nếu không có tin 24 giờ) và `risk_flags` của cùng ảnh chụp tin mà bảng live đang dùng.
  Web ghi "Tin 24h: tích cực / tiêu cực / trung tính (điểm)", so với biến động giá của chính phiên đó là "cùng chiều" hay
  "trái chiều", hoặc "Không có tin trong 72 giờ"; cờ rủi ro hiện thành cảnh báo. Câu chữ chỉ mô tả sự trùng hợp, không
  viết quan hệ nhân quả. Tin không tham gia phép tính dải giá.
- Khối chỉ số tài chính nền trên trang mã (E/P, ROE, D/E, tăng trưởng EPS và doanh thu, từ `mmf.fundamentals`): làm sau
  cùng, chỉ khi các phần trên đã xong; cần thêm một bảng Postgres nhỏ và một trường API.

Nếu kết quả đánh giá là "Không đạt" (mục 5): web vẫn bỏ Tăng/Giảm, nhưng huy hiệu lấy từ volcast đang chạy
(**Biến động mạnh / yếu so với rổ**, chỉ khi volcast dám gọi) và không có dải giá. Phần này chỉ dựng khi cần.

Phác thảo trang mã:

```
FPT   122.5
┌ T+2 ─────────────────────────┐  ┌ T+5 ─────────────────────────┐
│ ◔ Biến động thấp              │  │ ◕ Biến động cao               │
│ Dải giá 80%: 119.6 – 125.1    │  │ Dải giá 80%: 117.9 – 126.9    │
│ (−2.4% / +2.1%)               │  │ (−3.8% / +3.6%)               │
└───────────────────────────────┘  └───────────────────────────────┘
Mô hình thử nghiệm: dự báo mức biến động và dải giá, không dự báo chiều tăng giảm.
Dải giá: 8/10 lần giá đóng cửa sau h phiên được kỳ vọng nằm trong dải. Không phải giá mục tiêu.
```

## 7. Các cách đã cân nhắc

1. **HAR → dải giá (đã chọn).** Nhanh, khớp mọi kết quả đã đo. Mất: không phải mô hình sâu đa phương thức.
2. Cách 1 cộng một nhánh Fusion train trên cùng đích để so sánh. Giữ mạch đa phương thức trong báo cáo, nhưng thêm nửa
   ngày đến một ngày và ép thời gian phần web. Chủ dự án không chọn.
3. Chỉ trình bày lại nhãn mạnh/yếu của volcast. Vài giờ là xong nhưng không có con số để hành động.

Các quyết định của tôi (chủ dự án sửa được):
1. Huy hiệu chính trên web là mức biến động của rangecast (so với thường lệ của chính mã), không phải nhãn mạnh/yếu so
   với rổ của volcast. Lý do: một mô hình, một câu chuyện, và nó đi cùng dải giá. volcast tiếp tục chạy shadow.
2. Dải 80% (phân vị 10–90), một mức duy nhất.
3. Train từ 2019 trên snapshot lịch sử dài.
4. Bảng Postgres mới thay vì thêm cột vào `forecasts`; API giữ trường cũ ở giá trị null để mobile và PDF không vỡ.
5. Chỉ 15 mã đã được đánh giá, không mở rộng sang 5 mã còn lại của bảng quyết định.
6. Mức rủi ro theo phân vị 30/70 của chính mã trong 250 phiên.

## 8. Để người dựng tự quyết

- Tên module, tên hàm, cách chia file; ε của logarit; dạng bundle.
- Cách vẽ dải trên giao diện (chữ là đủ; đường trên biểu đồ là tùy chọn nếu còn giờ).
- Test: ít nhất một test "sửa giá tương lai không đổi đặc trưng và dải của quá khứ", một test độ phủ trên dữ liệu tổng hợp,
  test cầu nối theo mẫu `tests/fusion/test_stocksense.py`.

## 9. Kiểm tra khi xong

- Hai lệnh pytest của `CLAUDE.md` vẫn qua, cộng test mới.
- `rangecast predict` chạy trên kho giá hiện hành và ghi 30 dòng (15 mã × 2 h).
- Web (cổng 3100): trang mã, tổng quan thị trường và Chat không còn chữ Tăng/Đi ngang/Giảm của dự báo, hiện mức biến
  động và dải giá cho một mã thật; Trade Plan điền được điểm cắt lỗ từ dải. Test của StockSenseVN (vitest) vẫn qua.
