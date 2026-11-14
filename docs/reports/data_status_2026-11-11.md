# Báo cáo dữ liệu và chạy model, 2026-11-11 (cập nhật lúc 16:15)

Phạm vi: lấy dữ liệu mới nhất có thể và chạy các model trên đó. Holdout mmf không được chạy (xem mục 3.3).

## 1. Tóm tắt

- **Giá đã có phiên 2026-11-11 cho cả 20 mã** (1459 phiên mỗi mã), nhưng sau một sự cố. Cron sync 15:20 thất bại ở cả 20 mã vì `vnstock 6.0.0rc4` bị nhà cung cấp tắt từ 11/11/2026 (`VNSTOCK_RELEASE_DISABLED`). Giá được ghi lúc 15:51–15:52 bởi một lần sync khác, không phải của tôi (xem mục 5).
- **Tin, quote SSI, Finhay đều mới.** Tin đến 14:16 hôm nay, quote SSI lúc 15:00, netflows Finhay đến 11/11.
- **volcast v2.2 đã chạy trên dữ liệu mới nhất** (giá đã sync, lúc 16:12). Kết quả giống hệt bản dự báo 15:45 dựng từ bar quote SSI, nên bar quote và bar giá đồng nhất ở mức độ mà mô hình nhìn thấy.
- **Bảng quyết định phiên 11/11 chỉ có được sau khi tôi chạy tay** `volcast predict` và `ai_core.fusion.live` lúc 16:12. Cron 16:00 đã bỏ qua bước này (xem mục 5). Kết quả: 40 dòng, toàn `NO_TRADE`.
- **mmf không chạy trên dữ liệu mới.** Cách duy nhất là holdout, chỉ dùng một lần, do chủ dự án quyết định. Hiện có 8 phiên sau 30/10, cần 40 phiên đã trưởng thành.

## 2. Độ mới của dữ liệu

| Nguồn | Mới nhất | Ghi chú |
|---|---|---|
| Giá (20 mã) | 2026-11-11 (1459 phiên) | Ghi lúc 15:51–15:52 sau khi sync 15:20 lỗi |
| Tin (`nlp.db`) | 17,566+ tài liệu, bản mới nhất 2026-11-11 14:16 | Mới lưu: 24 bài ngày 10/11, 15 bài ngày 11/11 |
| Quote SSI | 2026-11-11 15:00 (675 dòng trong ngày) | Dùng cho dự báo live |
| Finhay netflows | đến 2026-11-11 (2565 dòng, từ 2026-07-28) | cron 15:40 chạy tốt |
| Finhay interbank | đến 2026-11-10 (4457 dòng) | |
| BCTC năm | 2019–2025 | Không đổi, không có BCTC 2026 |

Tôi chưa kiểm tra độ mới của chuỗi toàn cầu (S&P 500, VIX, Brent, USD/VND) vì mmf v1 không dùng chúng.

## 3. Model chạy trên dữ liệu mới

### 3.1 volcast v2.2, dự báo phiên 11/11

`p_strong` là xác suất mã có biên độ dịch chuyển |lợi suất h phiên| lớn hơn trung vị cắt ngang trong ngày. `call` chỉ có khi vượt ngưỡng bỏ phiếu, còn lại là bỏ qua (abstain). Đây không phải dự báo hướng giá.

| Mã | h=2 p_strong | h=2 call | h=5 p_strong | h=5 call |
|---|---|---|---|---|
| VRE | 0.579 | strong | 0.630 | strong |
| SSI | 0.535 | – | 0.575 | – |
| VIC | 0.542 | – | 0.550 | – |
| VPB | 0.482 | – | 0.510 | – |
| STB | 0.513 | – | 0.489 | – |
| HPG | 0.484 | – | 0.485 | – |
| MSN | 0.460 | – | 0.479 | – |
| FPT | 0.487 | – | 0.458 | – |
| MWG | 0.380 | weak | 0.449 | – |
| TCB | 0.415 | weak | 0.434 | – |
| GAS | 0.494 | – | 0.414 | – |
| ACB | 0.341 | weak | 0.369 | weak |
| VCB | 0.243 | weak | 0.325 | weak |
| MBB | 0.291 | weak | 0.321 | weak |
| VNM | 0.259 | weak | 0.229 | weak |

- Tổng: h=2 có 1 strong, 6 weak, 8 bỏ qua; h=5 có 1 strong, 4 weak, 10 bỏ qua.
- Phiên bản v2.2 hiện có 2 ngày dự báo (10/11 và 11/11). Shadow report cần ít nhất 40 phiên đếm riêng cho từng version, nên còn xa.
- Cron crosscheck tcbs vẫn bị bỏ qua vì thiếu `TCBS_API_KEY`, nên chưa có đối chiếu giá với nguồn thứ hai.

### 3.2 Bảng quyết định (`decision-v0-nogo`)

- Phiên 11/11: 40 dòng, tất cả `NO_TRADE`, lý do "no direction model (NO-GO)". 30/40 dòng có tin. 4 dòng mang cờ rủi ro (2 `prosecution`, 2 `sanction`). Tổng lịch sử: 10 phiên, tất cả `NO_TRADE`.
- Thí nghiệm hướng giá vẫn NO-GO ở cả h=2 và h=5 (`direction_report.md`), nên bảng sẽ không phát tín hiệu giao dịch.

### 3.3 mmf: vì sao không chạy

- Dữ liệu sau 30/10 có 8 phiên: 02/11 đến 11/11. Số phiên đã trưởng thành là 6 cho h=2 và 3 cho h=5. Luật cần 40 mỗi h (`--early` cần 20).
- Holdout chạy một lần cho mỗi h và chỉ do chủ dự án quyết định (`docs/mmf_thu_nghiem.md`). Tôi không chạy nó.
- Ước tính của tôi, giả sử không có ngày nghỉ bất thường và sync giá không đứt thêm: 40 phiên trưởng thành đạt khoảng **2026-12-29** cho h=2 và **2027-01-04** cho h=5. Sự cố hôm nay không làm mất phiên nào của kho giá, vì sync lấy bù từ 2026-10-14.
- mmf không có đường suy luận riêng cho dữ liệu mới ngoài holdout. Tôi không tạo ra một đường như vậy.

## 4. Kết quả mmf trên vùng phát triển (trước và sau calibration, CI theo fold)

Đây là vùng phát triển (fold A, B, C đã bị nhìn nhiều lần), không phải kết quả cuối. Chi tiết nằm ở `data/mmf/calibration_ci/` (bị gitignore): `metrics.csv`, `ci_per_fold.csv`, `temps.parquet`, `before.parquet`, `after.parquet`.

- Kết luận gộp không đổi (`mmf_dev_report.md`): fusion không chứng minh được hơn baseline mạnh nhất hay nhánh đơn tốt nhất. Ở T+5, bỏ nhánh text làm tốt hơn (+0.0082, CI +0.0037 đến +0.0121).
- Calibration (nhiệt độ T cộng alpha) giúp rõ nhất cho fusion ở T+5 (log-loss trung bình `fusion_concat` 1.1076 → 1.0874, do fold A quá tự tin với T≈3.5). Ở T+2 fold A, calibration làm xấu đi +0.009 đến +0.012 (CI chứa 0). Tôi không đổi quy trình đã đăng ký dựa trên số này.
- CI theo fold của "fusion trừ baseline tốt nhất": cả 6 ô đều chứa 0. Ở T+5 cả ba fold nghiêng về baseline (+0.012 đến +0.015).
- Nhánh text ở T+5: `fusion_concat` thua bản bỏ text ở cả ba fold, và CI loại trừ 0 ở fold A và B.
- Hạn chế: mỗi fold chỉ khoảng 3 khối bootstrap 20 phiên, 15 mã biến động cùng nhau, và CI chưa hiệu chỉnh cho việc so sánh nhiều ô. Chưa tính CI theo fold cho price-v2.

## 5. Vấn đề cần xử lý

1. **Lịch sử sự cố sync giá.** Lúc 15:20 cả 20 mã thất bại với `VNSTOCK_RELEASE_DISABLED`. Sau đó `requirements/vnstock6.txt`, `ai_core/prices/sync.py`, `ai_core/prices/config.py`, `tests/prices/test_sync.py` và `CLAUDE.md` bị sửa (chưa commit, không phải do tôi). Theo CLAUDE.md mới, sync giờ chạy trên `.venv` (vnstock 4.x, nguồn kbs, không cần key hay OTP), còn `.venv_6x` (vnstock 6.0.0, tcbs) dùng cho crosscheck. Crontab đã trỏ sync về `.venv`. CLAUDE.md cũng cảnh báo 4.0.9 có thể bị cắt tiếp.
   - Tôi đã thử chạy `jobs/sync_prices.py` bằng `~/.venv_6x` lúc 15:47. Nó bị kẹt ở mã đầu tiên (ACB, tcbs cần keyring và OTP) cho đến khi timeout 500 giây. Nó không ghi file giá nào, không còn tiến trình thừa.
   - Giá 11/11 trong kho do lần sync lúc 15:51–15:52 ghi, tôi chưa kiểm tra chi tiết lần đó. Tôi đã xác nhận cả 20 file có 1459 phiên đến 11/11 và dự báo volcast từ hai nguồn bar khớp nhau.
2. **Log sync gây hiểu nhầm.** Khi cả 3 lần gọi lỗi, `sync_prices` ghi `ERROR` rồi in "Không có bản ghi mới từ vnstock. Giữ nguyên ... phiên" và kết thúc bình thường. Cron không báo động. Nên cho job thoát khác 0 hoặc ghi cảnh báo rõ khi không lấy được dữ liệu của mã nào.
3. **Cron live bỏ lỡ cửa sổ predict/board khi sync trễ.** `jobs/live_update.sh` tính `hm` sau bước tin (khoảng 70 giây), nên lượt 16:00:01 đến bước predict lúc 16:01 và bỏ qua vì `hm > 1600`. Lượt 15:45 chạy trước khi giá đồng bộ xong. Kết quả là phiên 11/11 không có dòng board nếu không chạy tay. Cách sửa gọn: tính `hm` một lần ở đầu script, hoặc mở rộng cửa sổ predict thêm 15–30 phút.
4. **Cron crosscheck tcbs** thiếu `TCBS_API_KEY` (`vnstock-oapi keys set tcbs`), nên đối chiếu giá đang bị bỏ qua.
5. **Lỗi nhỏ trong `ai_core.news.cli quotes`:** `--model` mặc định trỏ tới `data/models/baseline.joblib` (đã không tồn tại sau restructure), gây traceback mỗi lượt (`live.log` có 35 traceback). Quote vẫn được lưu nhưng bước `sentiment_vs_price` bị bỏ qua. Cách sửa: đổi mặc định ở `ai_core/news/cli.py:73` sang `data/news/models/baseline.joblib`.
6. Finhay: khóa API vẫn chưa được xoay. Netflows và interbank vẫn chưa nối vào features.
