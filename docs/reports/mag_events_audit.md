# Kiểm toán lịch sự kiện cho R9 (trước khi chạy bất kỳ arm nào)

Ngày: 2026-11-12. Spec: `docs/superpowers/specs/2026-11-12-mag-har-events-design.md`. Không có nhãn nào được đọc trong bước này.

## File cố định

- `data/prices/events/2026-11-12.parquet`, tải bằng `python -m ai_core.prices.events` (vnstock 4, nguồn `vci`, mã `DIV, ISS, AGME, AGMR, EGME`, từ 2020-01-01).
- sha256 `6f7a7f21ce1e4b2feb81741c69069642e3a7f2c17bfeb8132e7ecf92cdc5863b`.
- 330 sự kiện: ISS 133, AGME 106, DIV 84, EGME 7, **AGMR 0** (nguồn không trả loại này). Theo mã: FPT 40, SSI 34, VNM 29, MBB 26, MWG 26, TCB 21, VPB 21, VCB 23, MSN 22, ACB 19, GAS 19, VIC 18, HPG 17, VRE 8, STB 7.
- Lần tải đầu bị 502 từ VCI; đã thêm thử lại (3 lần) vào `fetch`.

## Nghĩa các trường

- `AGME`, `EGME`: `display_date1` là **ngày họp**: các cuộc họp thường niên của FPT rơi vào 06–16/4, cách `public_date` 46–56 ngày (trung vị 51). `record_date` và `exright_date` là ngày chốt danh sách. Khớp với spec.
- `DIV`, `ISS`: `display_date1` luôn bằng `exright_date` (193/193). `exright_date` cách `public_date` trung vị 7 ngày, khớp nhịp thông báo VSDC khoảng 8 ngày trước ngày đăng ký cuối cùng.
- 18% sự kiện `ISS` không có `exright_date` (ESOP, phát hành riêng lẻ): chỉ được tính là thông báo (`ev_new5`).
- 39 sự kiện `DIV`/`ISS` có `public_date ≥ exright_date` (ví dụ FPT ISS: công bố 2026-06-29, GDKHQ 2026-06-24). Theo định nghĩa, chúng không bao giờ bật cờ trong cửa sổ, nên không gây rò rỉ. Điều này gợi ý `public_date` của VCI có thể là ngày cập nhật, không luôn là ngày công bố đầu tiên.

## Ngày công bố: VCI so với nguồn khác

Đối chiếu tự động: với mỗi sự kiện từ 2022, lấy bài sớm nhất trong `nlp.db` có mã trong tiêu đề, có từ khóa của loại sự kiện, và đăng trong khoảng [public_date − 30 ngày, record_date]. Có 49 sự kiện khớp, thuộc 8 mã (ACB, FPT, GAS, HPG, MSN, MWG, VIC, VNM). Bảng đầy đủ nằm trong workspace của plan.

| Chênh (ngày bài − public_date VCI) | Giá trị |
|---|---|
| trung vị | −4 |
| tứ phân vị 25% / 75% | −18 / +1 |
| min / max | −28 / +12 |
| số ca bài có trước VCI | 35/49 |

Ví dụ khớp đúng nội dung:

| Mã | Loại | public_date VCI | Bài đầu tiên | Chênh |
|---|---|---|---|---|
| HPG | DIV | 2022-06-10 | 2022-06-03 "HPG chốt quyền trả cổ tức tổng tỷ lệ 35%" | −7 |
| FPT | DIV | 2024-06-07 | 2024-05-20 "Cổ đông FPT sắp nhận 10% cổ tức tiền mặt…" | −18 |
| MWG | DIV | 2026-07-22 | 2026-07-16 "MWG chốt ngày chia cổ tức tiền mặt…" | −6 |
| HPG | DIV | 2026-05-05 | 2026-04-29 "Cổ đông HPG sắp nhận cổ tức tiền mặt…" | −6 |
| FPT | ISS | 2026-09-16 | 2026-09-10 "FPT sắp phát hành 171 triệu cổ phiếu thưởng" | −6 |
| ACB | AGME | 2026-02-12 | 2026-02-07 "ACB chuẩn bị họp đại hội cổ đông…" | −5 |
| ACB | DIV | 2026-06-05 | 2026-06-04 "ACB chốt quyền chia cổ tức năm 2025…" | −1 |
| ACB | DIV | 2024-05-24 | 2024-05-24 | 0 |
| FPT | AGME | 2026-02-05 | 2026-02-07 "FPT chốt ngày tổ chức ĐHĐCĐ thường niên năm 2026" | +2 |
| ACB | AGME | 2022-02-15 | 2022-02-18 "ACB chuẩn bị họp cổ đông…" | +3 |
| VNM | DIV | 2022-06-24 | 2022-07-03 (bài tổng hợp tuần) | +9 |

Thêm một mốc ngoài: theo đề xuất của chủ dự án, VSDC công bố ngày đăng ký cuối cùng 18/03/2026 cho ĐHĐCĐ của VNM ngày 25/02/2026. VCI ghi `public_date` 2026-02-27, tức muộn hơn 2 ngày.

**Kết luận cổng rò rỉ: qua.** Không có ca nào cho thấy `public_date` của VCI sớm hơn ngày thông tin thật sự công khai. Ngược lại, VCI thường ghi muộn hơn tin vài ngày, nên đặc trưng sự kiện thấy thông tin muộn hơn thị trường (thận trọng: thiệt về lực, không rò rỉ). Các ca chênh dương là bài viết sau thông báo, điều bình thường. Một số khớp tự động là khớp nhầm (ví dụ "FPT và NVIDIA hợp tác phát hành tập dữ liệu", "FPT Retail"), không ảnh hưởng kết luận.

Giới hạn còn lại, không kiểm soát được: sự kiện bị hủy có thể đã bị xóa khỏi nguồn, và ngày có thể đã bị sửa lùi. Ảnh chụp hằng ngày (Task 6) xử lý điều này cho dữ liệu tiến.

## Độ phủ (phiên 2022-01-01 đến 2026-10-30, 15 mã, 18 015 hàng mã-phiên)

| h | `ev_exr_in_h` = 1 | `ev_mtg_in_h` = 1 | có cờ bất kỳ | `ev_next` < 1 | `ev_new5` > 0 |
|--:|--:|--:|--:|--:|--:|
| 2 | 0.87% | 0.91% | 1.77% | 11.0% | 5.5% |
| 5 | 1.74% | 2.28% | 3.99% | 11.0% | 5.5% |

Độ phủ thấp hơn ước đoán trong spec (4% và 9%). Mỗi khối eval khoảng 60 phiên × 15 mã ≈ 900 hàng, nên chỉ có khoảng 16 (T+2) đến 36 (T+5) hàng có cờ trong một fold. **R9a có thể thiếu lực**; kết luận "Chưa chứng minh được" vì thế không đồng nghĩa với "không có tác động".
