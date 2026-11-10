# Thử nghiệm nhanh với model `mmf`

Chạy mọi lệnh từ gốc `AI_Core`, bằng `.venv_mmf/bin/python`. Dữ liệu đóng băng dùng cho vùng phát triển: `Pipeline_V1.2/stock/snapshots/24251acc1a5695bd`.

## Vòng lặp một thí nghiệm

```bash
SNAP=Pipeline_V1.2/stock/snapshots/24251acc1a5695bd
# 1. Chạy: thư mục riêng data/mmf/exp/<tên>/, không đụng run đã khóa
.venv_mmf/bin/python -m mmf.experiment --name thu1 --stock-dir $SNAP --h 5 --arms dl_price fusion_concat no_text --seeds 2
# 2. Báo cáo riêng, không ghi đè báo cáo chính
.venv_mmf/bin/python -m mmf.compare --run-dir data/mmf/exp/thu1/24251acc1a5695bd --out data/mmf/exp/thu1/report.md
```

- Baseline luôn chạy, nên mọi thí nghiệm luôn có đối chứng.
- `--arms`, `--seeds N` (N seed đầu) và `--h` giúp rút ngắn thời gian. Lượt chạy đầy đủ (2 h × 3 fold × 8 arm × 5 seed) mất khoảng 7 phút trên GPU.
- `--smoke` (1 epoch, 1 fold) dùng để kiểm tra code có chạy hay không, không dùng để đọc kết quả.
- Sau khi sửa code, đổi `--name` hoặc xóa thư mục cũ. Bộ chạy từ chối dùng lại file nếu dữ liệu hoặc cấu hình khác, nhưng nếu chỉ khác commit thì nó chỉ in cảnh báo.

## Sửa ở đâu

| Muốn thử | File |
|---|---|
| Đặc trưng giá, thị trường | `Pipeline_V1.2/src/price_features.py` (dùng chung với Pipeline, chạy lại test của nó) |
| Chỉ số tài chính | `mmf/fundamentals.py` (`FUND_FEATS`) |
| Đặc trưng tin tức | `mmf/text.py` (`TEXT_FEATS`) |
| Kiến trúc nhánh, cách hợp nhất | `mmf/models.py` |
| Thêm một arm (tập nhánh + kiểu fusion) | `ARMS` trong `mmf/config.py` |
| Thêm một baseline | `BASELINES` trong `mmf/config.py` và `mmf/baselines.py` |
| Siêu tham số (D, LR, LP, LT...) | `mmf/config.py` |

Chạy `.venv_mmf/bin/python -m pytest tests -q` sau mỗi thay đổi, vì các test chặn rò rỉ dữ liệu nằm ở đó. Đặc trưng mới phải chỉ dùng dữ liệu đến phiên t. Test `test_future_change_does_not_touch_past_features` sẽ báo lỗi nếu không.

Nghi ngờ rò rỉ (ví dụ AUC Tăng/Giảm ≥ 0.65 hoặc Macro-F1 ≥ 0.50 trên toàn mẫu eval) thì chạy: `.venv_mmf/bin/python -m mmf.leak_check --stock-dir $SNAP`.

## Ba điều không được làm

1. **Không chạy `mmf.holdout`** để thử. Mỗi h chỉ chạy được một lần, cho cấu hình đã khóa trong `docs/reports/mmf_lock.json`.
2. **Không ghi đè** `docs/reports/mmf_lock.json`, `mmf_dev_report.md` hay `mmf_preregistration.md`.
3. **Không gọi kết quả thử nghiệm là kết quả cuối.** Ba cửa sổ eval A, B, C đã bị nhìn nhiều lần. Một cấu hình mới chỉ được coi là đã kiểm chứng khi nó được khóa và đăng ký trong `mmf_preregistration.md` trước khi chạy vùng chấm cuối, tức là trước khi T+2 đủ 40 phiên, khoảng cuối tháng 12/2026. Hiện `mmf.holdout` chỉ đọc file khóa v1, nên muốn chấm thêm một phiên bản v2 trên cùng vùng thì cần sửa code trước mốc đó.
