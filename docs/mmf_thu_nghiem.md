# Thử nghiệm nhanh với model `mmf`

Chạy mọi lệnh từ gốc `AI_Core`, bằng `.venv_mmf/bin/python`. Dữ liệu đóng băng dùng cho vùng phát triển: `data/prices/snapshots/24251acc1a5695bd`.

## Vòng lặp một thí nghiệm

```bash
SNAP=data/prices/snapshots/24251acc1a5695bd
# 1. Chạy: thư mục riêng data/mmf/exp/<tên>/, không đụng run đã khóa
.venv_mmf/bin/python -m ai_core.models.mmf.experiment --name thu1 --stock-dir $SNAP --h 5 --arms dl_price fusion_concat no_text --seeds 2
# 2. Báo cáo riêng, không ghi đè báo cáo chính
.venv_mmf/bin/python -m ai_core.models.mmf.compare --run-dir data/mmf/exp/thu1/24251acc1a5695bd --out data/mmf/exp/thu1/report.md
```

- Baseline luôn chạy, nên mọi thí nghiệm luôn có đối chứng.
- `--arms`, `--seeds N` (N seed đầu) và `--h` giúp rút ngắn thời gian. Lượt chạy đầy đủ (2 h × 3 fold × 8 arm × 5 seed) mất khoảng 7 phút trên GPU.
- `--smoke` (1 epoch, 1 fold) dùng để kiểm tra code có chạy hay không, không dùng để đọc kết quả.
- Sau khi sửa code, đổi `--name` hoặc xóa thư mục cũ. Bộ chạy từ chối dùng lại file nếu dữ liệu hoặc cấu hình khác, nhưng nếu chỉ khác commit thì nó chỉ in cảnh báo.

## Sửa ở đâu

| Muốn thử | File |
|---|---|
| Đặc trưng giá, thị trường | `ai_core/prices/price_features.py` (dùng chung với Pipeline, chạy lại test của nó) |
| Chỉ số tài chính | `mmf/fundamentals.py` (`FUND_FEATS`) |
| Đặc trưng tin tức | `mmf/text.py` (`TEXT_FEATS`) |
| Kiến trúc nhánh, cách hợp nhất | `mmf/models.py` |
| Thêm một arm (tập nhánh + kiểu fusion) | `ARMS` trong `mmf/config.py` |
| Thêm một baseline | `BASELINES` trong `mmf/config.py` và `mmf/baselines.py` |
| Siêu tham số (D, LR, LP, LT...) | `mmf/config.py` |

Chạy `.venv_mmf/bin/python -m pytest tests -q` sau mỗi thay đổi, vì các test chặn rò rỉ dữ liệu nằm ở đó. Đặc trưng mới phải chỉ dùng dữ liệu đến phiên t. Test `test_future_change_does_not_touch_past_features` sẽ báo lỗi nếu không.

Nghi ngờ rò rỉ (ví dụ AUC Tăng/Giảm ≥ 0.65 hoặc Macro-F1 ≥ 0.50 trên toàn mẫu eval) thì chạy: `.venv_mmf/bin/python -m ai_core.models.mmf.leak_check --stock-dir $SNAP`.

## Variant v2 của nhánh giá

Ba variant chạy song song với các arm v1 và ghi kết quả thành arm tên `<model>@<variant>` trong cùng run dir:

| Variant | Khác v1 ở đâu |
|---|---|
| `base` | Dữ liệu y như v1, chỉ thêm mô hình hai tầng `hurdle_logit_price`, `hurdle_gbm_price` |
| `ext` | Train thêm trên các mã VN100 ngoài 15 mã; tune, calib, eval vẫn chỉ 15 mã |
| `ext_ctx` | `ext` cộng 19 đặc trưng bối cảnh (VNINDEX, độ rộng thị trường, S&P 500, VIX, Brent, USD/VND, hạng cắt ngang) |

```bash
# Dữ liệu phụ đã đóng băng: thư mục in ra bởi `python -m ai_core.models.mmf.aux freeze`
AUX=data/mmf/aux/<aux_id>
.venv_mmf/bin/python -m ai_core.models.mmf.experiment --name thu1 --stock-dir $SNAP                                  # arm v1, để có đối chứng
.venv_mmf/bin/python -m ai_core.models.mmf.experiment --name thu1 --stock-dir $SNAP --variants base ext ext_ctx --aux-dir $AUX
.venv_mmf/bin/python -m ai_core.models.mmf.compare --run-dir data/mmf/exp/thu1/24251acc1a5695bd --out data/mmf/exp/thu1/report.md
```

Báo cáo có thêm mục "R4: v2 so với v1" và "Từng bước cải tiến". Sửa ở đâu: đặc trưng bối cảnh trong `mmf/context.py`, mã ngoài và đóng băng trong `mmf/aux.py`, chuỗi toàn cầu trong `mmf/globaldata.py`, danh sách variant và ứng viên trong `mmf/config.py`. 12 ứng viên v2 đã đăng ký trong phụ lục 2026-11-10 của `mmf_preregistration.md`; thêm ứng viên mới thì phải ghi thêm phụ lục trước khi chạy.

## Trước lượt chấm cuối (chủ dự án chạy)

Khi có `docs/reports/mmf_lock_v2.json`, `ai_core.models.mmf.holdout` chấm thêm model v2 đã khóa (R4) trong cùng lượt chạy và đòi `--aux-dir`. Dữ liệu phụ phải mới ngang dữ liệu giá, nên chạy đúng thứ tự:

```bash
~/.venv_6x/bin/python jobs/sync_prices.py --stock-dir data/prices_ext --delay 4   # 85 mã ngoài
.venv_mmf/bin/python -m ai_core.models.mmf.aux index        # VNINDEX
.venv_mmf/bin/python -m ai_core.models.mmf.globaldata       # S&P 500, VIX, Brent, USD/VND
.venv_mmf/bin/python -m ai_core.models.mmf.aux freeze       # in ra đường dẫn aux
.venv_mmf/bin/python -m ai_core.models.mmf.holdout --h 2 --aux-dir <đường dẫn vừa in>
```

Thiếu `--aux-dir`, aux cũ hơn dữ liệu giá hoặc aux bị sửa sau khi đóng băng thì lệnh dừng trước khi huấn luyện và trước khi ghi file khóa chạy, nên có thể sửa rồi chạy lại.

## Ba điều không được làm

1. **Không chạy `ai_core.models.mmf.holdout`** để thử. Mỗi h chỉ chạy được một lần, cho cấu hình đã khóa trong `docs/reports/mmf_lock.json`.
2. **Không ghi đè** `docs/reports/mmf_lock.json`, `mmf_dev_report.md` hay `mmf_preregistration.md`.
3. **Không gọi kết quả thử nghiệm là kết quả cuối.** Ba cửa sổ eval A, B, C đã bị nhìn nhiều lần. Một cấu hình mới chỉ được coi là đã kiểm chứng khi nó được khóa và đăng ký trong `mmf_preregistration.md` trước khi chạy vùng chấm cuối, tức là trước khi T+2 đủ 40 phiên, khoảng cuối tháng 12/2026. `ai_core.models.mmf.holdout` đọc file khóa v1 và, nếu có, file khóa v2 của nhánh giá (`mmf_lock_v2.json`, viết bằng `ai_core.models.mmf.compare --lock-v2`); một phiên bản khác nữa thì cần sửa code và ghi phụ lục trước mốc đó.
