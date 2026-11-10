# Lớp dữ liệu giá

Mô hình dự báo cũ (Linear/NLinear/DLinear gộp 15 mã) đã bị gỡ khỏi repo; mã còn trong lịch sử git, ngay trước commit "remove price model". Lý do: không thắng baseline lặp giá cuối (trung vị MSE 1.02×), kappa test 0.02–0.03 và F1 chỉ hơn hoán vị 0.01–0.02. Chỉ còn lớp dữ liệu để dựng mô hình mới.

## Cấu trúc

```
sync_stock_parquet.py   đồng bộ giá từ vnstock (cron 15:20 các ngày trong tuần), kiểm tra, ghi stock/ và manifest
data/hose_holidays.csv  ngày nghỉ HOSE (cột date,verified; hiện toàn bộ verified=false, cần đối chiếu thông báo HOSE)
stock/                  {mã}_stock.parquet, manifest.json, snapshots/  (git bỏ qua)
src/
  config.py             đường dẫn, mã, đồng bộ, nhãn, fold, lịch
  data.py               đọc parquet -> Panel (trục ngày chung, OHLCV, lịch phiên, khoảng trống, dataset_id)
  validation.py         check_ohlcv, lịch phiên, ngày thường không giải thích được
  snapshot.py           content_hash, manifest, dataset_id, snapshot bất biến (python -m src.snapshot freeze)
  price_features.py     25 đặc trưng nhân quả giá/khối lượng/thị trường trên lịch phiên (phiên thiếu = NaN, không điền)
  labels.py             nhãn 3 lớp thích ứng theo biến động θ = K·σ60·√h, trạng thái ok/not_matured/missing_session/bad_sigma
  splits.py             fold train/tune/calib/eval theo ngày, purge theo label_end_date; mốc đóng băng trong config.DATE_FOLDS
  dataset.py            build_dataset(panel, h) và inference_features(panel): train và suy luận dùng chung một đường đặc trưng
  sessions.py           ngày nghỉ, phiên thứ n sau một ngày, phiên thiếu lúc now (ai_core.live dùng)
  stats.py              block bootstrap theo ngày (ai_core.direction dùng)
tests/                  test của lớp dữ liệu
```

## Đồng bộ và toàn vẹn

- Nguồn `SYNC_SOURCE = "kbs"`. Mỗi lần chạy tải chồng `SYNC_OVERLAP_SESSIONS = 20` phiên gần nhất để so với dữ liệu đã lưu.
- `clean_types` bỏ hàng vi phạm (`nonpositive_price`, `ohlc_order`, `volume_invalid`). Phiên đã lưu lệch OHLC quá `SYNC_BASIS_TOL` được coi là đổi chuẩn điều chỉnh; refetch đầy đủ bị từ chối nếu mất hàng vì kiểm tra.
- `stock/manifest.json` chứa số dòng, phiên cuối, băm nội dung từng mã và `dataset_id`. `build_panel` chỉ nhận `dataset_id` khi manifest khớp dữ liệu vừa đọc. `adjustment_policy` hiện là "unknown".
- Đóng băng dữ liệu đã dùng: `python -m src.snapshot freeze`.

## Dùng để dựng mô hình mới

```python
from src import config as C
from src.data import build_panel
from src.dataset import build_dataset
from src.splits import folds_from_config, split_samples

panel = build_panel(C.TICKERS)
ds = build_dataset(panel, h=5)
ds = ds[(ds.status == "ok") & ds.features_ok]
for f in folds_from_config():
    parts = split_samples(ds, f)      # train / tune / calib / eval, đã purge
```

Cần nhớ khi đánh giá: số mẫu hiệu dụng nhỏ (nhãn chồng lấn, 15 mã cùng chịu yếu tố thị trường); chọn model bằng log-loss hoặc Brier skill so với prior thay vì Macro-F1; bootstrap theo khối ngày.

## Đối chiếu với TCBS

`crosscheck_tcbs.py` chạy bằng `~/.venv_6x` (vnstock thế hệ 6). Nó chỉ đọc `stock/`, so 20 phiên gần nhất của mọi mã với TCBS Open API, rồi ghi nối mọi chỗ lệch vào `outputs/crosscheck_tcbs.csv`. Một chỗ lệch là giá lệch quá `SYNC_BASIS_TOL`, khối lượng lệch quá 2%, thiếu phiên ở một bên, hoặc lỗi tải. Phiên TCBS sống tối đa 24 giờ và không tự gia hạn: mỗi ngày cần `~/.venv_6x/bin/vnstock auth tcbs` với Smart OTP từ app TCInvest. Ngày nào chưa mở phiên thì lượt đối chiếu chỉ ghi "skipped", sync KBS vẫn chạy bình thường.

## Việc còn mở

- Đối chiếu 69 ngày nghỉ trong `data/hose_holidays.csv` với thông báo HOSE rồi đặt `verified=true`.
- Khai báo ngày nghỉ 2027 (Tết) vào `config.MARKET_HOLIDAYS` hoặc CSV.
