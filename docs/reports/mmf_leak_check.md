# Kiểm tra rò rỉ bằng nhãn train xáo trộn

Hoán vị nhãn train giữa các ngày (cùng một phép hoán vị cho mọi mã), giữ nguyên tune, calib, eval. Huấn luyện `dl_price` (LSTM, seed 59), 20 phép hoán vị cho mỗi (h, fold). Nếu nhãn hay thông tin tương lai lọt vào khâu huấn luyện, AUC trên eval sẽ lệch hẳn lên trên 0.5.

| h | fold | AUC Tăng/Giảm, nhãn thật | AUC nhãn xáo trộn: trung bình | độ lệch chuẩn | min | max | log-loss nhãn xáo trộn |
|---|---|---|---|---|---|---|---|
| 5 | A | 0.619 | 0.509 | 0.044 | 0.444 | 0.609 | 1.0983 |
| 5 | B | 0.475 | 0.445 | 0.022 | 0.408 | 0.502 | 1.0957 |
| 5 | C | 0.455 | 0.474 | 0.050 | 0.377 | 0.608 | 1.0950 |
| 2 | A | 0.556 | 0.489 | 0.035 | 0.414 | 0.538 | 1.0887 |
| 2 | B | 0.510 | 0.471 | 0.027 | 0.417 | 0.512 | 1.0848 |
| 2 | C | 0.550 | 0.492 | 0.029 | 0.429 | 0.541 | 1.0945 |

Kết luận: không có độ lệch lên trên 0.5 ở bất kỳ (h, fold) nào. Một lần chạy đơn lẻ có thể đạt 0.61 do ngẫu nhiên, vì eval chỉ khoảng 60 phiên và các mã biến động cùng nhau. Vì vậy một con số AUC đơn lẻ trên eval cần được đọc kèm độ lệch này. Phép kiểm tra này không phát hiện được đặc trưng dùng dữ liệu tương lai; phần đó do test trong `tests/test_mmf_frame.py`, `test_mmf_fundamentals.py`, `test_mmf_text.py` đảm nhận.

## Cách tái lập

```
.venv_mmf/bin/python -m mmf.leak_check --stock-dir Pipeline_V1.2/stock/snapshots/24251acc1a5695bd --perms 20 --h 5 2 --device cuda
```

Bảng được ghi ra `data/mmf/leak_check.md` (đổi bằng `--out`), không ghi đè file này. Phép kiểm tra dùng bộ mã hóa LSTM với seed 59 cho mọi fold, không dùng bộ mã hóa đã khóa theo fold và không dùng ensemble 5 seed, nên cột "AUC Tăng/Giảm, nhãn thật" ở đây khác AUC của `dl_price` trong `mmf_dev_report.md`.
