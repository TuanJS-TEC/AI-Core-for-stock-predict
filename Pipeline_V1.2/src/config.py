"""Cấu hình của lớp dữ liệu giá: nguồn, đồng bộ, lịch phiên, nhãn thích ứng và fold theo thời gian.

Mô hình dự báo cũ (Linear/NLinear/DLinear gộp 15 mã) đã bị gỡ; còn lại chỉ phần dữ liệu để dựng mô hình mới.
Các module khác dùng `from . import config as C` và không tự đặt hằng số.
"""
from pathlib import Path

# =====================================================================================================
# 1. Dữ liệu đầu vào
# =====================================================================================================
ROOT = Path(__file__).resolve().parents[1]
STOCK_DIR = ROOT / "stock"                  # thư mục chứa {mã}_stock.parquet (và bản .csv để xem định dạng)
STOCK_FILE = "{ticker}_stock.parquet"       # tên file dữ liệu của một mã
STOCK_CSV_GLOB = "*_stock.csv"              # mẫu tên file CSV gốc dùng khi đồng bộ dữ liệu

TICKERS = ["FPT", "VNM", "STB", "ACB", "SSI", "VIC", "HPG", "VCB", "MSN", "MBB", "GAS", "MWG", "VPB", "VRE", "TCB"]

TIME_COL = "time"                           # cột thời gian, một dòng mỗi phiên giao dịch
PRICE_COLS = ("open", "high", "low", "close")
VOLUME_COL = "volume"
RAW_COLS = (TIME_COL, *PRICE_COLS, VOLUME_COL)   # đủ 6 cột bắt buộc của mỗi file
TARGET_COL = "close"                        # cột giá dùng cho Panel.log_close

HORIZONS = (2, 5)                           # số phiên dự báo phía trước

# =====================================================================================================
# 2. Nhãn ba lớp thích ứng theo biến động (labels.py)
# =====================================================================================================
CLASS_DOWN, CLASS_FLAT, CLASS_UP = 0, 1, 2
N_CLASSES = 3
LABEL_NAMES = ("Giảm", "Đi ngang", "Tăng")  # theo thứ tự chỉ số lớp 0, 1, 2

# θ = K_ADAPT · σ · √h, σ = std (ddof=1) log-return 1 phiên trong SIGMA_WINDOW phiên kết thúc tại t.
K_ADAPT = 0.43
SIGMA_WINDOW = 60
LABEL_VERSION = "adaptive-k0.43-w60-v1"     # đổi khi đổi công thức/tham số nhãn

# =====================================================================================================
# 3. Fold theo thời gian (splits.py)
# =====================================================================================================
BLOCK_SESSIONS = 63                         # độ dài mỗi khối train/tune/calib/eval, tính theo phiên giao dịch
# (name, train_end, tune_end, calib_end, eval_end) cho fold A, B, C. Sinh một lần ngày 2026-11-09 bằng
# splits.make_date_folds(build_panel().calendar, "2026-10-30") trên dữ liệu thật (1457 phiên, 2021-01-04..2026-11-09).
# ĐÓNG BĂNG CÓ CHỦ ĐÍCH: không tính lại lúc chạy, để dữ liệu mới append không dời mốc.
DATE_FOLDS = (
    ("A", "2025-07-25", "2025-10-24", "2026-01-23", "2026-05-04"),
    ("B", "2025-10-24", "2026-01-23", "2026-05-04", "2026-07-30"),
    ("C", "2026-01-23", "2026-05-04", "2026-07-30", "2026-10-30"),
)

# Block bootstrap theo ngày trên vùng đánh giá (dùng bởi ai_core.direction)
BOOTSTRAP_BLOCK = 20                        # độ dài khối (phiên)
BOOTSTRAP_N = 500                           # số lần lấy mẫu
CI_LEVEL = 0.95                             # mức tin cậy (percentile)

# =====================================================================================================
# 4. Lịch phiên (sessions.py)
# =====================================================================================================
# Ngày nghỉ đã qua tự suy ra từ dữ liệu (ngày thường không có phiên); ngày nghỉ sắp tới phải khai báo ở đây theo thông
# báo của sở giao dịch (HOSE thường công bố lịch nghỉ Tết vào cuối năm trước).
MARKET_HOLIDAYS = ("2027-01-01",)
MARKET_CALENDAR_UNTIL = "2027-01-31"        # đã khai báo đủ ngày nghỉ đến ngày này; ngày mục tiêu sau đó chỉ là dự kiến
SESSION_DATA_READY = "15:00"                # giờ máy: từ giờ này phiên của hôm nay được coi là đã có dữ liệu để sync
HOLIDAYS_FILE = ROOT / "data" / "hose_holidays.csv"   # ngày nghỉ HOSE (cột date,verified); hợp với MARKET_HOLIDAYS

# =====================================================================================================
# 5. Đồng bộ dữ liệu từ vnstock (sync_stock_parquet.py, chạy riêng)
# =====================================================================================================
SYNC_INTERVAL = "1D"                        # dữ liệu theo phiên
SYNC_MAX_RETRIES = 3                        # số lần thử lại khi gọi API lỗi
SYNC_RETRY_BACKOFF = 2.0                    # chờ SYNC_RETRY_BACKOFF · lần_thử giây trước lần thử lại
SYNC_API_DELAY = 3.0                        # nghỉ (giây) giữa hai lần gọi API để tránh bị chặn: khách vnstock 20 lượt/phút, 20 mã
SYNC_BASIS_TOL = 0.005                      # nghìn đồng; lệch giá đóng cửa quá mức này ở phiên đã lưu = chuẩn điều chỉnh đã đổi
SYNC_RET_BAND = 0.072                       # |lợi suất ngày| lớn hơn mức này ở phiên mới = nghi đổi chuẩn điều chỉnh
SYNC_INIT_START = "2021-01-04"              # ngày bắt đầu tải lịch sử cho mã mới (--init)
SYNC_SOURCE = "kbs"                         # nguồn vnstock cho Quote(source=...); đã kiểm khớp parquet đã lưu, vci trả thêm phiên và giờ khác
SYNC_OVERLAP_SESSIONS = 20                  # tải chồng ngần này phiên gần nhất để so với dữ liệu đã lưu
SESSION_HOUR_UTC = 7                        # mỗi phiên ghi đúng một mốc: ngày + 07:00 UTC
