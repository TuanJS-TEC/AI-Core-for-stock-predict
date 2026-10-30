"""Cấu hình chung của project: mọi tham số, đường dẫn, tên file và thiết lập báo cáo đều lấy từ đây.

Project dùng một cấu hình duy nhất: huấn luyện gộp 15 mã, đầu vào chỉ là log giá đóng cửa, đầu phân loại trộn kênh
có embedding mã (phương án C ở bước 1 của plan_pooled_features.md), ngưỡng nhãn K_THETA = 0.43.

SEED chỉ được khai báo ở đây. Các module khác dùng `from . import config as C` và không tự đặt hằng số.
Kiến trúc mô hình nằm trong models.py, chỉ đọc kích thước/kernel từ file này.
"""
from pathlib import Path

# =====================================================================================================
# 1. Tái lập (reproducibility)
# =====================================================================================================
SEED = 59                                   # seed gốc duy nhất của project
SEEDS = [SEED + i for i in range(5)]        # 5 seed huấn luyện: 59..63, báo cáo mean ± std qua các seed này
DETERMINISTIC = True                        # torch.use_deterministic_algorithms; bắt buộc True, huấn luyện trên CPU (CLAUDE.md)

# =====================================================================================================
# 2. Dữ liệu đầu vào
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
TARGET_COL = "close"                        # kênh duy nhất của mô hình (log giá đóng cửa)

# Chia theo thứ tự thời gian, không xáo trộn: 70% train, 15% val, phần còn lại (15%) là test.
# Với 1447 dòng/mã cho ra 1012 / 217 / 218 (manual test trong CLAUDE.md).
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# =====================================================================================================
# 3. Nhãn phân loại
# =====================================================================================================
# r = log-return giữa bước T và điểm cuối cửa sổ. Nhãn: Giảm nếu r < -θ, Đi ngang nếu |r| ≤ θ, Tăng nếu r > θ.
# θ = K_THETA · std(r_train), tính riêng từng mã, từng T, chỉ từ train. K_THETA = 0.43 cho ba lớp gần cân bằng
# (mỗi lớp khoảng 1/3 nếu r xấp xỉ chuẩn), mức duy nhất có brier_skill test dương ở cả 4 (T, L) khi quét.
K_THETA = 0.43

CLASS_DOWN, CLASS_FLAT, CLASS_UP = 0, 1, 2
N_CLASSES = 3
LABEL_NAMES = ("Giảm", "Đi ngang", "Tăng")  # theo thứ tự chỉ số lớp 0, 1, 2

# =====================================================================================================
# 4. Mô hình (kiến trúc nằm trong models.py)
# =====================================================================================================
MODELS = ("Linear", "NLinear", "DLinear")   # họ kiến trúc của cả đầu hồi quy và đầu phân loại
LOOKBACKS = (30, 90)                        # L: số phiên trong cửa sổ đầu vào
HORIZONS = (2, 5)                           # T: số phiên dự báo phía trước
MA_KERNEL = 25                              # kernel moving average của DLinear, padding hai đầu giữ nguyên độ dài
CLS_TEMPORAL_DIM = 8                        # d: số chiều chiếu thời gian L → d của đầu phân loại
TICKER_EMB_DIM = 4                          # kích thước embedding mã, nối vào đầu vào tầng phân loại cuối

# =====================================================================================================
# 5. Huấn luyện
# =====================================================================================================
BATCH_SIZE = 128                            # mẫu của 15 mã trộn chung nên dùng batch lớn hơn huấn luyện từng mã
LR = 1e-3                                   # learning rate của Adam
MAX_EPOCHS = 100
PATIENCE = 10                               # số epoch không cải thiện trên val trước khi dừng (mỗi đầu riêng)
MIN_DELTA = 1e-9                            # mức giảm tối thiểu của MSE/NLL val để tính là cải thiện
WEIGHT_DECAYS = (0.0, 1e-2)                 # lưới weight decay của Adam
# Lưới nhỏ cho --quick: kiểm tra toàn bộ luồng trong vài phút, không dùng để kết luận
QUICK_GRID = {"horizons": (2,), "lookbacks": (30,), "models": ("NLinear",), "seeds": tuple(SEEDS[:2])}

# =====================================================================================================
# 6. Hiệu chỉnh xác suất và quy tắc chọn nhãn (metrics.py), fit trên val riêng từng mã
# =====================================================================================================
# Nhiệt độ: tìm kiếm chia ba theo beta = 1/T trong khoảng dưới (tức T trong [0.1, 100]), cực tiểu NLL val
TEMP_BETA_RANGE = (0.01, 10.0)
TEMP_SEARCH_ITERS = 80
# Quy tắc chọn nhãn argmax(p_c / prior_c^alpha); alpha chọn theo Macro-F1 trên val
ALPHA_GRID = tuple(round(0.1 * i, 1) for i in range(0, 11))     # 0.0 .. 1.0
PRIOR_EPS = 1e-6                            # sàn của prior để tránh chia cho 0

# =====================================================================================================
# 7. Thư mục và tên file đầu ra
# =====================================================================================================
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
RUN_NAME = f"pooled_embed_k{K_THETA:g}"     # thư mục kết quả (results/ và figures/) của lần chạy đầy đủ
QUICK_NAME = f"pooled_embed_quick_k{K_THETA:g}"   # thư mục của lần chạy thử nhanh (--quick)

EXPERIMENT_FILE = "experiment.json"         # cấu hình và chữ ký dữ liệu; khác thì chạy lại từ đầu
RUNS_FILE = "runs.csv"                      # chỉ số val/test của mọi lượt, theo từng mã
PREDS_FILE = "preds_all.csv"                # xác suất test của mọi lượt
STATES_FILE = "states.pt"                   # state_dict mọi lượt
VAL_SUMMARY_FILE = "val_summary.csv"        # chỉ số val trung bình qua mã theo (T, L, model, wd), dùng để chọn
FINAL_CONFIG_FILE = "final_config.csv"      # (model, wd) chọn bằng val cho từng (T, L)
TEST_PREDICTIONS_FILE = "test_predictions.csv"
CLASS_FREQ_FILE = "train_class_freq.csv"
FINAL_TABLE_FILE = "final_table.csv"
PER_TICKER_FILE = "per_ticker.csv"          # F1, kappa, F1 hoán vị theo từng (T, L, mã), trung bình qua seed
CONFUSION_FILE = "confusion_T{T}_L{L}.csv"
CONFUSION_NORM_FILE = "confusion_T{T}_L{L}_norm.csv"
CONFUSION_FIG_FILE = "confusion_T{T}.png"

# Model tốt nhất cho từng (mã, T, L): 15 mã x 2 T x 2 L = 60 file (predict.py). Sau mỗi lần chạy pipeline, lượt có
# MODEL_SELECT_METRIC trên val của mã cao nhất được so với model đang lưu (chấm lại trên val hiện tại); giữ model cao hơn.
MODELS_DIR = ROOT / "outputs" / "models"
MODEL_FILE = "{ticker}_T{T}_L{L}.pt"        # trọng số, scaler, tham số hiệu chỉnh, θ, chỉ số val/test
MODEL_REGISTRY_FILE = "registry.csv"        # một dòng mỗi model đang lưu
MODEL_HISTORY_FILE = "registry_history.csv" # mỗi lần cập nhật nối thêm một dòng mỗi (mã, T, L): new / kept / replaced
MODEL_SELECT_METRIC = "val_f1_macro"        # cùng tiêu chí chọn cấu hình (Macro-F1 val sau hiệu chỉnh alpha)
MODEL_MIN_GAIN = 1e-9                       # ứng viên phải cao hơn ít nhất mức này mới thay (bỏ qua sai số làm tròn CSV)
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_FILE = "predictions_{date}.csv" # dự đoán từ cửa sổ kết thúc tại phiên cuối {date} của dữ liệu
# Lịch giao dịch để tính ngày mục tiêu (phiên thứ T sau phiên cuối của dữ liệu): thứ Hai đến thứ Sáu trừ ngày nghỉ.
# Ngày nghỉ đã qua tự suy ra từ dữ liệu (ngày thường không có phiên); ngày nghỉ sắp tới phải khai báo ở đây theo thông
# báo của sở giao dịch (HOSE thường công bố lịch nghỉ Tết vào cuối năm trước).
MARKET_HOLIDAYS = ("2027-01-01",)
MARKET_CALENDAR_UNTIL = "2027-01-31"        # đã khai báo đủ ngày nghỉ đến ngày này; ngày mục tiêu sau đó chỉ là dự kiến
SESSION_DATA_READY = "15:00"                # giờ máy: từ giờ này phiên của hôm nay được coi là đã có dữ liệu để sync

# Đồ thị đường dự đoán (đầu hồi quy) và đường thực tế của từng model trong kho (report.plot_models): chỉ vẽ val và
# test, không vẽ train. MSE/MAE của model và baseline repeat ở thang chuẩn hóa và thang giá gốc.
MODEL_FIGURES_DIR = FIGURES_DIR / "models"
MODEL_FIG_FILE = "{ticker}_T{T}_L{L}.png"
MODEL_REGRESSION_FILE = RESULTS_DIR / "models_regression.csv"

# Cột xác suất (đã co về prior, dùng cho Brier) và điểm quyết định (dùng chọn nhãn), thứ tự Giảm, Đi ngang, Tăng
PROB_COLS = ("p_down", "p_flat", "p_up")
SCORE_COLS = ("s_down", "s_flat", "s_up")

# =====================================================================================================
# 8. Báo cáo và hình
# =====================================================================================================
SCALE_NOTE = "nhãn từ log-return thật, không phụ thuộc thang"   # ghi vào cột scale của bảng kết quả
STD_DDOF = 0                                # ddof của std qua seed
FIG_DPI = 150
FIG_INK = "#0b0b0b"
CONFUSION_FIGSIZE = (10.5, 4.6)
CONFUSION_CMAP = "Blues"
CONFUSION_TEXT_SWITCH = 0.6                 # ô có tỷ lệ lớn hơn ngưỡng này dùng chữ trắng
FIG_FONTSIZE = 9                            # chữ trong ô và tiêu đề chung
FIG_TITLE_FONTSIZE = 10                     # tiêu đề từng ô con
FIG_SUPTITLE_X = 0.01                       # vị trí ngang của tiêu đề chung (căn trái)
MODEL_FIGSIZE = (11.0, 4.8)                 # đồ thị dự đoán của một model
MODEL_ACTUAL_COLOR = FIG_INK                # đường thực tế
MODEL_PRED_COLOR = "#c8442f"                # đường dự đoán
MODEL_TEST_SHADE = "#e9eef5"                # nền vùng test
MODEL_LINE_WIDTH = 1.1
DISPLAY_WIDTH = 220                        # độ rộng in bảng pandas ra terminal
DISPLAY_DECIMALS = 4                        # số chữ số thập phân khi in bảng

# Baseline hoán vị, kappa và khoảng tin cậy (report.py)
PERM_N = 200                                # số lần hoán vị nhãn dự đoán cho mỗi (mã, seed) khi tính F1 hoán vị
BOOTSTRAP_BLOCK = 20                        # độ dài khối (phiên) của block bootstrap theo ngày trên test
BOOTSTRAP_N = 500                           # số lần lấy mẫu bootstrap
CI_LEVEL = 0.95                             # mức tin cậy của khoảng bootstrap (percentile)

# =====================================================================================================
# 9. Đồng bộ dữ liệu từ vnstock (sync_stock_parquet.py, chạy riêng ngoài pipeline)
# =====================================================================================================
SYNC_INTERVAL = "1D"                        # dữ liệu theo phiên
SYNC_MAX_RETRIES = 3                        # số lần thử lại khi gọi API lỗi
SYNC_RETRY_BACKOFF = 2.0                    # chờ SYNC_RETRY_BACKOFF · lần_thử giây trước lần thử lại
SYNC_API_DELAY = 1.0                        # nghỉ (giây) giữa hai lần gọi API để tránh bị chặn
SYNC_BASIS_TOL = 0.005                      # nghìn đồng; lệch giá đóng cửa quá mức này ở phiên đã lưu = chuẩn điều chỉnh đã đổi
SYNC_RET_BAND = 0.072                       # |lợi suất ngày| lớn hơn mức này ở phiên mới = nghi đổi chuẩn điều chỉnh
SYNC_INIT_START = "2021-01-04"                # ngày bắt đầu tải lịch sử cho mã mới (--init)
SESSION_HOUR_UTC = 7                        # mỗi phiên ghi đúng một mốc: ngày + 07:00 UTC (định dạng gốc trong docs/dataset.md)

# =====================================================================================================
# 10. Demo Streamlit (demo.py, chạy bằng `streamlit run demo.py`)
# =====================================================================================================
DEMO_CHART_START = "2026-10-01"             # đồ thị giá dự báo / giá thực tế bắt đầu từ phiên mục tiêu này đến hôm nay
DEMO_LABEL_COLORS = {"Giảm": "#c8442f", "Đi ngang": "#7a7a7a", "Tăng": "#2e8b57"}   # màu nhãn trong demo
DEMO_ACTUAL_COLOR = "#1f4e79"               # đường giá thực tế
DEMO_PRED_COLOR = "#e07b39"                 # đường giá dự báo (đầu hồi quy)
DEMO_REPEAT_COLOR = "#9a9a9a"               # baseline repeat
