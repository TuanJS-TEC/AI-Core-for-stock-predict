"""Script chuyển đổi các file CSV giá chứng khoán sang Apache Parquet
và cập nhật dữ liệu OHLCV mới nhất từ thư viện vnstock.
Đầu ra:
    Các file có định dạng [Mã CK]_stock.parquet được lưu tại thư mục stock.

Mỗi phiên có đúng một dòng: thời gian được chuẩn hóa về ngày + C.SESSION_HOUR_UTC giờ (UTC), rồi mới lọc trùng
theo ngày. Trước đây lọc trùng theo thời điểm gốc nên phiên vnstock trả về lúc 00:00 và phiên đã có lúc 07:00
của cùng một ngày cùng tồn tại (ví dụ 2026-10-16 bị lặp trong parquet của cả 15 mã).
Vì dữ liệu được cập nhật thường xuyên, số dòng và mốc train/val/test (split_sizes) thay đổi theo dữ liệu.

Dùng: python sync_stock_parquet.py              # lấy dữ liệu mới từ vnstock, chuẩn hóa, ghi parquet và CSV
      python sync_stock_parquet.py --init BID CTG  # mã mới: tải lịch sử từ C.SYNC_INIT_START rồi đồng bộ như thường
      python sync_stock_parquet.py --no-fetch   # chỉ chuẩn hóa và lọc trùng dữ liệu đang có, không gọi API
"""

import os
import glob
import time
import argparse
import logging
from datetime import datetime
import pandas as pd

from src import config as C

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Thử import vnstock
try:
    from vnstock import Quote
except ImportError:
    logger.error("Không tìm thấy thư viện vnstock! Vui lòng cài đặt: pip install vnstock")
    Quote = None

def wall_clock(s: pd.Series) -> pd.Series:
    """Thời gian không múi giờ, giữ giờ trên đồng hồ của từng giá trị.

    Chấp nhận cột lẫn giá trị có và không có múi giờ (ví dụ CSV đã chuẩn hóa về UTC ghép với dữ liệu vnstock
    không múi giờ, pandas để ở kiểu object), trường hợp mà pd.to_datetime báo lỗi nếu không có utc=True.
    """
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_localize(None)
    if pd.api.types.is_datetime64_dtype(s.dtype):
        return s
    stamps = [pd.Timestamp(v) for v in s]
    return pd.Series([v.tz_localize(None) if v.tzinfo is not None else v for v in stamps],
                     index=s.index, dtype="datetime64[ns]")


def normalize_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Một dòng mỗi phiên: thời gian = ngày + C.SESSION_HOUR_UTC giờ (UTC), lọc trùng theo ngày, sắp xếp.

    Thời gian có múi giờ được bỏ múi giờ nhưng giữ giờ trên đồng hồ trước khi lấy ngày, để phiên 00:00 giờ
    Việt Nam không bị lùi sang ngày hôm trước. Khi trùng ngày, giữ dòng xuất hiện sau cùng trong df
    (dữ liệu vnstock được ghép sau CSV nên được ưu tiên).
    """
    t = wall_clock(df[C.TIME_COL])
    out = df.copy()
    out[C.TIME_COL] = (t.dt.normalize() + pd.Timedelta(hours=C.SESSION_HOUR_UTC)).dt.tz_localize("UTC")
    out = out.drop_duplicates(subset=[C.TIME_COL], keep="last")
    return out.sort_values(C.TIME_COL, kind="stable").reset_index(drop=True)


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột và kiểu dữ liệu, chỉ giữ các cột chuẩn C.RAW_COLS."""
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    for col in C.PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[C.VOLUME_COL] = pd.to_numeric(df[C.VOLUME_COL], errors="coerce").fillna(0).astype("int64")
    return df[list(C.RAW_COLS)]


VN_TZ = "Asia/Ho_Chi_Minh"
MARKET_CLOSE_HOUR_VN = 15


def drop_open_session(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Bỏ dòng của ngày hôm nay (giờ Việt Nam) nếu `now` còn trước 15:00: phiên chưa chốt, giá/khối lượng còn đổi."""
    now = now.tz_convert(VN_TZ) if now.tzinfo is not None else now        # ponytail: now không múi giờ coi như giờ VN
    if now.hour >= MARKET_CLOSE_HOUR_VN or df.empty:
        return df
    return df[wall_clock(df[C.TIME_COL]).dt.normalize() != now.normalize().tz_localize(None)].reset_index(drop=True)


def fetch_latest_ohlcv(symbol: str, start_date: str, end_date: str, max_retries: int = C.SYNC_MAX_RETRIES) -> pd.DataFrame:
    """
    Lấy dữ liệu OHLCV từ thư viện vnstock cho một mã chứng khoán trong khoảng [start_date, end_date].
    Hỗ trợ cơ chế retry khi gặp sự cố mạng hoặc giới hạn API.
    """
    if Quote is None:
        logger.warning(f"Bỏ qua lấy dữ liệu vnstock cho {symbol} do chưa cài đặt thư viện.")
        return pd.DataFrame()

    for attempt in range(1, max_retries + 1):
        try:
            quote = Quote(symbol=symbol)
            df = quote.history(start=start_date, end=end_date, interval=C.SYNC_INTERVAL)
            if df is not None and not df.empty:
                return drop_open_session(df, pd.Timestamp.now(tz=VN_TZ))
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Lần thử {attempt}/{max_retries} lấy dữ liệu {symbol} thất bại: {e}")
            if attempt < max_retries:
                time.sleep(C.SYNC_RETRY_BACKOFF * attempt)
            else:
                logger.error(f"Không thể lấy dữ liệu mới cho {symbol} sau {max_retries} lần thử.")
    return pd.DataFrame()


def basis_changed(stored: pd.DataFrame, merged: pd.DataFrame) -> bool:
    """True nếu chuẩn điều chỉnh giá đã đổi (cổ tức, chia tách...) so với bản đã lưu. Cả hai đã qua normalize_sessions.

    Hai dấu hiệu: (1) phiên đã lưu có giá đóng cửa lệch quá C.SYNC_BASIS_TOL trong merged; (2) bước nhảy
    |pct_change| > C.SYNC_RET_BAND mà phiên sau mới hơn stored.time.max(). Bước nhảy đã nằm trong lịch sử
    (phiên bị thiếu thật) không tính lại mỗi ngày.
    """
    # ponytail: không phân biệt được bước nhảy do phiên thiếu với corporate action trong lần chạy đầu tiên; chấp nhận một lần tải lại.
    old = stored.set_index(C.TIME_COL)["close"]
    new = merged.set_index(C.TIME_COL)["close"]
    common = old.index.intersection(new.index)
    if ((new[common] - old[common]).abs() > C.SYNC_BASIS_TOL).any():
        return True
    jumps = new.pct_change().abs() > C.SYNC_RET_BAND
    return bool(jumps[jumps.index > old.index.max()].any())


REFETCH_START_SLACK = pd.Timedelta(days=5)   # bản tải lại được bắt đầu trễ hơn ngày đầu đã lưu tối đa chừng này


def refetch_full(symbol: str, stored: pd.DataFrame, end_date: str, delay: float):
    """Tải lại toàn bộ lịch sử từ ngày đầu đã lưu; trả bản đã chuẩn hóa, hoặc None nếu không được thay lịch sử.

    Bị từ chối (logger.error) khi rỗng (ví dụ bị giới hạn lượt gọi), bắt đầu trễ hơn ngày đầu đã lưu quá
    REFETCH_START_SLACK, hoặc kết thúc trước phiên cuối đã lưu: thay bằng bản cụt sẽ mất lịch sử vĩnh viễn.
    """
    full = fetch_latest_ohlcv(symbol, start_date=stored[C.TIME_COL].min().strftime("%Y-%m-%d"), end_date=end_date)
    time.sleep(delay)  # Tránh vượt quá giới hạn lượt gọi API
    if full is None or full.empty:
        logger.error(f"[{symbol}] tải lại toàn bộ lịch sử trả về rỗng; giữ nguyên dữ liệu đã lưu")
        return None
    full = normalize_sessions(clean_types(full))
    t, s = full[C.TIME_COL], stored[C.TIME_COL]
    if t.min() > s.min() + REFETCH_START_SLACK or t.max() < s.max():
        logger.error(f"[{symbol}] tải lại toàn bộ lịch sử bị cụt ({t.min():%Y-%m-%d}..{t.max():%Y-%m-%d}, "
                     f"đã lưu {s.min():%Y-%m-%d}..{s.max():%Y-%m-%d}); giữ nguyên dữ liệu đã lưu")
        return None
    return full


def process_stock_file(csv_path: str, output_dir: str, update_csv: bool = True,
                    delay_between_calls: float = C.SYNC_API_DELAY, fetch: bool = True,
                    full_refresh: bool = False):
    """
    Xử lý một file CSV của mã chứng khoán:
    1. Đọc dữ liệu lịch sử từ CSV.
    2. Xác định ngày giao dịch gần nhất hiện có.
    3. Lấy dữ liệu mới nhất từ vnstock đến ngày hiện tại (bỏ qua nếu fetch=False).
    4. Gộp dữ liệu, chuẩn hóa thời gian về một mốc mỗi phiên, loại bỏ trùng lặp theo ngày và sắp xếp.
    5. Lưu kết quả ra file [Mã CK]_stock.parquet (và tùy chọn cập nhật lại CSV).
    """
    file_name = os.path.basename(csv_path)
    # Lấy ticker (Ví dụ ACB_stock.csv -> ACB)
    symbol = file_name.split("_")[0].upper()
    parquet_path = os.path.join(output_dir, C.STOCK_FILE.format(ticker=symbol))

    logger.info(f"--- Đang xử lý mã: {symbol} ({file_name}) ---")

    # 1. Đọc dữ liệu từ file CSV gốc
    try:
        df_csv = pd.read_csv(csv_path)
        if df_csv.empty:
            logger.warning(f"File {csv_path} rỗng.")
            return
    except Exception as e:
        logger.error(f"Lỗi khi đọc {csv_path}: {e}")
        return

    # Chuẩn hóa tên cột
    df_csv.columns = [col.strip().lower() for col in df_csv.columns]
    missing_cols = [c for c in C.RAW_COLS if c not in df_csv.columns]
    if missing_cols:
        logger.error(f"File {file_name} thiếu các cột bắt buộc: {missing_cols}")
        return

    # Chuẩn hóa kiểu dữ liệu và thời gian (một dòng mỗi phiên)
    raw_count = len(df_csv)
    df_csv = normalize_sessions(clean_types(df_csv))
    if len(df_csv) < raw_count:
        logger.warning(f"[{symbol}] Bỏ {raw_count - len(df_csv)} dòng trùng ngày trong {file_name}")

    initial_count = len(df_csv)
    last_record_date = df_csv[C.TIME_COL].max()
    today = datetime.now()
    start_date_str = last_record_date.strftime("%Y-%m-%d")
    end_date_str = today.strftime("%Y-%m-%d")

    logger.info(f"[{symbol}] Dữ liệu hiện có: {initial_count} phiên (từ {df_csv[C.TIME_COL].min().strftime('%Y-%m-%d')} đến {last_record_date.strftime('%Y-%m-%d')})")

    # 2. Lấy dữ liệu mới nhất từ vnstock
    df_new = pd.DataFrame()
    if fetch and full_refresh:
        full = refetch_full(symbol, df_csv, end_date_str, delay_between_calls)
        if full is not None:
            logger.warning(f"[{symbol}] tải lại toàn bộ lịch sử (--full-refresh)")
            df_new = full
    elif fetch and start_date_str <= end_date_str:
        logger.info(f"[{symbol}] Gọi vnstock cập nhật từ {start_date_str} đến {end_date_str}...")
        df_new = fetch_latest_ohlcv(symbol, start_date=start_date_str, end_date=end_date_str)
        time.sleep(delay_between_calls)  # Tránh vượt quá giới hạn lượt gọi API

    # 3. Hợp nhất dữ liệu: ghép rồi lọc trùng theo ngày (phiên trùng ưu tiên bản ghi mới nhất từ vnstock)
    if df_new is not None and not df_new.empty:
        df_new = normalize_sessions(clean_types(df_new))      # cùng kiểu thời gian (UTC) với df_csv trước khi ghép
        df_merged = normalize_sessions(df_new if full_refresh else pd.concat([df_csv, df_new], ignore_index=True))
        if not full_refresh and basis_changed(df_csv, df_merged):
            logger.warning(f"[{symbol}] chuẩn điều chỉnh đổi, tải lại toàn bộ lịch sử")
            full = refetch_full(symbol, df_csv, end_date_str, delay_between_calls)
            if full is None:
                # Không ghi chuỗi lẫn hai chuẩn giá: ghi lại đúng dữ liệu đã lưu (như khi không có bản ghi mới).
                # Phiên hôm nay thiếu -> ai_core.live báo DATA_INVALID; mai phiên chồng lấn vẫn lệch nên tự tải lại.
                df_merged = df_csv
            else:
                df_merged = full
                if basis_changed(df_merged.iloc[:1], df_merged):
                    logger.error(f"[{symbol}] sau khi tải lại toàn bộ vẫn còn bước nhảy giá > {C.SYNC_RET_BAND:.1%}, vẫn ghi")
        new_records_count = len(df_merged) - initial_count
        logger.info(f"[{symbol}] Đã cập nhật thêm {new_records_count} bản ghi mới (Tổng: {len(df_merged)} phiên). Phiên mới nhất: {df_merged[C.TIME_COL].max().strftime('%Y-%m-%d')}")
    else:
        logger.info(f"[{symbol}] Không có bản ghi mới từ vnstock. Giữ nguyên {initial_count} phiên.")
        df_merged = df_csv

    # 4. Xuất ra định dạng Apache Parquet
    try:
        df_merged.to_parquet(parquet_path, engine="pyarrow", index=False)
        parquet_size_kb = os.path.getsize(parquet_path) / 1024
        logger.info(f"[{symbol}] Đã xuất thành công: {os.path.basename(parquet_path)} ({parquet_size_kb:.1f} KB)")
    except Exception as e:
        logger.error(f"[{symbol}] Lỗi khi lưu file Parquet {parquet_path}: {e}")

    # 5. Đồng bộ lại file CSV nếu được bật
    if update_csv:
        try:
            df_to_save = df_merged.copy()
            df_to_save[C.TIME_COL] = df_to_save[C.TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_to_save.to_csv(csv_path, index=False)
            logger.info(f"[{symbol}] Đã đồng bộ cập nhật lại {os.path.basename(csv_path)}")
        except Exception as e:
            logger.warning(f"[{symbol}] Không thể cập nhật lại CSV {csv_path}: {e}")


def init_stock_file(symbol: str, stock_dir: str, start: str = C.SYNC_INIT_START):
    """Tải lịch sử của mã mới từ `start` tới hôm nay, ghi {SYM}_stock.csv đã chuẩn hóa; trả đường dẫn.

    File đã tồn tại thì giữ nguyên. Tải rỗng thì trả None.
    """
    symbol = symbol.upper()
    path = os.path.join(stock_dir, f"{symbol}_stock.csv")
    if os.path.exists(path):
        return path
    df = fetch_latest_ohlcv(symbol, start_date=start, end_date=datetime.now().strftime("%Y-%m-%d"))
    if df is None or df.empty:
        logger.error(f"[{symbol}] Không tải được lịch sử từ {start}, không tạo file.")
        return None
    df = normalize_sessions(clean_types(df))
    df[C.TIME_COL] = df[C.TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(path, index=False)
    logger.info(f"[{symbol}] Đã khởi tạo {os.path.basename(path)}: {len(df)} phiên từ {df[C.TIME_COL].iloc[0]}")
    return path


def check_alignment(stock_dir: str, symbols) -> bool:
    """Cảnh báo nếu các mã không có cùng dãy ngày (ví dụ một mã lấy dữ liệu mới thất bại)."""
    days = {}
    for s in symbols:
        p = os.path.join(stock_dir, C.STOCK_FILE.format(ticker=s))
        if os.path.exists(p):
            t = pd.read_parquet(p, columns=[C.TIME_COL])[C.TIME_COL]
            days[s] = tuple(pd.to_datetime(t).dt.normalize())
    if len(set(days.values())) > 1:
        ref = max(days.values(), key=len)
        bad = {s: f"{len(d)} phiên, cuối {d[-1].date()}" for s, d in days.items() if d != ref}
        logger.warning(f"Các mã không cùng dãy ngày (đủ nhất: {len(ref)} phiên, cuối {ref[-1].date()}): {bad}. "
                       f"Chạy lại script để lấy đủ dữ liệu trước khi huấn luyện gộp.")
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chuyển định dạng CSV sang Parquet và cập nhật dữ liệu OHLCV mới nhất từ vnstock."
    )
    parser.add_argument(
        "--stock-dir",
        type=str,
        default=str(C.STOCK_DIR),
        help="Đường dẫn đến thư mục chứa các file dữ liệu cổ phiếu"
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Danh sách các mã cổ phiếu cụ thể cần cập nhật (ví dụ: --symbols ACB FPT HPG). Mặc định xử lý tất cả file trong thư mục."
    )
    parser.add_argument(
        "--update-csv",
        action="store_true",
        default=True,
        help="Đồng bộ cập nhật dữ liệu mới vào file CSV gốc (mặc định: True)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=C.SYNC_API_DELAY,
        help=f"Khoảng dừng (giây) giữa mỗi lần gọi API vnstock để tránh bị chặn IP (mặc định: {C.SYNC_API_DELAY} giây)"
    )

    parser.add_argument(
        "--full-refresh",
        nargs="+",
        default=[],
        metavar="SYM",
        help="Các mã tải lại toàn bộ lịch sử từ ngày đầu đã lưu và thay thế dữ liệu cũ (ví dụ sau khi đổi chuẩn điều chỉnh giá)"
    )

    parser.add_argument(
        "--init",
        nargs="+",
        default=[],
        metavar="SYM",
        help=f"Các mã mới: tải lịch sử từ {C.SYNC_INIT_START} và tạo file CSV nếu chưa có, trước khi đồng bộ"
    )

    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Không gọi vnstock, chỉ chuẩn hóa thời gian và lọc trùng dữ liệu đang có"
    )

    args = parser.parse_args(argv)

    stock_dir = os.path.abspath(args.stock_dir)
    if not os.path.exists(stock_dir):
        logger.error(f"Thư mục không tồn tại: {stock_dir}")
        return

    for sym in args.init:
        init_stock_file(sym, stock_dir)

    # Tìm tất cả file *_stock.csv
    csv_pattern = os.path.join(stock_dir, C.STOCK_CSV_GLOB)
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        # Thử tìm file .csv nói chung nếu không có hậu tố _stock.csv
        csv_files = [f for f in glob.glob(os.path.join(stock_dir, "*.csv"))]

    if not csv_files:
        logger.warning(f"Không tìm thấy file .csv nào trong {stock_dir}")
        return

    # Lọc theo danh sách symbols nếu có chỉ định
    if args.symbols:
        requested_symbols = [s.upper() for s in args.symbols]
        csv_files = [f for f in csv_files if os.path.basename(f).split("_")[0].upper() in requested_symbols]

    csv_files.sort()
    logger.info(f"Tìm thấy {len(csv_files)} file CSV cần xử lý trong: {stock_dir}")

    start_total_time = time.time()
    for idx, csv_file in enumerate(csv_files, 1):
        logger.info(f"\n[{idx}/{len(csv_files)}] Bắt đầu xử lý: {os.path.basename(csv_file)}")
        process_stock_file(
            csv_path=csv_file,
            output_dir=stock_dir,
            update_csv=args.update_csv,
            delay_between_calls=args.delay,
            fetch=not args.no_fetch,
            full_refresh=os.path.basename(csv_file).split("_")[0].upper() in {x.upper() for x in args.full_refresh}
        )
    elapsed_time = time.time() - start_total_time
    logger.info(f"Hoàn thành toàn bộ {len(csv_files)} mã chứng khoán trong {elapsed_time:.1f} giây.")
    logger.info(f"Tất cả file Apache Parquet đã sẵn sàng tại: {stock_dir}")
    return check_alignment(stock_dir, [os.path.basename(f).split("_")[0].upper() for f in csv_files])

if __name__ == "__main__":
    main()
